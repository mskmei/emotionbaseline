from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class VisualLogitAdapter(nn.Module):
    """Predicts a bounded correction to a frozen classifier's logits."""

    def __init__(self, visual_dim: int, n_classes: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.visual_norm = nn.LayerNorm(visual_dim)
        self.projector = nn.Sequential(
            nn.Linear(visual_dim + n_classes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, n_classes)
        nn.init.zeros_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, visual: torch.Tensor, base_log_prob: torch.Tensor):
        features = torch.cat([self.visual_norm(visual), base_log_prob.detach()], dim=-1)
        embedding = self.projector(features)
        return self.classifier(embedding), embedding


class TransferReliabilityGate(nn.Module):
    def __init__(self, n_classes: int, hidden_dim: int, dropout: float, bias: float, classwise: bool):
        super().__init__()
        self.n_classes = int(n_classes)
        self.classwise = bool(classwise)
        output_dim = n_classes if classwise else 1
        self.network = nn.Sequential(
            nn.Linear(10, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.constant_(self.network[-1].bias, bias)

    @staticmethod
    def _entropy(log_prob: torch.Tensor) -> torch.Tensor:
        return -(log_prob.exp() * log_prob).sum(dim=-1, keepdim=True) / math.log(log_prob.shape[-1])

    def forward(
        self,
        base_log_prob: torch.Tensor,
        source_log_prob: torch.Tensor,
        bobsl_log_prob: torch.Tensor,
        utterance_lengths: torch.Tensor,
    ) -> torch.Tensor:
        log_probs = [value.detach() for value in (base_log_prob, source_log_prob, bobsl_log_prob)]
        probabilities = [value.exp() for value in log_probs]
        entropies = [self._entropy(value) for value in log_probs]
        confidences = [value.max(dim=-1, keepdim=True).values for value in probabilities]
        disagreements = [
            0.5 * torch.abs(probabilities[0] - probabilities[1]).sum(dim=-1, keepdim=True),
            0.5 * torch.abs(probabilities[0] - probabilities[2]).sum(dim=-1, keepdim=True),
            0.5 * torch.abs(probabilities[1] - probabilities[2]).sum(dim=-1, keepdim=True),
        ]
        length_feature = torch.log1p(utterance_lengths.float()).unsqueeze(1) / math.log(51.0)
        features = torch.cat([*entropies, *confidences, *disagreements, length_feature], dim=-1)
        gate = torch.sigmoid(self.network(features))
        if not self.classwise:
            gate = gate.expand(-1, self.n_classes)
        return gate


class FrozenLogitResidualTransfer(nn.Module):
    """A backbone-agnostic source adapter plus bounded BOBSL residual.

    The backbone only needs to provide class log-probabilities and a visual feature.
    Source adaptation and BOBSL transfer are deliberately separated so BOBSL cannot
    overwrite the source decision function.
    """

    def __init__(
        self,
        visual_dim: int,
        n_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        source_scale: float = 1.0,
        transfer_scale: float = 0.5,
        gate_hidden_dim: int = 32,
        gate_dropout: float = 0.1,
        gate_bias: float = -3.0,
        classwise_gate: bool = True,
    ):
        super().__init__()
        self.source_adapter = VisualLogitAdapter(visual_dim, n_classes, hidden_dim, dropout)
        self.bobsl_adapter = VisualLogitAdapter(visual_dim, n_classes, hidden_dim, dropout)
        self.transfer_gate = TransferReliabilityGate(
            n_classes, gate_hidden_dim, gate_dropout, gate_bias, classwise_gate
        )
        self.source_scale = float(source_scale)
        self.transfer_scale = float(transfer_scale)
        self.n_classes = int(n_classes)

    def initialize_bobsl_from_source(self) -> None:
        self.bobsl_adapter.load_state_dict(self.source_adapter.state_dict())

    def freeze_source(self) -> None:
        for parameter in self.source_adapter.parameters():
            parameter.requires_grad = False

    def forward(
        self,
        base_log_prob: torch.Tensor,
        visual: torch.Tensor,
        utterance_lengths: torch.Tensor,
        enable_transfer: bool,
        detach_transfer_delta: bool = False,
    ) -> Dict[str, torch.Tensor]:
        source_residual, source_embedding = self.source_adapter(visual, base_log_prob)
        bobsl_residual, bobsl_embedding = self.bobsl_adapter(visual, base_log_prob)
        source_logits = base_log_prob + self.source_scale * torch.tanh(source_residual)
        bobsl_logits = base_log_prob + self.source_scale * torch.tanh(bobsl_residual)
        source_log_prob = F.log_softmax(source_logits, dim=-1)
        bobsl_log_prob = F.log_softmax(bobsl_logits, dim=-1)
        gate = self.transfer_gate(base_log_prob, source_log_prob, bobsl_log_prob, utterance_lengths)
        if not enable_transfer:
            gate = torch.zeros_like(gate)
        delta = torch.tanh(bobsl_residual - source_residual)
        if detach_transfer_delta:
            delta = delta.detach()
        fused_logits = source_logits + self.transfer_scale * gate * delta
        return {
            "fused_log_prob": F.log_softmax(fused_logits, dim=-1),
            "base_log_prob": base_log_prob,
            "source_log_prob": source_log_prob,
            "bobsl_log_prob": bobsl_log_prob,
            "source_logits": source_logits,
            "bobsl_logits": bobsl_logits,
            "source_embedding": source_embedding,
            "bobsl_embedding": bobsl_embedding,
            "source_residual": source_residual,
            "bobsl_residual": bobsl_residual,
            "transfer_delta": delta,
            "gate": gate,
        }


class SafeTransferGate(nn.Module):
    """Selects bounded corrections from a frozen source/joint model pair."""

    def __init__(
        self,
        visual_dim: int,
        n_classes: int,
        hidden_dim: int = 32,
        dropout: float = 0.1,
        bias: float = -2.0,
        max_scale: float = 1.0,
        mode: str = "reliability",
        visual_gate_dim: int = 16,
        fusion_space: str = "probability",
        delta_clip: float = 6.0,
    ):
        super().__init__()
        if mode not in {"global", "reliability", "visual"}:
            raise ValueError(f"Unsupported safe gate mode={mode}")
        self.mode = mode
        self.n_classes = int(n_classes)
        self.max_scale = float(max_scale)
        if fusion_space not in {"probability", "logprob"}:
            raise ValueError(f"Unsupported fusion_space={fusion_space}")
        self.fusion_space = fusion_space
        self.delta_clip = float(delta_clip)
        self.global_logits = nn.Parameter(torch.full((n_classes,), float(bias)))
        base_dim = 3 * n_classes + 6
        self.visual_projector = None
        if mode == "visual":
            self.visual_projector = nn.Sequential(
                nn.LayerNorm(visual_dim),
                nn.Linear(visual_dim, visual_gate_dim),
                nn.GELU(),
            )
            base_dim += visual_gate_dim
        if mode != "global":
            self.network = nn.Sequential(
                nn.Linear(base_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_classes),
            )
            nn.init.zeros_(self.network[-1].weight)
            nn.init.constant_(self.network[-1].bias, bias)
        else:
            self.network = None

    @staticmethod
    def reliability_features(
        source_log_prob: torch.Tensor,
        joint_log_prob: torch.Tensor,
        utterance_lengths: torch.Tensor,
    ) -> torch.Tensor:
        source_log_prob = source_log_prob.detach()
        joint_log_prob = joint_log_prob.detach()
        source_prob = source_log_prob.exp()
        joint_prob = joint_log_prob.exp()
        normalizer = math.log(source_log_prob.shape[-1])
        source_entropy = -(source_prob * source_prob.clamp_min(1e-8).log()).sum(dim=-1, keepdim=True) / normalizer
        joint_entropy = -(joint_prob * joint_prob.clamp_min(1e-8).log()).sum(dim=-1, keepdim=True) / normalizer
        source_confidence = source_prob.max(dim=-1, keepdim=True).values
        joint_confidence = joint_prob.max(dim=-1, keepdim=True).values
        disagreement = 0.5 * torch.abs(source_prob - joint_prob).sum(dim=-1, keepdim=True)
        length_feature = torch.log1p(utterance_lengths.float()).unsqueeze(1) / math.log(51.0)
        return torch.cat(
            [
                source_prob,
                joint_prob,
                torch.abs(source_prob - joint_prob),
                source_entropy,
                joint_entropy,
                source_confidence,
                joint_confidence,
                disagreement,
                length_feature,
            ],
            dim=-1,
        )

    def forward(
        self,
        source_log_prob: torch.Tensor,
        joint_log_prob: torch.Tensor,
        visual: torch.Tensor,
        utterance_lengths: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        source_finite = torch.isfinite(source_log_prob).all(dim=-1, keepdim=True)
        joint_finite = torch.isfinite(joint_log_prob).all(dim=-1, keepdim=True)
        source_clean = torch.nan_to_num(source_log_prob, nan=-30.0, posinf=0.0, neginf=-30.0)
        joint_clean = torch.nan_to_num(joint_log_prob, nan=-30.0, posinf=0.0, neginf=-30.0)
        source_clean = F.log_softmax(source_clean, dim=-1)
        joint_clean = F.log_softmax(joint_clean, dim=-1)
        source_clean = torch.where(source_finite, source_clean, joint_clean)
        joint_clean = torch.where(joint_finite, joint_clean, source_clean)
        if self.mode == "global":
            logits = self.global_logits.unsqueeze(0).expand(source_log_prob.shape[0], -1)
        else:
            features = self.reliability_features(source_clean, joint_clean, utterance_lengths)
            if self.visual_projector is not None:
                features = torch.cat([features, self.visual_projector(visual.detach())], dim=-1)
            logits = self.network(features)
        gate = torch.sigmoid(logits)
        if self.fusion_space == "probability":
            source_prob = source_clean.exp()
            joint_prob = joint_clean.exp()
            fused_prob = source_prob + self.max_scale * gate * (joint_prob - source_prob)
            fused_prob = fused_prob.clamp_min(1e-8)
            fused_prob = fused_prob / fused_prob.sum(dim=-1, keepdim=True)
            fused_log_prob = fused_prob.log()
        else:
            delta = (joint_clean - source_clean).clamp(-self.delta_clip, self.delta_clip)
            fused_log_prob = F.log_softmax(source_clean + self.max_scale * gate * delta, dim=-1)
        return {
            "fused_log_prob": fused_log_prob,
            "source_log_prob": source_clean,
            "joint_log_prob": joint_clean,
            "gate": gate,
        }
