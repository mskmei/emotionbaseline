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
