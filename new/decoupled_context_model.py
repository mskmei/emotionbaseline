from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from cesa import ContextDecoupledEmotionSubspaceAdapter


class ResidualFeedForward(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.network = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.network(self.norm(values))


class DecoupledLocalContextEmotionNet(nn.Module):
    """Single-backbone visual ERC model with isolated local transfer."""

    def __init__(
        self,
        visual_dim: int,
        speaker_dim: int,
        n_classes: int,
        emotion_basis: torch.Tensor,
        hidden_dim: int = 256,
        adapter_dim: int = 128,
        n_heads: int = 4,
        transformer_layers: int = 2,
        gru_layers: int = 1,
        dropout: float = 0.3,
        adapter_dropout: float = 0.2,
        residual_scale: float = 1.0,
        projection: str = "emotion",
        use_adapter: bool = True,
        use_transformer: bool = True,
        use_multiscale_conv: bool = True,
        use_uncertainty_gate: bool = True,
        max_seq_len: int = 192,
    ):
        super().__init__()
        self.n_classes = int(n_classes)
        self.use_adapter = bool(use_adapter)
        self.use_transformer = bool(use_transformer)
        self.use_multiscale_conv = bool(use_multiscale_conv)
        self.use_uncertainty_gate = bool(use_uncertainty_gate)
        self.adapter = ContextDecoupledEmotionSubspaceAdapter(
            visual_dim,
            emotion_basis,
            bottleneck_dim=adapter_dim,
            dropout=adapter_dropout,
            residual_scale=residual_scale,
            projection=projection,
        )
        self.visual_projection = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.local_encoder = nn.Sequential(
            ResidualFeedForward(hidden_dim, dropout),
            ResidualFeedForward(hidden_dim, dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.speaker_projection = nn.Linear(speaker_dim, hidden_dim, bias=False)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
        self.context_gru = nn.GRU(
            hidden_dim,
            hidden_dim // 2,
            num_layers=gru_layers,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=False,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.conv5 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2, groups=hidden_dim)
        self.conv_mix = nn.Linear(hidden_dim * 2, hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.local_classifier = nn.Linear(hidden_dim, n_classes)
        self.context_classifier = nn.Linear(hidden_dim, n_classes)
        gate_input_dim = hidden_dim * 3 + n_classes * 2 + 3
        self.uncertainty_gate = nn.Sequential(
            nn.LayerNorm(gate_input_dim),
            nn.Linear(gate_input_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes),
        )
        nn.init.zeros_(self.uncertainty_gate[-1].weight)
        nn.init.constant_(self.uncertainty_gate[-1].bias, -2.0)
        self.shift_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _entropy(logits: torch.Tensor) -> torch.Tensor:
        probability = F.softmax(logits, dim=-1)
        return -(probability * probability.clamp_min(1e-8).log()).sum(dim=-1, keepdim=True) / math.log(logits.shape[-1])

    def forward(
        self,
        visual: torch.Tensor,
        speakers: torch.Tensor,
        umask: torch.Tensor,
        lengths: Sequence[int],
    ) -> Dict[str, torch.Tensor]:
        if self.use_adapter:
            adapted_visual, delta = self.adapter(visual)
        else:
            adapted_visual = visual
            delta = torch.zeros_like(visual)
        local = self.local_encoder(self.visual_projection(adapted_visual))
        time_steps = local.shape[0]
        positions = torch.arange(time_steps, device=local.device).clamp_max(self.position_embedding.num_embeddings - 1)
        contextual_input = local + self.speaker_projection(speakers) + self.position_embedding(positions)[:, None, :]
        packed = nn.utils.rnn.pack_padded_sequence(contextual_input, lengths, enforce_sorted=False)
        packed_context, _ = self.context_gru(packed)
        gru_context, _ = nn.utils.rnn.pad_packed_sequence(packed_context, total_length=time_steps)
        context = local + gru_context
        padding_mask = ~umask.to(dtype=torch.bool)
        if self.use_transformer:
            context = context + self.transformer(context, src_key_padding_mask=padding_mask)
        valid_mask = umask.transpose(0, 1).unsqueeze(-1).to(dtype=context.dtype)
        context = context * valid_mask
        if self.use_multiscale_conv:
            batch_first = context.permute(1, 2, 0)
            conv = torch.cat([self.conv3(batch_first), self.conv5(batch_first)], dim=1).permute(2, 0, 1)
            context = context + self.conv_mix(conv)
        context = self.context_norm(context) * valid_mask
        local_logits = self.local_classifier(local)
        context_logits = self.context_classifier(context)
        local_probability = F.softmax(local_logits, dim=-1)
        context_probability = F.softmax(context_logits, dim=-1)
        disagreement = torch.abs(local_probability - context_probability).mean(dim=-1, keepdim=True)
        gate_features = torch.cat(
            [
                local,
                context,
                torch.abs(local - context),
                local_probability,
                context_probability,
                self._entropy(local_logits),
                self._entropy(context_logits),
                disagreement,
            ],
            dim=-1,
        )
        if self.use_uncertainty_gate:
            gate = torch.sigmoid(self.uncertainty_gate(gate_features))
        else:
            gate = torch.zeros_like(local_logits)
        fused_probability = (1.0 - gate) * context_probability + gate * local_probability
        fused_probability = fused_probability / fused_probability.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        previous = context[:-1]
        current = context[1:]
        shift_features = torch.cat([previous, current, torch.abs(current - previous)], dim=-1)
        shift_logits = self.shift_head(shift_features).squeeze(-1)
        return {
            "log_prob": fused_probability.clamp_min(1e-8).log(),
            "local_log_prob": F.log_softmax(local_logits, dim=-1),
            "context_log_prob": F.log_softmax(context_logits, dim=-1),
            "local_embedding": local,
            "context_embedding": context,
            "gate": gate,
            "adapter_delta": delta,
            "shift_logits": shift_logits,
        }
