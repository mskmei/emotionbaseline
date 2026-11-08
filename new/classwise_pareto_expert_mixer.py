from __future__ import annotations

import torch
import torch.nn as nn


class ClasswiseParetoExpertMixer(nn.Module):
    """Mixes anchor, source-only, and joint experts independently by class."""

    def __init__(self, classwise_weights: torch.Tensor):
        super().__init__()
        weights = torch.as_tensor(classwise_weights, dtype=torch.float32)
        if weights.ndim != 2 or weights.shape[1] != 3:
            raise ValueError("classwise_weights must have shape [n_classes, 3]")
        weights = weights.clamp_min(0)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        self.register_buffer("classwise_weights", weights)

    def forward(
        self,
        anchor_log_prob: torch.Tensor,
        source_log_prob: torch.Tensor,
        joint_log_prob: torch.Tensor,
    ) -> torch.Tensor:
        expert_prob = torch.stack(
            [anchor_log_prob.exp(), source_log_prob.exp(), joint_log_prob.exp()], dim=-1
        )
        fused_prob = (expert_prob * self.classwise_weights.unsqueeze(0)).sum(dim=-1)
        fused_prob = fused_prob.clamp_min(1e-12)
        fused_prob = fused_prob / fused_prob.sum(dim=-1, keepdim=True)
        return fused_prob.log()
