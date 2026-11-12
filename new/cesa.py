from __future__ import annotations

from typing import Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


def compute_emotion_subspace(
    features: Mapping[str, Sequence],
    labels: Mapping[str, Sequence[int]],
    keys: Sequence[str],
    n_classes: int,
    rank: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build an orthonormal source-emotion basis from class-center offsets."""
    sums = None
    counts = np.zeros(n_classes, dtype=np.int64)
    for key in keys:
        values = np.asarray(features[key], dtype=np.float64)
        targets = np.asarray(labels[key], dtype=np.int64)
        if values.ndim != 2 or len(values) != len(targets):
            raise ValueError(f"Misaligned feature/label sequence for {key}")
        if sums is None:
            sums = np.zeros((n_classes, values.shape[-1]), dtype=np.float64)
        for class_id in range(n_classes):
            keep = targets == class_id
            if np.any(keep):
                sums[class_id] += values[keep].sum(axis=0)
                counts[class_id] += int(keep.sum())
    if sums is None or np.any(counts == 0):
        raise ValueError(f"Cannot compute class centers; counts={counts.tolist()}")

    centers = sums / counts[:, None]
    global_center = np.average(centers, axis=0, weights=counts)
    centered = centers - global_center[None, :]
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    effective_rank = min(int(rank), n_classes - 1, right.shape[0])
    basis = right[:effective_rank].T.copy()
    return (
        torch.tensor(basis, dtype=torch.float32),
        torch.tensor(centers, dtype=torch.float32),
        torch.tensor(counts, dtype=torch.long),
    )


def compute_projection_statistics(
    features: Mapping[str, Sequence],
    keys: Sequence[str],
    basis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return source mean/std in the low-rank emotion coordinate system."""
    basis_array = torch.as_tensor(basis, dtype=torch.float32).cpu().numpy().astype(np.float64)
    projected = []
    for key in keys:
        values = np.asarray(features[key], dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != basis_array.shape[0]:
            raise ValueError(f"Unexpected feature shape for {key}: {values.shape}")
        projected.append(values @ basis_array)
    if not projected:
        raise ValueError("Cannot compute projection statistics without source features")
    coordinates = np.concatenate(projected, axis=0)
    mean = coordinates.mean(axis=0)
    scale = np.maximum(coordinates.std(axis=0), 1e-4)
    return torch.tensor(mean, dtype=torch.float32), torch.tensor(scale, dtype=torch.float32)


def prototype_alignment_loss(
    adapted_features: torch.Tensor,
    labels: torch.Tensor,
    basis: torch.Tensor,
    class_centers: torch.Tensor,
    coordinate_mean: torch.Tensor,
    coordinate_scale: torch.Tensor,
) -> torch.Tensor:
    """Align labeled samples to source class prototypes in standardized emotion coordinates."""
    basis = basis.to(device=adapted_features.device, dtype=adapted_features.dtype)
    class_centers = class_centers.to(device=adapted_features.device, dtype=adapted_features.dtype)
    coordinate_mean = coordinate_mean.to(device=adapted_features.device, dtype=adapted_features.dtype)
    coordinate_scale = coordinate_scale.to(device=adapted_features.device, dtype=adapted_features.dtype)
    sample_coordinates = (adapted_features @ basis - coordinate_mean) / coordinate_scale
    center_coordinates = (class_centers @ basis - coordinate_mean) / coordinate_scale
    targets = center_coordinates.index_select(0, labels.long())
    return torch.nn.functional.smooth_l1_loss(sample_coordinates, targets)


class ContextDecoupledEmotionSubspaceAdapter(nn.Module):
    """Zero-initialized utterance adapter constrained to source emotion axes."""

    def __init__(
        self,
        input_dim: int,
        basis: torch.Tensor,
        bottleneck_dim: int = 128,
        dropout: float = 0.1,
        residual_scale: float = 1.0,
        projection: str = "emotion",
    ):
        super().__init__()
        if projection not in {"emotion", "full"}:
            raise ValueError(f"Unsupported projection={projection}")
        basis = torch.as_tensor(basis, dtype=torch.float32)
        if basis.ndim != 2 or basis.shape[0] != input_dim:
            raise ValueError(f"Expected basis [{input_dim}, rank], got {tuple(basis.shape)}")
        self.projection = projection
        self.residual_scale = float(residual_scale)
        self.register_buffer("emotion_basis", basis)
        self.norm = nn.LayerNorm(input_dim)
        self.down = nn.Linear(input_dim, bottleneck_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        output_dim = basis.shape[1] if projection == "emotion" else input_dim
        self.up = nn.Linear(bottleneck_dim, output_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, visual: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.dropout(self.activation(self.down(self.norm(visual))))
        raw = torch.tanh(self.up(hidden))
        if self.projection == "emotion":
            delta = torch.matmul(raw, self.emotion_basis.transpose(0, 1))
        else:
            delta = raw
        delta = self.residual_scale * delta
        return visual + delta, delta


class EmotionDecisionResidualAdapter(nn.Module):
    """Zero-initialized visual residual in the shared emotion-logit space."""

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        bottleneck_dim: int = 64,
        dropout: float = 0.1,
        residual_scale: float = 1.0,
    ):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.norm = nn.LayerNorm(input_dim)
        self.down = nn.Linear(input_dim, bottleneck_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck_dim, n_classes)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, visual: torch.Tensor) -> torch.Tensor:
        hidden = self.dropout(self.activation(self.down(self.norm(visual))))
        return self.residual_scale * torch.tanh(self.up(hidden))
