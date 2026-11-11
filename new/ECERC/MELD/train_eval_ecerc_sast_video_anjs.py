#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from itertools import cycle
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
NEW_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(NEW_ROOT))

import train_eval_ecerc_video_anjs as base  # noqa: E402
from anjs_video_common import ANJS_LABELS  # noqa: E402
from cesa import (  # noqa: E402
    ContextDecoupledEmotionSubspaceAdapter,
    compute_emotion_subspace,
    compute_projection_statistics,
    prototype_alignment_loss,
)


def is_bobsl_key(key: str) -> bool:
    value = str(key)
    return value.startswith("joint_bobsl") or value.startswith("bobsl")


def make_dataset(path: Path, split: str, domain: str = "all") -> base.ECERCFeatureDataset:
    dataset = base.ECERCFeatureDataset(path, split)
    if domain == "source":
        dataset.keys = [key for key in dataset.keys if not is_bobsl_key(key)]
    elif domain == "bobsl":
        dataset.keys = [key for key in dataset.keys if is_bobsl_key(key)]
    elif domain != "all":
        raise ValueError(f"Unsupported domain={domain}")
    return dataset


def make_loader(dataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=base.ECERCFeatureDataset.collate_fn,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def flatten_seq(values: torch.Tensor, lengths: Sequence[int]) -> torch.Tensor:
    return torch.cat([values[:length, index] for index, length in enumerate(lengths)], dim=0)


def flatten_labels(labels: torch.Tensor, lengths: Sequence[int]) -> torch.Tensor:
    return torch.cat([labels[index, :length] for index, length in enumerate(lengths)], dim=0)


class SASTECERC(nn.Module):
    """Context-decoupled emotion-subspace adapter on a frozen ECERC backbone."""

    def __init__(self, backbone: nn.Module, visual_dim: int, basis: torch.Tensor, args):
        super().__init__()
        self.backbone = backbone
        self.adapter = ContextDecoupledEmotionSubspaceAdapter(
            visual_dim,
            basis,
            bottleneck_dim=args.adapter_dim,
            dropout=args.adapter_dropout,
            residual_scale=args.residual_scale,
            projection=args.projection,
        )

    def forward(self, emo_text, sem_text, visual, audio, qmask, umask, lengths, adapted: bool):
        emo_text = torch.zeros_like(emo_text)
        sem_text = torch.zeros_like(sem_text)
        audio = torch.zeros_like(audio)
        if adapted:
            visual, delta = self.adapter(visual)
        else:
            delta = torch.zeros_like(visual)
        evidence = torch.cat([emo_text, audio, visual], dim=-1)
        return self.backbone(evidence, sem_text, qmask, umask, lengths), delta


def adapter_parameters(model: SASTECERC) -> Iterable[nn.Parameter]:
    yield from model.adapter.parameters()


def load_backbone(args, info: Dict, checkpoint_path: Path, device: torch.device) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    for key in ("base_layer", "hidden_size"):
        if key in checkpoint_args:
            setattr(args, key, checkpoint_args[key])
    model = base.build_model(args, info, device)
    state = checkpoint.get("model_state_dict", checkpoint)
    clean = {key.removeprefix("module."): value for key, value in state.items()}
    incompatible = model.load_state_dict(clean, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={incompatible.missing_keys[:10]} "
            f"unexpected={incompatible.unexpected_keys[:10]}"
        )
    return model


def prepare_batch(data, device: torch.device):
    emo_text, sem_text, visual, audio, qmask, umask, labels = [item.to(device) for item in data[:-1]]
    lengths = base.sequence_lengths(umask)
    labels_flat = flatten_labels(labels, lengths)
    valid = labels_flat.ge(0)
    return emo_text, sem_text, visual, audio, qmask, umask, labels, lengths, labels_flat, valid


def result_from_arrays(losses, counts, golds, preds, probs, ids) -> Optional[Dict]:
    if not golds:
        return None
    from sklearn import metrics

    gold = np.asarray(golds, dtype=np.int64)
    pred = np.asarray(preds, dtype=np.int64)
    probability = np.concatenate(probs, axis=0)
    label_ids = list(range(len(ANJS_LABELS)))
    return {
        "loss": float(np.sum(losses) / max(np.sum(counts), 1)),
        "accuracy": float(metrics.accuracy_score(gold, pred)),
        "macro_f1": float(metrics.f1_score(gold, pred, labels=label_ids, average="macro", zero_division=0)),
        "weighted_f1": float(metrics.f1_score(gold, pred, labels=label_ids, average="weighted", zero_division=0)),
        "golds": gold,
        "preds": pred,
        "probs": probability,
        "ids": ids,
    }


@torch.no_grad()
def evaluate(model: SASTECERC, loader: DataLoader, device: torch.device, adapted: bool) -> Optional[Dict]:
    model.eval()
    losses: List[float] = []
    counts: List[int] = []
    golds: List[int] = []
    preds: List[int] = []
    probs: List[np.ndarray] = []
    ids: List[str] = []
    for data in loader:
        emo, sem, visual, audio, qmask, umask, labels, lengths, labels_flat, valid = prepare_batch(data, device)
        log_prob, _ = model(emo, sem, visual, audio, qmask, umask, lengths, adapted=adapted)
        current = log_prob[valid]
        target = labels_flat[valid]
        if not len(target):
            continue
        loss = F.nll_loss(current, target)
        losses.append(float(loss.item()) * len(target))
        counts.append(len(target))
        golds.extend(target.cpu().numpy().astype(int).tolist())
        preds.extend(current.argmax(dim=-1).cpu().numpy().astype(int).tolist())
        probs.append(current.exp().cpu().numpy().astype(np.float32))
        labels_cpu = labels.cpu()
        for batch_index, key in enumerate(data[-1]):
            for utterance_index in range(lengths[batch_index]):
                if int(labels_cpu[batch_index, utterance_index]) >= 0:
                    ids.append(f"{key}:utt{utterance_index + 1:02d}")
    return result_from_arrays(losses, counts, golds, preds, probs, ids)


def weighted_nll(log_prob, labels, weights, reduction="mean"):
    return F.nll_loss(log_prob, labels, weight=weights, reduction=reduction)


def train_epoch(
    model: SASTECERC,
    source_loader: DataLoader,
    bobsl_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    weights: Optional[torch.Tensor],
    basis: torch.Tensor,
    centers: torch.Tensor,
    coordinate_mean: torch.Tensor,
    coordinate_scale: torch.Tensor,
    device: torch.device,
    args,
) -> Dict[str, float]:
    model.backbone.eval()
    model.adapter.train()
    bobsl_iter = cycle(bobsl_loader)
    totals = {key: 0.0 for key in ("loss", "source", "bobsl", "safety", "stability", "prototype", "residual")}
    steps = 0
    for source_data in source_loader:
        optimizer.zero_grad(set_to_none=True)
        emo, sem, visual, audio, qmask, umask, _, lengths, labels, valid = prepare_batch(source_data, device)
        with torch.no_grad():
            base_log_prob, _ = model(emo, sem, visual, audio, qmask, umask, lengths, adapted=False)
        adapted_log_prob, source_delta = model(emo, sem, visual, audio, qmask, umask, lengths, adapted=True)
        targets = labels[valid]
        source_nll = weighted_nll(adapted_log_prob[valid], targets, weights, reduction="none")
        base_nll = weighted_nll(base_log_prob[valid], targets, weights, reduction="none")
        source_loss = source_nll.mean()
        safety_loss = F.relu(source_nll - base_nll + args.safety_margin).mean()
        stability_loss = F.kl_div(adapted_log_prob[valid], base_log_prob[valid].exp(), reduction="batchmean")
        source_delta_flat = flatten_seq(source_delta, lengths)[valid]
        residual_loss = source_delta_flat.pow(2).sum(dim=-1).mean()

        bobsl_data = next(bobsl_iter)
        b_emo, b_sem, b_visual, b_audio, b_qmask, b_umask, _, b_lengths, b_labels, b_valid = prepare_batch(
            bobsl_data, device
        )
        bobsl_log_prob, bobsl_delta = model(
            b_emo, b_sem, b_visual, b_audio, b_qmask, b_umask, b_lengths, adapted=True
        )
        bobsl_targets = b_labels[b_valid]
        bobsl_loss = weighted_nll(bobsl_log_prob[b_valid], bobsl_targets, weights)
        bobsl_features = flatten_seq(b_visual + bobsl_delta, b_lengths)[b_valid]
        prototype_loss = prototype_alignment_loss(
            bobsl_features, bobsl_targets, basis, centers, coordinate_mean, coordinate_scale
        )
        bobsl_delta_flat = flatten_seq(bobsl_delta, b_lengths)[b_valid]
        residual_loss = residual_loss + bobsl_delta_flat.pow(2).sum(dim=-1).mean()

        loss = (
            args.source_weight * source_loss
            + args.bobsl_weight * bobsl_loss
            + args.safety_weight * safety_loss
            + args.stability_weight * stability_loss
            + args.prototype_weight * prototype_loss
            + args.residual_weight * residual_loss
        )
        loss.backward()
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(list(adapter_parameters(model)), args.max_grad_norm)
        optimizer.step()
        values = {
            "loss": loss,
            "source": source_loss,
            "bobsl": bobsl_loss,
            "safety": safety_loss,
            "stability": stability_loss,
            "prototype": prototype_loss,
            "residual": residual_loss,
        }
        for key, value in values.items():
            totals[key] += float(value.detach().item())
        steps += 1
    return {key: value / max(steps, 1) for key, value in totals.items()}


def save_checkpoint(path: Path, model: SASTECERC, args, info, basis_meta, epoch, score):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "method": "SAST-ECERC",
            "label_names": ANJS_LABELS,
            "args": vars(args),
            "info": info,
            "basis_meta": basis_meta,
            "epoch": epoch,
            "source_score": score,
        },
        path,
    )


def train(args) -> None:
    base.seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    if device.type != "cuda":
        raise RuntimeError("SAST-ECERC requires CUDA")
    train_pkl = base.resolve_path(args.train_pkl, must_exist=True)
    external_pkl = base.resolve_path(args.external_test_pkl, must_exist=True)
    checkpoint_path = base.resolve_path(args.base_checkpoint, must_exist=True)
    out_dir = base.resolve_path(args.out_dir, must_exist=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = base.inspect_pkl(train_pkl)
    payload = base.load_payload(train_pkl)
    train_keys = list(payload["train_ids"])
    source_keys = [key for key in train_keys if not is_bobsl_key(key)]
    bobsl_keys = [key for key in train_keys if is_bobsl_key(key)]
    if not bobsl_keys:
        raise RuntimeError("SAST-ECERC requires a joint source/BOBSL pkl")
    basis, centers, counts = compute_emotion_subspace(
        payload["video_visual"], payload["emotion_labels"], source_keys, len(ANJS_LABELS), rank=args.subspace_rank
    )
    coordinate_mean, coordinate_scale = compute_projection_statistics(payload["video_visual"], source_keys, basis)
    basis_meta = {
        "rank": int(basis.shape[1]),
        "projection": args.projection,
        "source_counts": counts.tolist(),
        "coordinate_scale": coordinate_scale.tolist(),
        "source_dialogues": len(source_keys),
        "bobsl_dialogues": len(bobsl_keys),
    }

    backbone = load_backbone(args, info, checkpoint_path, device)
    model = SASTECERC(backbone, int(info["d_visual"]), basis, args).to(device)
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    source_train = make_dataset(train_pkl, "train", "source")
    bobsl_train = make_dataset(train_pkl, "train", "bobsl")
    source_test = make_dataset(train_pkl, "test")
    external_test = make_dataset(external_pkl, "test")
    source_loader = make_loader(source_train, args.source_batch_size, True, args.seed)
    bobsl_loader = make_loader(bobsl_train, args.bobsl_batch_size, True, args.seed + 17)
    source_test_loader = make_loader(source_test, args.eval_batch_size, False, args.seed)
    external_test_loader = make_loader(external_test, args.eval_batch_size, False, args.seed)
    source_labels = [int(label) for key in source_keys for label in payload["emotion_labels"][key] if int(label) >= 0]
    weights = base.make_class_weights(source_labels, len(ANJS_LABELS), device) if args.class_weight else None
    if weights is not None and args.class_weight_power != 1.0:
        weights = weights.pow(args.class_weight_power)
        weights = weights / weights.mean().clamp_min(1e-8)
    optimizer = torch.optim.AdamW(adapter_parameters(model), lr=args.lr, weight_decay=args.weight_decay)

    base_source = evaluate(model, source_test_loader, device, adapted=False)
    base_external = evaluate(model, external_test_loader, device, adapted=False)
    best_source = copy.deepcopy(base_source)
    best_external = copy.deepcopy(base_external)
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_score = float(base_source[args.selection_metric])
    rows: List[Dict] = []
    trainable = sum(parameter.numel() for parameter in adapter_parameters(model))
    print(f"[SAST-ECERC] source={len(source_train)} bobsl={len(bobsl_train)}")
    print(f"[SAST-ECERC] base source={base_source['weighted_f1']:.4f} external={base_external['weighted_f1']:.4f}")
    print(f"[SAST-ECERC] basis={json.dumps(basis_meta)} trainable={trainable}")

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_metrics = train_epoch(
            model, source_loader, bobsl_loader, optimizer, weights, basis, centers,
            coordinate_mean, coordinate_scale, device, args
        )
        source_result = evaluate(model, source_test_loader, device, adapted=True)
        external_result = evaluate(model, external_test_loader, device, adapted=True)
        score = float(source_result[args.selection_metric])
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_source = copy.deepcopy(source_result)
            best_external = copy.deepcopy(external_result)
            best_state = copy.deepcopy(model.state_dict())
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            "source_weighted_f1": source_result["weighted_f1"],
            "external_weighted_f1": external_result["weighted_f1"],
            "delta_source": source_result["weighted_f1"] - base_source["weighted_f1"],
            "delta_external": external_result["weighted_f1"] - base_external["weighted_f1"],
            "time_sec": round(time.time() - start, 2),
        }
        rows.append(row)
        print(
            f"[SAST-ECERC] epoch={epoch} source={row['source_weighted_f1']:.4f} ({row['delta_source']:+.4f}) "
            f"external={row['external_weighted_f1']:.4f} ({row['delta_external']:+.4f}) "
            f"loss={train_metrics['loss']:.4f} time={row['time_sec']}s"
        )

    with (out_dir / "epoch_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    common = {
        "script": "new/ECERC/MELD/train_eval_ecerc_sast_video_anjs.py",
        "method": "SAST-ECERC",
        "seed": args.seed,
        "train_pkl": str(train_pkl),
        "external_test_pkl": str(external_pkl),
        "base_checkpoint": str(checkpoint_path),
        "basis_meta": basis_meta,
        "best_epoch": best_epoch,
        "base_source_weighted_f1": base_source["weighted_f1"],
        "base_external_weighted_f1": base_external["weighted_f1"],
    }
    base.save_report(base_source, out_dir, "source_test_base", {**common, "selection": "base"})
    base.save_report(base_external, out_dir, "external_test_base", {**common, "selection": "base"})
    base.save_report(best_source, out_dir, "source_test_best", {**common, "selection": "source"})
    base.save_report(best_external, out_dir, "external_test_best", {**common, "selection": "source"})
    model.load_state_dict(best_state)
    save_checkpoint(out_dir / "model_best.pt", model, args, info, basis_meta, best_epoch, best_score)
    summary = {
        **common,
        "base_source": base_source["weighted_f1"],
        "base_external": base_external["weighted_f1"],
        "best_source": best_source["weighted_f1"],
        "best_external": best_external["weighted_f1"],
        "delta_source": best_source["weighted_f1"] - base_source["weighted_f1"],
        "delta_external": best_external["weighted_f1"] - base_external["weighted_f1"],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"[SAST-ECERC] done epoch={best_epoch} source={best_source['weighted_f1']:.4f} "
        f"external={best_external['weighted_f1']:.4f}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Single-checkpoint SAST adapter for ECERC video models.")
    parser.add_argument("--train_pkl", required=True)
    parser.add_argument("--external_test_pkl", required=True)
    parser.add_argument("--base_checkpoint", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--source_batch_size", type=int, default=8)
    parser.add_argument("--bobsl_batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--adapter_dim", type=int, default=128)
    parser.add_argument("--adapter_dropout", type=float, default=0.2)
    parser.add_argument("--subspace_rank", type=int, default=3)
    parser.add_argument("--projection", choices=["emotion", "full"], default="emotion")
    parser.add_argument("--residual_scale", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--source_weight", type=float, default=1.0)
    parser.add_argument("--bobsl_weight", type=float, default=1.0)
    parser.add_argument("--safety_weight", type=float, default=0.5)
    parser.add_argument("--safety_margin", type=float, default=0.0)
    parser.add_argument("--stability_weight", type=float, default=0.1)
    parser.add_argument("--prototype_weight", type=float, default=0.1)
    parser.add_argument("--residual_weight", type=float, default=1e-3)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--class_weight", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--class_weight_power",
        type=float,
        default=1.0,
        help="Exponent for smoothly interpolating between uniform (0) and inverse-frequency (1) weights.",
    )
    parser.add_argument("--selection_metric", choices=["weighted_f1", "macro_f1"], default="weighted_f1")
    parser.add_argument("--base_layer", type=int, default=1)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--no_cuda", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
