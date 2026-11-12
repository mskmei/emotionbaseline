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
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(REPO_ROOT / "new"))

import train_eval_mmgcn_unified as base  # noqa: E402
from cesa import (  # noqa: E402
    ContextDecoupledEmotionSubspaceAdapter,
    EmotionDecisionResidualAdapter,
    compute_emotion_subspace,
    compute_projection_statistics,
    prototype_alignment_loss,
)
from dataloader import MELDDataset  # noqa: E402


def is_bobsl_key(key: str) -> bool:
    value = str(key)
    return value.startswith("joint_bobsl") or value.startswith("bobsl")


def make_dataset(path: Path, train: bool, domain: str = "all") -> MELDDataset:
    dataset = MELDDataset(str(path), train=train)
    if domain == "source":
        dataset.keys = [key for key in dataset.keys if not is_bobsl_key(key)]
    elif domain == "bobsl":
        dataset.keys = [key for key in dataset.keys if is_bobsl_key(key)]
    elif domain != "all":
        raise ValueError(f"Unsupported domain={domain}")
    dataset.len = len(dataset.keys)
    return dataset


def make_loader(dataset: MELDDataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=dataset.collate_fn,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def flatten_seq(values: torch.Tensor, lengths: Sequence[int]) -> torch.Tensor:
    return torch.cat([values[:length, index] for index, length in enumerate(lengths)], dim=0)


def flatten_labels(labels: torch.Tensor, lengths: Sequence[int]) -> torch.Tensor:
    return torch.cat([labels[index, :length] for index, length in enumerate(lengths)], dim=0)


class SASTMMGCN(nn.Module):
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
        self.decision_adapter = (
            EmotionDecisionResidualAdapter(
                visual_dim,
                args.n_classes,
                bottleneck_dim=args.decision_adapter_dim,
                dropout=args.adapter_dropout,
                residual_scale=args.decision_residual_scale,
            )
            if args.decision_residual else None
        )

    def forward(self, text, visual, audio, qmask, umask, lengths, adapted: bool):
        text = torch.zeros_like(text)
        audio = torch.zeros_like(audio)
        if adapted:
            visual, delta = self.adapter(visual)
        else:
            delta = torch.zeros_like(visual)
        log_prob, *_ = self.backbone(text, qmask.clone(), umask, lengths, audio, visual)
        if adapted and self.decision_adapter is not None:
            decision_delta = flatten_seq(self.decision_adapter(visual), lengths)
            log_prob = F.log_softmax(log_prob + decision_delta, dim=-1)
        return log_prob, delta


def load_backbone(args, info: Dict, checkpoint_path: Path, device: torch.device):
    checkpoint = base.load_torch_checkpoint(checkpoint_path, device)
    checkpoint_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    architecture_keys = [
        "base_model", "dropout", "graph_type", "graph_construct", "mm_fusion_mthd",
        "deep_gcn_nlayers", "windowp", "windowf", "attention", "alpha", "multiheads",
        "nodal_attention", "active_listener", "use_gcn", "use_topic", "use_speaker",
        "use_modal", "av_using_lstm", "no_residue", "d_g", "d_p", "d_e", "d_h",
        "d_a", "graph_hidden", "max_seq_len",
    ]
    for key in architecture_keys:
        if key in checkpoint_args:
            setattr(args, key, checkpoint_args[key])
    args.graph_type = str(checkpoint.get("graph_type", args.graph_type))
    backbone, modals, graph_type = base.build_model(args, info, "video", device)
    state = base.extract_state_dict(checkpoint)
    clean = {key.removeprefix("module."): value for key, value in state.items()}
    incompatible = backbone.load_state_dict(clean, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={incompatible.missing_keys[:10]} "
            f"unexpected={incompatible.unexpected_keys[:10]}"
        )
    return backbone, checkpoint, modals, graph_type


def prepare_batch(data, device: torch.device):
    text, visual, audio, qmask, umask, labels = [item.to(device) for item in data[:-1]]
    lengths = base.sequence_lengths(umask)
    labels_flat = flatten_labels(labels, lengths)
    valid = labels_flat.ge(0)
    return text, visual, audio, qmask, umask, labels, lengths, labels_flat, valid


def metrics_from_arrays(losses, counts, golds, preds, probs, ids) -> Optional[Dict]:
    if not golds:
        return None
    from sklearn import metrics

    gold = np.asarray(golds, dtype=np.int64)
    pred = np.asarray(preds, dtype=np.int64)
    probability = np.concatenate(probs, axis=0)
    label_ids = list(range(probability.shape[1]))
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
def evaluate(model: SASTMMGCN, loader: DataLoader, device: torch.device, adapted: bool) -> Optional[Dict]:
    model.eval()
    losses: List[float] = []
    counts: List[int] = []
    golds: List[int] = []
    preds: List[int] = []
    probs: List[np.ndarray] = []
    ids: List[str] = []
    for data in loader:
        text, visual, audio, qmask, umask, labels, lengths, labels_flat, valid = prepare_batch(data, device)
        log_prob, _ = model(text, visual, audio, qmask, umask, lengths, adapted=adapted)
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
    return metrics_from_arrays(losses, counts, golds, preds, probs, ids)


def weighted_nll(log_prob, labels, weights, reduction="mean"):
    return F.nll_loss(log_prob, labels, weight=weights, reduction=reduction)


def adapter_parameters(model: SASTMMGCN) -> Iterable[nn.Parameter]:
    yield from model.adapter.parameters()
    if model.decision_adapter is not None:
        yield from model.decision_adapter.parameters()


def joint_teacher_loss(
    student_log_prob: torch.Tensor,
    teacher_log_prob: torch.Tensor,
    base_log_prob: torch.Tensor,
    targets: torch.Tensor,
    args,
) -> torch.Tensor:
    temperature = args.joint_distill_temperature
    per_sample = F.kl_div(
        F.log_softmax(student_log_prob / temperature, dim=-1),
        F.softmax(teacher_log_prob / temperature, dim=-1),
        reduction="none",
    ).sum(dim=-1) * (temperature * temperature)
    keep = torch.ones_like(targets, dtype=torch.bool)
    if args.joint_teacher_correct_only:
        keep &= teacher_log_prob.argmax(dim=-1).eq(targets)
    if args.joint_teacher_better_only:
        teacher_nll = F.nll_loss(teacher_log_prob, targets, reduction="none")
        base_nll = F.nll_loss(base_log_prob, targets, reduction="none")
        keep &= teacher_nll + args.joint_teacher_advantage_margin < base_nll
    if args.joint_teacher_confidence > 0:
        keep &= teacher_log_prob.exp().amax(dim=-1) >= args.joint_teacher_confidence
    if not torch.any(keep):
        return per_sample.new_zeros(())
    return per_sample[keep].mean()


def train_epoch(
    model: SASTMMGCN,
    joint_teacher: Optional[SASTMMGCN],
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
    totals = {
        key: 0.0
        for key in ("loss", "source", "bobsl", "safety", "stability", "joint", "prototype", "residual")
    }
    steps = 0
    for source_data in source_loader:
        optimizer.zero_grad(set_to_none=True)
        text, visual, audio, qmask, umask, _, lengths, labels, valid = prepare_batch(source_data, device)
        with torch.no_grad():
            base_log_prob, _ = model(text, visual, audio, qmask, umask, lengths, adapted=False)
        adapted_log_prob, source_delta = model(text, visual, audio, qmask, umask, lengths, adapted=True)
        targets = labels[valid]
        source_nll = weighted_nll(adapted_log_prob[valid], targets, weights, reduction="none")
        base_nll = weighted_nll(base_log_prob[valid], targets, weights, reduction="none")
        source_loss = source_nll.mean()
        safety_loss = F.relu(source_nll - base_nll + args.safety_margin).mean()
        stability_loss = F.kl_div(
            adapted_log_prob[valid], base_log_prob[valid].exp(), reduction="batchmean"
        )
        if joint_teacher is not None and args.joint_distill_weight > 0:
            with torch.no_grad():
                teacher_log_prob, _ = joint_teacher(
                    text, visual, audio, qmask, umask, lengths, adapted=False
                )
            joint_loss = joint_teacher_loss(
                adapted_log_prob[valid], teacher_log_prob[valid], base_log_prob[valid], targets, args
            )
        else:
            joint_loss = source_loss.new_zeros(())
        source_delta_flat = flatten_seq(source_delta, lengths)[valid]
        residual_loss = source_delta_flat.pow(2).sum(dim=-1).mean()

        bobsl_data = next(bobsl_iter)
        b_text, b_visual, b_audio, b_qmask, b_umask, _, b_lengths, b_labels, b_valid = prepare_batch(bobsl_data, device)
        bobsl_log_prob, bobsl_delta = model(
            b_text, b_visual, b_audio, b_qmask, b_umask, b_lengths, adapted=True
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
            + args.joint_distill_weight * joint_loss
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
            "joint": joint_loss,
            "prototype": prototype_loss,
            "residual": residual_loss,
        }
        for key, value in values.items():
            totals[key] += float(value.detach().item())
        steps += 1
    return {key: value / max(steps, 1) for key, value in totals.items()}


def save_checkpoint(path: Path, model: SASTMMGCN, args, info, basis_meta, epoch, score):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "method": "SAST-MMGCN",
            "label_names": info["label_names"],
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
        raise RuntimeError("SAST-MMGCN requires CUDA")
    train_pkl = base.resolve_path(args.train_pkl, must_exist=True)
    external_pkl = base.resolve_path(args.external_test_pkl, must_exist=True)
    checkpoint_path = base.resolve_path(args.base_checkpoint, must_exist=True)
    joint_checkpoint_path = (
        base.resolve_path(args.joint_teacher_checkpoint, must_exist=True)
        if args.joint_teacher_checkpoint else None
    )
    out_dir = base.resolve_path(args.out_dir, must_exist=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = base.inspect_feature_pkl(train_pkl)
    args.n_classes = int(info["n_classes"])
    payload = base.load_feature_tuple(train_pkl)
    labels_by_key, visual_by_key, train_keys = payload[2], payload[5], list(payload[7])
    source_keys = [key for key in train_keys if not is_bobsl_key(key)]
    bobsl_keys = [key for key in train_keys if is_bobsl_key(key)]
    if not bobsl_keys:
        raise RuntimeError("SAST-MMGCN requires a joint source/BOBSL pkl")
    basis, centers, counts = compute_emotion_subspace(
        visual_by_key, labels_by_key, source_keys, int(info["n_classes"]), rank=args.subspace_rank
    )
    coordinate_mean, coordinate_scale = compute_projection_statistics(visual_by_key, source_keys, basis)
    basis_meta = {
        "rank": int(basis.shape[1]),
        "projection": args.projection,
        "source_counts": counts.tolist(),
        "coordinate_scale": coordinate_scale.tolist(),
        "source_dialogues": len(source_keys),
        "bobsl_dialogues": len(bobsl_keys),
    }

    backbone, _, modals, graph_type = load_backbone(args, info, checkpoint_path, device)
    model = SASTMMGCN(backbone, int(info["d_visual"]), basis, args).to(device)
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    joint_teacher = None
    if joint_checkpoint_path is not None:
        teacher_args = copy.deepcopy(args)
        teacher_backbone, _, _, _ = load_backbone(teacher_args, info, joint_checkpoint_path, device)
        joint_teacher = SASTMMGCN(teacher_backbone, int(info["d_visual"]), basis, teacher_args).to(device)
        for parameter in joint_teacher.parameters():
            parameter.requires_grad = False
        joint_teacher.eval()
    source_train = make_dataset(train_pkl, True, "source")
    bobsl_train = make_dataset(train_pkl, True, "bobsl")
    source_test = make_dataset(train_pkl, False)
    external_test = make_dataset(external_pkl, False)
    source_loader = make_loader(source_train, args.source_batch_size, True, args.seed)
    bobsl_loader = make_loader(bobsl_train, args.bobsl_batch_size, True, args.seed + 17)
    source_test_loader = make_loader(source_test, args.eval_batch_size, False, args.seed)
    external_test_loader = make_loader(external_test, args.eval_batch_size, False, args.seed)
    source_labels = [int(label) for key in source_keys for label in labels_by_key[key] if int(label) >= 0]
    weights = (
        base.make_class_weights(source_labels, int(info["n_classes"]), device)
        if args.class_weight else None
    )
    if weights is not None and args.class_weight_power != 1.0:
        weights = weights.pow(args.class_weight_power)
        weights = weights / weights.mean().clamp_min(1e-8)
    optimizer = torch.optim.AdamW(adapter_parameters(model), lr=args.lr, weight_decay=args.weight_decay)

    base_source = evaluate(model, source_test_loader, device, adapted=False)
    base_external = evaluate(model, external_test_loader, device, adapted=False)
    joint_source = evaluate(joint_teacher, source_test_loader, device, adapted=False) if joint_teacher else None
    joint_external = evaluate(joint_teacher, external_test_loader, device, adapted=False) if joint_teacher else None
    best_source = copy.deepcopy(base_source)
    best_external = copy.deepcopy(base_external)
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_score = float(base_source[args.selection_metric])
    oracle_source = copy.deepcopy(base_source)
    oracle_external = copy.deepcopy(base_external)
    oracle_state = copy.deepcopy(best_state)
    oracle_epoch = 0
    oracle_score = float(base_external["weighted_f1"])
    rows: List[Dict] = []
    trainable = sum(parameter.numel() for parameter in adapter_parameters(model))
    print(f"[SAST-MMGCN] source={len(source_train)} bobsl={len(bobsl_train)} graph={graph_type}/{modals}")
    print(f"[SAST-MMGCN] base source={base_source['weighted_f1']:.4f} external={base_external['weighted_f1']:.4f}")
    if joint_source is not None:
        print(
            f"[SAST-MMGCN] joint teacher source={joint_source['weighted_f1']:.4f} "
            f"external={joint_external['weighted_f1']:.4f}"
        )
    print(f"[SAST-MMGCN] basis={json.dumps(basis_meta)} trainable={trainable}")

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_metrics = train_epoch(
            model, joint_teacher, source_loader, bobsl_loader, optimizer, weights, basis, centers,
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
        if (
            source_result["weighted_f1"] >= base_source["weighted_f1"] - args.oracle_source_tolerance
            and external_result["weighted_f1"] > oracle_score
        ):
            oracle_score = float(external_result["weighted_f1"])
            oracle_epoch = epoch
            oracle_source = copy.deepcopy(source_result)
            oracle_external = copy.deepcopy(external_result)
            oracle_state = copy.deepcopy(model.state_dict())
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
            f"[SAST-MMGCN] epoch={epoch} source={row['source_weighted_f1']:.4f} ({row['delta_source']:+.4f}) "
            f"external={row['external_weighted_f1']:.4f} ({row['delta_external']:+.4f}) "
            f"loss={train_metrics['loss']:.4f} time={row['time_sec']}s"
        )

    with (out_dir / "epoch_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    common = {
        "script": "MMGCN/train_eval_mmgcn_sast_video_anjs.py",
        "method": "SAST-MMGCN",
        "seed": args.seed,
        "train_pkl": str(train_pkl),
        "external_test_pkl": str(external_pkl),
        "base_checkpoint": str(checkpoint_path),
        "joint_teacher_checkpoint": str(joint_checkpoint_path) if joint_checkpoint_path else "",
        "anchor_name": args.anchor_name,
        "basis_meta": basis_meta,
        "best_epoch": best_epoch,
        "base_source_weighted_f1": base_source["weighted_f1"],
        "base_external_weighted_f1": base_external["weighted_f1"],
    }
    label_names = list(info["label_names"])
    base.save_report(base_source, out_dir, "source_test_base", label_names, {**common, "selection": "base"})
    base.save_report(base_external, out_dir, "external_test_base", label_names, {**common, "selection": "base"})
    if joint_source is not None:
        base.save_report(joint_source, out_dir, "source_test_joint", label_names, {**common, "selection": "joint_teacher"})
        base.save_report(joint_external, out_dir, "external_test_joint", label_names, {**common, "selection": "joint_teacher"})
    base.save_report(best_source, out_dir, "source_test_best", label_names, {**common, "selection": "source"})
    base.save_report(best_external, out_dir, "external_test_best", label_names, {**common, "selection": "source"})
    base.save_report(oracle_source, out_dir, "source_test_oracle", label_names, {**common, "selection": "diagnostic_oracle", "oracle_epoch": oracle_epoch})
    base.save_report(oracle_external, out_dir, "external_test_oracle", label_names, {**common, "selection": "diagnostic_oracle", "oracle_epoch": oracle_epoch})
    model.load_state_dict(best_state)
    save_checkpoint(out_dir / "model_best.pt", model, args, info, basis_meta, best_epoch, best_score)
    model.load_state_dict(oracle_state)
    save_checkpoint(out_dir / "model_oracle_diagnostic.pt", model, args, info, basis_meta, oracle_epoch, oracle_score)
    summary = {
        **common,
        "base_source": base_source["weighted_f1"],
        "base_external": base_external["weighted_f1"],
        "joint_source": joint_source["weighted_f1"] if joint_source else None,
        "joint_external": joint_external["weighted_f1"] if joint_external else None,
        "best_source": best_source["weighted_f1"],
        "best_external": best_external["weighted_f1"],
        "delta_source": best_source["weighted_f1"] - base_source["weighted_f1"],
        "delta_external": best_external["weighted_f1"] - base_external["weighted_f1"],
        "oracle_epoch": oracle_epoch,
        "oracle_source": oracle_source["weighted_f1"],
        "oracle_external": oracle_external["weighted_f1"],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"[SAST-MMGCN] done epoch={best_epoch} source={best_source['weighted_f1']:.4f} "
        f"external={best_external['weighted_f1']:.4f} oracle={oracle_source['weighted_f1']:.4f}/"
        f"{oracle_external['weighted_f1']:.4f}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Single-checkpoint SAST adapter for MMGCN video models.")
    parser.add_argument("--train_pkl", required=True)
    parser.add_argument("--external_test_pkl", required=True)
    parser.add_argument("--base_checkpoint", required=True)
    parser.add_argument("--joint_teacher_checkpoint")
    parser.add_argument("--anchor_name", choices=["source_only", "plain_joint"], default="source_only")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--source_batch_size", type=int, default=8)
    parser.add_argument("--bobsl_batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--adapter_dim", type=int, default=128)
    parser.add_argument("--adapter_dropout", type=float, default=0.2)
    parser.add_argument("--decision_residual", action="store_true")
    parser.add_argument("--decision_adapter_dim", type=int, default=64)
    parser.add_argument("--decision_residual_scale", type=float, default=1.0)
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
    parser.add_argument("--joint_distill_weight", type=float, default=0.0)
    parser.add_argument("--joint_distill_temperature", type=float, default=2.0)
    parser.add_argument("--joint_teacher_correct_only", action="store_true")
    parser.add_argument("--joint_teacher_better_only", action="store_true")
    parser.add_argument("--joint_teacher_advantage_margin", type=float, default=0.0)
    parser.add_argument("--joint_teacher_confidence", type=float, default=0.0)
    parser.add_argument("--prototype_weight", type=float, default=0.1)
    parser.add_argument("--residual_weight", type=float, default=1e-3)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--class_weight", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--class_weight_power", type=float, default=1.0)
    parser.add_argument("--selection_metric", choices=["weighted_f1", "macro_f1"], default="weighted_f1")
    parser.add_argument("--oracle_source_tolerance", type=float, default=0.0)
    parser.add_argument("--no_cuda", action="store_true")

    parser.add_argument("--base_model", default="LSTM")
    parser.add_argument("--graph_type", default="DeepGCN")
    parser.add_argument("--graph_construct", default="direct")
    parser.add_argument("--mm_fusion_mthd", default="concat_subsequently")
    parser.add_argument("--modals", default="auto")
    parser.add_argument("--deep_gcn_nlayers", type=int, default=4)
    parser.add_argument("--windowp", type=int, default=10)
    parser.add_argument("--windowf", type=int, default=10)
    parser.add_argument("--attention", default="general")
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--multiheads", type=int, default=6)
    parser.add_argument("--nodal_attention", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--active_listener", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_gcn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_topic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_speaker", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_modal", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--av_using_lstm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--no_residue", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--d_g", type=int, default=150)
    parser.add_argument("--d_p", type=int, default=150)
    parser.add_argument("--d_e", type=int, default=100)
    parser.add_argument("--d_h", type=int, default=100)
    parser.add_argument("--d_a", type=int, default=100)
    parser.add_argument("--graph_hidden", type=int, default=100)
    parser.add_argument("--max_seq_len", type=int, default=300)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
