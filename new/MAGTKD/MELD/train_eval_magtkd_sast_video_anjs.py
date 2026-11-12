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
REPO_ROOT = THIS_DIR.parents[2]
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(REPO_ROOT / "new"))

import train_eval_magtkd_video_anjs as base  # noqa: E402
from anjs_video_common import ANJS_LABELS  # noqa: E402
from cesa import (  # noqa: E402
    ContextDecoupledEmotionSubspaceAdapter,
    EmotionDecisionResidualAdapter,
    compute_emotion_subspace,
    compute_projection_statistics,
    interpolate_module_state_,
    prototype_alignment_loss,
)
from model import Transformer_Based_Model  # noqa: E402


def is_bobsl_key(key: str) -> bool:
    value = str(key)
    return value.startswith("joint_bobsl") or value.startswith("bobsl")


def make_dataset(path: Path, split: str, domain: str = "all") -> base.MAGTKDFeatureDataset:
    dataset = base.MAGTKDFeatureDataset(path, split)
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
        collate_fn=base.MAGTKDFeatureDataset.collate_fn,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def flatten_seq(values: torch.Tensor, lengths: Sequence[int]) -> torch.Tensor:
    return torch.cat([values[:length, index] for index, length in enumerate(lengths)], dim=0)


def flatten_labels(labels: torch.Tensor, lengths: Sequence[int]) -> torch.Tensor:
    return torch.cat([labels[index, :length] for index, length in enumerate(lengths)], dim=0)


class SASTMAGTKD(nn.Module):
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
                len(ANJS_LABELS),
                bottleneck_dim=args.decision_adapter_dim,
                dropout=args.adapter_dropout,
                residual_scale=args.decision_residual_scale,
            )
            if args.decision_residual else None
        )

    def forward(self, text, audio_kd, video_kd, qmask, umask, lengths, adapted: bool):
        text = torch.zeros_like(text)
        audio_kd = torch.zeros_like(audio_kd)
        if adapted:
            video_kd, delta = self.adapter(video_kd)
        else:
            delta = torch.zeros_like(video_kd)
        _, _, video_logits, *_ = self.backbone(
            text, audio_kd, video_kd, umask, qmask.clone(), lengths
        )
        if adapted and self.decision_adapter is not None:
            decision_delta = self.decision_adapter(video_kd).permute(1, 0, 2)
            video_logits = video_logits + decision_delta
        return video_logits, delta


def load_backbone(args, checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    for key in ("hidden_dim", "n_head", "dropout", "temp", "clsNum"):
        if key in checkpoint_args:
            setattr(args, key, checkpoint_args[key])
    args.clsNum = len(ANJS_LABELS)
    backbone = Transformer_Based_Model(args).to(device)
    state = checkpoint.get("model_state_dict", checkpoint)
    clean = {key.removeprefix("module."): value for key, value in state.items()}
    incompatible = backbone.load_state_dict(clean, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={incompatible.missing_keys[:10]} "
            f"unexpected={incompatible.unexpected_keys[:10]}"
        )
    return backbone


def prepare_batch(data, device: torch.device):
    text, audio, video, audio_kd, video_kd, qmask, umask, labels = [item.to(device) for item in data[:-1]]
    lengths = base.sequence_lengths(umask)
    labels_flat = flatten_labels(labels, lengths)
    valid_flat = labels_flat.ge(0)
    valid_padded = umask.bool() & labels.ge(0)
    return text, audio, video, audio_kd, video_kd, qmask, umask, labels, lengths, labels_flat, valid_flat, valid_padded


def metric_result(losses, counts, golds, preds, probs, ids) -> Optional[Dict]:
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
def evaluate(model: SASTMAGTKD, loader: DataLoader, device: torch.device, adapted: bool) -> Optional[Dict]:
    model.eval()
    losses: List[float] = []
    counts: List[int] = []
    golds: List[int] = []
    preds: List[int] = []
    probs: List[np.ndarray] = []
    ids: List[str] = []
    for data in loader:
        (
            text, _, _, audio_kd, video_kd, qmask, umask, labels, lengths,
            _, _, valid,
        ) = prepare_batch(data, device)
        logits, _ = model(text, audio_kd, video_kd, qmask, umask, lengths, adapted=adapted)
        current = logits[valid]
        target = labels[valid]
        if not len(target):
            continue
        loss = F.cross_entropy(current, target)
        probability = torch.softmax(current, dim=-1)
        losses.append(float(loss.item()) * len(target))
        counts.append(len(target))
        golds.extend(target.cpu().numpy().astype(int).tolist())
        preds.extend(probability.argmax(dim=-1).cpu().numpy().astype(int).tolist())
        probs.append(probability.cpu().numpy().astype(np.float32))
        labels_cpu = labels.cpu()
        for batch_index, key in enumerate(data[-1]):
            for utterance_index in range(lengths[batch_index]):
                if int(labels_cpu[batch_index, utterance_index]) >= 0:
                    ids.append(f"{key}:utt{utterance_index + 1:02d}")
    return metric_result(losses, counts, golds, preds, probs, ids)


def weighted_ce(logits, labels, weights, reduction="mean"):
    return F.cross_entropy(logits, labels, weight=weights, reduction=reduction)


def adapter_parameters(model: SASTMAGTKD) -> Iterable[nn.Parameter]:
    yield from model.adapter.parameters()
    if model.decision_adapter is not None:
        yield from model.decision_adapter.parameters()


def joint_teacher_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    base_logits: torch.Tensor,
    targets: torch.Tensor,
    args,
) -> torch.Tensor:
    temperature = args.joint_distill_temperature
    per_sample = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction="none",
    ).sum(dim=-1) * (temperature * temperature)
    keep = torch.ones_like(targets, dtype=torch.bool)
    if args.joint_teacher_correct_only:
        keep &= teacher_logits.argmax(dim=-1).eq(targets)
    if args.joint_teacher_better_only:
        teacher_nll = F.cross_entropy(teacher_logits, targets, reduction="none")
        base_nll = F.cross_entropy(base_logits, targets, reduction="none")
        keep &= teacher_nll + args.joint_teacher_advantage_margin < base_nll
    if args.joint_teacher_confidence > 0:
        keep &= F.softmax(teacher_logits, dim=-1).amax(dim=-1) >= args.joint_teacher_confidence
    if not torch.any(keep):
        return per_sample.new_zeros(())
    return per_sample[keep].mean()


def train_epoch(
    model: SASTMAGTKD,
    joint_teacher: Optional[SASTMAGTKD],
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
        for key in (
            "loss", "source", "bobsl", "safety", "stability", "joint", "peer_bobsl", "prototype", "residual"
        )
    }
    steps = 0
    for source_data in source_loader:
        optimizer.zero_grad(set_to_none=True)
        (
            text, _, _, audio_kd, video_kd, qmask, umask, _, lengths,
            labels_flat, valid_flat, valid,
        ) = prepare_batch(source_data, device)
        with torch.no_grad():
            base_logits, _ = model(text, audio_kd, video_kd, qmask, umask, lengths, adapted=False)
        adapted_logits, source_delta = model(text, audio_kd, video_kd, qmask, umask, lengths, adapted=True)
        targets = labels_flat[valid_flat]
        source_current = adapted_logits[valid]
        base_current = base_logits[valid]
        source_nll = weighted_ce(source_current, targets, weights, reduction="none")
        base_nll = weighted_ce(base_current, targets, weights, reduction="none")
        source_loss = source_nll.mean()
        safety_loss = F.relu(source_nll - base_nll + args.safety_margin).mean()
        stability_loss = F.kl_div(
            F.log_softmax(source_current, dim=-1), F.softmax(base_current, dim=-1), reduction="batchmean"
        )
        if joint_teacher is not None and args.joint_distill_weight > 0:
            with torch.no_grad():
                teacher_logits, _ = joint_teacher(
                    text, audio_kd, video_kd, qmask, umask, lengths, adapted=False
                )
            joint_loss = joint_teacher_loss(
                source_current, teacher_logits[valid], base_current, targets, args
            )
        else:
            joint_loss = source_loss.new_zeros(())
        source_delta_flat = flatten_seq(source_delta, lengths)[valid_flat]
        residual_loss = source_delta_flat.pow(2).sum(dim=-1).mean()

        bobsl_data = next(bobsl_iter)
        (
            b_text, _, _, b_audio_kd, b_video_kd, b_qmask, b_umask, _, b_lengths,
            b_labels_flat, b_valid_flat, b_valid,
        ) = prepare_batch(bobsl_data, device)
        bobsl_logits, bobsl_delta = model(
            b_text, b_audio_kd, b_video_kd, b_qmask, b_umask, b_lengths, adapted=True
        )
        bobsl_targets = b_labels_flat[b_valid_flat]
        bobsl_loss = weighted_ce(bobsl_logits[b_valid], bobsl_targets, weights)
        if joint_teacher is not None and args.peer_bobsl_distill_weight > 0:
            with torch.no_grad():
                bobsl_base_logits, _ = model(
                    b_text, b_audio_kd, b_video_kd, b_qmask, b_umask, b_lengths, adapted=False
                )
                bobsl_teacher_logits, _ = joint_teacher(
                    b_text, b_audio_kd, b_video_kd, b_qmask, b_umask, b_lengths, adapted=False
                )
            peer_bobsl_loss = joint_teacher_loss(
                bobsl_logits[b_valid], bobsl_teacher_logits[b_valid],
                bobsl_base_logits[b_valid], bobsl_targets, args
            )
        else:
            peer_bobsl_loss = bobsl_loss.new_zeros(())
        bobsl_features = flatten_seq(b_video_kd + bobsl_delta, b_lengths)[b_valid_flat]
        prototype_loss = prototype_alignment_loss(
            bobsl_features, bobsl_targets, basis, centers, coordinate_mean, coordinate_scale
        )
        bobsl_delta_flat = flatten_seq(bobsl_delta, b_lengths)[b_valid_flat]
        residual_loss = residual_loss + bobsl_delta_flat.pow(2).sum(dim=-1).mean()
        loss = (
            args.source_weight * source_loss
            + args.bobsl_weight * bobsl_loss
            + args.safety_weight * safety_loss
            + args.stability_weight * stability_loss
            + args.joint_distill_weight * joint_loss
            + args.peer_bobsl_distill_weight * peer_bobsl_loss
            + args.prototype_weight * prototype_loss
            + args.residual_weight * residual_loss
        )
        loss.backward()
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(list(adapter_parameters(model)), args.max_grad_norm)
        optimizer.step()
        for key, value in {
            "loss": loss,
            "source": source_loss,
            "bobsl": bobsl_loss,
            "safety": safety_loss,
            "stability": stability_loss,
            "joint": joint_loss,
            "peer_bobsl": peer_bobsl_loss,
            "prototype": prototype_loss,
            "residual": residual_loss,
        }.items():
            totals[key] += float(value.detach().item())
        steps += 1
    return {key: value / max(steps, 1) for key, value in totals.items()}


def save_checkpoint(path: Path, model: SASTMAGTKD, args, info, basis_meta, epoch, score):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "method": "SAST-MAGTKD",
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
        raise RuntimeError("SAST-MAGTKD requires CUDA")
    train_pkl = base.resolve_path(args.train_pkl, must_exist=True)
    external_pkl = base.resolve_path(args.external_test_pkl, must_exist=True)
    checkpoint_path = base.resolve_path(args.base_checkpoint, must_exist=True)
    joint_checkpoint_path = (
        base.resolve_path(args.joint_teacher_checkpoint, must_exist=True)
        if args.joint_teacher_checkpoint else None
    )
    out_dir = base.resolve_path(args.out_dir, must_exist=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = base.inspect_pkl(train_pkl)
    payload = base.load_payload(train_pkl)
    train_keys = list(payload["train_vids"])
    source_keys = [key for key in train_keys if not is_bobsl_key(key)]
    bobsl_keys = [key for key in train_keys if is_bobsl_key(key)]
    if not bobsl_keys:
        raise RuntimeError("SAST-MAGTKD requires a joint source/BOBSL pkl")
    basis, centers, counts = compute_emotion_subspace(
        payload["video_kd"], payload["labels"], source_keys, len(ANJS_LABELS), rank=args.subspace_rank
    )
    coordinate_mean, coordinate_scale = compute_projection_statistics(payload["video_kd"], source_keys, basis)
    basis_meta = {
        "rank": int(basis.shape[1]),
        "projection": args.projection,
        "source_counts": counts.tolist(),
        "coordinate_scale": coordinate_scale.tolist(),
        "source_dialogues": len(source_keys),
        "bobsl_dialogues": len(bobsl_keys),
    }

    backbone = load_backbone(args, checkpoint_path, device)
    model = SASTMAGTKD(backbone, int(info["d_visual"]), basis, args).to(device)
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    joint_teacher = None
    if joint_checkpoint_path is not None:
        teacher_args = copy.deepcopy(args)
        teacher_backbone = load_backbone(teacher_args, joint_checkpoint_path, device)
        joint_teacher = SASTMAGTKD(teacher_backbone, int(info["d_visual"]), basis, teacher_args).to(device)
        for parameter in joint_teacher.parameters():
            parameter.requires_grad = False
        joint_teacher.eval()
    source_train = make_dataset(train_pkl, "train", "source")
    bobsl_train = make_dataset(train_pkl, "train", "bobsl")
    source_test = make_dataset(train_pkl, "test")
    external_test = make_dataset(external_pkl, "test")
    source_loader = make_loader(source_train, args.source_batch_size, True, args.seed)
    bobsl_loader = make_loader(bobsl_train, args.bobsl_batch_size, True, args.seed + 17)
    source_test_loader = make_loader(source_test, args.eval_batch_size, False, args.seed)
    external_test_loader = make_loader(external_test, args.eval_batch_size, False, args.seed)
    source_labels = [int(label) for key in source_keys for label in payload["labels"][key] if int(label) >= 0]
    weights = base.make_class_weights(source_labels, len(ANJS_LABELS), device) if args.class_weight else None
    if weights is not None and args.class_weight_power != 1.0:
        weights = weights.pow(args.class_weight_power)
        weights = weights / weights.mean().clamp_min(1e-8)
    optimizer = torch.optim.AdamW(adapter_parameters(model), lr=args.lr, weight_decay=args.weight_decay)

    base_source = evaluate(model, source_test_loader, device, adapted=False)
    base_external = evaluate(model, external_test_loader, device, adapted=False)
    joint_source = evaluate(joint_teacher, source_test_loader, device, adapted=False) if joint_teacher else None
    joint_external = evaluate(joint_teacher, external_test_loader, device, adapted=False) if joint_teacher else None
    if args.backbone_mix_alpha > 0:
        if joint_teacher is None:
            raise ValueError("--backbone_mix_alpha requires --joint_teacher_checkpoint")
        interpolate_module_state_(model.backbone, joint_teacher.backbone, args.backbone_mix_alpha)
    anchor_source = evaluate(model, source_test_loader, device, adapted=False)
    anchor_external = evaluate(model, external_test_loader, device, adapted=False)
    best_source = copy.deepcopy(anchor_source)
    best_external = copy.deepcopy(anchor_external)
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_score = float(anchor_source[args.selection_metric])
    oracle_source = copy.deepcopy(anchor_source)
    oracle_external = copy.deepcopy(anchor_external)
    oracle_state = copy.deepcopy(best_state)
    oracle_epoch = 0
    oracle_score = float(anchor_external["weighted_f1"])
    rows: List[Dict] = []
    trainable = sum(parameter.numel() for parameter in adapter_parameters(model))
    print(f"[SAST-MAGTKD] source={len(source_train)} bobsl={len(bobsl_train)} trainable={trainable}")
    print(f"[SAST-MAGTKD] base source={base_source['weighted_f1']:.4f} external={base_external['weighted_f1']:.4f}")
    if joint_source is not None:
        print(
            f"[SAST-MAGTKD] joint teacher source={joint_source['weighted_f1']:.4f} "
            f"external={joint_external['weighted_f1']:.4f}"
        )
    if args.backbone_mix_alpha > 0:
        print(
            f"[SAST-MAGTKD] mixed anchor alpha={args.backbone_mix_alpha:.3f} "
            f"source={anchor_source['weighted_f1']:.4f} external={anchor_external['weighted_f1']:.4f}"
        )
    print(f"[SAST-MAGTKD] basis={json.dumps(basis_meta)}")

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
            source_result["weighted_f1"] >= anchor_source["weighted_f1"] - args.oracle_source_tolerance
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
            f"[SAST-MAGTKD] epoch={epoch} source={row['source_weighted_f1']:.4f} ({row['delta_source']:+.4f}) "
            f"external={row['external_weighted_f1']:.4f} ({row['delta_external']:+.4f}) "
            f"loss={train_metrics['loss']:.4f} time={row['time_sec']}s"
        )

    with (out_dir / "epoch_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    common = {
        "script": "new/MAGTKD/MELD/train_eval_magtkd_sast_video_anjs.py",
        "method": "SAST-MAGTKD",
        "seed": args.seed,
        "train_pkl": str(train_pkl),
        "external_test_pkl": str(external_pkl),
        "base_checkpoint": str(checkpoint_path),
        "joint_teacher_checkpoint": str(joint_checkpoint_path) if joint_checkpoint_path else "",
        "anchor_name": args.anchor_name,
        "backbone_mix_alpha": args.backbone_mix_alpha,
        "anchor_source_weighted_f1": anchor_source["weighted_f1"],
        "anchor_external_weighted_f1": anchor_external["weighted_f1"],
        "basis_meta": basis_meta,
        "best_epoch": best_epoch,
        "base_source_weighted_f1": base_source["weighted_f1"],
        "base_external_weighted_f1": base_external["weighted_f1"],
    }
    base.save_report(base_source, out_dir, "source_test_base", {**common, "selection": "base"})
    base.save_report(base_external, out_dir, "external_test_base", {**common, "selection": "base"})
    if joint_source is not None:
        base.save_report(joint_source, out_dir, "source_test_joint", {**common, "selection": "joint_teacher"})
        base.save_report(joint_external, out_dir, "external_test_joint", {**common, "selection": "joint_teacher"})
    base.save_report(anchor_source, out_dir, "source_test_anchor", {**common, "selection": "mixed_anchor"})
    base.save_report(anchor_external, out_dir, "external_test_anchor", {**common, "selection": "mixed_anchor"})
    base.save_report(best_source, out_dir, "source_test_best", {**common, "selection": "source"})
    base.save_report(best_external, out_dir, "external_test_best", {**common, "selection": "source"})
    base.save_report(oracle_source, out_dir, "source_test_oracle", {**common, "selection": "diagnostic_oracle", "oracle_epoch": oracle_epoch})
    base.save_report(oracle_external, out_dir, "external_test_oracle", {**common, "selection": "diagnostic_oracle", "oracle_epoch": oracle_epoch})
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
        "anchor_source": anchor_source["weighted_f1"],
        "anchor_external": anchor_external["weighted_f1"],
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
        f"[SAST-MAGTKD] done epoch={best_epoch} source={best_source['weighted_f1']:.4f} "
        f"external={best_external['weighted_f1']:.4f} oracle={oracle_source['weighted_f1']:.4f}/"
        f"{oracle_external['weighted_f1']:.4f}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Single-checkpoint SAST adapter for MAGTKD video models.")
    parser.add_argument("--train_pkl", required=True)
    parser.add_argument("--external_test_pkl", required=True)
    parser.add_argument("--base_checkpoint", required=True)
    parser.add_argument("--joint_teacher_checkpoint")
    parser.add_argument("--anchor_name", choices=["source_only", "plain_joint"], default="source_only")
    parser.add_argument("--backbone_mix_alpha", type=float, default=0.0)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--source_batch_size", type=int, default=16)
    parser.add_argument("--bobsl_batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=32)
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
    parser.add_argument("--peer_bobsl_distill_weight", type=float, default=0.0)
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
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--temp", type=float, default=2.0)
    parser.add_argument("--clsNum", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
