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
from typing import Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import train_eval_decoupled_context_anjs as base  # noqa: E402
from anjs_video_common import ANJS_LABELS, resolve_path  # noqa: E402
from cesa import ContextDecoupledEmotionSubspaceAdapter, compute_emotion_subspace  # noqa: E402
from decoupled_context_model import DecoupledLocalContextEmotionNet  # noqa: E402


class DLCENCESA(nn.Module):
    def __init__(self, backbone: nn.Module, visual_dim: int, basis: torch.Tensor, args):
        super().__init__()
        self.backbone = backbone
        self.disable_transfer_gate = bool(args.disable_transfer_gate)
        self.transfer_adapter = ContextDecoupledEmotionSubspaceAdapter(
            visual_dim,
            basis,
            bottleneck_dim=args.transfer_dim,
            dropout=args.transfer_dropout,
            residual_scale=args.transfer_scale,
            projection=args.projection,
        )
        self.transfer_head = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, args.transfer_hidden_dim),
            nn.GELU(),
            nn.Dropout(args.transfer_dropout),
            nn.Linear(args.transfer_hidden_dim, len(ANJS_LABELS)),
        )
        gate_input_dim = len(ANJS_LABELS) * 2 + 4
        self.transfer_gate = nn.Sequential(
            nn.LayerNorm(gate_input_dim),
            nn.Linear(gate_input_dim, args.gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(args.transfer_dropout),
            nn.Linear(args.gate_hidden_dim, len(ANJS_LABELS)),
        )
        nn.init.zeros_(self.transfer_gate[-1].weight)
        nn.init.constant_(self.transfer_gate[-1].bias, args.gate_bias)

    def forward(self, visual, speakers, mask, lengths, adapted: bool = True):
        if adapted:
            visual, delta = self.transfer_adapter(visual)
        else:
            delta = torch.zeros_like(visual)
        outputs = self.backbone(visual, speakers, mask, lengths)
        outputs["transfer_visual"] = visual
        outputs["transfer_delta"] = delta
        if adapted and not self.disable_transfer_gate:
            backbone_probability = outputs["log_prob"].exp()
            local_log_prob = self.local_log_prob(visual)
            local_probability = local_log_prob.exp()
            backbone_entropy = -(backbone_probability * outputs["log_prob"]).sum(dim=-1, keepdim=True)
            local_entropy = -(local_probability * local_log_prob).sum(dim=-1, keepdim=True)
            disagreement = torch.abs(backbone_probability - local_probability).mean(dim=-1, keepdim=True)
            delta_norm = delta.pow(2).mean(dim=-1, keepdim=True).sqrt()
            gate_features = torch.cat(
                [backbone_probability, local_probability, backbone_entropy, local_entropy, disagreement, delta_norm],
                dim=-1,
            )
            transfer_gate = torch.sigmoid(self.transfer_gate(gate_features))
            fused_probability = (1.0 - transfer_gate) * backbone_probability + transfer_gate * local_probability
            fused_probability = fused_probability / fused_probability.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            outputs["backbone_log_prob"] = outputs["log_prob"]
            outputs["log_prob"] = fused_probability.clamp_min(1e-8).log()
            outputs["transfer_local_log_prob"] = local_log_prob
            outputs["transfer_gate"] = transfer_gate
        else:
            outputs["transfer_gate"] = torch.zeros_like(outputs["log_prob"])
        return outputs

    def local_log_prob(self, visual):
        return F.log_softmax(self.transfer_head(visual), dim=-1)


def trainable_parameters(model: DLCENCESA) -> Iterable[nn.Parameter]:
    yield from model.transfer_adapter.parameters()
    yield from model.transfer_head.parameters()
    if not model.disable_transfer_gate:
        yield from model.transfer_gate.parameters()


def build_backbone(checkpoint: Dict, payload: Dict, source_keys, device: torch.device):
    saved_args = checkpoint.get("args", {})
    first_key = source_keys[0]
    visual_dim = int(np.asarray(payload["video_visual"][first_key]).shape[-1])
    speaker_dim = int(np.asarray(payload["speakers"][first_key]).shape[-1])
    rank = int(saved_args.get("subspace_rank", 3))
    basis, _, _ = compute_emotion_subspace(
        payload["video_visual"], payload["labels"], source_keys, len(ANJS_LABELS), rank
    )
    backbone = DecoupledLocalContextEmotionNet(
        visual_dim,
        speaker_dim,
        len(ANJS_LABELS),
        basis,
        hidden_dim=int(saved_args.get("hidden_dim", 256)),
        adapter_dim=int(saved_args.get("adapter_dim", 128)),
        n_heads=int(saved_args.get("n_heads", 4)),
        transformer_layers=int(saved_args.get("transformer_layers", 2)),
        gru_layers=int(saved_args.get("gru_layers", 1)),
        dropout=float(saved_args.get("dropout", 0.3)),
        adapter_dropout=float(saved_args.get("adapter_dropout", 0.2)),
        residual_scale=float(saved_args.get("residual_scale", 1.0)),
        projection=saved_args.get("projection", "emotion"),
        use_adapter=bool(saved_args.get("use_adapter", True)),
        use_transformer=bool(saved_args.get("use_transformer", True)),
        use_multiscale_conv=bool(saved_args.get("use_multiscale_conv", True)),
        use_uncertainty_gate=bool(saved_args.get("use_uncertainty_gate", True)),
    ).to(device)
    state = checkpoint.get("model_state_dict", checkpoint)
    backbone.load_state_dict({key.removeprefix("module."): value for key, value in state.items()}, strict=True)
    return backbone, visual_dim


def weighted_nll(log_prob, target, weights, reduction="mean"):
    return F.nll_loss(log_prob, target, weight=weights, reduction=reduction)


def train_epoch(model, source_loader, bobsl_loader, optimizer, weights, device, args):
    model.backbone.eval()
    # cuDNN requires an RNN training reserve buffer to backpropagate into the
    # input adapter. This GRU has one layer and therefore no recurrent dropout.
    model.backbone.context_gru.train()
    model.transfer_adapter.train()
    model.transfer_head.train()
    model.transfer_gate.train()
    bobsl_iterator = cycle(bobsl_loader)
    totals = {key: 0.0 for key in ("loss", "source", "bobsl", "local", "safety", "residual")}
    steps = 0
    for source_data in source_loader:
        optimizer.zero_grad(set_to_none=True)
        visual, speakers, mask, labels, lengths = base.move_batch(source_data, device)
        outputs = model(visual, speakers, mask, lengths, adapted=True)
        with torch.no_grad():
            baseline = model(visual, speakers, mask, lengths, adapted=False)
        targets = base.flatten_labels(labels, lengths)
        valid = targets.ge(0)
        targets = targets[valid]
        adapted_log_prob = base.flatten(outputs["log_prob"], lengths)[valid]
        baseline_log_prob = base.flatten(baseline["log_prob"], lengths)[valid]
        source_nll = weighted_nll(adapted_log_prob, targets, weights, reduction="none")
        baseline_nll = weighted_nll(baseline_log_prob, targets, weights, reduction="none")
        source_loss = source_nll.mean()
        safety_loss = F.relu(source_nll - baseline_nll + args.safety_margin).mean()
        source_visual = base.flatten(outputs["transfer_visual"], lengths)[valid]
        source_local = weighted_nll(model.local_log_prob(source_visual), targets, weights)
        source_delta = base.flatten(outputs["transfer_delta"], lengths)[valid]
        residual_loss = source_delta.pow(2).sum(dim=-1).mean()

        bobsl_data = next(bobsl_iterator)
        b_visual, b_speakers, b_mask, b_labels, b_lengths = base.move_batch(bobsl_data, device)
        b_outputs = model(b_visual, b_speakers, b_mask, b_lengths, adapted=True)
        b_targets = base.flatten_labels(b_labels, b_lengths)
        b_valid = b_targets.ge(0)
        b_targets = b_targets[b_valid]
        b_adapted = base.flatten(b_outputs["transfer_visual"], b_lengths)[b_valid]
        bobsl_loss = weighted_nll(model.local_log_prob(b_adapted), b_targets, weights)
        b_delta = base.flatten(b_outputs["transfer_delta"], b_lengths)[b_valid]
        residual_loss = residual_loss + b_delta.pow(2).sum(dim=-1).mean()

        loss = (
            args.source_weight * source_loss
            + args.bobsl_weight * bobsl_loss
            + args.local_source_weight * source_local
            + args.safety_weight * safety_loss
            + args.residual_weight * residual_loss
        )
        loss.backward()
        if args.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(list(trainable_parameters(model)), args.max_grad_norm)
        optimizer.step()
        for key, value in {
            "loss": loss, "source": source_loss, "bobsl": bobsl_loss,
            "local": source_local, "safety": safety_loss, "residual": residual_loss,
        }.items():
            totals[key] += float(value.detach().item())
        steps += 1
    return {key: value / max(steps, 1) for key, value in totals.items()}


@torch.no_grad()
def evaluate(model, loader, device, adapted: bool):
    model.eval()
    golds, preds, probabilities, gates, ids = [], [], [], [], []
    for data in loader:
        visual, speakers, mask, labels, lengths = base.move_batch(data, device)
        outputs = model(visual, speakers, mask, lengths, adapted=adapted)
        target = base.flatten_labels(labels, lengths)
        valid = target.ge(0)
        target = target[valid]
        log_prob = base.flatten(outputs["log_prob"], lengths)[valid]
        gate = base.flatten(outputs["transfer_gate"], lengths)[valid]
        prediction = log_prob.argmax(dim=-1)
        golds.extend(target.cpu().numpy().astype(int).tolist())
        preds.extend(prediction.cpu().numpy().astype(int).tolist())
        probabilities.append(log_prob.exp().cpu().numpy().astype(np.float32))
        gates.append(gate.cpu().numpy().astype(np.float32))
        labels_cpu = labels.cpu()
        for batch_index, key in enumerate(data[-1]):
            for utterance_index in range(lengths[batch_index]):
                if int(labels_cpu[batch_index, utterance_index]) >= 0:
                    ids.append(f"{key}:utt{utterance_index + 1:02d}")
    from sklearn import metrics

    gold = np.asarray(golds, dtype=np.int64)
    pred = np.asarray(preds, dtype=np.int64)
    probability = np.concatenate(probabilities)
    label_ids = list(range(len(ANJS_LABELS)))
    return {
        "loss": float(F.nll_loss(torch.tensor(np.log(np.maximum(probability, 1e-8))), torch.tensor(gold)).item()),
        "accuracy": float(metrics.accuracy_score(gold, pred)),
        "macro_f1": float(metrics.f1_score(gold, pred, labels=label_ids, average="macro", zero_division=0)),
        "weighted_f1": float(metrics.f1_score(gold, pred, labels=label_ids, average="weighted", zero_division=0)),
        "golds": gold, "preds": pred, "probs": probability,
        "gate": np.concatenate(gates), "ids": ids,
    }


def save_checkpoint(path, model, args, epoch, score, metadata):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "method": "DLCEN-CESA",
            "args": vars(args),
            "epoch": epoch,
            "source_score": score,
            "metadata": metadata,
        },
        path,
    )


def train(args):
    base.seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    train_path = resolve_path(args.train_pkl, must_exist=True)
    external_path = resolve_path(args.external_test_pkl, must_exist=True)
    checkpoint_path = resolve_path(args.base_checkpoint, must_exist=True)
    out_dir = resolve_path(args.out_dir, must_exist=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = base.load_payload(train_path)
    external_payload = base.load_payload(external_path)
    source_keys = [key for key in payload["train_ids"] if not base.is_bobsl(str(key))]
    bobsl_keys = [key for key in payload["train_ids"] if base.is_bobsl(str(key))]
    checkpoint = torch.load(checkpoint_path, map_location=device)
    backbone, visual_dim = build_backbone(checkpoint, payload, source_keys, device)
    transfer_basis, _, counts = compute_emotion_subspace(
        payload["video_visual"], payload["labels"], source_keys, len(ANJS_LABELS), args.subspace_rank
    )
    model = DLCENCESA(backbone, visual_dim, transfer_basis, args).to(device)
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False

    source_train = base.VisualDialogueDataset(payload, "train", "source")
    bobsl_train = base.VisualDialogueDataset(payload, "train", "bobsl")
    source_test = base.VisualDialogueDataset(payload, "test")
    external_test = base.VisualDialogueDataset(external_payload, "test")
    source_loader = base.make_loader(source_train, args.source_batch_size, True, args)
    bobsl_loader = base.make_loader(bobsl_train, args.bobsl_batch_size, True, args, 19)
    source_test_loader = base.make_loader(source_test, args.eval_batch_size, False, args)
    external_loader = base.make_loader(external_test, args.eval_batch_size, False, args)
    weights = base.class_weights(payload, source_keys, device) if args.class_weight else None
    optimizer = torch.optim.AdamW(trainable_parameters(model), lr=args.lr, weight_decay=args.weight_decay)

    base_source = evaluate(model, source_test_loader, device, adapted=False)
    base_external = evaluate(model, external_loader, device, adapted=False)
    best_source, best_external = base.snapshot(base_source), base.snapshot(base_external)
    best_state = copy.deepcopy(model.state_dict())
    best_score, best_epoch = base_source[args.selection_metric], 0
    rows = []
    print(f"[DLCEN-CESA] source={len(source_train)} bobsl={len(bobsl_train)} counts={counts.tolist()}")
    print(f"[DLCEN-CESA] base source={base_source['weighted_f1']:.4f} external={base_external['weighted_f1']:.4f}")
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        training = train_epoch(model, source_loader, bobsl_loader, optimizer, weights, device, args)
        source_result = evaluate(model, source_test_loader, device, adapted=True)
        external_result = evaluate(model, external_loader, device, adapted=True)
        if source_result[args.selection_metric] > best_score:
            best_score = source_result[args.selection_metric]
            best_epoch = epoch
            best_source, best_external = base.snapshot(source_result), base.snapshot(external_result)
            best_state = copy.deepcopy(model.state_dict())
        row = {
            "epoch": epoch, **{f"train_{key}": value for key, value in training.items()},
            "source_weighted_f1": source_result["weighted_f1"],
            "external_weighted_f1": external_result["weighted_f1"],
            "delta_source": source_result["weighted_f1"] - base_source["weighted_f1"],
            "delta_external": external_result["weighted_f1"] - base_external["weighted_f1"],
            "time_sec": round(time.time() - started, 2),
        }
        rows.append(row)
        print(
            f"[DLCEN-CESA] epoch={epoch} source={row['source_weighted_f1']:.4f} ({row['delta_source']:+.4f}) "
            f"external={row['external_weighted_f1']:.4f} ({row['delta_external']:+.4f}) time={row['time_sec']}s"
        )
    with (out_dir / "epoch_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "method": "DLCEN-CESA", "seed": args.seed, "best_epoch": best_epoch,
        "selection": "source", "base_checkpoint": str(checkpoint_path), "args": vars(args),
    }
    base.write_report(base_source, out_dir, "source_test_base", metadata)
    base.write_report(base_external, out_dir, "external_test_base", metadata)
    base.write_report(best_source, out_dir, "source_test_best", metadata)
    base.write_report(best_external, out_dir, "external_test_best", metadata)
    model.load_state_dict(best_state)
    save_checkpoint(out_dir / "model_best.pt", model, args, best_epoch, best_score, metadata)
    print(
        f"[DLCEN-CESA] done epoch={best_epoch} source={best_source['weighted_f1']:.4f} "
        f"external={best_external['weighted_f1']:.4f} out={out_dir}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Adapt a source-trained DLCEN with context-decoupled CESA.")
    parser.add_argument("--train_pkl", required=True)
    parser.add_argument("--external_test_pkl", required=True)
    parser.add_argument("--base_checkpoint", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--source_batch_size", type=int, default=16)
    parser.add_argument("--bobsl_batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--transfer_dim", type=int, default=128)
    parser.add_argument("--transfer_hidden_dim", type=int, default=128)
    parser.add_argument("--gate_hidden_dim", type=int, default=64)
    parser.add_argument("--gate_bias", type=float, default=-3.0)
    parser.add_argument("--disable_transfer_gate", action="store_true")
    parser.add_argument("--transfer_dropout", type=float, default=0.2)
    parser.add_argument("--transfer_scale", type=float, default=1.0)
    parser.add_argument("--subspace_rank", type=int, default=3)
    parser.add_argument("--projection", choices=["emotion", "full"], default="emotion")
    parser.add_argument("--source_weight", type=float, default=1.0)
    parser.add_argument("--bobsl_weight", type=float, default=1.0)
    parser.add_argument("--local_source_weight", type=float, default=0.1)
    parser.add_argument("--safety_weight", type=float, default=0.5)
    parser.add_argument("--safety_margin", type=float, default=0.0)
    parser.add_argument("--residual_weight", type=float, default=0.001)
    parser.add_argument("--class_weight", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--selection_metric", choices=["weighted_f1", "macro_f1"], default="weighted_f1")
    parser.add_argument("--no_cuda", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
