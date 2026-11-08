#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import pickle
import random
import sys
import time
from itertools import cycle
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))

from anjs_video_common import ANJS_LABELS, resolve_path  # noqa: E402
from cesa import compute_emotion_subspace  # noqa: E402
from decoupled_context_model import DecoupledLocalContextEmotionNet  # noqa: E402


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_payload(path: Path) -> Dict:
    payload = pickle.load(path.open("rb"), encoding="latin1")
    required = ["video_visual", "speakers", "labels", "train_ids", "test_ids"]
    if not isinstance(payload, dict) or any(key not in payload for key in required):
        raise ValueError(f"Unsupported feature payload: {path}")
    payload.setdefault("valid_ids", [])
    return payload


def is_bobsl(key: str) -> bool:
    return str(key).startswith("joint_bobsl") or str(key).startswith("bobsl")


class VisualDialogueDataset(Dataset):
    def __init__(self, payload: Dict, split: str, domain: str = "all"):
        self.payload = payload
        self.keys = list(payload[f"{split}_ids"])
        if domain == "source":
            self.keys = [key for key in self.keys if not is_bobsl(str(key))]
        elif domain == "bobsl":
            self.keys = [key for key in self.keys if is_bobsl(str(key))]
        elif domain != "all":
            raise ValueError(domain)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index: int):
        key = self.keys[index]
        return (
            torch.tensor(np.asarray(self.payload["video_visual"][key]), dtype=torch.float32),
            torch.tensor(np.asarray(self.payload["speakers"][key]), dtype=torch.float32),
            torch.ones(len(self.payload["labels"][key]), dtype=torch.float32),
            torch.tensor(np.asarray(self.payload["labels"][key]), dtype=torch.long),
            key,
        )

    @staticmethod
    def collate(data):
        columns = list(zip(*data))
        return (
            pad_sequence(columns[0]),
            pad_sequence(columns[1]),
            pad_sequence(columns[2], batch_first=True),
            pad_sequence(columns[3], batch_first=True, padding_value=-100),
            list(columns[4]),
        )


def make_loader(dataset, batch_size: int, shuffle: bool, args, offset: int = 0):
    generator = torch.Generator().manual_seed(args.seed + offset)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=VisualDialogueDataset.collate,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def lengths_from_mask(mask: torch.Tensor) -> List[int]:
    return [int(row.sum().item()) for row in mask]


def flatten(values: torch.Tensor, lengths: Sequence[int]) -> torch.Tensor:
    return torch.cat([values[:length, index] for index, length in enumerate(lengths)], dim=0)


def flatten_labels(labels: torch.Tensor, lengths: Sequence[int]) -> torch.Tensor:
    return torch.cat([labels[index, :length] for index, length in enumerate(lengths)], dim=0)


def class_weights(payload: Dict, source_keys: Sequence[str], device: torch.device) -> torch.Tensor:
    labels = [int(value) for key in source_keys for value in payload["labels"][key] if int(value) >= 0]
    counts = np.bincount(np.asarray(labels), minlength=len(ANJS_LABELS)).astype(np.float32)
    weights = counts.sum() / (len(ANJS_LABELS) * np.maximum(counts, 1))
    return torch.tensor(weights / weights.mean(), dtype=torch.float32, device=device)


def nll(log_prob: torch.Tensor, target: torch.Tensor, weights: Optional[torch.Tensor]):
    return F.nll_loss(log_prob, target, weight=weights)


def shift_loss(outputs: Dict[str, torch.Tensor], labels: torch.Tensor, lengths: Sequence[int]) -> torch.Tensor:
    logits = outputs["shift_logits"]
    values = []
    targets = []
    for batch_index, length in enumerate(lengths):
        if length < 2:
            continue
        current = labels[batch_index, :length]
        valid = current[:-1].ge(0) & current[1:].ge(0)
        if torch.any(valid):
            values.append(logits[: length - 1, batch_index][valid])
            targets.append((current[:-1][valid] != current[1:][valid]).float())
    if not values:
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(torch.cat(values), torch.cat(targets))


def source_objective(outputs, labels, lengths, weights, args):
    target = flatten_labels(labels, lengths)
    valid = target.ge(0)
    target = target[valid]
    fused = flatten(outputs["log_prob"], lengths)[valid]
    local = flatten(outputs["local_log_prob"], lengths)[valid]
    context = flatten(outputs["context_log_prob"], lengths)[valid]
    fused_loss = nll(fused, target, weights)
    local_loss = nll(local, target, weights)
    context_loss = nll(context, target, weights)
    transition_loss = shift_loss(outputs, labels, lengths)
    probability = fused.exp()
    one_hot = F.one_hot(target, num_classes=len(ANJS_LABELS)).float()
    calibration = (probability - one_hot).pow(2).sum(dim=-1).mean()
    residual = flatten(outputs["adapter_delta"], lengths)[valid].pow(2).sum(dim=-1).mean()
    loss = (
        fused_loss
        + args.local_loss_weight * local_loss
        + args.context_loss_weight * context_loss
        + args.shift_loss_weight * transition_loss
        + args.calibration_weight * calibration
        + args.residual_weight * residual
    )
    return loss, {
        "fused": fused_loss,
        "local": local_loss,
        "context": context_loss,
        "shift": transition_loss,
        "calibration": calibration,
        "residual": residual,
    }


def bobsl_objective(model, visual, speakers, mask, labels, lengths, weights, args):
    outputs = model(visual, speakers, mask, lengths)
    target = flatten_labels(labels, lengths)
    valid = target.ge(0)
    target = target[valid]
    local = flatten(outputs["local_log_prob"], lengths)[valid]
    local_loss = nll(local, target, weights)
    residual = flatten(outputs["adapter_delta"], lengths)[valid].pow(2).sum(dim=-1).mean()
    if args.bobsl_context_loss_weight > 0:
        context = flatten(outputs["log_prob"], lengths)[valid]
        context_loss = nll(context, target, weights)
    else:
        context_loss = local_loss * 0.0
    loss = local_loss + args.bobsl_context_loss_weight * context_loss + args.residual_weight * residual
    return loss, {"bobsl_local": local_loss, "bobsl_context": context_loss}


def move_batch(data, device: torch.device):
    visual, speakers, mask, labels = [value.to(device) for value in data[:-1]]
    return visual, speakers, mask, labels, lengths_from_mask(mask)


def train_epoch(model, source_loader, bobsl_loader, optimizer, weights, device, args):
    model.train()
    bobsl_iterator = cycle(bobsl_loader) if bobsl_loader is not None else None
    totals: Dict[str, float] = {}
    steps = 0
    for source_data in source_loader:
        optimizer.zero_grad(set_to_none=True)
        visual, speakers, mask, labels, lengths = move_batch(source_data, device)
        outputs = model(visual, speakers, mask, lengths)
        source_loss, parts = source_objective(outputs, labels, lengths, weights, args)
        loss = source_loss
        if bobsl_iterator is not None:
            bobsl_data = next(bobsl_iterator)
            b_visual, b_speakers, b_mask, b_labels, b_lengths = move_batch(bobsl_data, device)
            bobsl_loss, bobsl_parts = bobsl_objective(
                model, b_visual, b_speakers, b_mask, b_labels, b_lengths, weights, args
            )
            loss = loss + args.bobsl_weight * bobsl_loss
            parts.update(bobsl_parts)
        loss.backward()
        if args.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        parts["total"] = loss
        for key, value in parts.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().item())
        steps += 1
    return {key: value / max(steps, 1) for key, value in totals.items()}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    golds: List[int] = []
    preds: List[int] = []
    probabilities: List[np.ndarray] = []
    ids: List[str] = []
    gates: List[np.ndarray] = []
    for data in loader:
        visual, speakers, mask, labels, lengths = move_batch(data, device)
        outputs = model(visual, speakers, mask, lengths)
        target = flatten_labels(labels, lengths)
        valid = target.ge(0)
        log_prob = flatten(outputs["log_prob"], lengths)[valid]
        gate = flatten(outputs["gate"], lengths)[valid]
        target = target[valid]
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
    gold = np.asarray(golds, dtype=np.int64)
    pred = np.asarray(preds, dtype=np.int64)
    probability = np.concatenate(probabilities)
    gate = np.concatenate(gates)
    label_ids = list(range(len(ANJS_LABELS)))
    return {
        "loss": float(F.nll_loss(torch.tensor(np.log(np.maximum(probability, 1e-8))), torch.tensor(gold)).item()),
        "accuracy": float(metrics.accuracy_score(gold, pred)),
        "macro_f1": float(metrics.f1_score(gold, pred, labels=label_ids, average="macro", zero_division=0)),
        "weighted_f1": float(metrics.f1_score(gold, pred, labels=label_ids, average="weighted", zero_division=0)),
        "golds": gold,
        "preds": pred,
        "probs": probability,
        "gate": gate,
        "ids": ids,
    }


def snapshot(result: Dict) -> Dict:
    return {key: value.copy() if isinstance(value, np.ndarray) else list(value) if isinstance(value, list) else value for key, value in result.items()}


def write_report(result: Dict, out_dir: Path, prefix: str, metadata: Dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    label_ids = list(range(len(ANJS_LABELS)))
    report = metrics.classification_report(
        result["golds"], result["preds"], labels=label_ids, target_names=ANJS_LABELS, digits=4, zero_division=0
    )
    (out_dir / f"{prefix}_classification_report.txt").write_text(report, encoding="utf-8")
    per_p, per_r, per_f, per_s = metrics.precision_recall_fscore_support(
        result["golds"], result["preds"], labels=label_ids, zero_division=0
    )
    summary = {
        **metadata,
        "n_samples": len(result["golds"]),
        "accuracy": result["accuracy"],
        "macro_f1": result["macro_f1"],
        "weighted_f1": result["weighted_f1"],
        "mean_local_gate": result["gate"].mean(axis=0).tolist(),
        "per_class": {
            label: {"precision": float(per_p[i]), "recall": float(per_r[i]), "f1": float(per_f[i]), "support": int(per_s[i])}
            for i, label in enumerate(ANJS_LABELS)
        },
    }
    (out_dir / f"{prefix}_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (out_dir / f"{prefix}_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item_id", "gold_id", "pred_id"] + [f"prob_{label}" for label in ANJS_LABELS])
        for item_id, gold, pred, probability in zip(result["ids"], result["golds"], result["preds"], result["probs"]):
            writer.writerow([item_id, int(gold), int(pred), *probability.tolist()])
    return summary


def train(args) -> None:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    train_path = resolve_path(args.train_pkl, must_exist=True)
    external_path = resolve_path(args.external_test_pkl, must_exist=True)
    out_dir = resolve_path(args.out_dir, must_exist=False)
    payload = load_payload(train_path)
    external_payload = load_payload(external_path)
    source_keys = [key for key in payload["train_ids"] if not is_bobsl(str(key))]
    bobsl_keys = [key for key in payload["train_ids"] if is_bobsl(str(key))]
    first_key = source_keys[0]
    visual_dim = int(np.asarray(payload["video_visual"][first_key]).shape[-1])
    speaker_dim = int(np.asarray(payload["speakers"][first_key]).shape[-1])
    basis, centers, counts = compute_emotion_subspace(
        payload["video_visual"], payload["labels"], source_keys, len(ANJS_LABELS), args.subspace_rank
    )
    model = DecoupledLocalContextEmotionNet(
        visual_dim,
        speaker_dim,
        len(ANJS_LABELS),
        basis,
        hidden_dim=args.hidden_dim,
        adapter_dim=args.adapter_dim,
        n_heads=args.n_heads,
        transformer_layers=args.transformer_layers,
        gru_layers=args.gru_layers,
        dropout=args.dropout,
        adapter_dropout=args.adapter_dropout,
        residual_scale=args.residual_scale,
        projection=args.projection,
        use_adapter=args.use_adapter,
        use_transformer=args.use_transformer,
        use_multiscale_conv=args.use_multiscale_conv,
        use_uncertainty_gate=args.use_uncertainty_gate,
    ).to(device)
    source_train = VisualDialogueDataset(payload, "train", "source")
    bobsl_train = VisualDialogueDataset(payload, "train", "bobsl") if bobsl_keys and args.bobsl_weight > 0 else None
    source_test = VisualDialogueDataset(payload, "test")
    external_test = VisualDialogueDataset(external_payload, "test")
    source_loader = make_loader(source_train, args.batch_size, True, args)
    bobsl_loader = make_loader(bobsl_train, args.bobsl_batch_size, True, args, 19) if bobsl_train is not None else None
    source_test_loader = make_loader(source_test, args.eval_batch_size, False, args)
    external_loader = make_loader(external_test, args.eval_batch_size, False, args)
    weights = class_weights(payload, source_keys, device) if args.class_weight else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)

    rows = []
    best_score = -1.0
    best_epoch = 0
    best_state = None
    best_source = None
    best_external = None
    print(
        f"[DLCEN] source={len(source_train)} bobsl={len(bobsl_train) if bobsl_train else 0} "
        f"visual={visual_dim} speaker={speaker_dim} basis_counts={counts.tolist()}"
    )
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        training = train_epoch(model, source_loader, bobsl_loader, optimizer, weights, device, args)
        scheduler.step()
        source_result = evaluate(model, source_test_loader, device)
        external_result = evaluate(model, external_loader, device)
        score = source_result[args.selection_metric]
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_source = snapshot(source_result)
            best_external = snapshot(external_result)
        row = {
            "epoch": epoch,
            "train_loss": training["total"],
            "source_weighted_f1": source_result["weighted_f1"],
            "source_macro_f1": source_result["macro_f1"],
            "external_weighted_f1": external_result["weighted_f1"],
            "external_macro_f1": external_result["macro_f1"],
            "lr": optimizer.param_groups[0]["lr"],
            "time_sec": round(time.time() - start, 2),
        }
        rows.append(row)
        print(
            f"[DLCEN] epoch={epoch} train={row['train_loss']:.4f} source={row['source_weighted_f1']:.4f} "
            f"external={row['external_weighted_f1']:.4f} time={row['time_sec']}s"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "epoch_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    model.load_state_dict(best_state)
    metadata = {
        "method": "DLCEN",
        "seed": args.seed,
        "best_epoch": best_epoch,
        "selection_metric": args.selection_metric,
        "train_pkl": str(train_path),
        "external_test_pkl": str(external_path),
        "source_dialogues": len(source_keys),
        "bobsl_dialogues": len(bobsl_keys) if bobsl_loader is not None else 0,
        "args": vars(args),
    }
    write_report(best_source, out_dir, "source_test_best", metadata)
    write_report(best_external, out_dir, "external_test_best", metadata)
    torch.save(
        {"model_state_dict": best_state, "method": "DLCEN", "args": vars(args), "metadata": metadata},
        out_dir / "model_best.pt",
    )
    print(
        f"[DLCEN] done epoch={best_epoch} source={best_source['weighted_f1']:.4f} "
        f"external={best_external['weighted_f1']:.4f} out={out_dir}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the single-checkpoint decoupled local-context visual ERC model.")
    parser.add_argument("--train_pkl", required=True)
    parser.add_argument("--external_test_pkl", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--bobsl_batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--adapter_dim", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--transformer_layers", type=int, default=2)
    parser.add_argument("--gru_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--adapter_dropout", type=float, default=0.2)
    parser.add_argument("--residual_scale", type=float, default=1.0)
    parser.add_argument("--subspace_rank", type=int, default=3)
    parser.add_argument("--projection", choices=["emotion", "full"], default="emotion")
    parser.add_argument("--use_adapter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_transformer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_multiscale_conv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_uncertainty_gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bobsl_weight", type=float, default=0.5)
    parser.add_argument("--bobsl_context_loss_weight", type=float, default=0.0)
    parser.add_argument("--local_loss_weight", type=float, default=0.25)
    parser.add_argument("--context_loss_weight", type=float, default=0.25)
    parser.add_argument("--shift_loss_weight", type=float, default=0.1)
    parser.add_argument("--calibration_weight", type=float, default=0.05)
    parser.add_argument("--residual_weight", type=float, default=0.01)
    parser.add_argument("--class_weight", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--selection_metric", choices=["weighted_f1", "macro_f1"], default="weighted_f1")
    parser.add_argument("--no_cuda", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

