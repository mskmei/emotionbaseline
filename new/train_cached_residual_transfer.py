#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from residual_transfer_module import FrozenLogitResidualTransfer


LABELS = ("A", "N", "J", "S")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class LogitCacheDataset(Dataset):
    def __init__(self, path: str | Path):
        payload = np.load(path, allow_pickle=True)
        self.base_log_prob = torch.from_numpy(payload["base_log_prob"].astype(np.float32))
        self.visual = torch.from_numpy(payload["visual"].astype(np.float32))
        self.labels = torch.from_numpy(payload["labels"].astype(np.int64))
        self.lengths = torch.from_numpy(payload["lengths"].astype(np.float32))
        self.domain = torch.from_numpy(payload["domain"].astype(bool))
        self.ids = payload["ids"].astype(str)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return (
            self.base_log_prob[index],
            self.visual[index],
            self.labels[index],
            self.lengths[index],
            self.domain[index],
            self.ids[index],
        )


def class_balanced_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if not len(labels):
        return logits.sum() * 0.0
    losses = F.cross_entropy(logits, labels, reduction="none")
    per_class = [losses[labels == class_id].mean() for class_id in range(len(LABELS)) if torch.any(labels == class_id)]
    return torch.stack(per_class).mean()


def make_train_loader(dataset: LogitCacheDataset, args: argparse.Namespace) -> DataLoader:
    domain = dataset.domain.numpy()
    generator = torch.Generator().manual_seed(args.seed)
    if domain.any() and (~domain).any():
        source_count = int((~domain).sum())
        bobsl_count = int(domain.sum())
        weights = np.where(domain, 0.5 / bobsl_count, 0.5 / source_count)
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=2 * min(bobsl_count, max(source_count, args.min_domain_samples)),
            replacement=True,
            generator=generator,
        )
        return DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=0)


def metric_dict(golds: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(metrics.accuracy_score(golds, preds)),
        "macro_f1": float(metrics.f1_score(golds, preds, labels=range(4), average="macro", zero_division=0)),
        "weighted_f1": float(metrics.f1_score(golds, preds, labels=range(4), average="weighted", zero_division=0)),
    }


def run_epoch(
    model: FrozenLogitResidualTransfer,
    loader: DataLoader,
    device: torch.device,
    has_bobsl: bool,
    args: argparse.Namespace,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, object]:
    training = optimizer is not None
    model.train(training)
    totals: List[float] = []
    counts: List[int] = []
    golds: List[np.ndarray] = []
    ids: List[str] = []
    branch_probs: Dict[str, List[np.ndarray]] = {name: [] for name in ("fused", "base", "source", "bobsl")}
    gates: List[np.ndarray] = []
    deltas: List[np.ndarray] = []
    for base_log_prob, visual, labels, lengths, domain, item_ids in loader:
        base_log_prob = base_log_prob.to(device)
        visual = visual.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)
        domain = domain.to(device)
        if training:
            optimizer.zero_grad()
        output = model(
            base_log_prob,
            visual,
            lengths,
            enable_transfer=has_bobsl,
            detach_transfer_delta=training and has_bobsl,
        )
        source = ~domain
        if training:
            source_loss = class_balanced_loss(output["fused_log_prob"][source], labels[source])
            source_expert_loss = class_balanced_loss(output["source_logits"][source], labels[source])
            base_distill = F.kl_div(
                output["source_log_prob"][source],
                output["base_log_prob"][source].detach().exp(),
                reduction="batchmean",
            )
            if has_bobsl:
                bobsl_loss = class_balanced_loss(output["bobsl_logits"][domain], labels[domain])
                cross_source = class_balanced_loss(output["bobsl_logits"][source], labels[source])
                stability = F.kl_div(
                    output["fused_log_prob"][source],
                    output["source_log_prob"][source].detach().exp(),
                    reduction="batchmean",
                )
                gate_penalty = output["gate"][source].mean()
            else:
                zero = output["fused_log_prob"].sum() * 0.0
                bobsl_loss = cross_source = stability = gate_penalty = zero
            residual_penalty = output["source_residual"][source].pow(2).mean()
            loss = (
                source_loss
                + args.source_expert_weight * source_expert_loss
                + args.base_distill_weight * base_distill
                + args.bobsl_weight * bobsl_loss
                + args.cross_source_weight * cross_source
                + args.stability_weight * stability
                + args.gate_penalty * gate_penalty
                + args.source_residual_penalty * residual_penalty
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad], args.max_grad_norm
            )
            optimizer.step()
        else:
            loss = F.nll_loss(output["fused_log_prob"], labels)
        totals.append(float(loss.detach()) * len(labels))
        counts.append(len(labels))
        golds.append(labels.detach().cpu().numpy())
        ids.extend(str(value) for value in item_ids)
        for name, key in (
            ("fused", "fused_log_prob"),
            ("base", "base_log_prob"),
            ("source", "source_log_prob"),
            ("bobsl", "bobsl_log_prob"),
        ):
            branch_probs[name].append(output[key].exp().detach().cpu().numpy())
        gates.append(output["gate"].detach().cpu().numpy())
        deltas.append(output["transfer_delta"].detach().cpu().numpy())
    gold_array = np.concatenate(golds)
    result: Dict[str, object] = {
        "loss": sum(totals) / sum(counts),
        "golds": gold_array,
        "ids": ids,
        "gate": np.concatenate(gates),
        "delta": np.concatenate(deltas),
    }
    for name, values in branch_probs.items():
        probabilities = np.concatenate(values)
        predictions = probabilities.argmax(axis=1)
        result[name] = {**metric_dict(gold_array, predictions), "probs": probabilities, "preds": predictions}
    return result


def snapshot(result: Dict[str, object]) -> Dict[str, object]:
    output: Dict[str, object] = {}
    for key, value in result.items():
        if isinstance(value, np.ndarray):
            output[key] = value.copy()
        elif isinstance(value, list):
            output[key] = list(value)
        elif isinstance(value, dict):
            output[key] = {name: item.copy() if isinstance(item, np.ndarray) else item for name, item in value.items()}
        else:
            output[key] = value
    return output


def save_result(result: Dict[str, object], out_dir: Path, prefix: str, extra: Mapping[str, object]) -> None:
    summary: Dict[str, object] = {
        **extra,
        "n": len(result["golds"]),
        "loss": result["loss"],
        "mean_gate": np.asarray(result["gate"]).mean(axis=0).tolist(),
        "mean_abs_delta": np.abs(np.asarray(result["delta"])).mean(axis=0).tolist(),
    }
    for branch in ("fused", "base", "source", "bobsl"):
        summary[branch] = {
            metric: float(result[branch][metric]) for metric in ("accuracy", "macro_f1", "weighted_f1")
        }
    (out_dir / f"{prefix}_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (out_dir / f"{prefix}_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["item_id", "gold_id", "gold_label"]
            + [f"gate_{label}" for label in LABELS]
            + [f"delta_{label}" for label in LABELS]
            + [f"{branch}_pred" for branch in ("fused", "base", "source", "bobsl")]
        )
        for index, item_id in enumerate(result["ids"]):
            gold = int(result["golds"][index])
            writer.writerow(
                [item_id, gold, LABELS[gold]]
                + result["gate"][index].tolist()
                + result["delta"][index].tolist()
                + [int(result[branch]["preds"][index]) for branch in ("fused", "base", "source", "bobsl")]
            )


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    train_dataset = LogitCacheDataset(args.train_cache)
    source_dataset = LogitCacheDataset(args.source_test_cache)
    external_dataset = LogitCacheDataset(args.external_test_cache)
    has_bobsl = bool(train_dataset.domain.any())
    model = FrozenLogitResidualTransfer(
        visual_dim=train_dataset.visual.shape[1],
        n_classes=4,
        hidden_dim=args.adapter_hidden_dim,
        dropout=args.adapter_dropout,
        source_scale=args.source_scale,
        transfer_scale=args.transfer_scale,
        gate_hidden_dim=args.gate_hidden_dim,
        gate_dropout=args.gate_dropout,
        gate_bias=args.gate_bias,
        classwise_gate=args.classwise_gate,
    ).to(device)
    if args.init_source_checkpoint:
        checkpoint = torch.load(args.init_source_checkpoint, map_location=device, weights_only=False)
        source_state = {
            key: value for key, value in checkpoint["model_state_dict"].items() if key.startswith("source_adapter.")
        }
        model.load_state_dict(source_state, strict=False)
        model.initialize_bobsl_from_source()
        model.freeze_source()
    elif has_bobsl:
        raise ValueError("Joint cache requires --init_source_checkpoint")
    train_loader = make_train_loader(train_dataset, args)
    source_loader = DataLoader(source_dataset, batch_size=args.batch_size * 2, shuffle=False)
    external_loader = DataLoader(external_dataset, batch_size=args.batch_size * 2, shuffle=False)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    initial_source = run_epoch(model, source_loader, device, has_bobsl, args)
    initial_external = run_epoch(model, external_loader, device, has_bobsl, args)
    best_score = float(initial_source["fused"]["weighted_f1"])
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    best_source = snapshot(initial_source)
    best_external = snapshot(initial_external)
    epoch_rows = []
    print(f"[cached-transfer] epoch=0 source={best_score:.4f} external={initial_external['fused']['weighted_f1']:.4f}")
    for epoch in range(1, args.epochs + 1):
        train_result = run_epoch(model, train_loader, device, has_bobsl, args, optimizer)
        source_result = run_epoch(model, source_loader, device, has_bobsl, args)
        external_result = run_epoch(model, external_loader, device, has_bobsl, args)
        score = float(source_result["fused"]["weighted_f1"])
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_source = snapshot(source_result)
            best_external = snapshot(external_result)
        epoch_rows.append(
            {
                "epoch": epoch,
                "train_wf1": train_result["fused"]["weighted_f1"],
                "source_wf1": score,
                "external_wf1": external_result["fused"]["weighted_f1"],
                "source_gate": float(np.asarray(source_result["gate"]).mean()),
                "external_gate": float(np.asarray(external_result["gate"]).mean()),
            }
        )
        print(
            f"[cached-transfer] epoch={epoch} source={score:.4f} external={external_result['fused']['weighted_f1']:.4f}"
        )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "epoch_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(epoch_rows[0]))
        writer.writeheader()
        writer.writerows(epoch_rows)
    extra = {"seed": args.seed, "best_epoch": best_epoch, "args": vars(args), "has_bobsl": has_bobsl}
    save_result(best_source, out_dir, "source_test_best", {**extra, "split": "source"})
    save_result(best_external, out_dir, "external_test_best", {**extra, "split": "external"})
    torch.save(
        {"model_state_dict": best_state, "args": vars(args), "best_epoch": best_epoch, "best_score": best_score},
        out_dir / "model_best.pt",
    )
    print(
        f"[cached-transfer] best_epoch={best_epoch} source={best_source['fused']['weighted_f1']:.4f} "
        f"external={best_external['fused']['weighted_f1']:.4f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_cache", required=True)
    parser.add_argument("--source_test_cache", required=True)
    parser.add_argument("--external_test_cache", required=True)
    parser.add_argument("--init_source_checkpoint", default="")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=35)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--adapter_hidden_dim", type=int, default=128)
    parser.add_argument("--adapter_dropout", type=float, default=0.2)
    parser.add_argument("--source_scale", type=float, default=0.25)
    parser.add_argument("--transfer_scale", type=float, default=0.5)
    parser.add_argument("--gate_hidden_dim", type=int, default=32)
    parser.add_argument("--gate_dropout", type=float, default=0.1)
    parser.add_argument("--gate_bias", type=float, default=-3.0)
    parser.add_argument("--classwise_gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--source_expert_weight", type=float, default=0.25)
    parser.add_argument("--base_distill_weight", type=float, default=2.0)
    parser.add_argument("--bobsl_weight", type=float, default=1.0)
    parser.add_argument("--cross_source_weight", type=float, default=0.0)
    parser.add_argument("--stability_weight", type=float, default=0.2)
    parser.add_argument("--gate_penalty", type=float, default=0.01)
    parser.add_argument("--source_residual_penalty", type=float, default=0.01)
    parser.add_argument("--min_domain_samples", type=int, default=5000)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--no_cuda", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
