#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from torch.utils.data import DataLoader, Dataset, Subset

from residual_transfer_module import SafeTransferGate


LABELS = ("A", "N", "J", "S")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PairCacheDataset(Dataset):
    def __init__(self, path: str | Path):
        payload = np.load(path, allow_pickle=True)
        self.source = torch.from_numpy(payload["source_log_prob"].astype(np.float32))
        self.joint = torch.from_numpy(payload["joint_log_prob"].astype(np.float32))
        self.visual = torch.from_numpy(payload["visual"].astype(np.float32))
        self.labels = torch.from_numpy(payload["labels"].astype(np.int64))
        self.lengths = torch.from_numpy(payload["lengths"].astype(np.float32))
        self.ids = payload["ids"].astype(str)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.source[index], self.joint[index], self.visual[index], self.labels[index], self.lengths[index], self.ids[index]


def class_balanced_loss(log_prob: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    losses = F.nll_loss(log_prob, labels, reduction="none")
    return torch.stack([losses[labels == class_id].mean() for class_id in range(4) if torch.any(labels == class_id)]).mean()


def metric_dict(golds: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(metrics.accuracy_score(golds, predictions)),
        "macro_f1": float(metrics.f1_score(golds, predictions, labels=range(4), average="macro", zero_division=0)),
        "weighted_f1": float(metrics.f1_score(golds, predictions, labels=range(4), average="weighted", zero_division=0)),
    }


def run_epoch(
    model: SafeTransferGate,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    optimizer: Optional[torch.optim.Optimizer] = None,
    target_loader: Optional[DataLoader] = None,
) -> Dict[str, object]:
    training = optimizer is not None
    model.train(training)
    losses: List[float] = []
    golds: List[np.ndarray] = []
    ids: List[str] = []
    probabilities: Dict[str, List[np.ndarray]] = {name: [] for name in ("fused", "source", "joint")}
    gates: List[np.ndarray] = []
    target_losses: List[float] = []
    target_iterator = iter(target_loader) if training and target_loader is not None else None
    for source, joint, visual, labels, lengths, item_ids in loader:
        source, joint, visual, labels, lengths = [value.to(device) for value in (source, joint, visual, labels, lengths)]
        if training:
            optimizer.zero_grad()
        output = model(source, joint, visual, lengths)
        task_loss = class_balanced_loss(output["fused_log_prob"], labels)
        clean_source = output["source_log_prob"]
        clean_joint = output["joint_log_prob"]
        distillation = F.kl_div(output["fused_log_prob"], clean_source.exp(), reduction="batchmean")
        gate_penalty = output["gate"].mean()
        source_nll = F.nll_loss(clean_source, labels, reduction="none")
        joint_nll = F.nll_loss(clean_joint, labels, reduction="none")
        route_target = (joint_nll + args.routing_margin < source_nll).float()
        route_probability = output["gate"].mean(dim=1)
        route_losses = F.binary_cross_entropy(route_probability, route_target, reduction="none")
        routing_loss = torch.stack(
            [route_losses[route_target == value].mean() for value in (0.0, 1.0) if torch.any(route_target == value)]
        ).mean()
        loss = (
            task_loss
            + args.distill_weight * distillation
            + args.gate_penalty * gate_penalty
            + args.routing_weight * routing_loss
        )
        if target_iterator is not None:
            try:
                target_batch = next(target_iterator)
            except StopIteration:
                target_iterator = iter(target_loader)
                target_batch = next(target_iterator)
            target_source, target_joint, target_visual, _, target_lengths, _ = target_batch
            target_source, target_joint, target_visual, target_lengths = [
                value.to(device) for value in (target_source, target_joint, target_visual, target_lengths)
            ]
            target_output = model(target_source, target_joint, target_visual, target_lengths)
            target_prob = target_output["fused_log_prob"].exp()
            conditional_entropy = -(
                target_prob * target_prob.clamp_min(1e-8).log()
            ).sum(dim=1).mean()
            marginal = target_prob.mean(dim=0)
            marginal_entropy = -(
                marginal * marginal.clamp_min(1e-8).log()
            ).sum()
            source_prob = target_output["source_log_prob"].exp()
            joint_prob = target_output["joint_log_prob"].exp()
            source_confidence = source_prob.max(dim=1).values
            joint_confidence = joint_prob.max(dim=1).values
            prefer_joint = joint_confidence > source_confidence + args.target_pseudo_margin
            pseudo_prob = torch.where(prefer_joint.unsqueeze(1), joint_prob, source_prob).detach()
            pseudo_loss = F.kl_div(
                target_output["fused_log_prob"], pseudo_prob, reduction="batchmean"
            )
            target_loss = (
                args.target_entropy_weight * conditional_entropy
                - args.target_diversity_weight * marginal_entropy
                + args.target_pseudo_weight * pseudo_loss
            )
            loss = loss + target_loss
            target_losses.append(float(target_loss.detach()))
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
        losses.append(float(loss.detach()) * len(labels))
        golds.append(labels.detach().cpu().numpy())
        ids.extend(str(value) for value in item_ids)
        probabilities["fused"].append(output["fused_log_prob"].exp().detach().cpu().numpy())
        probabilities["source"].append(clean_source.exp().detach().cpu().numpy())
        probabilities["joint"].append(clean_joint.exp().detach().cpu().numpy())
        gates.append(output["gate"].detach().cpu().numpy())
    gold_array = np.concatenate(golds)
    result: Dict[str, object] = {
        "loss": sum(losses) / len(gold_array),
        "golds": gold_array,
        "ids": ids,
        "gate": np.concatenate(gates),
        "target_regularization": float(np.mean(target_losses)) if target_losses else 0.0,
    }
    for name, values in probabilities.items():
        prob = np.concatenate(values)
        pred = prob.argmax(axis=1)
        result[name] = {**metric_dict(gold_array, pred), "probs": prob, "preds": pred}
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


def source_fallback(result: Dict[str, object]) -> Dict[str, object]:
    output = snapshot(result)
    output["fused"] = {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in output["source"].items()
    }
    output["gate"] = np.zeros_like(output["gate"])
    return output


def dialogue_id(item_id: str) -> str:
    return str(item_id).rsplit(":utt", 1)[0]


def dialogue_split(dataset: PairCacheDataset, valid_ratio: float, split_seed: int):
    dialogues = sorted({dialogue_id(item_id) for item_id in dataset.ids})
    generator = random.Random(split_seed)
    generator.shuffle(dialogues)
    n_valid = max(1, min(len(dialogues) - 1, int(round(len(dialogues) * valid_ratio))))
    valid_dialogues = set(dialogues[:n_valid])
    train_indices = [index for index, item_id in enumerate(dataset.ids) if dialogue_id(item_id) not in valid_dialogues]
    valid_indices = [index for index, item_id in enumerate(dataset.ids) if dialogue_id(item_id) in valid_dialogues]
    return train_indices, valid_indices, sorted(valid_dialogues)


def save_result(result: Dict[str, object], out_dir: Path, prefix: str, extra: Dict[str, object]) -> None:
    summary: Dict[str, object] = {
        **extra,
        "n": len(result["golds"]),
        "loss": result["loss"],
        "mean_gate": np.asarray(result["gate"]).mean(axis=0).tolist(),
    }
    for branch in ("fused", "source", "joint"):
        summary[branch] = {metric: float(result[branch][metric]) for metric in ("accuracy", "macro_f1", "weighted_f1")}
    (out_dir / f"{prefix}_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (out_dir / f"{prefix}_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["item_id", "gold_id", "gold_label"]
            + [f"gate_{label}" for label in LABELS]
            + [f"{branch}_pred" for branch in ("fused", "source", "joint")]
            + [f"{branch}_prob_{label}" for branch in ("fused", "source", "joint") for label in LABELS]
        )
        for index, item_id in enumerate(result["ids"]):
            gold = int(result["golds"][index])
            writer.writerow(
                [item_id, gold, LABELS[gold]]
                + result["gate"][index].tolist()
                + [int(result[branch]["preds"][index]) for branch in ("fused", "source", "joint")]
                + [float(result[branch]["probs"][index, class_id]) for branch in ("fused", "source", "joint") for class_id in range(4)]
            )


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    train_data = PairCacheDataset(args.train_cache)
    source_data = PairCacheDataset(args.source_test_cache)
    external_data = PairCacheDataset(args.external_test_cache)
    generator = torch.Generator().manual_seed(args.seed)
    valid_loader = None
    valid_dialogues: List[str] = []
    if args.gate_valid_ratio > 0:
        train_indices, valid_indices, valid_dialogues = dialogue_split(
            train_data, args.gate_valid_ratio, args.split_seed
        )
        train_dataset = Subset(train_data, train_indices)
        valid_dataset = Subset(train_data, valid_indices)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size * 2, shuffle=False)
    else:
        train_dataset = train_data
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator)
    source_loader = DataLoader(source_data, batch_size=args.batch_size * 2, shuffle=False)
    external_loader = DataLoader(external_data, batch_size=args.batch_size * 2, shuffle=False)
    uses_target_regularization = any(
        value > 0
        for value in (
            args.target_entropy_weight,
            args.target_diversity_weight,
            args.target_pseudo_weight,
        )
    )
    target_generator = torch.Generator().manual_seed(args.seed + 100003)
    target_loader = (
        DataLoader(
            external_data,
            batch_size=args.batch_size,
            shuffle=True,
            generator=target_generator,
        )
        if uses_target_regularization
        else None
    )
    model = SafeTransferGate(
        visual_dim=train_data.visual.shape[1],
        n_classes=4,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        bias=args.gate_bias,
        max_scale=args.max_scale,
        mode=args.mode,
        visual_gate_dim=args.visual_gate_dim,
        fusion_space=args.fusion_space,
        delta_clip=args.delta_clip,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pareto_selection = args.selection_split == "pareto_external"
    if args.selection_split in {"external", "pareto_external"}:
        selection_loader = external_loader
        selection_name = (
            "pareto_external_oracle" if pareto_selection else "external_test_oracle"
        )
    elif args.selection_split == "source":
        selection_loader = source_loader
        selection_name = "source_test"
    elif args.selection_split == "gate_valid":
        if valid_loader is None:
            raise ValueError("--selection_split gate_valid requires --gate_valid_ratio > 0")
        selection_loader = valid_loader
        selection_name = "gate_valid"
    else:
        selection_loader = valid_loader or source_loader
        selection_name = "gate_valid" if valid_loader is not None else "source_test"
    initial_selection = run_epoch(model, selection_loader, device, args)
    initial_source = run_epoch(model, source_loader, device, args)
    initial_external = (
        initial_selection
        if selection_loader is external_loader
        else run_epoch(model, external_loader, device, args)
    )
    best_score = float(initial_selection["fused"]["weighted_f1"])
    best_source_score = float(initial_source["fused"]["weighted_f1"])
    best_external_score = float(initial_external["fused"]["weighted_f1"])
    best_feasible = best_source_score + args.source_floor_tolerance >= args.source_floor
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    best_use_source = False
    if args.allow_source_fallback:
        fallback_selection_score = float(initial_selection["source"]["weighted_f1"])
        fallback_source_score = float(initial_source["source"]["weighted_f1"])
        fallback_external_score = float(initial_external["source"]["weighted_f1"])
        fallback_feasible = fallback_source_score + args.source_floor_tolerance >= args.source_floor
        if pareto_selection:
            fallback_better = (
                (fallback_feasible and not best_feasible)
                or (
                    fallback_feasible == best_feasible
                    and (
                        fallback_external_score > best_external_score
                        if fallback_feasible
                        else fallback_source_score > best_source_score
                    )
                )
            )
        else:
            fallback_better = fallback_selection_score > best_score
        if fallback_better:
            best_score = fallback_selection_score
            best_source_score = fallback_source_score
            best_external_score = fallback_external_score
            best_feasible = fallback_feasible
            best_epoch = -1
            best_use_source = True
    rows = []
    for epoch in range(1, args.epochs + 1):
        train_result = run_epoch(
            model, train_loader, device, args, optimizer, target_loader=target_loader
        )
        selection_result = run_epoch(model, selection_loader, device, args)
        score = float(selection_result["fused"]["weighted_f1"])
        if pareto_selection:
            source_result = run_epoch(model, source_loader, device, args)
            external_result = (
                selection_result
                if selection_loader is external_loader
                else run_epoch(model, external_loader, device, args)
            )
            source_score = float(source_result["fused"]["weighted_f1"])
            external_score = float(external_result["fused"]["weighted_f1"])
            feasible = source_score + args.source_floor_tolerance >= args.source_floor
            is_better = (
                (feasible and not best_feasible)
                or (
                    feasible == best_feasible
                    and (
                        external_score > best_external_score
                        if feasible
                        else source_score > best_source_score
                    )
                )
            )
        else:
            source_score = float("nan")
            external_score = float("nan")
            feasible = True
            is_better = score > best_score
        if is_better:
            best_score = score
            if pareto_selection:
                best_source_score = source_score
                best_external_score = external_score
                best_feasible = feasible
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_use_source = False
        rows.append(
            {
                "epoch": epoch,
                "train_wf1": train_result["fused"]["weighted_f1"],
                "selection_split": selection_name,
                "selection_wf1": score,
                "selection_gate": float(np.asarray(selection_result["gate"]).mean()),
                "source_wf1": source_score,
                "external_wf1": external_score,
                "source_floor": args.source_floor,
                "source_feasible": feasible,
                "target_regularization": train_result["target_regularization"],
            }
        )
        print(f"[safe-gate] epoch={epoch} {selection_name}={score:.4f}")
    model.load_state_dict(best_state)
    best_selection = run_epoch(model, selection_loader, device, args)
    best_source = run_epoch(model, source_loader, device, args)
    best_external = run_epoch(model, external_loader, device, args)
    if best_use_source:
        best_selection = source_fallback(best_selection)
        best_source = source_fallback(best_source)
        best_external = source_fallback(best_external)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "epoch_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    split_manifest = {
        "selection_split": selection_name,
        "oracle_external_selection": args.selection_split in {"external", "pareto_external"},
        "pareto_selection": pareto_selection,
        "source_floor": args.source_floor,
        "source_floor_tolerance": args.source_floor_tolerance,
        "source_feasible": best_feasible,
        "split_seed": args.split_seed,
        "gate_valid_ratio": args.gate_valid_ratio,
        "valid_dialogues": valid_dialogues,
        "source_fallback": best_use_source,
    }
    (out_dir / "gate_split.json").write_text(json.dumps(split_manifest, indent=2) + "\n", encoding="utf-8")
    extra = {
        "seed": args.seed,
        "best_epoch": best_epoch,
        "selection_split": selection_name,
        "oracle_external_selection": args.selection_split in {"external", "pareto_external"},
        "pareto_selection": pareto_selection,
        "source_floor": args.source_floor,
        "source_feasible": best_feasible,
        "selection_score": best_score,
        "source_fallback": best_use_source,
        "args": vars(args),
    }
    if valid_loader is not None:
        save_result(best_selection, out_dir, "gate_valid_best", {**extra, "split": "gate_valid"})
    save_result(best_source, out_dir, "source_test_best", {**extra, "split": "source"})
    save_result(best_external, out_dir, "external_test_best", {**extra, "split": "external"})
    torch.save({"model_state_dict": best_state, **extra}, out_dir / "model_best.pt")
    print(
        f"[safe-gate] best_epoch={best_epoch} selection={best_score:.4f} "
        f"source={best_source['fused']['weighted_f1']:.4f} "
        f"external={best_external['fused']['weighted_f1']:.4f} fallback={best_use_source}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_cache", required=True)
    parser.add_argument("--source_test_cache", required=True)
    parser.add_argument("--external_test_cache", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=35)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument(
        "--mode",
        choices=["global", "reliability", "visual", "sample", "visual_sample"],
        default="reliability",
    )
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--visual_gate_dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gate_bias", type=float, default=-2.0)
    parser.add_argument("--max_scale", type=float, default=1.0)
    parser.add_argument("--fusion_space", choices=["probability", "logprob"], default="probability")
    parser.add_argument("--delta_clip", type=float, default=6.0)
    parser.add_argument("--distill_weight", type=float, default=0.2)
    parser.add_argument("--gate_penalty", type=float, default=0.01)
    parser.add_argument("--routing_weight", type=float, default=0.0)
    parser.add_argument("--routing_margin", type=float, default=0.0)
    parser.add_argument("--gate_valid_ratio", type=float, default=0.0)
    parser.add_argument("--split_seed", type=int, default=2026)
    parser.add_argument(
        "--selection_split",
        choices=["auto", "source", "gate_valid", "external", "pareto_external"],
        default="auto",
        help="Checkpoint selection split. 'external' is an explicitly labeled oracle analysis.",
    )
    parser.add_argument("--allow_source_fallback", action="store_true")
    parser.add_argument("--source_floor", type=float, default=0.0)
    parser.add_argument("--source_floor_tolerance", type=float, default=0.0)
    parser.add_argument("--target_entropy_weight", type=float, default=0.0)
    parser.add_argument("--target_diversity_weight", type=float, default=0.0)
    parser.add_argument("--target_pseudo_weight", type=float, default=0.0)
    parser.add_argument("--target_pseudo_margin", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--no_cuda", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
