#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(REPO_ROOT / "new"))

from residual_transfer_module import FrozenLogitResidualTransfer  # noqa: E402
from train_eval_domain_routed_cmerc import (  # noqa: E402
    ANJS_LABELS,
    CMERCVisualDataset,
    build_model,
    class_balanced_loss,
    flatten_domain_mask,
    flatten_seq_first,
    inspect_pkl,
    load_checkpoint,
    make_eval_loader,
    make_train_loader,
    metrics_from_arrays,
    namespace_from_checkpoint,
    resolve_path,
    seed_everything,
    sequence_lengths,
)


BRANCHES = ("fused", "base", "source", "bobsl")


class ResidualTransferCMERC(nn.Module):
    def __init__(self, context_model: nn.Module, visual_dim: int, args: argparse.Namespace, has_bobsl: bool):
        super().__init__()
        self.context_model = context_model
        self.transfer = FrozenLogitResidualTransfer(
            visual_dim=visual_dim,
            n_classes=len(ANJS_LABELS),
            hidden_dim=args.adapter_hidden_dim,
            dropout=args.adapter_dropout,
            source_scale=args.source_scale,
            transfer_scale=args.transfer_scale,
            gate_hidden_dim=args.gate_hidden_dim,
            gate_dropout=args.gate_dropout,
            gate_bias=args.gate_bias,
            classwise_gate=args.classwise_gate,
        )
        self.has_bobsl = bool(has_bobsl)
        for parameter in self.context_model.parameters():
            parameter.requires_grad = False
        self.context_model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.context_model.eval()
        return self

    def forward(
        self,
        texts: Sequence[torch.Tensor],
        qmask: torch.Tensor,
        umask: torch.Tensor,
        lengths: Sequence[int],
        acouf: torch.Tensor,
        visuf: torch.Tensor,
        epoch: int,
        detach_transfer_delta: bool,
    ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            base_log_prob, *_ = self.context_model(texts, qmask, umask, lengths, acouf, visuf, epoch)
        visual = flatten_seq_first(visuf, lengths)
        utterance_lengths = torch.cat(
            [torch.full((length,), length, dtype=torch.float32, device=visuf.device) for length in lengths]
        )
        return self.transfer(
            base_log_prob,
            visual,
            utterance_lengths,
            enable_transfer=self.has_bobsl,
            detach_transfer_delta=detach_transfer_delta,
        )


def run_epoch(
    model: ResidualTransferCMERC,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, object]:
    training = optimizer is not None
    model.train(training)
    losses: List[float] = []
    loss_weights: List[int] = []
    golds_all: List[int] = []
    ids_all: List[str] = []
    preds: Dict[str, List[int]] = {branch: [] for branch in BRANCHES}
    probs: Dict[str, List[np.ndarray]] = {branch: [] for branch in BRANCHES}
    gates: List[np.ndarray] = []
    deltas: List[np.ndarray] = []

    for data in dataloader:
        if training:
            optimizer.zero_grad()
        text1, text2, text3, text4, visuf, acouf, qmask, umask, labels = [
            value.to(device) for value in data[:-1]
        ]
        texts = [torch.zeros_like(text) for text in (text1, text2, text3, text4)]
        acouf = torch.zeros_like(acouf)
        lengths = sequence_lengths(umask)
        labels_flat = torch.cat([labels[index][:length] for index, length in enumerate(lengths)])
        valid = labels_flat.ge(0)
        bobsl = flatten_domain_mask(data[-1], lengths, device) & valid
        source = ~bobsl & valid
        outputs = model(
            texts,
            qmask,
            umask,
            lengths,
            acouf,
            visuf,
            epoch,
            detach_transfer_delta=training and model.has_bobsl,
        )

        if training:
            source_loss = class_balanced_loss(outputs["fused_log_prob"][source], labels_flat[source])
            source_expert_loss = class_balanced_loss(outputs["source_logits"][source], labels_flat[source])
            if model.has_bobsl:
                bobsl_loss = class_balanced_loss(outputs["bobsl_logits"][bobsl], labels_flat[bobsl])
                cross_source_loss = class_balanced_loss(outputs["bobsl_logits"][source], labels_flat[source])
                stability = F.kl_div(
                    outputs["fused_log_prob"][source],
                    outputs["source_log_prob"][source].detach().exp(),
                    reduction="batchmean",
                )
                gate_penalty = outputs["gate"][source].mean()
            else:
                zero = outputs["fused_log_prob"].sum() * 0.0
                bobsl_loss = zero
                cross_source_loss = zero
                stability = zero
                gate_penalty = zero
            base_distillation = F.kl_div(
                outputs["source_log_prob"][source],
                outputs["base_log_prob"][source].detach().exp(),
                reduction="batchmean",
            )
            source_residual_penalty = outputs["source_residual"][source].pow(2).mean()
            loss = (
                source_loss
                + args.source_expert_weight * source_expert_loss
                + args.bobsl_weight * bobsl_loss
                + args.cross_source_weight * cross_source_loss
                + args.stability_weight * stability
                + args.gate_penalty * gate_penalty
                + args.base_distill_weight * base_distillation
                + args.source_residual_penalty * source_residual_penalty
            )
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad], args.max_grad_norm
                )
            optimizer.step()
        else:
            loss = class_balanced_loss(outputs["fused_log_prob"][valid], labels_flat[valid])

        valid_count = int(valid.sum().item())
        losses.append(float(loss.detach().item()) * valid_count)
        loss_weights.append(valid_count)
        golds = labels_flat[valid].detach().cpu().numpy().astype(np.int64)
        golds_all.extend(golds.tolist())
        branch_log_probs = {
            "fused": outputs["fused_log_prob"],
            "base": outputs["base_log_prob"],
            "source": outputs["source_log_prob"],
            "bobsl": outputs["bobsl_log_prob"],
        }
        for branch, log_prob in branch_log_probs.items():
            selected = log_prob[valid]
            preds[branch].extend(selected.argmax(dim=-1).detach().cpu().numpy().astype(np.int64).tolist())
            probs[branch].append(selected.exp().detach().cpu().numpy().astype(np.float32))
        gates.append(outputs["gate"][valid].detach().cpu().numpy().astype(np.float32))
        deltas.append(outputs["transfer_delta"][valid].detach().cpu().numpy().astype(np.float32))
        label_matrix = labels.detach().cpu()
        for batch_index, vid in enumerate(data[-1]):
            for utterance_index in range(lengths[batch_index]):
                if int(label_matrix[batch_index, utterance_index]) >= 0:
                    ids_all.append(f"{vid}:utt{utterance_index + 1:02d}")

    gold_array = np.asarray(golds_all, dtype=np.int64)
    result: Dict[str, object] = {
        "loss": float(np.sum(losses) / max(np.sum(loss_weights), 1)),
        "golds": gold_array,
        "ids": ids_all,
        "gate": np.concatenate(gates, axis=0),
        "delta": np.concatenate(deltas, axis=0),
    }
    for branch in BRANCHES:
        pred_array = np.asarray(preds[branch], dtype=np.int64)
        result[branch] = {
            **metrics_from_arrays(gold_array, pred_array),
            "preds": pred_array,
            "probs": np.concatenate(probs[branch], axis=0),
        }
    return result


def snapshot(result: Dict[str, object]) -> Dict[str, object]:
    copied: Dict[str, object] = {
        "loss": result["loss"],
        "golds": result["golds"].copy(),
        "ids": list(result["ids"]),
        "gate": result["gate"].copy(),
        "delta": result["delta"].copy(),
    }
    for branch in BRANCHES:
        copied[branch] = {
            key: value.copy() if isinstance(value, np.ndarray) else value for key, value in result[branch].items()
        }
    return copied


def save_result(
    result: Dict[str, object], out_dir: Path, prefix: str, extra: Dict[str, object]
) -> Dict[str, object]:
    summary: Dict[str, object] = {
        **extra,
        "n": len(result["golds"]),
        "loss": result["loss"],
        "mean_gate": np.asarray(result["gate"]).mean(axis=0).tolist(),
        "mean_abs_delta": np.abs(np.asarray(result["delta"])).mean(axis=0).tolist(),
    }
    for branch in BRANCHES:
        summary[branch] = {
            metric: float(result[branch][metric]) for metric in ("accuracy", "macro_f1", "weighted_f1")
        }
        summary[branch]["pred_counts"] = {
            label: int(np.sum(result[branch]["preds"] == class_id))
            for class_id, label in enumerate(ANJS_LABELS)
        }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{prefix}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (out_dir / f"{prefix}_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["item_id", "gold_id", "gold_label"]
            + [f"gate_{label}" for label in ANJS_LABELS]
            + [f"delta_{label}" for label in ANJS_LABELS]
            + [f"{branch}_pred" for branch in BRANCHES]
            + [f"{branch}_prob_{label}" for branch in BRANCHES for label in ANJS_LABELS]
        )
        for index, item_id in enumerate(result["ids"]):
            writer.writerow(
                [item_id, int(result["golds"][index]), ANJS_LABELS[int(result["golds"][index])]]
                + result["gate"][index].tolist()
                + result["delta"][index].tolist()
                + [int(result[branch]["preds"][index]) for branch in BRANCHES]
                + [
                    float(result[branch]["probs"][index, class_id])
                    for branch in BRANCHES
                    for class_id in range(len(ANJS_LABELS))
                ]
            )
    return summary


def save_checkpoint(
    path: Path, model: ResidualTransferCMERC, args: argparse.Namespace, epoch: int, score: float
) -> None:
    state = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if not key.startswith("context_model.")
    }
    torch.save(
        {
            "model_state_dict": state,
            "context_checkpoint": str(Path(args.context_checkpoint).resolve()),
            "args": vars(args),
            "epoch": epoch,
            "best_score": score,
            "baseline": "ResidualTransferCMERC",
        },
        path,
    )


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CMERC graph implementation requires CUDA.")
    train_pkl = resolve_path(args.train_pkl, must_exist=True)
    source_pkl = resolve_path(args.source_pkl, must_exist=True)
    external_pkl = resolve_path(args.external_test_pkl, must_exist=True)
    context_checkpoint_path = resolve_path(args.context_checkpoint, must_exist=True)
    out_dir = resolve_path(args.out_dir, must_exist=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = inspect_pkl(train_pkl)
    for path in (source_pkl, external_pkl):
        other = inspect_pkl(path)
        for key in ("d_text", "d_audio", "d_visual", "n_speakers", "n_classes"):
            if int(info[key]) != int(other[key]):
                raise RuntimeError(f"Feature mismatch for {key}: train={info[key]} other={other[key]}")

    context_checkpoint = load_checkpoint(context_checkpoint_path, device)
    context_args = namespace_from_checkpoint(context_checkpoint, args)
    context_model = build_model(context_args, info, device)
    context_model.load_state_dict(context_checkpoint["model_state_dict"], strict=True)
    train_dataset = CMERCVisualDataset(train_pkl, "train")
    has_bobsl = any("bobsl" in str(key).lower() for key in train_dataset.keys)
    model = ResidualTransferCMERC(context_model, int(info["d_visual"]), args, has_bobsl).to(device)
    if args.init_source_checkpoint:
        init_path = resolve_path(args.init_source_checkpoint, must_exist=True)
        init_checkpoint = load_checkpoint(init_path, device)
        state = init_checkpoint.get("model_state_dict") or {}
        source_state = {
            key: value for key, value in state.items() if key.startswith("transfer.source_adapter.")
        }
        missing, unexpected = model.load_state_dict(source_state, strict=False)
        model.transfer.initialize_bobsl_from_source()
        model.transfer.freeze_source()
        print(
            f"[RT-CMERC] init_source={init_path} loaded={len(source_state)} "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
    elif has_bobsl:
        raise ValueError("Joint training requires --init_source_checkpoint")

    train_loader = make_train_loader(args, train_dataset)
    source_loader = make_eval_loader(args, CMERCVisualDataset(source_pkl, "test"))
    external_loader = make_eval_loader(args, CMERCVisualDataset(external_pkl, "test"))
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    print(f"[RT-CMERC] train_pkl={train_pkl} has_bobsl={has_bobsl}")
    print(f"[RT-CMERC] context_checkpoint={context_checkpoint_path}")
    print(f"[RT-CMERC] out_dir={out_dir} device={device}")

    initial_source = run_epoch(model, source_loader, device, 0, args)
    initial_external = run_epoch(model, external_loader, device, 0, args)
    best_score = float(initial_source["fused"][args.selection_metric])
    best_epoch = 0
    best_state = copy.deepcopy(
        {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
            if not key.startswith("context_model.")
        }
    )
    best_source = snapshot(initial_source)
    best_external = snapshot(initial_external)
    epoch_rows: List[Dict[str, object]] = [
        {
            "epoch": 0,
            "time_sec": 0.0,
            "train_fused_wf1": float("nan"),
            "source_fused_wf1": initial_source["fused"]["weighted_f1"],
            "source_base_wf1": initial_source["base"]["weighted_f1"],
            "source_adapter_wf1": initial_source["source"]["weighted_f1"],
            "source_bobsl_wf1": initial_source["bobsl"]["weighted_f1"],
            "source_gate": float(np.asarray(initial_source["gate"]).mean()),
            "external_fused_wf1": initial_external["fused"]["weighted_f1"],
            "external_base_wf1": initial_external["base"]["weighted_f1"],
            "external_adapter_wf1": initial_external["source"]["weighted_f1"],
            "external_bobsl_wf1": initial_external["bobsl"]["weighted_f1"],
            "external_gate": float(np.asarray(initial_external["gate"]).mean()),
        }
    ]
    print(
        f"[RT-CMERC] epoch=0 source={best_score:.4f} "
        f"external={initial_external['fused']['weighted_f1']:.4f}"
    )
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_result = run_epoch(model, train_loader, device, epoch, args, optimizer)
        source_result = run_epoch(model, source_loader, device, epoch, args)
        external_result = run_epoch(model, external_loader, device, epoch, args)
        score = float(source_result["fused"][args.selection_metric])
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(
                {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                    if not key.startswith("context_model.")
                }
            )
            best_source = snapshot(source_result)
            best_external = snapshot(external_result)
            print(f"[RT-CMERC] new best epoch={epoch} source_{args.selection_metric}={score:.4f}")
        row = {
            "epoch": epoch,
            "time_sec": round(time.time() - start, 2),
            "train_fused_wf1": train_result["fused"]["weighted_f1"],
            "source_fused_wf1": source_result["fused"]["weighted_f1"],
            "source_base_wf1": source_result["base"]["weighted_f1"],
            "source_adapter_wf1": source_result["source"]["weighted_f1"],
            "source_bobsl_wf1": source_result["bobsl"]["weighted_f1"],
            "source_gate": float(np.asarray(source_result["gate"]).mean()),
            "external_fused_wf1": external_result["fused"]["weighted_f1"],
            "external_base_wf1": external_result["base"]["weighted_f1"],
            "external_adapter_wf1": external_result["source"]["weighted_f1"],
            "external_bobsl_wf1": external_result["bobsl"]["weighted_f1"],
            "external_gate": float(np.asarray(external_result["gate"]).mean()),
        }
        epoch_rows.append(row)
        print(
            f"[RT-CMERC] epoch={epoch} source={row['source_fused_wf1']:.4f} "
            f"external={row['external_fused_wf1']:.4f} gate={row['source_gate']:.3f}/"
            f"{row['external_gate']:.3f} time={row['time_sec']}s"
        )

    with (out_dir / "epoch_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(epoch_rows[0]))
        writer.writeheader()
        writer.writerows(epoch_rows)
    extra = {
        "baseline": "ResidualTransferCMERC",
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_source_score": best_score,
        "train_pkl": str(train_pkl),
        "source_pkl": str(source_pkl),
        "external_test_pkl": str(external_pkl),
        "context_checkpoint": str(context_checkpoint_path),
        "has_bobsl": has_bobsl,
        "args": vars(args),
    }
    source_summary = save_result(best_source, out_dir, "source_test_best", {**extra, "split": "source"})
    external_summary = save_result(
        best_external, out_dir, "external_test_best", {**extra, "split": "external"}
    )
    model.load_state_dict(best_state, strict=False)
    save_checkpoint(out_dir / "model_best.pt", model, args, best_epoch, best_score)
    print(
        f"[RT-CMERC] best_epoch={best_epoch} source_wf1={source_summary['fused']['weighted_f1']:.4f} "
        f"external_wf1={external_summary['fused']['weighted_f1']:.4f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CMERC with a reusable bounded residual transfer module.")
    parser.add_argument("--train_pkl", required=True)
    parser.add_argument("--source_pkl", required=True)
    parser.add_argument("--external_test_pkl", required=True)
    parser.add_argument("--context_checkpoint", required=True)
    parser.add_argument("--init_source_checkpoint", default="")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=35)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--context_dropout", type=float, default=0.4)
    parser.add_argument("--adapter_hidden_dim", type=int, default=256)
    parser.add_argument("--adapter_dropout", type=float, default=0.2)
    parser.add_argument("--source_scale", type=float, default=1.0)
    parser.add_argument("--transfer_scale", type=float, default=0.5)
    parser.add_argument("--gate_hidden_dim", type=int, default=32)
    parser.add_argument("--gate_dropout", type=float, default=0.1)
    parser.add_argument("--gate_bias", type=float, default=-3.0)
    parser.add_argument("--classwise_gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--source_expert_weight", type=float, default=0.25)
    parser.add_argument("--bobsl_weight", type=float, default=1.0)
    parser.add_argument("--cross_source_weight", type=float, default=0.0)
    parser.add_argument("--stability_weight", type=float, default=0.2)
    parser.add_argument("--gate_penalty", type=float, default=0.01)
    parser.add_argument("--base_distill_weight", type=float, default=1.0)
    parser.add_argument("--source_residual_penalty", type=float, default=0.001)
    parser.add_argument("--source_repeat", type=int, default=2)
    parser.add_argument("--min_domain_samples_per_epoch", type=int, default=500)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument(
        "--selection_metric", choices=["weighted_f1", "macro_f1", "accuracy"], default="weighted_f1"
    )
    parser.add_argument("--no_cuda", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
