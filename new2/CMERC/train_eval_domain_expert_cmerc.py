#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_eval_domain_routed_cmerc import (
    ANJS_LABELS,
    CMERCVisualDataset,
    CosineLocalExpert,
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
    prototype_alignment,
    resolve_path,
    seed_everything,
    sequence_lengths,
)


BRANCHES = ("fused", "context", "source_local", "bobsl_local")


class MultiExpertGate(nn.Module):
    def __init__(self, hidden_dim: int, biases: Sequence[float]):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(10, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 3))
        nn.init.zeros_(self.network[-1].weight)
        with torch.no_grad():
            self.network[-1].bias.copy_(torch.tensor(list(biases), dtype=torch.float32))

    @staticmethod
    def reliability_features(
        context_log_prob: torch.Tensor,
        source_log_prob: torch.Tensor,
        bobsl_log_prob: torch.Tensor,
        utterance_lengths: torch.Tensor,
    ) -> torch.Tensor:
        log_probs = [value.detach() for value in (context_log_prob, source_log_prob, bobsl_log_prob)]
        probs = [value.exp() for value in log_probs]
        normalizer = math.log(float(len(ANJS_LABELS)))
        entropies = [-(prob * log_prob).sum(dim=-1, keepdim=True) / normalizer for prob, log_prob in zip(probs, log_probs)]
        confidences = [prob.max(dim=-1, keepdim=True).values for prob in probs]
        disagreements = [
            0.5 * torch.abs(probs[0] - probs[1]).sum(dim=-1, keepdim=True),
            0.5 * torch.abs(probs[0] - probs[2]).sum(dim=-1, keepdim=True),
            0.5 * torch.abs(probs[1] - probs[2]).sum(dim=-1, keepdim=True),
        ]
        length_feature = torch.log1p(utterance_lengths.float()).unsqueeze(1) / math.log(51.0)
        return torch.cat([*entropies, *confidences, *disagreements, length_feature], dim=1)

    def forward(
        self,
        context_log_prob: torch.Tensor,
        source_log_prob: torch.Tensor,
        bobsl_log_prob: torch.Tensor,
        utterance_lengths: torch.Tensor,
        has_bobsl: bool,
    ) -> torch.Tensor:
        features = self.reliability_features(
            context_log_prob, source_log_prob, bobsl_log_prob, utterance_lengths
        )
        logits = self.network(features)
        if not has_bobsl:
            logits = logits.clone()
            logits[:, 2] = -1e4
        return torch.softmax(logits, dim=-1)


class ResidualBOBSLGate(nn.Module):
    """Adds BOBSL evidence without changing the source-trained context/local ratio."""

    def __init__(self, hidden_dim: int, bias: float):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(10, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.constant_(self.network[-1].bias, bias)

    def forward(
        self,
        context_log_prob: torch.Tensor,
        source_log_prob: torch.Tensor,
        bobsl_log_prob: torch.Tensor,
        utterance_lengths: torch.Tensor,
    ) -> torch.Tensor:
        features = MultiExpertGate.reliability_features(
            context_log_prob, source_log_prob, bobsl_log_prob, utterance_lengths
        )
        return torch.sigmoid(self.network(features))


class DomainExpertCMERC(nn.Module):
    def __init__(self, context_model: nn.Module, visual_dim: int, args: argparse.Namespace, has_bobsl: bool):
        super().__init__()
        self.context_model = context_model
        expert_args = (visual_dim, args.local_hidden_dim, args.local_embedding_dim, args.local_dropout, args.cosine_scale)
        self.source_expert = CosineLocalExpert(*expert_args)
        self.bobsl_expert = CosineLocalExpert(*expert_args)
        self.gate = MultiExpertGate(args.gate_hidden_dim, args.gate_biases)
        self.bobsl_gate = ResidualBOBSLGate(args.gate_hidden_dim, args.residual_gate_bias)
        self.residual_bobsl_gate = bool(args.residual_bobsl_gate)
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
    ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            context_log_prob, *_ = self.context_model(texts, qmask, umask, lengths, acouf, visuf, epoch)
        visual = flatten_seq_first(visuf, lengths)
        source_logits, source_embedding = self.source_expert(visual)
        bobsl_logits, bobsl_embedding = self.bobsl_expert(visual)
        source_log_prob = F.log_softmax(source_logits, dim=-1)
        bobsl_log_prob = F.log_softmax(bobsl_logits, dim=-1)
        utterance_lengths = torch.cat(
            [torch.full((length,), length, dtype=torch.float32, device=visuf.device) for length in lengths]
        )
        if self.residual_bobsl_gate:
            base_gate = self.gate(
                context_log_prob, source_log_prob, bobsl_log_prob, utterance_lengths, False
            )
            if self.has_bobsl:
                bobsl_weight = self.bobsl_gate(
                    context_log_prob, source_log_prob, bobsl_log_prob, utterance_lengths
                )
            else:
                bobsl_weight = torch.zeros_like(base_gate[:, :1])
            gate = torch.cat([base_gate[:, :2] * (1.0 - bobsl_weight), bobsl_weight], dim=1)
        else:
            gate = self.gate(
                context_log_prob, source_log_prob, bobsl_log_prob, utterance_lengths, self.has_bobsl
            )
        probabilities = torch.stack(
            [context_log_prob.exp(), source_log_prob.exp(), bobsl_log_prob.exp()], dim=1
        )
        fused_log_prob = torch.log(torch.clamp((gate.unsqueeze(-1) * probabilities).sum(dim=1), min=1e-12))
        return {
            "fused_log_prob": fused_log_prob,
            "context_log_prob": context_log_prob,
            "source_local_log_prob": source_log_prob,
            "bobsl_local_log_prob": bobsl_log_prob,
            "source_logits": source_logits,
            "bobsl_logits": bobsl_logits,
            "source_embedding": source_embedding,
            "bobsl_embedding": bobsl_embedding,
            "gate": gate,
        }


def expert_alignment(model: DomainExpertCMERC) -> torch.Tensor:
    source = F.normalize(model.source_expert.prototypes, dim=-1)
    bobsl = F.normalize(model.bobsl_expert.prototypes, dim=-1)
    return (1.0 - torch.sum(source * bobsl, dim=-1)).mean()


def routed_log_prob(
    outputs: Dict[str, torch.Tensor],
    keep: torch.Tensor,
    train_expert: str,
) -> torch.Tensor:
    gate = outputs["gate"][keep]
    context_prob = outputs["context_log_prob"][keep].exp().detach()
    source_prob = outputs["source_local_log_prob"][keep].exp()
    bobsl_prob = outputs["bobsl_local_log_prob"][keep].exp()
    if train_expert == "source":
        bobsl_prob = bobsl_prob.detach()
    elif train_expert == "bobsl":
        source_prob = source_prob.detach()
    probabilities = torch.stack([context_prob, source_prob, bobsl_prob], dim=1)
    return torch.log(torch.clamp((gate.unsqueeze(-1) * probabilities).sum(dim=1), min=1e-12))


def run_epoch(
    model: DomainExpertCMERC,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict:
    training = optimizer is not None
    model.train(training)
    losses: List[float] = []
    loss_weights: List[int] = []
    golds_all: List[int] = []
    preds: Dict[str, List[int]] = {branch: [] for branch in BRANCHES}
    probs: Dict[str, List[np.ndarray]] = {branch: [] for branch in BRANCHES}
    ids_all: List[str] = []
    gates: List[np.ndarray] = []

    for data in dataloader:
        if training:
            optimizer.zero_grad()
        text1, text2, text3, text4, visuf, acouf, qmask, umask, labels = [value.to(device) for value in data[:-1]]
        raw_texts = [text1, text2, text3, text4]
        if args.modality == "visual":
            texts = [torch.zeros_like(text) for text in raw_texts]
        elif args.modality == "tv":
            texts = raw_texts
        else:
            raise ValueError(f"Unsupported modality={args.modality}")
        acouf = torch.zeros_like(acouf)
        lengths = sequence_lengths(umask)
        labels_flat = torch.cat([labels[index][:length] for index, length in enumerate(lengths)])
        valid = labels_flat.ge(0)
        bobsl = flatten_domain_mask(data[-1], lengths, device) & valid
        source = ~bobsl & valid
        outputs = model(texts, qmask, umask, lengths, acouf, visuf, epoch)

        if training:
            source_fused = routed_log_prob(outputs, source, "source")
            bobsl_fused = routed_log_prob(outputs, bobsl, "bobsl")
            source_fused_loss = class_balanced_loss(source_fused, labels_flat[source])
            bobsl_fused_loss = class_balanced_loss(bobsl_fused, labels_flat[bobsl])
            source_expert_loss = class_balanced_loss(outputs["source_logits"][source], labels_flat[source])
            bobsl_expert_loss = class_balanced_loss(outputs["bobsl_logits"][bobsl], labels_flat[bobsl])
            cross_source_loss = class_balanced_loss(outputs["bobsl_logits"][source], labels_flat[source])
            if torch.any(source):
                consistency = F.kl_div(
                    source_fused,
                    outputs["context_log_prob"][source].exp(),
                    reduction="batchmean",
                )
                context_gate_penalty = (1.0 - outputs["gate"][source, 0]).mean()
            else:
                consistency = outputs["gate"].sum() * 0.0
                context_gate_penalty = outputs["gate"].sum() * 0.0
            alignment_loss = expert_alignment(model) if model.has_bobsl else outputs["gate"].sum() * 0.0
            embedding_alignment_loss = (
                prototype_alignment(outputs["bobsl_embedding"], labels_flat, bobsl)
                if model.has_bobsl
                else outputs["gate"].sum() * 0.0
            )
            loss = (
                source_fused_loss
                + args.bobsl_fused_weight * bobsl_fused_loss
                + args.local_source_weight * source_expert_loss
                + args.bobsl_weight * bobsl_expert_loss
                + args.cross_source_weight * cross_source_loss
                + args.alignment_weight * alignment_loss
                + args.embedding_alignment_weight * embedding_alignment_loss
                + args.distill_weight * consistency
                + args.gate_penalty * context_gate_penalty
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
        for branch in BRANCHES:
            branch_log_prob = outputs[f"{branch}_log_prob"][valid]
            preds[branch].extend(branch_log_prob.argmax(dim=-1).detach().cpu().numpy().astype(np.int64).tolist())
            probs[branch].append(branch_log_prob.exp().detach().cpu().numpy().astype(np.float32))
        gates.append(outputs["gate"][valid].detach().cpu().numpy().astype(np.float32))
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
    }
    for branch in BRANCHES:
        pred_array = np.asarray(preds[branch], dtype=np.int64)
        result[branch] = {
            **metrics_from_arrays(gold_array, pred_array),
            "preds": pred_array,
            "probs": np.concatenate(probs[branch], axis=0),
        }
    return result


def snapshot(result: Dict) -> Dict:
    copied: Dict[str, object] = {
        "loss": result["loss"],
        "golds": result["golds"].copy(),
        "ids": list(result["ids"]),
        "gate": result["gate"].copy(),
    }
    for branch in BRANCHES:
        copied[branch] = {
            key: value.copy() if isinstance(value, np.ndarray) else value for key, value in result[branch].items()
        }
    return copied


def save_result(result: Dict, out_dir: Path, prefix: str, extra: Dict[str, object]) -> Dict[str, object]:
    summary: Dict[str, object] = {
        **extra,
        "n": len(result["golds"]),
        "loss": result["loss"],
        "mean_gate": {
            name: float(result["gate"][:, index].mean())
            for index, name in enumerate(("context", "source_local", "bobsl_local"))
        },
    }
    for branch in BRANCHES:
        summary[branch] = {
            metric: float(result[branch][metric]) for metric in ("accuracy", "macro_f1", "weighted_f1")
        }
        summary[branch]["pred_counts"] = {
            label: int(np.sum(result[branch]["preds"] == class_id)) for class_id, label in enumerate(ANJS_LABELS)
        }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{prefix}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (out_dir / f"{prefix}_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["item_id", "gold_id", "gold_label", "gate_context", "gate_source", "gate_bobsl"]
            + [f"{branch}_pred" for branch in BRANCHES]
            + [f"{branch}_prob_{label}" for branch in BRANCHES for label in ANJS_LABELS]
        )
        for index, item_id in enumerate(result["ids"]):
            writer.writerow(
                [item_id, int(result["golds"][index]), ANJS_LABELS[int(result["golds"][index])], *result["gate"][index].tolist()]
                + [int(result[branch]["preds"][index]) for branch in BRANCHES]
                + [
                    float(result[branch]["probs"][index, class_id])
                    for branch in BRANCHES
                    for class_id in range(len(ANJS_LABELS))
                ]
            )
    return summary


def save_checkpoint(path: Path, model: DomainExpertCMERC, args: argparse.Namespace, epoch: int, score: float) -> None:
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
            "baseline": "DomainExpertCMERC",
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

    checkpoint = load_checkpoint(context_checkpoint_path, device)
    context_args = namespace_from_checkpoint(checkpoint, args)
    context_model = build_model(context_args, info, device)
    context_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    train_dataset = CMERCVisualDataset(train_pkl, "train")
    has_bobsl = any("bobsl" in str(key).lower() for key in train_dataset.keys)
    model = DomainExpertCMERC(context_model, int(info["d_visual"]), args, has_bobsl).to(device)
    if args.init_source_expert_checkpoint:
        init_path = resolve_path(args.init_source_expert_checkpoint, must_exist=True)
        init_checkpoint = load_checkpoint(init_path, device)
        init_state = init_checkpoint.get("model_state_dict") or {}
        compatible = {
            key: value
            for key, value in init_state.items()
            if key.startswith("source_expert.") or key.startswith("gate.")
        }
        missing, unexpected = model.load_state_dict(compatible, strict=False)
        print(
            f"[DE-CMERC] init_source_expert={init_path} loaded={len(compatible)} "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
    if args.freeze_source_expert:
        for parameter in model.source_expert.parameters():
            parameter.requires_grad = False
    if args.freeze_base_gate:
        for parameter in model.gate.parameters():
            parameter.requires_grad = False
    train_loader = make_train_loader(args, train_dataset)
    source_loader = make_eval_loader(args, CMERCVisualDataset(source_pkl, "test"))
    external_loader = make_eval_loader(args, CMERCVisualDataset(external_pkl, "test"))
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print(f"[DE-CMERC] train_pkl={train_pkl} has_bobsl={has_bobsl}")
    print(f"[DE-CMERC] modality={args.modality}")
    print(f"[DE-CMERC] context_checkpoint={context_checkpoint_path}")
    print(f"[DE-CMERC] out_dir={out_dir} device={device}")
    best_score = -1.0
    best_epoch = 0
    best_state = None
    best_source = None
    best_external = None
    epoch_rows: List[Dict[str, object]] = []
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
                {key: value.detach().cpu() for key, value in model.state_dict().items() if not key.startswith("context_model.")}
            )
            best_source = snapshot(source_result)
            best_external = snapshot(external_result)
            print(f"[DE-CMERC] new best epoch={epoch} source_{args.selection_metric}={score:.4f}")
        row = {
            "epoch": epoch,
            "time_sec": round(time.time() - start, 2),
            "train_fused_wf1": train_result["fused"]["weighted_f1"],
            "source_fused_wf1": source_result["fused"]["weighted_f1"],
            "source_context_wf1": source_result["context"]["weighted_f1"],
            "source_source_local_wf1": source_result["source_local"]["weighted_f1"],
            "source_bobsl_local_wf1": source_result["bobsl_local"]["weighted_f1"],
            "source_gate_context": float(source_result["gate"][:, 0].mean()),
            "source_gate_bobsl": float(source_result["gate"][:, 2].mean()),
            "external_fused_wf1": external_result["fused"]["weighted_f1"],
            "external_context_wf1": external_result["context"]["weighted_f1"],
            "external_source_local_wf1": external_result["source_local"]["weighted_f1"],
            "external_bobsl_local_wf1": external_result["bobsl_local"]["weighted_f1"],
            "external_gate_context": float(external_result["gate"][:, 0].mean()),
            "external_gate_bobsl": float(external_result["gate"][:, 2].mean()),
        }
        epoch_rows.append(row)
        print(
            f"[DE-CMERC] epoch={epoch} source={row['source_fused_wf1']:.4f} external={row['external_fused_wf1']:.4f} "
            f"gate_source_c/b={row['source_gate_context']:.3f}/{row['source_gate_bobsl']:.3f} "
            f"gate_external_c/b={row['external_gate_context']:.3f}/{row['external_gate_bobsl']:.3f} time={row['time_sec']}s"
        )

    with (out_dir / "epoch_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(epoch_rows[0]))
        writer.writeheader()
        writer.writerows(epoch_rows)
    extra = {
        "baseline": "DomainExpertCMERC",
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
    external_summary = save_result(best_external, out_dir, "external_test_best", {**extra, "split": "external"})
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
        save_checkpoint(out_dir / "model_best.pt", model, args, best_epoch, best_score)
    print(
        f"[DE-CMERC] best_epoch={best_epoch} source_wf1={source_summary['fused']['weighted_f1']:.4f} "
        f"external_wf1={external_summary['fused']['weighted_f1']:.4f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CMERC with separate source/BOBSL local experts and reliability routing.")
    parser.add_argument("--train_pkl", required=True)
    parser.add_argument("--source_pkl", required=True)
    parser.add_argument("--external_test_pkl", required=True)
    parser.add_argument("--context_checkpoint", required=True)
    parser.add_argument("--init_source_expert_checkpoint", default="")
    parser.add_argument("--freeze_source_expert", action="store_true")
    parser.add_argument("--freeze_base_gate", action="store_true")
    parser.add_argument("--modality", choices=["visual", "tv"], default="visual")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=35)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--context_dropout", type=float, default=0.4)
    parser.add_argument("--local_hidden_dim", type=int, default=256)
    parser.add_argument("--local_embedding_dim", type=int, default=128)
    parser.add_argument("--local_dropout", type=float, default=0.2)
    parser.add_argument("--cosine_scale", type=float, default=10.0)
    parser.add_argument("--gate_hidden_dim", type=int, default=32)
    parser.add_argument("--gate_biases", type=float, nargs=3, default=[2.0, 0.0, -1.0])
    parser.add_argument("--residual_bobsl_gate", action="store_true")
    parser.add_argument("--residual_gate_bias", type=float, default=-3.0)
    parser.add_argument("--local_source_weight", type=float, default=0.5)
    parser.add_argument("--bobsl_weight", type=float, default=0.5)
    parser.add_argument("--bobsl_fused_weight", type=float, default=0.25)
    parser.add_argument("--cross_source_weight", type=float, default=0.0)
    parser.add_argument("--alignment_weight", type=float, default=0.05)
    parser.add_argument("--embedding_alignment_weight", type=float, default=0.0)
    parser.add_argument("--distill_weight", type=float, default=0.3)
    parser.add_argument("--gate_penalty", type=float, default=0.02)
    parser.add_argument("--source_repeat", type=int, default=2)
    parser.add_argument("--min_domain_samples_per_epoch", type=int, default=500)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--selection_metric", choices=["weighted_f1", "macro_f1", "accuracy"], default="weighted_f1")
    parser.add_argument("--no_cuda", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
