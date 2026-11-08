#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from torch.utils.data import DataLoader, WeightedRandomSampler

from train_eval_cmerc_visual_anjs import (
    ANJS_LABELS,
    CMERCVisualDataset,
    build_model,
    inspect_pkl,
    resolve_path,
    sequence_lengths,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_checkpoint(path: Path, device: torch.device) -> Dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def namespace_from_checkpoint(checkpoint: Dict, fallback_args: argparse.Namespace) -> argparse.Namespace:
    values = dict(checkpoint.get("args") or {})
    defaults = {
        "modality": "visual",
        "dropout": fallback_args.context_dropout,
        "base_model": "GRU",
        "graph_type": "hyper",
        "graph_construct": "direct",
        "mm_fusion_mthd": "concat_DHT",
        "norm": "BN",
        "num_L": 3,
        "num_K": 3,
        "windowp": 10,
        "windowf": 10,
        "attention": "general",
        "alpha": 0.2,
        "multiheads": 6,
        "use_residue": False,
        "deep_gcn_nlayers": 4,
        "D_g": 1024,
        "D_p": 150,
        "D_e": 100,
        "D_h": 100,
        "D_a": 100,
        "graph_hidden_size": 512,
        "max_seq_len": 200,
        "no_cuda": False,
    }
    for key, value in defaults.items():
        values.setdefault(key, value)
    return argparse.Namespace(**values)


def flatten_seq_first(tensor: torch.Tensor, lengths: Sequence[int]) -> torch.Tensor:
    return torch.cat([tensor[:length, batch_index] for batch_index, length in enumerate(lengths)], dim=0)


def flatten_domain_mask(vids: Sequence[str], lengths: Sequence[int], device: torch.device) -> torch.Tensor:
    values: List[bool] = []
    for vid, length in zip(vids, lengths):
        values.extend(["bobsl" in str(vid).lower()] * int(length))
    return torch.tensor(values, dtype=torch.bool, device=device)


def class_balanced_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if not len(labels):
        return logits.sum() * 0.0
    losses = F.cross_entropy(logits, labels, reduction="none")
    class_losses = [losses[labels == class_id].mean() for class_id in range(len(ANJS_LABELS)) if torch.any(labels == class_id)]
    return torch.stack(class_losses).mean()


def prototype_alignment(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    bobsl_mask: torch.Tensor,
) -> torch.Tensor:
    losses: List[torch.Tensor] = []
    for class_id in range(len(ANJS_LABELS)):
        source_keep = (labels == class_id) & ~bobsl_mask
        bobsl_keep = (labels == class_id) & bobsl_mask
        if torch.any(source_keep) and torch.any(bobsl_keep):
            source_center = F.normalize(embeddings[source_keep].mean(dim=0), dim=0)
            bobsl_center = F.normalize(embeddings[bobsl_keep].mean(dim=0), dim=0)
            losses.append(1.0 - torch.sum(source_center * bobsl_center))
    return torch.stack(losses).mean() if losses else embeddings.sum() * 0.0


class CosineLocalExpert(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int, dropout: float, scale: float):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.prototypes = nn.Parameter(torch.empty(len(ANJS_LABELS), embedding_dim))
        nn.init.xavier_uniform_(self.prototypes)
        self.scale = float(scale)

    def forward(self, visual: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embedding = F.normalize(self.projector(visual), dim=-1)
        prototypes = F.normalize(self.prototypes, dim=-1)
        return self.scale * embedding @ prototypes.T, embedding


class ReliabilityGate(nn.Module):
    def __init__(self, hidden_dim: int, bias: float):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(7, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.constant_(self.network[-1].bias, bias)

    def forward(
        self,
        context_log_prob: torch.Tensor,
        local_logits: torch.Tensor,
        utterance_lengths: torch.Tensor,
    ) -> torch.Tensor:
        context_prob = context_log_prob.exp()
        local_prob = local_logits.softmax(dim=-1)
        normalizer = math.log(float(len(ANJS_LABELS)))
        context_entropy = -(context_prob * context_log_prob).sum(dim=-1, keepdim=True) / normalizer
        local_entropy = -(local_prob * F.log_softmax(local_logits, dim=-1)).sum(dim=-1, keepdim=True) / normalizer
        context_confidence = context_prob.max(dim=-1, keepdim=True).values
        local_confidence = local_prob.max(dim=-1, keepdim=True).values
        disagreement = 0.5 * torch.abs(context_prob - local_prob).sum(dim=-1, keepdim=True)
        confidence_margin = local_confidence - context_confidence
        length_feature = torch.log1p(utterance_lengths.float()).unsqueeze(1) / math.log(51.0)
        features = torch.cat(
            [
                context_entropy,
                local_entropy,
                context_confidence,
                local_confidence,
                disagreement,
                confidence_margin,
                length_feature,
            ],
            dim=1,
        )
        return torch.sigmoid(self.network(features)).squeeze(1)


class DomainRoutedCMERC(nn.Module):
    def __init__(self, context_model: nn.Module, visual_dim: int, args: argparse.Namespace):
        super().__init__()
        self.context_model = context_model
        self.local_expert = CosineLocalExpert(
            visual_dim,
            args.local_hidden_dim,
            args.local_embedding_dim,
            args.local_dropout,
            args.cosine_scale,
        )
        self.gate = ReliabilityGate(args.gate_hidden_dim, args.gate_bias)
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
        flat_visual = flatten_seq_first(visuf, lengths)
        local_logits, local_embedding = self.local_expert(flat_visual)
        utterance_lengths = torch.cat(
            [torch.full((length,), length, dtype=torch.float32, device=visuf.device) for length in lengths]
        )
        gate = self.gate(context_log_prob.detach(), local_logits, utterance_lengths)
        fused_logits = (1.0 - gate.unsqueeze(1)) * context_log_prob.detach() + gate.unsqueeze(1) * local_logits
        return {
            "fused_log_prob": F.log_softmax(fused_logits, dim=-1),
            "context_log_prob": context_log_prob,
            "local_log_prob": F.log_softmax(local_logits, dim=-1),
            "local_logits": local_logits,
            "local_embedding": local_embedding,
            "gate": gate,
        }


def make_train_loader(args: argparse.Namespace, dataset: CMERCVisualDataset) -> DataLoader:
    bobsl = np.asarray(["bobsl" in str(key).lower() for key in dataset.keys], dtype=bool)
    source_count = int((~bobsl).sum())
    bobsl_count = int(bobsl.sum())
    generator = torch.Generator().manual_seed(args.seed)
    if source_count and bobsl_count:
        per_domain = min(bobsl_count, max(args.min_domain_samples_per_epoch, source_count * args.source_repeat))
        weights = np.where(bobsl, 0.5 / bobsl_count, 0.5 / source_count)
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=2 * per_domain,
            replacement=True,
            generator=generator,
        )
        print(
            f"[DR-CMERC] domain sampler source_dialogues={source_count} bobsl_dialogues={bobsl_count} "
            f"samples_per_epoch={2 * per_domain}"
        )
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            collate_fn=CMERCVisualDataset.collate_fn,
            num_workers=args.num_workers,
        )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=CMERCVisualDataset.collate_fn,
        num_workers=args.num_workers,
    )


def make_eval_loader(args: argparse.Namespace, dataset: CMERCVisualDataset) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=CMERCVisualDataset.collate_fn,
        num_workers=args.num_workers,
    )


def metrics_from_arrays(golds: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    label_ids = list(range(len(ANJS_LABELS)))
    return {
        "accuracy": float(metrics.accuracy_score(golds, preds)),
        "macro_f1": float(metrics.f1_score(golds, preds, labels=label_ids, average="macro", zero_division=0)),
        "weighted_f1": float(metrics.f1_score(golds, preds, labels=label_ids, average="weighted", zero_division=0)),
    }


def run_epoch(
    model: DomainRoutedCMERC,
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
    branch_golds: List[int] = []
    branch_preds: Dict[str, List[int]] = {name: [] for name in ("fused", "context", "local")}
    branch_probs: Dict[str, List[np.ndarray]] = {name: [] for name in branch_preds}
    ids_all: List[str] = []
    gates: List[np.ndarray] = []
    bobsl_flags: List[np.ndarray] = []

    for data in dataloader:
        if training:
            optimizer.zero_grad()
        text1, text2, text3, text4, visuf, acouf, qmask, umask, labels = [value.to(device) for value in data[:-1]]
        texts = [torch.zeros_like(text) for text in (text1, text2, text3, text4)]
        acouf = torch.zeros_like(acouf)
        lengths = sequence_lengths(umask)
        labels_flat = torch.cat([labels[index][:length] for index, length in enumerate(lengths)])
        valid = labels_flat.ge(0)
        domain_mask = flatten_domain_mask(data[-1], lengths, device)
        outputs = model(texts, qmask, umask, lengths, acouf, visuf, epoch)

        if training:
            source = valid & ~domain_mask
            bobsl = valid & domain_mask
            fused_source_loss = class_balanced_loss(outputs["fused_log_prob"][source], labels_flat[source])
            local_source_loss = class_balanced_loss(outputs["local_logits"][source], labels_flat[source])
            local_bobsl_loss = class_balanced_loss(outputs["local_logits"][bobsl], labels_flat[bobsl])
            alignment_loss = prototype_alignment(
                outputs["local_embedding"][valid], labels_flat[valid], domain_mask[valid]
            )
            if torch.any(source):
                consistency = F.kl_div(
                    outputs["fused_log_prob"][source],
                    outputs["context_log_prob"][source].exp(),
                    reduction="batchmean",
                )
                gate_penalty = outputs["gate"][source].mean()
            else:
                consistency = outputs["gate"].sum() * 0.0
                gate_penalty = outputs["gate"].sum() * 0.0
            loss = (
                fused_source_loss
                + args.local_source_weight * local_source_loss
                + args.bobsl_weight * local_bobsl_loss
                + args.alignment_weight * alignment_loss
                + args.distill_weight * consistency
                + args.gate_penalty * gate_penalty
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
        gold = labels_flat[valid].detach().cpu().numpy().astype(np.int64)
        branch_golds.extend(gold.tolist())
        for branch in branch_preds:
            log_prob = outputs[f"{branch}_log_prob"][valid]
            branch_preds[branch].extend(log_prob.argmax(dim=-1).detach().cpu().numpy().astype(np.int64).tolist())
            branch_probs[branch].append(log_prob.exp().detach().cpu().numpy().astype(np.float32))
        gates.append(outputs["gate"][valid].detach().cpu().numpy().astype(np.float32))
        bobsl_flags.append(domain_mask[valid].detach().cpu().numpy().astype(bool))
        label_matrix = labels.detach().cpu()
        for batch_index, vid in enumerate(data[-1]):
            for utterance_index in range(lengths[batch_index]):
                if int(label_matrix[batch_index, utterance_index]) >= 0:
                    ids_all.append(f"{vid}:utt{utterance_index + 1:02d}")

    golds = np.asarray(branch_golds, dtype=np.int64)
    result: Dict[str, object] = {
        "loss": float(np.sum(losses) / max(np.sum(loss_weights), 1)),
        "golds": golds,
        "ids": ids_all,
        "gate": np.concatenate(gates),
        "bobsl": np.concatenate(bobsl_flags),
    }
    for branch in branch_preds:
        predictions = np.asarray(branch_preds[branch], dtype=np.int64)
        result[branch] = {
            **metrics_from_arrays(golds, predictions),
            "preds": predictions,
            "probs": np.concatenate(branch_probs[branch], axis=0),
        }
    return result


def snapshot(result: Dict) -> Dict:
    copied = {"loss": result["loss"], "golds": result["golds"].copy(), "ids": list(result["ids"]), "gate": result["gate"].copy()}
    for branch in ("fused", "context", "local"):
        copied[branch] = {
            key: value.copy() if isinstance(value, np.ndarray) else value for key, value in result[branch].items()
        }
    return copied


def save_result(result: Dict, out_dir: Path, prefix: str, extra: Dict[str, object]) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    golds = result["golds"]
    summary: Dict[str, object] = {**extra, "n": len(golds), "loss": result["loss"], "mean_gate": float(np.mean(result["gate"]))}
    for branch in ("fused", "context", "local"):
        summary[branch] = {
            metric: float(result[branch][metric]) for metric in ("accuracy", "macro_f1", "weighted_f1")
        }
        summary[branch]["pred_counts"] = {
            label: int(np.sum(result[branch]["preds"] == class_id)) for class_id, label in enumerate(ANJS_LABELS)
        }
    (out_dir / f"{prefix}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (out_dir / f"{prefix}_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["item_id", "gold_id", "gold_label", "gate"]
            + [f"{branch}_pred" for branch in ("fused", "context", "local")]
            + [f"{branch}_prob_{label}" for branch in ("fused", "context", "local") for label in ANJS_LABELS]
        )
        for index, item_id in enumerate(result["ids"]):
            writer.writerow(
                [item_id, int(golds[index]), ANJS_LABELS[int(golds[index])], float(result["gate"][index])]
                + [int(result[branch]["preds"][index]) for branch in ("fused", "context", "local")]
                + [
                    float(result[branch]["probs"][index, class_id])
                    for branch in ("fused", "context", "local")
                    for class_id in range(len(ANJS_LABELS))
                ]
            )
    return summary


def save_checkpoint(path: Path, model: DomainRoutedCMERC, args: argparse.Namespace, epoch: int, score: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    trainable_state = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if not key.startswith("context_model.")
    }
    torch.save(
        {
            "model_state_dict": trainable_state,
            "context_checkpoint": str(Path(args.context_checkpoint).resolve()),
            "args": vars(args),
            "epoch": epoch,
            "best_score": score,
            "baseline": "DomainRoutedCMERC",
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
    source_info = inspect_pkl(source_pkl)
    external_info = inspect_pkl(external_pkl)
    for other in (source_info, external_info):
        for key in ("d_text", "d_audio", "d_visual", "n_speakers", "n_classes"):
            if int(info[key]) != int(other[key]):
                raise RuntimeError(f"Feature mismatch for {key}: train={info[key]} other={other[key]}")

    context_checkpoint = load_checkpoint(context_checkpoint_path, device)
    context_args = namespace_from_checkpoint(context_checkpoint, args)
    context_model = build_model(context_args, info, device)
    context_model.load_state_dict(context_checkpoint["model_state_dict"], strict=True)
    model = DomainRoutedCMERC(context_model, int(info["d_visual"]), args).to(device)
    train_dataset = CMERCVisualDataset(train_pkl, "train")
    source_dataset = CMERCVisualDataset(source_pkl, "test")
    external_dataset = CMERCVisualDataset(external_pkl, "test")
    train_loader = make_train_loader(args, train_dataset)
    source_loader = make_eval_loader(args, source_dataset)
    external_loader = make_eval_loader(args, external_dataset)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print(f"[DR-CMERC] train_pkl={train_pkl}")
    print(f"[DR-CMERC] source_pkl={source_pkl}")
    print(f"[DR-CMERC] external_pkl={external_pkl}")
    print(f"[DR-CMERC] context_checkpoint={context_checkpoint_path}")
    print(f"[DR-CMERC] out_dir={out_dir} device={device}")
    best_score = -1.0
    best_epoch = 0
    best_state = None
    best_source = None
    best_external = None
    rows: List[Dict[str, object]] = []
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
            print(f"[DR-CMERC] new best epoch={epoch} source_{args.selection_metric}={score:.4f}")
        row = {
            "epoch": epoch,
            "time_sec": round(time.time() - start, 2),
            "train_fused_wf1": train_result["fused"]["weighted_f1"],
            "source_fused_wf1": source_result["fused"]["weighted_f1"],
            "source_context_wf1": source_result["context"]["weighted_f1"],
            "source_local_wf1": source_result["local"]["weighted_f1"],
            "source_gate": float(np.mean(source_result["gate"])),
            "external_fused_wf1": external_result["fused"]["weighted_f1"],
            "external_context_wf1": external_result["context"]["weighted_f1"],
            "external_local_wf1": external_result["local"]["weighted_f1"],
            "external_gate": float(np.mean(external_result["gate"])),
        }
        rows.append(row)
        print(
            f"[DR-CMERC] epoch={epoch} train={row['train_fused_wf1']:.4f} "
            f"source={row['source_fused_wf1']:.4f} external={row['external_fused_wf1']:.4f} "
            f"gate_source={row['source_gate']:.3f} gate_external={row['external_gate']:.3f} time={row['time_sec']}s"
        )

    with (out_dir / "epoch_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    extra = {
        "baseline": "DomainRoutedCMERC",
        "seed": args.seed,
        "best_epoch": best_epoch,
        "selection_metric": args.selection_metric,
        "best_source_score": best_score,
        "train_pkl": str(train_pkl),
        "source_pkl": str(source_pkl),
        "external_test_pkl": str(external_pkl),
        "context_checkpoint": str(context_checkpoint_path),
        "args": vars(args),
    }
    source_summary = save_result(best_source, out_dir, "source_test_best", {**extra, "split": "source"})
    external_summary = save_result(best_external, out_dir, "external_test_best", {**extra, "split": "external"})
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
        save_checkpoint(out_dir / "model_best.pt", model, args, best_epoch, best_score)
    print(
        f"[DR-CMERC] best_epoch={best_epoch} source_wf1={source_summary['fused']['weighted_f1']:.4f} "
        f"external_wf1={external_summary['fused']['weighted_f1']:.4f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Domain-routed CMERC with a BOBSL-local expert and frozen source context expert.")
    parser.add_argument("--train_pkl", required=True)
    parser.add_argument("--source_pkl", required=True)
    parser.add_argument("--external_test_pkl", required=True)
    parser.add_argument("--context_checkpoint", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=35)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--context_dropout", type=float, default=0.4)
    parser.add_argument("--local_hidden_dim", type=int, default=256)
    parser.add_argument("--local_embedding_dim", type=int, default=128)
    parser.add_argument("--local_dropout", type=float, default=0.2)
    parser.add_argument("--cosine_scale", type=float, default=10.0)
    parser.add_argument("--gate_hidden_dim", type=int, default=16)
    parser.add_argument("--gate_bias", type=float, default=-2.0)
    parser.add_argument("--local_source_weight", type=float, default=0.5)
    parser.add_argument("--bobsl_weight", type=float, default=0.5)
    parser.add_argument("--alignment_weight", type=float, default=0.1)
    parser.add_argument("--distill_weight", type=float, default=0.3)
    parser.add_argument("--gate_penalty", type=float, default=0.02)
    parser.add_argument("--source_repeat", type=int, default=4)
    parser.add_argument("--min_domain_samples_per_epoch", type=int, default=1000)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--selection_metric", choices=["weighted_f1", "macro_f1", "accuracy"], default="weighted_f1")
    parser.add_argument("--no_cuda", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
