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
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(REPO_ROOT / "new"))

import train_eval_cmerc_visual_anjs as base_script  # noqa: E402
from anjs_video_common import ANJS_LABELS, resolve_path  # noqa: E402
from cesa import ContextDecoupledEmotionSubspaceAdapter, compute_emotion_subspace  # noqa: E402


def is_bobsl_key(key: str) -> bool:
    return str(key).startswith("joint_bobsl") or str(key).startswith("bobsl")


def make_dataset(path: Path, split: str, domain: str = "all") -> base_script.CMERCVisualDataset:
    dataset = base_script.CMERCVisualDataset(path, split)
    if domain == "source":
        dataset.keys = [key for key in dataset.keys if not is_bobsl_key(str(key))]
    elif domain == "bobsl":
        dataset.keys = [key for key in dataset.keys if is_bobsl_key(str(key))]
    elif domain != "all":
        raise ValueError(f"Unsupported domain={domain}")
    return dataset


def make_loader(dataset, batch_size: int, shuffle: bool, args, seed_offset: int = 0):
    generator = torch.Generator()
    generator.manual_seed(args.seed + seed_offset)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=base_script.CMERCVisualDataset.collate_fn,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def flatten_seq_first(values: torch.Tensor, lengths: Sequence[int]) -> torch.Tensor:
    return torch.cat([values[:length, batch_index] for batch_index, length in enumerate(lengths)], dim=0)


def flatten_labels(labels: torch.Tensor, lengths: Sequence[int]) -> torch.Tensor:
    return torch.cat([labels[batch_index, :length] for batch_index, length in enumerate(lengths)], dim=0)


def weighted_nll(log_prob: torch.Tensor, labels: torch.Tensor, weights: Optional[torch.Tensor], reduction: str = "mean"):
    return F.nll_loss(log_prob, labels, weight=weights, reduction=reduction)


class CESACMERC(nn.Module):
    def __init__(self, base_model: nn.Module, visual_dim: int, basis: torch.Tensor, args):
        super().__init__()
        self.base_model = base_model
        self.adapter = ContextDecoupledEmotionSubspaceAdapter(
            visual_dim,
            basis,
            bottleneck_dim=args.adapter_dim,
            dropout=args.adapter_dropout,
            residual_scale=args.residual_scale,
            projection=args.projection,
        )
        self.local_head = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, args.local_hidden_dim),
            nn.GELU(),
            nn.Dropout(args.adapter_dropout),
            nn.Linear(args.local_hidden_dim, len(ANJS_LABELS)),
        )

    def adapt(self, visual: torch.Tensor):
        return self.adapter(visual)

    def context_forward(self, texts, qmask, umask, lengths, audio, visual, epoch):
        return self.base_model(texts, qmask, umask, lengths, audio, visual, epoch)[0]

    def local_log_prob(self, visual: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.local_head(visual), dim=-1)


def load_base_checkpoint(model: nn.Module, path: Path, device: torch.device) -> Dict:
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    clean = {key.removeprefix("module."): value for key, value in state.items()}
    incompatible = model.load_state_dict(clean, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Base checkpoint mismatch: missing={incompatible.missing_keys[:10]} "
            f"unexpected={incompatible.unexpected_keys[:10]}"
        )
    return checkpoint


def prepare_batch(data, device: torch.device):
    text1, text2, text3, text4, visual, audio, qmask, umask, labels = [value.to(device) for value in data[:-1]]
    texts, visual, audio = base_script.prepare_inputs([text1, text2, text3, text4], visual, audio, "visual")
    lengths = base_script.sequence_lengths(umask)
    labels_flat = flatten_labels(labels, lengths)
    valid = labels_flat.ge(0)
    return texts, visual, audio, qmask, umask, labels, lengths, labels_flat, valid


@torch.no_grad()
def evaluate(model: CESACMERC, loader: DataLoader, device: torch.device, epoch: int, adapted: bool) -> Optional[Dict]:
    model.eval()
    losses: List[float] = []
    counts: List[int] = []
    golds: List[int] = []
    preds: List[int] = []
    probs: List[np.ndarray] = []
    ids: List[str] = []
    for data in loader:
        texts, visual, audio, qmask, umask, labels, lengths, labels_flat, valid = prepare_batch(data, device)
        if adapted:
            visual, _ = model.adapt(visual)
        log_prob = model.context_forward(texts, qmask, umask, lengths, audio, visual, epoch)
        if not torch.any(valid):
            continue
        current = log_prob[valid]
        target = labels_flat[valid]
        loss = F.nll_loss(current, target)
        pred = current.argmax(dim=-1)
        losses.append(float(loss.item()) * len(target))
        counts.append(len(target))
        golds.extend(target.cpu().numpy().astype(int).tolist())
        preds.extend(pred.cpu().numpy().astype(int).tolist())
        probs.append(current.exp().cpu().numpy().astype(np.float32))
        label_matrix = labels.cpu()
        for batch_index, key in enumerate(data[-1]):
            for utterance_index in range(lengths[batch_index]):
                if int(label_matrix[batch_index, utterance_index]) >= 0:
                    ids.append(f"{key}:utt{utterance_index + 1:02d}")
    if not golds:
        return None
    gold_array = np.asarray(golds, dtype=np.int64)
    pred_array = np.asarray(preds, dtype=np.int64)
    probability_array = np.concatenate(probs, axis=0)
    from sklearn import metrics

    label_ids = list(range(len(ANJS_LABELS)))
    return {
        "loss": float(np.sum(losses) / max(np.sum(counts), 1)),
        "accuracy": float(metrics.accuracy_score(gold_array, pred_array)),
        "macro_f1": float(metrics.f1_score(gold_array, pred_array, labels=label_ids, average="macro", zero_division=0)),
        "weighted_f1": float(metrics.f1_score(gold_array, pred_array, labels=label_ids, average="weighted", zero_division=0)),
        "golds": gold_array,
        "preds": pred_array,
        "probs": probability_array,
        "ids": ids,
    }


def adapter_parameters(model: CESACMERC) -> Iterable[nn.Parameter]:
    yield from model.adapter.parameters()
    yield from model.local_head.parameters()


def train_adapter_epoch(
    model: CESACMERC,
    source_loader: DataLoader,
    bobsl_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    weights: Optional[torch.Tensor],
    device: torch.device,
    epoch: int,
    args,
) -> Dict[str, float]:
    model.base_model.eval()
    model.adapter.train()
    model.local_head.train()
    bobsl_iter = cycle(bobsl_loader)
    totals: Dict[str, float] = {
        key: 0.0 for key in ("loss", "source", "bobsl", "bobsl_context", "local", "safety", "residual")
    }
    steps = 0
    for source_data in source_loader:
        bobsl_data = next(bobsl_iter)
        optimizer.zero_grad(set_to_none=True)

        texts, source_visual, audio, qmask, umask, _, lengths, source_labels, source_valid = prepare_batch(source_data, device)
        adapted_visual, source_delta = model.adapt(source_visual)
        adapted_log_prob = model.context_forward(texts, qmask, umask, lengths, audio, adapted_visual, epoch)
        with torch.no_grad():
            base_log_prob = model.context_forward(texts, qmask, umask, lengths, audio, source_visual, epoch)
        source_targets = source_labels[source_valid]
        source_nll = weighted_nll(adapted_log_prob[source_valid], source_targets, weights, reduction="none")
        base_nll = weighted_nll(base_log_prob[source_valid], source_targets, weights, reduction="none")
        source_loss = source_nll.mean()
        safety_loss = F.relu(source_nll - base_nll + args.safety_margin).mean()
        source_adapted_flat = flatten_seq_first(adapted_visual, lengths)[source_valid]
        source_local = weighted_nll(model.local_log_prob(source_adapted_flat), source_targets, weights)
        source_delta_flat = flatten_seq_first(source_delta, lengths)[source_valid]
        residual_loss = source_delta_flat.pow(2).sum(dim=-1).mean()

        (
            bobsl_texts,
            bobsl_visual,
            bobsl_audio,
            bobsl_qmask,
            bobsl_umask,
            _,
            bobsl_lengths,
            bobsl_labels,
            bobsl_valid,
        ) = prepare_batch(bobsl_data, device)
        bobsl_adapted, bobsl_delta = model.adapt(bobsl_visual)
        bobsl_flat = flatten_seq_first(bobsl_adapted, bobsl_lengths)[bobsl_valid]
        bobsl_targets = bobsl_labels[bobsl_valid]
        bobsl_loss = weighted_nll(model.local_log_prob(bobsl_flat), bobsl_targets, weights)
        if args.bobsl_context_weight > 0:
            bobsl_context_log_prob = model.context_forward(
                bobsl_texts,
                bobsl_qmask,
                bobsl_umask,
                bobsl_lengths,
                bobsl_audio,
                bobsl_adapted,
                epoch,
            )
            bobsl_context_loss = weighted_nll(
                bobsl_context_log_prob[bobsl_valid], bobsl_targets, weights
            )
        else:
            bobsl_context_loss = bobsl_loss.new_zeros(())
        bobsl_delta_flat = flatten_seq_first(bobsl_delta, bobsl_lengths)[bobsl_valid]
        residual_loss = residual_loss + bobsl_delta_flat.pow(2).sum(dim=-1).mean()

        loss = (
            args.source_weight * source_loss
            + args.bobsl_weight * bobsl_loss
            + args.bobsl_context_weight * bobsl_context_loss
            + args.local_source_weight * source_local
            + args.safety_weight * safety_loss
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
            "bobsl_context": bobsl_context_loss,
            "local": source_local,
            "safety": safety_loss,
            "residual": residual_loss,
        }
        for key, value in values.items():
            totals[key] += float(value.detach().item())
        steps += 1
    return {key: value / max(steps, 1) for key, value in totals.items()}


def save_cesa_checkpoint(path: Path, model: CESACMERC, args, info: Dict, basis_meta: Dict, epoch: int, score: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "method": "CESA-CMERC",
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
    base_script.seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CMERC requires CUDA")
    train_pkl = resolve_path(args.train_pkl, must_exist=True)
    external_pkl = resolve_path(args.external_test_pkl, must_exist=True)
    base_checkpoint = resolve_path(args.base_checkpoint, must_exist=True)
    out_dir = resolve_path(args.out_dir, must_exist=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = base_script.inspect_pkl(train_pkl)
    payload = base_script.load_payload(train_pkl)
    source_keys = [key for key in payload["train_ids"] if not is_bobsl_key(str(key))]
    bobsl_keys = [key for key in payload["train_ids"] if is_bobsl_key(str(key))]
    if not bobsl_keys:
        raise RuntimeError("CESA training requires a joint source/BOBSL pkl")
    basis, centers, counts = compute_emotion_subspace(
        payload["video_visual"], payload["labels"], source_keys, len(ANJS_LABELS), rank=args.subspace_rank
    )
    basis_meta = {
        "rank": int(basis.shape[1]),
        "source_counts": counts.tolist(),
        "source_center_norms": centers.norm(dim=1).tolist(),
        "source_dialogues": len(source_keys),
        "bobsl_dialogues": len(bobsl_keys),
    }

    checkpoint = torch.load(base_checkpoint, map_location="cpu")
    checkpoint_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    architecture_keys = [
        "base_model", "D_g", "D_p", "D_e", "D_h", "D_a", "graph_hidden_size", "max_seq_len",
        "windowp", "windowf", "attention", "graph_type", "alpha", "multiheads", "graph_construct",
        "use_residue", "deep_gcn_nlayers", "norm", "num_L", "num_K", "mm_fusion_mthd", "dropout",
    ]
    for key in architecture_keys:
        if key in checkpoint_args:
            setattr(args, key, checkpoint_args[key])
    base_model = base_script.build_model(args, info, device)
    load_base_checkpoint(base_model, base_checkpoint, device)
    model = CESACMERC(base_model, int(info["d_visual"]), basis, args).to(device)
    for parameter in model.base_model.parameters():
        parameter.requires_grad = False

    source_train = make_dataset(train_pkl, "train", "source")
    bobsl_train = make_dataset(train_pkl, "train", "bobsl")
    source_test = make_dataset(train_pkl, "test")
    external_test = make_dataset(external_pkl, "test")
    source_train_loader = make_loader(source_train, args.source_batch_size, True, args)
    bobsl_train_loader = make_loader(bobsl_train, args.bobsl_batch_size, True, args, seed_offset=17)
    source_test_loader = make_loader(source_test, args.eval_batch_size, False, args)
    external_test_loader = make_loader(external_test, args.eval_batch_size, False, args)

    source_labels = [int(label) for key in source_keys for label in payload["labels"][key] if int(label) >= 0]
    weights = base_script.make_class_weights(source_labels, len(ANJS_LABELS), device) if args.class_weight else None
    optimizer = torch.optim.AdamW(adapter_parameters(model), lr=args.lr, weight_decay=args.weight_decay)

    base_source = evaluate(model, source_test_loader, device, 0, adapted=False)
    base_external = evaluate(model, external_test_loader, device, 0, adapted=False)
    best_source = copy.deepcopy(base_source)
    best_external = copy.deepcopy(base_external)
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_score = float(base_source[args.selection_metric])
    oracle_state = copy.deepcopy(best_state)
    oracle_source = copy.deepcopy(base_source)
    oracle_external = copy.deepcopy(base_external)
    oracle_epoch = 0
    oracle_score = float(base_source["weighted_f1"] + base_external["weighted_f1"])
    rows: List[Dict] = []
    print(f"[CESA] train_pkl={train_pkl} source={len(source_train)} bobsl={len(bobsl_train)}")
    print(f"[CESA] base_checkpoint={base_checkpoint}")
    print(f"[CESA] basis={json.dumps(basis_meta)}")
    print(f"[CESA] base source={base_source['weighted_f1']:.4f} external={base_external['weighted_f1']:.4f}")

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_metrics = train_adapter_epoch(
            model, source_train_loader, bobsl_train_loader, optimizer, weights, device, epoch, args
        )
        source_result = evaluate(model, source_test_loader, device, epoch, adapted=True)
        external_result = evaluate(model, external_test_loader, device, epoch, adapted=True)
        source_score = float(source_result[args.selection_metric])
        if source_score > best_score:
            best_score = source_score
            best_epoch = epoch
            best_source = copy.deepcopy(source_result)
            best_external = copy.deepcopy(external_result)
            best_state = copy.deepcopy(model.state_dict())
        paired_score = float(source_result["weighted_f1"] + external_result["weighted_f1"])
        if source_result["weighted_f1"] >= base_source["weighted_f1"] and paired_score > oracle_score:
            oracle_score = paired_score
            oracle_epoch = epoch
            oracle_source = copy.deepcopy(source_result)
            oracle_external = copy.deepcopy(external_result)
            oracle_state = copy.deepcopy(model.state_dict())
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            "source_weighted_f1": source_result["weighted_f1"],
            "source_macro_f1": source_result["macro_f1"],
            "external_weighted_f1": external_result["weighted_f1"],
            "external_macro_f1": external_result["macro_f1"],
            "delta_source": source_result["weighted_f1"] - base_source["weighted_f1"],
            "delta_external": external_result["weighted_f1"] - base_external["weighted_f1"],
            "time_sec": round(time.time() - start, 2),
        }
        rows.append(row)
        print(
            f"[CESA] epoch={epoch} source={row['source_weighted_f1']:.4f} ({row['delta_source']:+.4f}) "
            f"external={row['external_weighted_f1']:.4f} ({row['delta_external']:+.4f}) "
            f"loss={train_metrics['loss']:.4f} time={row['time_sec']}s"
        )

    final_source = copy.deepcopy(source_result)
    final_external = copy.deepcopy(external_result)
    final_state = copy.deepcopy(model.state_dict())
    with (out_dir / "epoch_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    common = {
        "script": "new2/CMERC/train_eval_cmerc_cesa_visual_anjs.py",
        "method": "CESA-CMERC",
        "seed": args.seed,
        "train_pkl": str(train_pkl),
        "external_test_pkl": str(external_pkl),
        "base_checkpoint": str(base_checkpoint),
        "basis_meta": basis_meta,
        "selection_metric": args.selection_metric,
        "best_epoch": best_epoch,
        "base_source_weighted_f1": base_source["weighted_f1"],
        "base_external_weighted_f1": base_external["weighted_f1"],
    }
    base_script.save_report(base_source, out_dir, "source_test_base", {**common, "selection": "base"})
    base_script.save_report(base_external, out_dir, "external_test_base", {**common, "selection": "base"})
    base_script.save_report(final_source, out_dir, "source_test_final", {**common, "selection": "fixed_final"})
    base_script.save_report(final_external, out_dir, "external_test_final", {**common, "selection": "fixed_final"})
    base_script.save_report(best_source, out_dir, "source_test_best", {**common, "selection": "source"})
    base_script.save_report(best_external, out_dir, "external_test_best", {**common, "selection": "source"})
    base_script.save_report(oracle_source, out_dir, "source_test_oracle", {**common, "selection": "diagnostic_oracle", "oracle_epoch": oracle_epoch})
    base_script.save_report(oracle_external, out_dir, "external_test_oracle", {**common, "selection": "diagnostic_oracle", "oracle_epoch": oracle_epoch})
    model.load_state_dict(best_state)
    save_cesa_checkpoint(out_dir / "model_best.pt", model, args, info, basis_meta, best_epoch, best_score)
    model.load_state_dict(final_state)
    save_cesa_checkpoint(
        out_dir / "model_final.pt", model, args, info, basis_meta, args.epochs, final_source[args.selection_metric]
    )
    model.load_state_dict(oracle_state)
    save_cesa_checkpoint(out_dir / "model_oracle_diagnostic.pt", model, args, info, basis_meta, oracle_epoch, oracle_score)
    print(
        f"[CESA] done best_epoch={best_epoch} source={best_source['weighted_f1']:.4f} "
        f"external={best_external['weighted_f1']:.4f} oracle_epoch={oracle_epoch} "
        f"oracle={oracle_source['weighted_f1']:.4f}/{oracle_external['weighted_f1']:.4f} out={out_dir}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a single-checkpoint CESA adapter on frozen CMERC.")
    parser.add_argument("--train_pkl", required=True)
    parser.add_argument("--external_test_pkl", required=True)
    parser.add_argument("--base_checkpoint", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--source_batch_size", type=int, default=8)
    parser.add_argument("--bobsl_batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--adapter_dim", type=int, default=128)
    parser.add_argument("--local_hidden_dim", type=int, default=128)
    parser.add_argument("--adapter_dropout", type=float, default=0.2)
    parser.add_argument("--residual_scale", type=float, default=1.0)
    parser.add_argument("--subspace_rank", type=int, default=3)
    parser.add_argument("--projection", choices=["emotion", "full"], default="emotion")
    parser.add_argument("--source_weight", type=float, default=1.0)
    parser.add_argument("--bobsl_weight", type=float, default=1.0)
    parser.add_argument(
        "--bobsl_context_weight",
        type=float,
        default=0.0,
        help="Naive-joint ablation: pass length-one BOBSL samples through the frozen context model.",
    )
    parser.add_argument("--local_source_weight", type=float, default=0.25)
    parser.add_argument("--safety_weight", type=float, default=1.0)
    parser.add_argument("--safety_margin", type=float, default=0.0)
    parser.add_argument("--residual_weight", type=float, default=0.01)
    parser.add_argument("--class_weight", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--selection_metric", choices=["weighted_f1", "macro_f1"], default="weighted_f1")
    parser.add_argument("--no_cuda", action="store_true")

    # CMERC architecture defaults; checkpoint metadata replaces these before loading.
    parser.add_argument("--base_model", default="GRU")
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--graph_type", default="hyper")
    parser.add_argument("--graph_construct", default="direct")
    parser.add_argument("--mm_fusion_mthd", default="concat_DHT")
    parser.add_argument("--norm", default="BN")
    parser.add_argument("--num_L", type=int, default=3)
    parser.add_argument("--num_K", type=int, default=3)
    parser.add_argument("--windowp", type=int, default=10)
    parser.add_argument("--windowf", type=int, default=10)
    parser.add_argument("--attention", default="general")
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--multiheads", type=int, default=6)
    parser.add_argument("--use_residue", action="store_true")
    parser.add_argument("--deep_gcn_nlayers", type=int, default=4)
    parser.add_argument("--D_g", type=int, default=1024)
    parser.add_argument("--D_p", type=int, default=150)
    parser.add_argument("--D_e", type=int, default=100)
    parser.add_argument("--D_h", type=int, default=100)
    parser.add_argument("--D_a", type=int, default=100)
    parser.add_argument("--graph_hidden_size", type=int, default=100)
    parser.add_argument("--max_seq_len", type=int, default=110)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
