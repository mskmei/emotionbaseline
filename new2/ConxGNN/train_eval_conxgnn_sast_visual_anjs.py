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
from sklearn import metrics

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(REPO_ROOT / "new"))

import src  # noqa: E402
import train_eval_conxgnn_visual_anjs as base_script  # noqa: E402
from anjs_video_common import ANJS_LABELS, resolve_path  # noqa: E402
from cesa import ContextDecoupledEmotionSubspaceAdapter, compute_emotion_subspace  # noqa: E402
from src.TensorGraph import TensorGraph  # noqa: E402


def is_bobsl_sample(sample: Dict) -> bool:
    sample_id = str(sample.get("id", ""))
    return sample_id.startswith("joint_bobsl") or sample_id.startswith("bobsl")


def split_samples(samples: Sequence[Dict]) -> tuple[List[Dict], List[Dict]]:
    source = [sample for sample in samples if not is_bobsl_sample(sample)]
    bobsl = [sample for sample in samples if is_bobsl_sample(sample)]
    return source, bobsl


def flatten_batch(values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    return torch.cat([values[index, : int(length)] for index, length in enumerate(lengths.tolist())], dim=0)


def conx_forward_logits(model: nn.Module, data: Dict) -> torch.Tensor:
    # Keep the adapted visual tensor attached to autograd. The baseline helper
    # deep-copies its input, which PyTorch does not support for non-leaf tensors.
    graph_data = dict(data)
    base_tensor_graph = TensorGraph(graph_data, model.modalities)
    _uni_loss, tensor_graph = model.uniencoder.get_loss(base_tensor_graph)
    tensor_graph.padded_dict2multimodal_features()
    graph_out, _ = model.graph_model.get_loss(tensor_graph)
    clf = model.clf
    graph_out.node_concat2multimodal_features()
    if clf.use_sa:
        graph_out, _ = clf.UnimodalSelfAttention(graph_out)
    if clf.use_ca:
        graph_out = clf.CrossModalAttention(graph_out)
    graph_out.multimodal_features2node_concat()
    graph_out.node_concat2dim_concat()
    graph_out.data = clf.fc_layer(graph_out.data)
    return clf.mlp_layer(torch.relu(graph_out.data))


class SASTConxGNN(nn.Module):
    """Source-Anchored Subspace Transfer with a frozen ConxGNN backbone."""

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

    def forward(self, data: Dict, adapted: bool) -> tuple[torch.Tensor, torch.Tensor]:
        current = dict(data)
        current["text_tensor"] = torch.zeros_like(current["text_tensor"])
        current["audio_tensor"] = torch.zeros_like(current["audio_tensor"])
        if adapted:
            current["visual_tensor"], delta = self.adapter(current["visual_tensor"])
        else:
            delta = torch.zeros_like(current["visual_tensor"])
        return conx_forward_logits(self.base_model, current), delta


def adapter_parameters(model: SASTConxGNN) -> Iterable[nn.Parameter]:
    yield from model.adapter.parameters()


def make_dataset(samples: Sequence[Dict], batch_size: int, info: Dict) -> base_script.ConxVisualDataset:
    return base_script.ConxVisualDataset(samples, batch_size, info["d_text"], info["d_audio"], info["d_visual"])


def build_basis(source_samples: Sequence[Dict], rank: int):
    features = {str(sample["id"]): sample["visual"] for sample in source_samples}
    labels = {str(sample["id"]): sample["labels"] for sample in source_samples}
    return compute_emotion_subspace(features, labels, list(features), len(ANJS_LABELS), rank=rank)


def load_base_model(args, info: Dict, source_train, source_test, checkpoint: Dict, device: torch.device):
    checkpoint_args = checkpoint.get("args", {})
    architecture_keys = [
        "config_yaml", "drop_rate", "hidden_size", "inter_size", "weight_decay",
        "ce_loss_param", "cb_loss_param", "focal_gamma", "loss", "class_weight",
    ]
    for key in architecture_keys:
        if key in checkpoint_args:
            setattr(args, key, checkpoint_args[key])
    args.device = str(device)
    empty_dev = make_dataset([], args.eval_batch_size, info)
    config = base_script.build_config(args, info, source_train, empty_dev, source_test)
    model = src.MainModel(config).to(device)
    state = checkpoint.get("model_state_dict", checkpoint)
    incompatible = model.load_state_dict(
        {key.removeprefix("module."): value for key, value in state.items()}, strict=False
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={incompatible.missing_keys[:10]} "
            f"unexpected={incompatible.unexpected_keys[:10]}"
        )
    return model


def move_batch(raw: Dict, device: torch.device) -> Dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in raw.items()
    }


def metric_result(losses, counts, golds, preds, probs, ids) -> Optional[Dict]:
    if not golds:
        return None
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
def evaluate(model: SASTConxGNN, dataset, device: torch.device, adapted: bool) -> Optional[Dict]:
    model.eval()
    losses: List[float] = []
    counts: List[int] = []
    golds: List[int] = []
    preds: List[int] = []
    probs: List[np.ndarray] = []
    ids: List[str] = []
    for index in range(len(dataset)):
        raw = dataset[index]
        data = move_batch(raw, device)
        labels = data["label_tensor"]
        valid = labels.ge(0)
        logits, _ = model(data, adapted=adapted)
        current = logits[valid]
        target = labels[valid]
        if not len(target):
            continue
        loss = F.cross_entropy(current, target)
        probability = torch.softmax(current, dim=-1)
        pred = probability.argmax(dim=-1)
        losses.append(float(loss.item()) * len(target))
        counts.append(len(target))
        golds.extend(target.cpu().numpy().astype(int).tolist())
        preds.extend(pred.cpu().numpy().astype(int).tolist())
        probs.append(probability.cpu().numpy().astype(np.float32))
        mask = valid.cpu().numpy().astype(bool).tolist()
        ids.extend([item_id for item_id, keep in zip(raw["item_ids"], mask) if keep])
    return metric_result(losses, counts, golds, preds, probs, ids)


def weighted_ce(logits, labels, weights, reduction="mean"):
    return F.cross_entropy(logits, labels, weight=weights, reduction=reduction)


def train_epoch(model, source_set, bobsl_set, optimizer, class_weights, device, args, epoch):
    model.base_model.eval()
    model.adapter.train()
    source_set.shuffle(args.seed + epoch)
    bobsl_set.shuffle(args.seed + epoch + 1000)
    bobsl_indices = cycle(range(len(bobsl_set)))
    totals = {key: 0.0 for key in ["loss", "source", "bobsl", "safety", "residual"]}
    steps = 0
    for source_index in range(len(source_set)):
        optimizer.zero_grad(set_to_none=True)
        source = move_batch(source_set[source_index], device)
        source_labels = source["label_tensor"]
        source_valid = source_labels.ge(0)
        with torch.no_grad():
            source_base, _ = model(source, adapted=False)
        source_adapted, source_delta = model(source, adapted=True)
        source_nll = weighted_ce(
            source_adapted[source_valid], source_labels[source_valid], class_weights, reduction="none"
        )
        base_nll = weighted_ce(
            source_base[source_valid], source_labels[source_valid], class_weights, reduction="none"
        )
        source_loss = source_nll.mean()
        safety_loss = F.relu(source_nll - base_nll + args.safety_margin).mean()
        source_delta = flatten_batch(source_delta, source["text_len_tensor"])
        source_residual = source_delta[source_valid].pow(2).sum(dim=-1).mean()

        bobsl = move_batch(bobsl_set[next(bobsl_indices)], device)
        bobsl_labels = bobsl["label_tensor"]
        bobsl_valid = bobsl_labels.ge(0)
        bobsl_logits, bobsl_delta = model(bobsl, adapted=True)
        bobsl_loss = weighted_ce(bobsl_logits[bobsl_valid], bobsl_labels[bobsl_valid], class_weights)
        bobsl_delta = flatten_batch(bobsl_delta, bobsl["text_len_tensor"])
        bobsl_residual = bobsl_delta[bobsl_valid].pow(2).sum(dim=-1).mean()
        residual_loss = source_residual + bobsl_residual

        loss = (
            args.source_weight * source_loss
            + args.bobsl_weight * bobsl_loss
            + args.safety_weight * safety_loss
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
            "residual": residual_loss,
        }.items():
            totals[key] += float(value.detach().item())
        steps += 1
    return {key: value / max(steps, 1) for key, value in totals.items()}


def save_checkpoint(path: Path, model, args, info, basis_meta, epoch, score):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "method": "SAST-ConxGNN",
            "label_names": ANJS_LABELS,
            "args": vars(args),
            "info": info,
            "basis_meta": basis_meta,
            "epoch": epoch,
            "source_score": score,
        },
        path,
    )


def train(args):
    base_script.seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    if device.type != "cuda":
        raise RuntimeError("ConxGNN requires CUDA")
    train_pkl = resolve_path(args.train_pkl, must_exist=True)
    external_pkl = resolve_path(args.external_test_pkl, must_exist=True)
    checkpoint_path = resolve_path(args.base_checkpoint, must_exist=True)
    out_dir = resolve_path(args.out_dir, must_exist=False)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = base_script.load_payload(train_pkl)
    external_payload = base_script.load_payload(external_pkl)
    info = base_script.inspect_pkl(train_pkl)
    source_samples, bobsl_samples = split_samples(payload["train"])
    if not bobsl_samples:
        raise RuntimeError("SAST requires a joint source/BOBSL pkl")
    basis, centers, counts = build_basis(source_samples, args.subspace_rank)
    basis_meta = {
        "rank": int(basis.shape[1]),
        "projection": args.projection,
        "source_counts": counts.tolist(),
        "source_center_norms": centers.norm(dim=1).tolist(),
        "source_dialogues": len(source_samples),
        "bobsl_dialogues": len(bobsl_samples),
    }

    source_train = make_dataset(source_samples, args.source_batch_size, info)
    bobsl_train = make_dataset(bobsl_samples, args.bobsl_batch_size, info)
    source_test = make_dataset(payload["test"], args.eval_batch_size, info)
    external_test = make_dataset(external_payload["test"], args.eval_batch_size, info)
    for name, dataset in [
        ("source_train", source_train), ("bobsl_train", bobsl_train),
        ("source_test", source_test), ("external_test", external_test),
    ]:
        base_script.validate_hyperedge_capacity(name, dataset)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    base_model = load_base_model(args, info, source_train, source_test, checkpoint, device)
    model = SASTConxGNN(base_model, int(info["d_visual"]), basis, args).to(device)
    for parameter in model.base_model.parameters():
        parameter.requires_grad = False
    labels = [int(label) for sample in source_samples for label in sample["labels"] if int(label) >= 0]
    class_weights = (
        base_script.make_class_weights(labels, len(ANJS_LABELS), device)
        if args.class_weight and not args.unweighted_adapter
        else None
    )
    optimizer = torch.optim.AdamW(adapter_parameters(model), lr=args.lr, weight_decay=args.weight_decay)

    base_source = evaluate(model, source_test, device, adapted=False)
    base_external = evaluate(model, external_test, device, adapted=False)
    best_source = copy.deepcopy(base_source)
    best_external = copy.deepcopy(base_external)
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_score = float(base_source[args.selection_metric])
    rows = []
    trainable = sum(parameter.numel() for parameter in adapter_parameters(model) if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    print(f"[SAST] train_pkl={train_pkl} source={len(source_samples)} bobsl={len(bobsl_samples)}")
    print(f"[SAST] base_checkpoint={checkpoint_path}")
    print(f"[SAST] basis={json.dumps(basis_meta)} trainable={trainable}/{total}")
    print(f"[SAST] base source={base_source['weighted_f1']:.4f} external={base_external['weighted_f1']:.4f}")

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_metrics = train_epoch(
            model, source_train, bobsl_train, optimizer, class_weights, device, args, epoch
        )
        source_result = evaluate(model, source_test, device, adapted=True)
        external_result = evaluate(model, external_test, device, adapted=True)
        score = float(source_result[args.selection_metric])
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_source = copy.deepcopy(source_result)
            best_external = copy.deepcopy(external_result)
            best_state = copy.deepcopy(model.state_dict())
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
            f"[SAST] epoch={epoch} source={row['source_weighted_f1']:.4f} ({row['delta_source']:+.4f}) "
            f"external={row['external_weighted_f1']:.4f} ({row['delta_external']:+.4f}) "
            f"loss={train_metrics['loss']:.4f} time={row['time_sec']}s"
        )

    with (out_dir / "epoch_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    common = {
        "script": "new2/ConxGNN/train_eval_conxgnn_sast_visual_anjs.py",
        "method": "SAST-ConxGNN",
        "seed": args.seed,
        "train_pkl": str(train_pkl),
        "external_test_pkl": str(external_pkl),
        "base_checkpoint": str(checkpoint_path),
        "basis_meta": basis_meta,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "selection_metric": args.selection_metric,
        "best_epoch": best_epoch,
        "base_source_weighted_f1": base_source["weighted_f1"],
        "base_external_weighted_f1": base_external["weighted_f1"],
    }
    final_source = source_result
    final_external = external_result
    base_script.save_report(base_source, out_dir, "source_test_base", {**common, "selection": "base"})
    base_script.save_report(base_external, out_dir, "external_test_base", {**common, "selection": "base"})
    base_script.save_report(final_source, out_dir, "source_test_final", {**common, "selection": "fixed_final"})
    base_script.save_report(final_external, out_dir, "external_test_final", {**common, "selection": "fixed_final"})
    base_script.save_report(best_source, out_dir, "source_test_best", {**common, "selection": "source"})
    base_script.save_report(best_external, out_dir, "external_test_best", {**common, "selection": "source"})
    model.load_state_dict(best_state)
    save_checkpoint(out_dir / "model_best.pt", model, args, info, basis_meta, best_epoch, best_score)
    summary = {
        **common,
        "base_source": base_source["weighted_f1"],
        "base_external": base_external["weighted_f1"],
        "best_source": best_source["weighted_f1"],
        "best_external": best_external["weighted_f1"],
        "delta_source": best_source["weighted_f1"] - base_source["weighted_f1"],
        "delta_external": best_external["weighted_f1"] - base_external["weighted_f1"],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"[SAST] done epoch={best_epoch} source={best_source['weighted_f1']:.4f} "
        f"external={best_external['weighted_f1']:.4f} out={out_dir}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Source-Anchored Subspace Transfer for ConxGNN.")
    parser.add_argument("--train_pkl", required=True)
    parser.add_argument("--external_test_pkl", required=True)
    parser.add_argument("--base_checkpoint", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--config_yaml", default=str(THIS_DIR / "configs" / "meld.yaml"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--source_batch_size", type=int, default=8)
    parser.add_argument("--bobsl_batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--adapter_dim", type=int, default=128)
    parser.add_argument("--adapter_dropout", type=float, default=0.2)
    parser.add_argument("--subspace_rank", type=int, default=1)
    parser.add_argument("--projection", choices=["emotion", "full"], default="emotion")
    parser.add_argument("--residual_scale", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--source_weight", type=float, default=1.0)
    parser.add_argument("--bobsl_weight", type=float, default=1.0)
    parser.add_argument("--safety_weight", type=float, default=0.5)
    parser.add_argument("--safety_margin", type=float, default=0.0)
    parser.add_argument("--residual_weight", type=float, default=1e-3)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--selection_metric", choices=["weighted_f1", "macro_f1", "accuracy"], default="weighted_f1")
    parser.add_argument("--class_weight", action="store_true")
    parser.add_argument("--unweighted_adapter", action="store_true")
    parser.add_argument("--no_cuda", action="store_true")
    # Filled from the source checkpoint before model construction.
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--drop_rate", type=float, default=0.8)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--inter_size", type=int, default=256)
    parser.add_argument("--loss", default="ce")
    parser.add_argument("--focal_gamma", type=float, default=2.5)
    parser.add_argument("--ce_loss_param", type=float, default=1.0)
    parser.add_argument("--cb_loss_param", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
