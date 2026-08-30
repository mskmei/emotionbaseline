#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import pickle
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import metrics
from torch.utils.data import DataLoader, Subset

from dataloader import MELDDataset
from model import DialogueGCNModel


MODALITY_ALIASES = {
    "full": "full",
    "avl": "full",
    "text": "text",
    "t": "text",
    "l": "text",
    "video": "video",
    "visual": "video",
    "v": "video",
    "tv": "tv",
    "vt": "tv",
    "vl": "tv",
    "text_video": "tv",
    "video_text": "tv",
}


class FocalNLLLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else torch.empty(0))

    def forward(self, log_probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        true_log_probs = log_probs.gather(1, labels.view(-1, 1)).squeeze(1)
        pt = true_log_probs.exp().clamp_min(1e-8)
        loss = -((1.0 - pt) ** self.gamma) * true_log_probs
        if self.weight.numel() > 0:
            loss = loss * self.weight.gather(0, labels)
        return loss.mean()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_path(raw: str, must_exist: bool = False) -> Path:
    path = Path(raw).expanduser()
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, Path(__file__).resolve().parent / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    if must_exist:
        raise FileNotFoundError(f"Path not found: {raw}. Checked: {', '.join(str(x) for x in candidates)}")
    return candidates[0].resolve()


def normalize_modality(value: str) -> str:
    key = (value or "tv").strip().lower()
    if key not in MODALITY_ALIASES:
        valid = ", ".join(sorted(MODALITY_ALIASES))
        raise ValueError(f"Unsupported modality={value!r}; valid values: {valid}")
    return MODALITY_ALIASES[key]


def parse_modalities(raw_values: Sequence[str]) -> List[str]:
    out: List[str] = []
    for value in raw_values:
        for part in str(value).replace(",", " ").split():
            modality = normalize_modality(part)
            if modality not in out:
                out.append(modality)
    return out or ["tv"]


def load_feature_tuple(path: Path):
    return pickle.load(path.open("rb"), encoding="latin1")


def inspect_feature_pkl(path: Path) -> Dict:
    obj = load_feature_tuple(path)
    video_speakers = obj[1]
    video_labels = obj[2]
    video_text = obj[3]
    video_audio = obj[4]
    video_visual = obj[5]
    train_vid = obj[7]
    test_vid = obj[8]
    meta = obj[9] if len(obj) > 9 and isinstance(obj[9], dict) else {}
    keys = list(train_vid) + list(test_vid)
    if not keys:
        raise RuntimeError(f"No dialogue keys in {path}")

    first_key = keys[0]
    d_text = int(np.asarray(video_text[first_key]).shape[-1])
    d_audio = int(np.asarray(video_audio[first_key]).shape[-1])
    d_visual = int(np.asarray(video_visual[first_key]).shape[-1])
    n_speakers = int(np.asarray(video_speakers[first_key]).shape[-1])

    valid_labels: List[int] = []
    for key in keys:
        valid_labels.extend([int(x) for x in video_labels[key] if int(x) >= 0])
    if not valid_labels:
        raise RuntimeError(f"No valid labels in {path}")

    n_classes = int(meta.get("n_classes") or (max(valid_labels) + 1))
    label_names = list(meta.get("label_names") or [str(i) for i in range(n_classes)])
    if len(label_names) < n_classes:
        label_names.extend(str(i) for i in range(len(label_names), n_classes))

    return {
        "path": str(path),
        "d_text": d_text,
        "d_audio": d_audio,
        "d_visual": d_visual,
        "n_speakers": n_speakers,
        "n_classes": n_classes,
        "label_names": label_names[:n_classes],
        "train_dialogues": len(train_vid),
        "test_dialogues": len(test_vid),
        "valid_label_count": len(valid_labels),
        "valid_label_counts": np.bincount(np.asarray(valid_labels), minlength=n_classes).tolist(),
        "meta": meta,
    }


def collect_train_labels(path: Path) -> List[int]:
    obj = load_feature_tuple(path)
    video_labels = obj[2]
    train_vid = obj[7]
    labels: List[int] = []
    for key in train_vid:
        labels.extend([int(x) for x in video_labels[key] if int(x) >= 0])
    return labels


def make_class_weights(labels: Sequence[int], n_classes: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=n_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    total = float(counts.sum())
    weights = total / (float(n_classes) * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_loaders(args, train_pkl: Path, external_test_pkl: Optional[Path]):
    trainset = MELDDataset(str(train_pkl), train=True)
    source_testset = MELDDataset(str(train_pkl), train=False)

    indices = list(range(len(trainset)))
    valid_size = int(max(0.0, min(args.valid_ratio, 0.9)) * len(indices))
    valid_indices = indices[:valid_size]
    train_indices = indices[valid_size:]
    if args.max_train_dialogues > 0:
        train_indices = train_indices[: args.max_train_dialogues]
    if args.max_source_test_dialogues > 0:
        source_test_indices = list(range(min(len(source_testset), args.max_source_test_dialogues)))
    else:
        source_test_indices = list(range(len(source_testset)))

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        Subset(trainset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=trainset.collate_fn,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available() and not args.no_cuda,
        generator=generator,
    )
    valid_loader = None
    if valid_indices:
        valid_loader = DataLoader(
            Subset(trainset, valid_indices),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=trainset.collate_fn,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available() and not args.no_cuda,
        )
    source_test_loader = DataLoader(
        Subset(source_testset, source_test_indices),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=source_testset.collate_fn,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available() and not args.no_cuda,
    )

    external_loader = None
    if external_test_pkl is not None:
        external_testset = MELDDataset(str(external_test_pkl), train=False)
        external_indices = list(range(len(external_testset)))
        if args.max_external_test_dialogues > 0:
            external_indices = external_indices[: args.max_external_test_dialogues]
        external_loader = DataLoader(
            Subset(external_testset, external_indices),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=external_testset.collate_fn,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available() and not args.no_cuda,
        )

    return train_loader, valid_loader, source_test_loader, external_loader


def resolve_modals(modality: str, graph_type: str, requested: str) -> Tuple[str, str]:
    if requested != "auto":
        modals = requested
    elif modality == "full":
        modals = "avl"
    elif modality == "tv":
        modals = "vl"
    elif modality == "text":
        modals = "l"
    elif modality == "video":
        modals = "v"
    else:
        raise ValueError(f"Unsupported modality={modality}")

    resolved_graph = graph_type
    if graph_type in {"MMGCN", "MMGCN2"} and len(modals) < 2:
        resolved_graph = "relation"
        print(f"[MMGCN] graph_type={graph_type} does not support single modal {modals}; using relation graph.")
    return modals, resolved_graph


def zero_unused_modalities(textf, visuf, acouf, modality: str):
    if modality == "text":
        return textf, torch.zeros_like(visuf), torch.zeros_like(acouf)
    if modality == "video":
        return torch.zeros_like(textf), visuf, torch.zeros_like(acouf)
    if modality == "tv":
        return textf, visuf, torch.zeros_like(acouf)
    return textf, visuf, acouf


def build_model(args, info: Dict, modality: str, device: torch.device) -> Tuple[DialogueGCNModel, str, str]:
    modals, graph_type = resolve_modals(modality, args.graph_type, args.modals)
    if graph_type in {"MMGCN", "MMGCN2"} and device.type != "cuda":
        raise RuntimeError("MMGCN/model_mm.py uses CUDA tensors internally. Run on the GPU server or set --graph_type relation.")

    model = DialogueGCNModel(
        args.base_model,
        int(info["d_text"]),
        args.d_g,
        args.d_p,
        args.d_e,
        args.d_h,
        args.d_a,
        args.graph_hidden,
        n_speakers=int(info["n_speakers"]),
        max_seq_len=args.max_seq_len,
        window_past=args.windowp,
        window_future=args.windowf,
        n_classes=int(info["n_classes"]),
        listener_state=args.active_listener,
        context_attention=args.attention,
        dropout=args.dropout,
        nodal_attention=args.nodal_attention,
        no_cuda=args.no_cuda,
        graph_type=graph_type,
        use_topic=args.use_topic,
        alpha=args.alpha,
        multiheads=args.multiheads,
        graph_construct=args.graph_construct,
        use_GCN=args.use_gcn,
        use_residue=not args.no_residue,
        D_m_v=int(info["d_visual"]),
        D_m_a=int(info["d_audio"]),
        modals=modals,
        att_type=args.mm_fusion_mthd,
        av_using_lstm=args.av_using_lstm,
        Deep_GCN_nlayers=args.deep_gcn_nlayers,
        dataset="MELD",
        use_speaker=args.use_speaker,
        use_modal=args.use_modal,
    )
    return model.to(device), modals, graph_type


def sequence_lengths(umask: torch.Tensor) -> List[int]:
    lengths: List[int] = []
    for row in umask:
        idx = (row == 1).nonzero(as_tuple=False)
        lengths.append(0 if idx.numel() == 0 else int(idx[-1].item()) + 1)
    return lengths


def run_epoch(
    model: DialogueGCNModel,
    loss_function: nn.Module,
    dataloader: Optional[DataLoader],
    device: torch.device,
    modality: str,
    optimizer: Optional[optim.Optimizer] = None,
    max_grad_norm: float = 0.0,
) -> Optional[Dict]:
    if dataloader is None:
        return None

    train = optimizer is not None
    model.train() if train else model.eval()
    losses: List[float] = []
    loss_weights: List[int] = []
    golds_all: List[int] = []
    preds_all: List[int] = []
    probs_all: List[np.ndarray] = []
    ids_all: List[str] = []

    for data in dataloader:
        if train:
            optimizer.zero_grad()

        textf, visuf, acouf, qmask, umask, label = [d.to(device) for d in data[:-1]]
        textf, visuf, acouf = zero_unused_modalities(textf, visuf, acouf, modality)
        lengths = sequence_lengths(umask)

        log_prob, *_ = model(textf, qmask, umask, lengths, acouf, visuf)
        labels_flat = torch.cat([label[j][: lengths[j]] for j in range(len(label))])
        valid = labels_flat.ge(0)
        valid_count = int(valid.sum().item())
        if valid_count == 0:
            continue

        loss = loss_function(log_prob[valid], labels_flat[valid])
        if train:
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        with torch.no_grad():
            valid_log_prob = log_prob[valid]
            pred = torch.argmax(valid_log_prob, dim=1)
            losses.append(float(loss.item()) * valid_count)
            loss_weights.append(valid_count)
            golds_all.extend(labels_flat[valid].detach().cpu().numpy().astype(int).tolist())
            preds_all.extend(pred.detach().cpu().numpy().astype(int).tolist())
            probs_all.append(valid_log_prob.exp().detach().cpu().numpy().astype(np.float32))

            labels_matrix = label.detach().cpu()
            vids = data[-1]
            for batch_idx, vid in enumerate(vids):
                for utt_idx in range(lengths[batch_idx]):
                    if int(labels_matrix[batch_idx, utt_idx]) >= 0:
                        ids_all.append(f"{vid}:utt{utt_idx + 1:02d}")

    if not golds_all:
        return None

    golds = np.asarray(golds_all, dtype=np.int64)
    preds = np.asarray(preds_all, dtype=np.int64)
    probs = np.concatenate(probs_all, axis=0)
    label_ids = list(range(probs.shape[1]))
    return {
        "loss": float(np.sum(losses) / max(np.sum(loss_weights), 1)),
        "accuracy": float(metrics.accuracy_score(golds, preds)),
        "macro_f1": float(metrics.f1_score(golds, preds, labels=label_ids, average="macro", zero_division=0)),
        "weighted_f1": float(metrics.f1_score(golds, preds, labels=label_ids, average="weighted", zero_division=0)),
        "golds": golds,
        "preds": preds,
        "probs": probs,
        "ids": ids_all,
    }


def result_snapshot(result: Optional[Dict]) -> Optional[Dict]:
    if result is None:
        return None
    return {
        key: value.copy() if isinstance(value, np.ndarray) else list(value) if isinstance(value, list) else value
        for key, value in result.items()
    }


def write_confusion_matrix(cm: np.ndarray, labels: Sequence[str], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("labels: " + ",".join(labels) + "\n")
        for row in cm:
            f.write(" ".join(str(int(x)) for x in row) + "\n")


def save_report(result: Optional[Dict], out_dir: Path, prefix: str, label_names: Sequence[str], extra: Dict) -> Optional[Dict]:
    if result is None:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    label_ids = list(range(len(label_names)))
    golds = result["golds"]
    preds = result["preds"]
    probs = result["probs"]
    cm = metrics.confusion_matrix(golds, preds, labels=label_ids)
    report = metrics.classification_report(
        golds,
        preds,
        labels=label_ids,
        target_names=list(label_names),
        digits=4,
        zero_division=0,
    )
    (out_dir / f"{prefix}_classification_report.txt").write_text(report, encoding="utf-8")
    np.save(out_dir / f"{prefix}_confusion_matrix.npy", cm)
    write_confusion_matrix(cm, label_names, out_dir / f"{prefix}_confusion_matrix.txt")

    with (out_dir / f"{prefix}_predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["item_id", "gold_id", "gold_label", "pred_id", "pred_label", "correct"]
            + [f"prob_{name}" for name in label_names]
        )
        for item_id, gold, pred, prob in zip(result["ids"], golds, preds, probs):
            writer.writerow(
                [item_id, int(gold), label_names[int(gold)], int(pred), label_names[int(pred)], int(gold == pred)]
                + [float(x) for x in prob]
            )

    per_p, per_r, per_f1, per_support = metrics.precision_recall_fscore_support(
        golds,
        preds,
        labels=label_ids,
        zero_division=0,
    )
    summary = {
        **extra,
        "n_samples": int(len(golds)),
        "loss": float(result["loss"]),
        "accuracy": float(result["accuracy"]),
        "macro_f1": float(result["macro_f1"]),
        "weighted_f1": float(result["weighted_f1"]),
        "gold_counts": dict(zip(label_names, np.bincount(golds, minlength=len(label_names)).astype(int).tolist())),
        "pred_counts": dict(zip(label_names, np.bincount(preds, minlength=len(label_names)).astype(int).tolist())),
        "per_class": {
            label_names[i]: {
                "precision": float(per_p[i]),
                "recall": float(per_r[i]),
                "f1": float(per_f1[i]),
                "support": int(per_support[i]),
                "mean_pred_prob": float(probs[:, i].mean()) if len(probs) else 0.0,
            }
            for i in label_ids
        },
    }
    (out_dir / f"{prefix}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def build_loss(args, train_labels: Sequence[int], n_classes: int, device: torch.device) -> nn.Module:
    weight = None if args.no_class_weight else make_class_weights(train_labels, n_classes, device)
    if weight is not None:
        print(f"[MMGCN] class_weights={weight.detach().cpu().numpy().round(4).tolist()}")
    if args.loss == "nll":
        return nn.NLLLoss(weight=weight)
    if args.loss == "focal":
        return FocalNLLLoss(gamma=args.focal_gamma, weight=weight)
    raise ValueError(f"Unsupported loss: {args.loss}")


def train_one_modality(args, modality: str, train_pkl: Path, external_test_pkl: Optional[Path], device: torch.device, root_out: Path) -> None:
    seed_everything(args.seed)
    info = inspect_feature_pkl(train_pkl)
    if external_test_pkl is not None:
        external_info = inspect_feature_pkl(external_test_pkl)
        for key in ["d_text", "d_audio", "d_visual", "n_speakers", "n_classes"]:
            if int(info[key]) != int(external_info[key]):
                raise RuntimeError(
                    f"Source/target feature mismatch for {key}: source={info[key]} "
                    f"target={external_info[key]}. Rebuild both pkl files with the same command."
                )
    if args.n_classes > 0:
        info["n_classes"] = args.n_classes
        info["label_names"] = args.label_names[: args.n_classes] if args.label_names else [str(i) for i in range(args.n_classes)]

    train_loader, valid_loader, source_test_loader, external_loader = build_loaders(args, train_pkl, external_test_pkl)
    model, modals, graph_type = build_model(args, info, modality, device)
    train_labels = collect_train_labels(train_pkl)
    loss_function = build_loss(args, train_labels, int(info["n_classes"]), device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    out_dir = root_out / modality
    out_dir.mkdir(parents=True, exist_ok=True)
    epoch_rows: List[Dict] = []
    best_state = None
    best_epoch = 0
    best_source = None
    best_external = None
    best_score = -1.0
    final_source = None
    final_external = None

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_result = run_epoch(
            model,
            loss_function,
            train_loader,
            device,
            modality,
            optimizer=optimizer,
            max_grad_norm=args.max_grad_norm,
        )
        valid_result = run_epoch(model, loss_function, valid_loader, device, modality)

        source_result = None
        external_result = None
        if epoch % max(args.eval_every, 1) == 0 or epoch == args.epochs:
            source_result = run_epoch(model, loss_function, source_test_loader, device, modality)
            external_result = run_epoch(model, loss_function, external_loader, device, modality)
            final_source = source_result
            final_external = external_result

            select_result = external_result if external_result is not None else source_result
            select_score = select_result["weighted_f1"] if select_result is not None else -1.0
            if select_score > best_score:
                best_score = select_score
                best_epoch = epoch
                best_source = result_snapshot(source_result)
                best_external = result_snapshot(external_result)
                best_state = copy.deepcopy(model.state_dict())

        row = {
            "epoch": epoch,
            "time_sec": round(time.time() - start, 2),
            "train_loss": train_result["loss"] if train_result else None,
            "train_acc": train_result["accuracy"] if train_result else None,
            "train_macro_f1": train_result["macro_f1"] if train_result else None,
            "train_weighted_f1": train_result["weighted_f1"] if train_result else None,
            "valid_loss": valid_result["loss"] if valid_result else None,
            "valid_acc": valid_result["accuracy"] if valid_result else None,
            "valid_macro_f1": valid_result["macro_f1"] if valid_result else None,
            "valid_weighted_f1": valid_result["weighted_f1"] if valid_result else None,
            "source_test_loss": source_result["loss"] if source_result else None,
            "source_test_acc": source_result["accuracy"] if source_result else None,
            "source_test_macro_f1": source_result["macro_f1"] if source_result else None,
            "source_test_weighted_f1": source_result["weighted_f1"] if source_result else None,
            "external_test_loss": external_result["loss"] if external_result else None,
            "external_test_acc": external_result["accuracy"] if external_result else None,
            "external_test_macro_f1": external_result["macro_f1"] if external_result else None,
            "external_test_weighted_f1": external_result["weighted_f1"] if external_result else None,
        }
        epoch_rows.append(row)

        msg = (
            f"[MMGCN][{modality}] epoch={epoch}"
            f" train_wf1={(train_result or {}).get('weighted_f1', float('nan')):.4f}"
        )
        if valid_result:
            msg += f" valid_wf1={valid_result['weighted_f1']:.4f}"
        if source_result:
            msg += f" source_wf1={source_result['weighted_f1']:.4f}"
        if external_result:
            msg += f" external_wf1={external_result['weighted_f1']:.4f}"
        msg += f" time={row['time_sec']}s"
        print(msg)

    metrics_path = out_dir / f"mmgcn_unified_{modality}_epoch_metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(epoch_rows[0]))
        writer.writeheader()
        writer.writerows(epoch_rows)

    label_names = list(info["label_names"])
    common_extra = {
        "script": "MMGCN/train_eval_mmgcn_unified.py",
        "train_pkl": str(train_pkl),
        "external_test_pkl": str(external_test_pkl) if external_test_pkl else "",
        "modality": modality,
        "modals": modals,
        "graph_type": graph_type,
        "label_names": label_names,
        "dims": {
            "text": int(info["d_text"]),
            "audio": int(info["d_audio"]),
            "visual": int(info["d_visual"]),
            "n_speakers": int(info["n_speakers"]),
            "n_classes": int(info["n_classes"]),
        },
        "epochs": args.epochs,
        "best_epoch_by_external_or_source_weighted_f1": best_epoch,
        "epoch_metrics_csv": str(metrics_path),
    }
    final_source_summary = save_report(final_source, out_dir, "source_test_final", label_names, {**common_extra, "split": "source_test", "selection": "final"})
    final_external_summary = save_report(final_external, out_dir, "external_test_final", label_names, {**common_extra, "split": "external_test", "selection": "final"})
    best_source_summary = save_report(best_source, out_dir, "source_test_best", label_names, {**common_extra, "split": "source_test", "selection": "best"})
    best_external_summary = save_report(best_external, out_dir, "external_test_best", label_names, {**common_extra, "split": "external_test", "selection": "best"})

    if best_state is not None:
        torch.save(
            {
                "model_state_dict": best_state,
                "modality": modality,
                "modals": modals,
                "graph_type": graph_type,
                "info": info,
                "args": vars(args),
            },
            out_dir / "model_best.pt",
        )

    print(f"[MMGCN][{modality}] reports saved to {out_dir}")
    if final_external_summary:
        print(
            f"[MMGCN][{modality}] final external acc={final_external_summary['accuracy']:.4f} "
            f"macro_f1={final_external_summary['macro_f1']:.4f} "
            f"weighted_f1={final_external_summary['weighted_f1']:.4f}"
        )
    if best_external_summary:
        print(
            f"[MMGCN][{modality}] best_epoch={best_epoch} external acc={best_external_summary['accuracy']:.4f} "
            f"macro_f1={best_external_summary['macro_f1']:.4f} "
            f"weighted_f1={best_external_summary['weighted_f1']:.4f}"
        )
    elif final_source_summary:
        print(
            f"[MMGCN][{modality}] final source acc={final_source_summary['accuracy']:.4f} "
            f"macro_f1={final_source_summary['macro_f1']:.4f} "
            f"weighted_f1={final_source_summary['weighted_f1']:.4f}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Train/evaluate MMGCN on same-origin ANJS4 features.")
    parser.add_argument("--train_pkl", type=str, required=True)
    parser.add_argument("--external_test_pkl", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="./MMGCN/saved/unified_meld_to_ejsl")
    parser.add_argument("--modalities", nargs="+", default=["tv"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--valid_ratio", type=float, default=0.0)
    parser.add_argument("--max_train_dialogues", type=int, default=0)
    parser.add_argument("--max_source_test_dialogues", type=int, default=0)
    parser.add_argument("--max_external_test_dialogues", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--l2", type=float, default=0.00003)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--loss", choices=["nll", "focal"], default="focal")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--no_class_weight", action="store_true")
    parser.add_argument("--n_classes", type=int, default=0)
    parser.add_argument("--label_names", nargs="*", default=[])

    parser.add_argument("--base_model", default="LSTM")
    parser.add_argument("--graph_type", default="MMGCN")
    parser.add_argument("--graph_construct", default="direct")
    parser.add_argument("--mm_fusion_mthd", default="concat_subsequently")
    parser.add_argument("--modals", default="auto", help="auto uses vl for tv, avl for full, l/v for single-modal runs.")
    parser.add_argument("--deep_gcn_nlayers", type=int, default=4)
    parser.add_argument("--windowp", type=int, default=10)
    parser.add_argument("--windowf", type=int, default=10)
    parser.add_argument("--attention", default="general")
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--multiheads", type=int, default=6)
    parser.add_argument("--nodal_attention", action="store_true", default=True)
    parser.add_argument("--active_listener", action="store_true", default=False)
    parser.add_argument("--use_gcn", action="store_true", default=False)
    parser.add_argument("--use_topic", action="store_true", default=False)
    parser.add_argument("--use_speaker", action="store_true", default=True)
    parser.add_argument("--use_modal", action="store_true", default=False)
    parser.add_argument("--av_using_lstm", action="store_true", default=False)
    parser.add_argument("--no_residue", action="store_true", default=False)
    parser.add_argument("--d_g", type=int, default=150)
    parser.add_argument("--d_p", type=int, default=150)
    parser.add_argument("--d_e", type=int, default=100)
    parser.add_argument("--d_h", type=int, default=100)
    parser.add_argument("--d_a", type=int, default=100)
    parser.add_argument("--graph_hidden", type=int, default=100)
    parser.add_argument("--max_seq_len", type=int, default=300)
    parser.add_argument("--no_cuda", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_pkl = resolve_path(args.train_pkl, must_exist=True)
    external_test_pkl = resolve_path(args.external_test_pkl, must_exist=True) if args.external_test_pkl else None
    out_dir = resolve_path(args.out_dir, must_exist=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    info = inspect_feature_pkl(train_pkl)
    print(f"[MMGCN] train_pkl={train_pkl}")
    print(f"[MMGCN] external_test_pkl={external_test_pkl}")
    print(f"[MMGCN] out_dir={out_dir}")
    print(f"[MMGCN] device={device}")
    print(f"[MMGCN] feature_info={json.dumps(info, ensure_ascii=False, indent=2)}")

    for modality in parse_modalities(args.modalities):
        train_one_modality(args, modality, train_pkl, external_test_pkl, device, out_dir)


if __name__ == "__main__":
    main()
