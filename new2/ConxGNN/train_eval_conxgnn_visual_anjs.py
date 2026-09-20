#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(REPO_ROOT / "new"))

import src  # noqa: E402
from anjs_video_common import ANJS_LABELS, resolve_path  # noqa: E402
from src.loss.FocalLoss import FocalLoss  # noqa: E402
from src.TensorGraph import TensorGraph  # noqa: E402


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_payload(path: Path) -> Dict:
    obj = pickle.load(path.open("rb"), encoding="latin1")
    if not isinstance(obj, dict):
        raise ValueError(f"Expected ConxGNN visual dict: {path}")
    for split in ["train", "test"]:
        if split not in obj:
            raise ValueError(f"Missing split {split} in {path}")
    obj.setdefault("dev", [])
    obj.setdefault("meta", {})
    return obj


class ConxVisualDataset:
    def __init__(self, samples: Sequence[Dict], batch_size: int, text_dim: int, audio_dim: int, visual_dim: int):
        self.samples = list(samples)
        self.batch_size = int(batch_size)
        self.text_dim = int(text_dim)
        self.audio_dim = int(audio_dim)
        self.visual_dim = int(visual_dim)
        self.num_batches = int(math.ceil(len(self.samples) / max(self.batch_size, 1))) if self.samples else 0
        self.metadata = self.build_metadata()
        self.padded_batches = [self.padding(self.raw_batch(i)) for i in range(self.num_batches)]

    def build_metadata(self) -> Dict:
        int_classes = {idx: 0 for idx in range(len(ANJS_LABELS))}
        total = 0
        for sample in self.samples:
            for label in sample["labels"]:
                label = int(label)
                if label >= 0:
                    int_classes[label] += 1
                    total += 1
        emo_classes = {ANJS_LABELS[idx]: int_classes[idx] for idx in int_classes}
        return {
            "batch_size": self.batch_size,
            "num_batches": self.num_batches,
            "dataset": "iemocap_4",
            "embedding_dim": {"a": self.audio_dim, "t": self.text_dim, "v": self.visual_dim},
            "int_classes": int_classes,
            "emo_classes": emo_classes,
            "total_num_sentences": total,
            "speaker_to_idx": {"M": 0, "F": 1},
        }

    def __len__(self) -> int:
        return self.num_batches

    def __getitem__(self, index: int) -> Dict:
        return self.padded_batches[index]

    def raw_batch(self, index: int) -> List[Dict]:
        return self.samples[index * self.batch_size:(index + 1) * self.batch_size]

    def shuffle(self, seed: int) -> None:
        rng = random.Random(seed)
        rng.shuffle(self.samples)
        self.padded_batches = [self.padding(self.raw_batch(i)) for i in range(self.num_batches)]

    def padding(self, samples: Sequence[Dict]) -> Dict:
        batch_size = len(samples)
        lengths = [len(sample["labels"]) for sample in samples]
        max_len = max(lengths) if lengths else 0
        text_tensor = torch.zeros((batch_size, max_len, self.text_dim), dtype=torch.float32)
        audio_tensor = torch.zeros((batch_size, max_len, self.audio_dim), dtype=torch.float32)
        visual_tensor = torch.zeros((batch_size, max_len, self.visual_dim), dtype=torch.float32)
        speaker_tensor = torch.zeros((batch_size, max_len), dtype=torch.long)
        labels: List[int] = []
        utterance_texts: List[List[str]] = []
        item_ids: List[str] = []
        for i, sample in enumerate(samples):
            cur_len = lengths[i]
            text_tensor[i, :cur_len, :] = torch.tensor(np.asarray(sample["text"], dtype=np.float32))
            audio_tensor[i, :cur_len, :] = torch.tensor(np.asarray(sample["audio"], dtype=np.float32))
            visual_tensor[i, :cur_len, :] = torch.tensor(np.asarray(sample["visual"], dtype=np.float32))
            speaker_tensor[i, :cur_len] = torch.tensor(np.asarray(sample["speakers"], dtype=np.int64) % 2)
            labels.extend(int(x) for x in sample["labels"])
            utterance_texts.append(list(sample.get("sentence", [""] * cur_len)))
            sample_id = str(sample.get("id", f"sample_{i}"))
            for utt_idx in range(cur_len):
                item_ids.append(f"{sample_id}:utt{utt_idx + 1:02d}")
        return {
            "text_len_tensor": torch.tensor(lengths, dtype=torch.long),
            "text_tensor": text_tensor,
            "audio_tensor": audio_tensor,
            "visual_tensor": visual_tensor,
            "speakers_tensor": speaker_tensor,
            "label_tensor": torch.tensor(labels, dtype=torch.long),
            "utterance_texts": utterance_texts,
            "item_ids": item_ids,
        }


def infer_dims(payload: Dict) -> Dict[str, int]:
    samples = list(payload.get("train", [])) + list(payload.get("test", [])) + list(payload.get("dev", []))
    if not samples:
        raise ValueError("Feature pkl has no samples.")
    sample = samples[0]
    return {
        "d_text": int(np.asarray(sample["text"]).shape[-1]),
        "d_audio": int(np.asarray(sample["audio"]).shape[-1]),
        "d_visual": int(np.asarray(sample["visual"]).shape[-1]),
        "n_classes": len(ANJS_LABELS),
    }


def inspect_pkl(path: Path) -> Dict:
    payload = load_payload(path)
    dims = infer_dims(payload)
    valid_labels: List[int] = []
    for split in ["train", "dev", "test"]:
        for sample in payload.get(split, []):
            valid_labels.extend(int(x) for x in sample["labels"] if int(x) >= 0)
    if not valid_labels:
        raise RuntimeError(f"No valid labels in {path}")
    return {
        "path": str(path),
        **dims,
        "label_names": ANJS_LABELS,
        "train_dialogues": len(payload.get("train", [])),
        "dev_dialogues": len(payload.get("dev", [])),
        "test_dialogues": len(payload.get("test", [])),
        "valid_label_count": len(valid_labels),
        "valid_label_counts": np.bincount(np.asarray(valid_labels, dtype=np.int64), minlength=len(ANJS_LABELS)).astype(int).tolist(),
        "meta": payload.get("meta", {}),
    }


def collect_train_labels(payload: Dict) -> List[int]:
    labels: List[int] = []
    for sample in payload.get("train", []):
        labels.extend(int(x) for x in sample["labels"] if int(x) >= 0)
    return labels


def make_class_weights(labels: Sequence[int], n_classes: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=n_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (float(n_classes) * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_config(args, info: Dict, trainset: ConxVisualDataset, devset: ConxVisualDataset, source_set: ConxVisualDataset):
    config = src.utils.load_yaml(args.config_yaml)
    config["dataset"] = "iemocap_4"
    config["device"] = args.device
    config["seed"] = args.seed
    config["epochs"] = args.epochs
    config["batch_size"] = args.batch_size
    config["learning_rate"] = args.lr
    config["weight_decay"] = args.weight_decay
    config["drop_rate"] = args.drop_rate
    config["hidden_size"] = args.hidden_size
    config["inter_size"] = args.inter_size
    config["modalities"] = "atv"
    config["class_weight"] = False
    config["classi_loss"] = "CE"
    config["wandb"] = False
    config["backup"] = None
    config["from_begin"] = True
    config["pretrain_unimodal_epochs"] = 0
    config["use_contrastive_unimodal"] = False
    config["use_crossmodal"] = False
    config["use_modal_weighting"] = False
    config["use_hgr_loss"] = False
    config["dataset_embedding_dims"] = {
        "iemocap": {"a": 100, "t": 768, "v": 512},
        "iemocap_4": {"a": int(info["d_audio"]), "t": int(info["d_text"]), "v": int(info["d_visual"])},
        "meld_m3net": {"a": 300, "t": 600, "t1": 1024, "t2": 1024, "t3": 1024, "t4": 1024, "v": 342},
        "mosei": {"a": 80, "t": 768, "v": 35},
    }
    config["trainset_metadata"] = trainset.metadata
    config["devset_metadata"] = devset.metadata
    config["testset_metadata"] = source_set.metadata
    config["training_status"] = {"cur_epoch": 1, "cur_iter": 1}
    return config


def move_batch(data: Dict, device: torch.device) -> Dict:
    moved = {}
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def conx_forward_logits(model: nn.Module, data: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_padded_tensor_data = copy.deepcopy(data)
    base_tensor_graph = TensorGraph(batch_padded_tensor_data, model.modalities)
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
    fc_outputs = torch.relu(graph_out.data)
    logits = clf.mlp_layer(fc_outputs)
    return logits, fc_outputs


def run_epoch(
    model: nn.Module,
    cls_loss: nn.Module,
    dataset: Optional[ConxVisualDataset],
    device: torch.device,
    ce_loss_param: float,
    cb_loss_param: float,
    optimizer: Optional[torch.optim.Optimizer] = None,
    max_grad_norm: float = 0.0,
) -> Optional[Dict]:
    if dataset is None or len(dataset) == 0:
        return None
    train = optimizer is not None
    model.train() if train else model.eval()
    losses: List[float] = []
    loss_weights: List[int] = []
    golds_all: List[int] = []
    preds_all: List[int] = []
    probs_all: List[np.ndarray] = []
    ids_all: List[str] = []
    for idx in range(len(dataset)):
        if train:
            optimizer.zero_grad()
        raw_data = dataset[idx]
        data = move_batch(raw_data, device)
        labels = data["label_tensor"]
        valid = labels.ge(0)
        valid_count = int(valid.sum().item())
        if valid_count == 0:
            continue
        with torch.set_grad_enabled(train):
            logits, fc_outputs = conx_forward_logits(model, data)
            loss = ce_loss_param * cls_loss(logits[valid], labels[valid])
            if cb_loss_param > 0 and valid_count > 1:
                loss = loss + cb_loss_param * model.clf.CBContrastiveLoss(fc_outputs[valid], labels[valid])
            if train:
                loss.backward()
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
        with torch.no_grad():
            prob = torch.softmax(logits[valid], dim=-1)
            pred = torch.argmax(logits[valid], dim=-1)
            losses.append(float(loss.item()) * valid_count)
            loss_weights.append(valid_count)
            golds_all.extend(labels[valid].detach().cpu().numpy().astype(int).tolist())
            preds_all.extend(pred.detach().cpu().numpy().astype(int).tolist())
            probs_all.append(prob.detach().cpu().numpy().astype(np.float32))
            item_ids = raw_data["item_ids"]
            valid_mask = valid.detach().cpu().numpy().astype(bool).tolist()
            ids_all.extend([item_id for item_id, keep in zip(item_ids, valid_mask) if keep])
    if not golds_all:
        return None
    golds = np.asarray(golds_all, dtype=np.int64)
    preds = np.asarray(preds_all, dtype=np.int64)
    probs = np.concatenate(probs_all, axis=0)
    label_ids = list(range(len(ANJS_LABELS)))
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
    return {key: value.copy() if isinstance(value, np.ndarray) else list(value) if isinstance(value, list) else value for key, value in result.items()}


def write_confusion_matrix(cm: np.ndarray, labels: Sequence[str], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("labels: " + ",".join(labels) + "\n")
        for row in cm:
            f.write(" ".join(str(int(x)) for x in row) + "\n")


def save_report(result: Optional[Dict], out_dir: Path, prefix: str, extra: Dict) -> Optional[Dict]:
    if result is None:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    label_ids = list(range(len(ANJS_LABELS)))
    golds = result["golds"]
    preds = result["preds"]
    probs = result["probs"]
    cm = metrics.confusion_matrix(golds, preds, labels=label_ids)
    report = metrics.classification_report(golds, preds, labels=label_ids, target_names=ANJS_LABELS, digits=4, zero_division=0)
    (out_dir / f"{prefix}_classification_report.txt").write_text(report, encoding="utf-8")
    np.save(out_dir / f"{prefix}_confusion_matrix.npy", cm)
    write_confusion_matrix(cm, ANJS_LABELS, out_dir / f"{prefix}_confusion_matrix.txt")
    with (out_dir / f"{prefix}_predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["item_id", "gold_id", "gold_label", "pred_id", "pred_label", "correct"] + [f"prob_{name}" for name in ANJS_LABELS])
        for item_id, gold, pred, prob in zip(result["ids"], golds, preds, probs):
            writer.writerow([item_id, int(gold), ANJS_LABELS[int(gold)], int(pred), ANJS_LABELS[int(pred)], int(gold == pred)] + [float(x) for x in prob])
    per_p, per_r, per_f1, per_support = metrics.precision_recall_fscore_support(golds, preds, labels=label_ids, zero_division=0)
    summary = {
        **extra,
        "n_samples": int(len(golds)),
        "loss": float(result["loss"]),
        "accuracy": float(result["accuracy"]),
        "macro_f1": float(result["macro_f1"]),
        "weighted_f1": float(result["weighted_f1"]),
        "gold_counts": dict(zip(ANJS_LABELS, np.bincount(golds, minlength=len(ANJS_LABELS)).astype(int).tolist())),
        "pred_counts": dict(zip(ANJS_LABELS, np.bincount(preds, minlength=len(ANJS_LABELS)).astype(int).tolist())),
        "per_class": {
            ANJS_LABELS[i]: {
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


def select_result(args, valid_result: Optional[Dict], source_result: Optional[Dict], external_result: Optional[Dict]) -> Tuple[Optional[Dict], str]:
    candidates = {"valid": valid_result, "source": source_result, "external": external_result}
    selected = candidates.get(args.selection_split)
    if selected is not None:
        return selected, args.selection_split
    for key in ["source", "valid", "external"]:
        if candidates[key] is not None:
            return candidates[key], key
    return None, "none"


def save_checkpoint(path: Path, model: nn.Module, args, info: Dict, epoch: int, best_epoch: int, best_score: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "baseline": "ConxGNN",
            "modality": "visual",
            "label_names": ANJS_LABELS,
            "info": info,
            "args": vars(args),
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_score": best_score,
        },
        path,
    )


def train(args) -> None:
    seed_everything(args.seed)
    train_pkl = resolve_path(args.train_pkl, must_exist=True)
    external_pkl = resolve_path(args.external_test_pkl, must_exist=True) if args.external_test_pkl else None
    out_dir = resolve_path(args.out_dir, must_exist=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and not args.no_cuda else "cpu")
    if device.type != "cuda":
        raise RuntimeError("ConxGNN graph code is expected to run on GPU for these experiments.")
    args.device = str(device)

    source_payload = load_payload(train_pkl)
    external_payload = load_payload(external_pkl) if external_pkl is not None else None
    info = inspect_pkl(train_pkl)
    if external_pkl is not None:
        external_info = inspect_pkl(external_pkl)
        for key in ["d_text", "d_audio", "d_visual", "n_classes"]:
            if int(info[key]) != int(external_info[key]):
                raise RuntimeError(f"Source/target feature mismatch for {key}: source={info[key]} target={external_info[key]}")

    trainset = ConxVisualDataset(source_payload.get("train", []), args.batch_size, info["d_text"], info["d_audio"], info["d_visual"])
    devset = ConxVisualDataset(source_payload.get("dev", []), args.batch_size, info["d_text"], info["d_audio"], info["d_visual"])
    source_set = ConxVisualDataset(source_payload.get("test", []), args.batch_size, info["d_text"], info["d_audio"], info["d_visual"])
    external_set = None
    if external_payload is not None:
        external_set = ConxVisualDataset(external_payload.get("test", []), args.batch_size, info["d_text"], info["d_audio"], info["d_visual"])

    config = build_config(args, info, trainset, devset, source_set)
    print(f"[CONXGNN] train_pkl={train_pkl}")
    print(f"[CONXGNN] external_test_pkl={external_pkl}")
    print(f"[CONXGNN] out_dir={out_dir}")
    print(f"[CONXGNN] device={device}")
    print(f"[CONXGNN] feature_info={json.dumps(info, ensure_ascii=False, indent=2)}")
    print(f"[CONXGNN] model_dataset={config.dataset} modalities={config.modalities}")

    model = src.MainModel(config).to(device)
    train_labels = collect_train_labels(source_payload)
    class_weights = make_class_weights(train_labels, len(ANJS_LABELS), device) if args.class_weight and args.loss == "ce" else None
    if class_weights is not None:
        print(f"[CONXGNN] class_weights={class_weights.detach().cpu().numpy().round(4).tolist()}")
    cls_loss = FocalLoss(gamma=args.focal_gamma) if args.loss == "focal" else nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    epoch_rows: List[Dict] = []
    best_state = None
    best_epoch = 0
    best_score = -1.0
    best_selection_split = "none"
    best_source = None
    best_external = None
    final_source = None
    final_external = None

    for epoch in range(1, args.epochs + 1):
        config.training_status["cur_epoch"] = epoch
        trainset.shuffle(args.seed + epoch)
        start = time.time()
        train_result = run_epoch(model, cls_loss, trainset, device, args.ce_loss_param, args.cb_loss_param, optimizer=optimizer, max_grad_norm=args.max_grad_norm)
        valid_result = run_epoch(model, cls_loss, devset, device, args.ce_loss_param, args.cb_loss_param)
        source_result = run_epoch(model, cls_loss, source_set, device, args.ce_loss_param, args.cb_loss_param)
        external_result = run_epoch(model, cls_loss, external_set, device, args.ce_loss_param, args.cb_loss_param)
        final_source = source_result
        final_external = external_result
        selected, selection_split = select_result(args, valid_result, source_result, external_result)
        selected_score = float(selected[args.selection_metric]) if selected is not None else -1.0
        if selected_score > best_score:
            best_score = selected_score
            best_epoch = epoch
            best_selection_split = selection_split
            best_source = result_snapshot(source_result)
            best_external = result_snapshot(external_result)
            best_state = copy.deepcopy(model.state_dict())
            print(f"[CONXGNN][visual] new best epoch={epoch} split={selection_split} {args.selection_metric}={best_score:.4f}")
        if args.save_epoch_every > 0 and (epoch % args.save_epoch_every == 0 or epoch == args.epochs):
            save_checkpoint(out_dir / "checkpoints" / f"epoch_{epoch:03d}.pt", model, args, info, epoch, best_epoch, best_score)
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
        print(
            f"[CONXGNN][visual] epoch={epoch} "
            f"train_wf1={(train_result or {}).get('weighted_f1', float('nan')):.4f} "
            f"source_wf1={(source_result or {}).get('weighted_f1', float('nan')):.4f} "
            f"external_wf1={(external_result or {}).get('weighted_f1', float('nan')):.4f} "
            f"time={row['time_sec']}s"
        )

    metrics_path = out_dir / "conxgnn_visual_epoch_metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(epoch_rows[0]))
        writer.writeheader()
        writer.writerows(epoch_rows)

    common_extra = {
        "script": "new2/ConxGNN/train_eval_conxgnn_visual_anjs.py",
        "baseline": "ConxGNN",
        "modality": "visual",
        "train_pkl": str(train_pkl),
        "external_test_pkl": str(external_pkl) if external_pkl else "",
        "label_names": ANJS_LABELS,
        "dims": {
            "text": int(info["d_text"]),
            "audio": int(info["d_audio"]),
            "visual": int(info["d_visual"]),
            "n_classes": int(info["n_classes"]),
        },
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "dropout": args.drop_rate,
        "loss": args.loss,
        "focal_gamma": args.focal_gamma,
        "class_weight": bool(args.class_weight),
        "ce_loss_param": args.ce_loss_param,
        "cb_loss_param": args.cb_loss_param,
        "selection_split": args.selection_split,
        "selection_metric": args.selection_metric,
        "best_selection_split": best_selection_split,
        "best_epoch_by_selection_metric": best_epoch,
        "best_selection_score": best_score,
        "epoch_metrics_csv": str(metrics_path),
    }
    save_report(final_source, out_dir, "source_test_final", {**common_extra, "split": "source_test", "selection": "final"})
    save_report(final_external, out_dir, "external_test_final", {**common_extra, "split": "external_test", "selection": "final"})
    save_report(best_source, out_dir, "source_test_best", {**common_extra, "split": "source_test", "selection": "best"})
    best_external_summary = save_report(best_external, out_dir, "external_test_best", {**common_extra, "split": "external_test", "selection": "best"})
    if best_state is not None:
        model.load_state_dict(best_state)
        save_checkpoint(out_dir / "model_best.pt", model, args, info, best_epoch, best_epoch, best_score)
    if best_external_summary:
        print(
            f"[CONXGNN][visual] best_epoch={best_epoch} external acc={best_external_summary['accuracy']:.4f} "
            f"macro_f1={best_external_summary['macro_f1']:.4f} weighted_f1={best_external_summary['weighted_f1']:.4f}"
        )
    print(f"[CONXGNN][visual] reports saved to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate ConxGNN on internally aligned ANJS4 visual-only features.")
    parser.add_argument("--train_pkl", type=str, required=True)
    parser.add_argument("--external_test_pkl", type=str, default="")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--config_yaml", type=str, default=str(THIS_DIR / "configs" / "meld.yaml"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-8)
    parser.add_argument("--drop_rate", type=float, default=0.8)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--inter_size", type=int, default=256)
    parser.add_argument("--loss", choices=["focal", "ce"], default="ce")
    parser.add_argument("--focal_gamma", type=float, default=2.5)
    parser.add_argument("--class_weight", action="store_true")
    parser.add_argument("--ce_loss_param", type=float, default=1.0)
    parser.add_argument("--cb_loss_param", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=0.0)
    parser.add_argument("--selection_split", choices=["valid", "source", "external"], default="source")
    parser.add_argument("--selection_metric", choices=["weighted_f1", "macro_f1", "accuracy"], default="weighted_f1")
    parser.add_argument("--save_epoch_every", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_cuda", action="store_true")
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
