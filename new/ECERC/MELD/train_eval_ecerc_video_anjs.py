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
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

THIS_DIR = Path(__file__).resolve().parent
NEW_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(NEW_ROOT))

from anjs_video_common import ANJS_LABELS, resolve_path  # noqa: E402
from loss import Loss  # noqa: E402
from model import ECERC  # noqa: E402


ECERC_FIELDS = [
    "speakers",
    "emotion_labels",
    "sentiment_labels",
    "eroberta1",
    "eroberta2",
    "eroberta3",
    "eroberta4",
    "sroberta1",
    "sroberta2",
    "sroberta3",
    "sroberta4",
    "video_audio",
    "video_visual",
    "sentences",
    "train_ids",
    "test_ids",
    "valid_ids",
]


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
    if not isinstance(obj, (tuple, list)) or len(obj) != len(ECERC_FIELDS):
        raise ValueError(f"Expected ECERC feature tuple with {len(ECERC_FIELDS)} fields: {path}")
    return dict(zip(ECERC_FIELDS, obj))


class ECERCFeatureDataset(Dataset):
    def __init__(self, path: str | Path, split: str):
        self.path = resolve_path(path, must_exist=True)
        self.payload = load_payload(self.path)
        if split == "train":
            self.keys = list(self.payload["train_ids"])
        elif split == "test":
            self.keys = list(self.payload["test_ids"])
        elif split == "valid":
            self.keys = list(self.payload["valid_ids"])
        else:
            raise ValueError(f"Unsupported split: {split}")

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int):
        vid = self.keys[index]
        return (
            torch.FloatTensor(np.asarray(self.payload["eroberta1"][vid], dtype=np.float32)),
            torch.FloatTensor(np.asarray(self.payload["sroberta1"][vid], dtype=np.float32)),
            torch.FloatTensor(np.asarray(self.payload["video_visual"][vid], dtype=np.float32)),
            torch.FloatTensor(np.asarray(self.payload["video_audio"][vid], dtype=np.float32)),
            torch.FloatTensor(np.asarray(self.payload["speakers"][vid], dtype=np.float32)),
            torch.FloatTensor([1] * len(self.payload["emotion_labels"][vid])),
            torch.LongTensor(np.asarray(self.payload["emotion_labels"][vid], dtype=np.int64)),
            vid,
        )

    @staticmethod
    def collate_fn(data):
        columns = list(zip(*data))
        return [
            pad_sequence(list(columns[0])),
            pad_sequence(list(columns[1])),
            pad_sequence(list(columns[2])),
            pad_sequence(list(columns[3])),
            pad_sequence(list(columns[4])),
            pad_sequence(list(columns[5]), batch_first=True),
            pad_sequence(list(columns[6]), batch_first=True),
            list(columns[7]),
        ]


def inspect_pkl(path: Path) -> Dict:
    payload = load_payload(path)
    keys = list(payload["train_ids"]) + list(payload["test_ids"]) + list(payload["valid_ids"])
    if not keys:
        raise RuntimeError(f"No dialogue keys in {path}")
    first = keys[0]
    valid_labels: List[int] = []
    for key in keys:
        valid_labels.extend(int(x) for x in payload["emotion_labels"][key] if int(x) >= 0)
    if not valid_labels:
        raise RuntimeError(f"No valid labels in {path}")
    return {
        "path": str(path),
        "d_text": int(np.asarray(payload["eroberta1"][first]).shape[-1]),
        "d_audio": int(np.asarray(payload["video_audio"][first]).shape[-1]),
        "d_visual": int(np.asarray(payload["video_visual"][first]).shape[-1]),
        "n_speakers": int(np.asarray(payload["speakers"][first]).shape[-1]),
        "n_classes": len(ANJS_LABELS),
        "label_names": ANJS_LABELS,
        "train_dialogues": len(payload["train_ids"]),
        "test_dialogues": len(payload["test_ids"]),
        "valid_dialogues": len(payload["valid_ids"]),
        "valid_label_count": len(valid_labels),
        "valid_label_counts": np.bincount(np.asarray(valid_labels, dtype=np.int64), minlength=len(ANJS_LABELS)).astype(int).tolist(),
    }


def collect_train_labels(path: Path) -> List[int]:
    payload = load_payload(path)
    labels: List[int] = []
    for key in payload["train_ids"]:
        labels.extend(int(x) for x in payload["emotion_labels"][key] if int(x) >= 0)
    return labels


def make_class_weights(labels: Sequence[int], n_classes: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=n_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (float(n_classes) * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_loaders(args, train_pkl: Path, external_test_pkl: Optional[Path]):
    trainset = ECERCFeatureDataset(train_pkl, "train")
    source_testset = ECERCFeatureDataset(train_pkl, "test")
    validset = ECERCFeatureDataset(train_pkl, "valid")

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=ECERCFeatureDataset.collate_fn,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available() and not args.no_cuda,
        generator=generator,
    )
    valid_loader = None
    if len(validset) > 0:
        valid_loader = DataLoader(
            validset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=ECERCFeatureDataset.collate_fn,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available() and not args.no_cuda,
        )
    source_loader = DataLoader(
        source_testset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=ECERCFeatureDataset.collate_fn,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available() and not args.no_cuda,
    )

    external_loader = None
    if external_test_pkl is not None:
        external_testset = ECERCFeatureDataset(external_test_pkl, "test")
        external_loader = DataLoader(
            external_testset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=ECERCFeatureDataset.collate_fn,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available() and not args.no_cuda,
        )
    return train_loader, valid_loader, source_loader, external_loader


def build_model(args, info: Dict, device: torch.device) -> ECERC:
    if device.type != "cuda":
        raise RuntimeError("ECERC/MELD/model.py assumes CUDA masks internally. Run this script on a GPU or patch ECERC forward for CPU.")
    input_size = int(info["d_text"]) + int(info["d_audio"]) + int(info["d_visual"])
    model = ECERC(
        args,
        d_t=int(info["d_text"]),
        d_a=int(info["d_audio"]),
        d_v=int(info["d_visual"]),
        base_layer=args.base_layer,
        input_size=input_size,
        hidden_size=args.hidden_size,
        n_speakers=int(info["n_speakers"]),
        n_classes=int(info["n_classes"]),
        cuda_flag=True,
    )
    return model.to(device)


def sequence_lengths(umask: torch.Tensor) -> List[int]:
    lengths: List[int] = []
    for row in umask:
        idx = (row == 1).nonzero(as_tuple=False)
        lengths.append(0 if idx.numel() == 0 else int(idx[-1].item()) + 1)
    return lengths


def prepare_video_only_inputs(emo_text, sem_text, visual, audio):
    return torch.zeros_like(emo_text), torch.zeros_like(sem_text), visual, torch.zeros_like(audio)


def run_epoch(
    model: ECERC,
    loss_function: nn.Module,
    dataloader: Optional[DataLoader],
    device: torch.device,
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
        emo_text, sem_text, visual, audio, qmask, umask, label = [d.to(device) for d in data[:-1]]
        emo_text, sem_text, visual, audio = prepare_video_only_inputs(emo_text, sem_text, visual, audio)
        lengths = sequence_lengths(umask)
        u_e = torch.cat([emo_text, audio, visual], dim=-1)

        with torch.set_grad_enabled(train):
            log_prob = model(u_e, sem_text, qmask, umask, lengths)
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

            label_matrix = label.detach().cpu()
            vids = data[-1]
            for batch_idx, vid in enumerate(vids):
                for utt_idx in range(lengths[batch_idx]):
                    if int(label_matrix[batch_idx, utt_idx]) >= 0:
                        ids_all.append(f"{vid}:utt{utt_idx + 1:02d}")

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
    return {
        key: value.copy() if isinstance(value, np.ndarray) else list(value) if isinstance(value, list) else value
        for key, value in result.items()
    }


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
    if args.selection_split == "auto":
        for key in ["source", "valid", "external"]:
            if candidates[key] is not None:
                return candidates[key], key
        return None, "none"
    selected = candidates.get(args.selection_split)
    if selected is not None:
        return selected, args.selection_split
    for key in ["source", "valid", "external"]:
        if candidates[key] is not None:
            return candidates[key], key
    return None, "none"


def save_checkpoint(path: Path, model: ECERC, args, info: Dict, epoch: int, best_epoch: int, best_score: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "baseline": "ECERC",
            "modality": "video",
            "label_names": ANJS_LABELS,
            "info": info,
            "args": vars(args),
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_score": best_score,
        },
        path,
    )


def metric_value(result: Dict, metric: str) -> float:
    return float(result[metric])


def train(args) -> None:
    seed_everything(args.seed)
    train_pkl = resolve_path(args.train_pkl, must_exist=True)
    external_pkl = resolve_path(args.external_test_pkl, must_exist=True) if args.external_test_pkl else None
    out_dir = resolve_path(args.out_dir, must_exist=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    info = inspect_pkl(train_pkl)
    if external_pkl is not None:
        external_info = inspect_pkl(external_pkl)
        for key in ["d_text", "d_audio", "d_visual", "n_speakers", "n_classes"]:
            if int(info[key]) != int(external_info[key]):
                raise RuntimeError(f"Source/target feature mismatch for {key}: source={info[key]} target={external_info[key]}")

    print(f"[ECERC] train_pkl={train_pkl}")
    print(f"[ECERC] external_test_pkl={external_pkl}")
    print(f"[ECERC] out_dir={out_dir}")
    print(f"[ECERC] device={device}")
    print(f"[ECERC] feature_info={json.dumps(info, ensure_ascii=False, indent=2)}")

    train_loader, valid_loader, source_loader, external_loader = build_loaders(args, train_pkl, external_pkl)
    model = build_model(args, info, device)
    train_labels = collect_train_labels(train_pkl)
    class_weights = make_class_weights(train_labels, int(info["n_classes"]), device) if args.class_weight else None
    if class_weights is not None:
        print(f"[ECERC] class_weights={class_weights.detach().cpu().numpy().round(4).tolist()}")
    loss_function = Loss(gamma=args.loss_gamma, alpha=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

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
        start = time.time()
        train_result = run_epoch(model, loss_function, train_loader, device, optimizer=optimizer, max_grad_norm=args.max_grad_norm)
        valid_result = run_epoch(model, loss_function, valid_loader, device)
        source_result = run_epoch(model, loss_function, source_loader, device)
        external_result = run_epoch(model, loss_function, external_loader, device)
        final_source = source_result
        final_external = external_result

        selected, selection_split = select_result(args, valid_result, source_result, external_result)
        selected_score = metric_value(selected, args.selection_metric) if selected is not None else -1.0
        if selected_score > best_score:
            best_score = selected_score
            best_epoch = epoch
            best_selection_split = selection_split
            best_source = result_snapshot(source_result)
            best_external = result_snapshot(external_result)
            best_state = copy.deepcopy(model.state_dict())
            print(f"[ECERC][video] new best epoch={epoch} split={selection_split} {args.selection_metric}={best_score:.4f}")

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
            f"[ECERC][video] epoch={epoch} "
            f"train_wf1={(train_result or {}).get('weighted_f1', float('nan')):.4f} "
            f"source_wf1={(source_result or {}).get('weighted_f1', float('nan')):.4f} "
            f"external_wf1={(external_result or {}).get('weighted_f1', float('nan')):.4f} "
            f"time={row['time_sec']}s"
        )

    metrics_path = out_dir / "ecerc_video_epoch_metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(epoch_rows[0]))
        writer.writeheader()
        writer.writerows(epoch_rows)

    common_extra = {
        "script": "new/ECERC/MELD/train_eval_ecerc_video_anjs.py",
        "baseline": "ECERC",
        "modality": "video",
        "train_pkl": str(train_pkl),
        "external_test_pkl": str(external_pkl) if external_pkl else "",
        "label_names": ANJS_LABELS,
        "dims": {
            "text": int(info["d_text"]),
            "audio": int(info["d_audio"]),
            "visual": int(info["d_visual"]),
            "n_speakers": int(info["n_speakers"]),
            "n_classes": int(info["n_classes"]),
        },
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "lr": args.lr,
        "l2": args.l2,
        "loss_gamma": args.loss_gamma,
        "class_weight": bool(args.class_weight),
        "max_grad_norm": args.max_grad_norm,
        "base_layer": args.base_layer,
        "hidden_size": args.hidden_size,
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
            f"[ECERC][video] best_epoch={best_epoch} external acc={best_external_summary['accuracy']:.4f} "
            f"macro_f1={best_external_summary['macro_f1']:.4f} weighted_f1={best_external_summary['weighted_f1']:.4f}"
        )
    print(f"[ECERC][video] reports saved to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate ECERC on internally aligned ANJS4 video-only features.")
    parser.add_argument("--train_pkl", type=str, required=True)
    parser.add_argument("--external_test_pkl", type=str, default="")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--l2", type=float, default=0.0002)
    parser.add_argument("--loss_gamma", type=float, default=1.0)
    parser.add_argument("--class_weight", action="store_true")
    parser.add_argument("--max_grad_norm", type=float, default=0.0)
    parser.add_argument("--base_layer", type=int, default=1)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--selection_split", choices=["auto", "valid", "source", "external"], default="source")
    parser.add_argument("--selection_metric", choices=["weighted_f1", "macro_f1", "accuracy"], default="weighted_f1")
    parser.add_argument("--save_epoch_every", type=int, default=0)
    parser.add_argument("--no_cuda", action="store_true")
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()

