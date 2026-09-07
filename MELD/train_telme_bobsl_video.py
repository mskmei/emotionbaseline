#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import Student_Video
from train_telme_meld_modalities import partially_load_state_dict
from utils import video_processor


LABELS = ["A", "N", "J", "S"]
EMOTION_TO_ID = {
    "anger": 0,
    "neutral": 1,
    "joy": 2,
    "sadness": 3,
}


@dataclass
class BobslSample:
    sample_id: str
    label: int
    video_path: Optional[Path]
    frame_dir: Optional[Path]
    text: str


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_path(raw: str, must_exist: bool = False) -> Path:
    path = Path(raw).expanduser()
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, Path(__file__).resolve().parent.parent / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    if must_exist:
        raise FileNotFoundError(f"Path not found: {raw}. Checked: {', '.join(str(x) for x in candidates)}")
    return candidates[0].resolve()


def resolve_csv(raw: str, bobsl_root: Path, default_name: str) -> Path:
    if raw:
        return resolve_path(raw, must_exist=True)
    path = bobsl_root / default_name
    if not path.exists():
        raise FileNotFoundError(f"Missing BOBSL split CSV: {path}")
    return path.resolve()


def normalize_text(text: str) -> str:
    return " ".join(str(text or "").replace("\t", " ").split())


def resolve_video_path(bobsl_root: Path, video_subdir: str, stem: str, clip_name: str) -> Optional[Path]:
    candidates = [
        bobsl_root / video_subdir / stem / clip_name,
        bobsl_root / "video" / stem / clip_name,
        bobsl_root / stem / clip_name,
        bobsl_root / clip_name,
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    return candidates[0]


def resolve_frame_dir(bobsl_root: Path, frame_subdir: str, stem: str) -> Optional[Path]:
    path = bobsl_root / frame_subdir / stem
    return path.resolve() if path.exists() else path


def read_split(
    csv_path: Path,
    split_name: str,
    bobsl_root: Path,
    video_subdir: str,
    frame_subdir: str,
    min_score: float,
    limit: int,
) -> List[BobslSample]:
    samples: List[BobslSample] = []
    skipped_emotion = 0
    skipped_score = 0
    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            emotion = str(row.get("emotion", "")).strip().lower()
            if emotion not in EMOTION_TO_ID:
                skipped_emotion += 1
                continue
            try:
                score = float(row.get("score", 0.0) or 0.0)
            except ValueError:
                score = 0.0
            if score < min_score:
                skipped_score += 1
                continue
            stem = str(row.get("stem", "")).strip()
            clip_name = str(row.get("clip_name", "")).strip()
            if not stem or not clip_name:
                skipped_emotion += 1
                continue
            samples.append(
                BobslSample(
                    sample_id=f"{split_name}:{stem}/{clip_name}",
                    label=EMOTION_TO_ID[emotion],
                    video_path=resolve_video_path(bobsl_root, video_subdir, stem, clip_name),
                    frame_dir=resolve_frame_dir(bobsl_root, frame_subdir, stem),
                    text=normalize_text(str(row.get("text", ""))),
                )
            )
            if limit > 0 and len(samples) >= limit:
                break

    counts = np.bincount(np.asarray([x.label for x in samples], dtype=np.int64), minlength=len(LABELS)).tolist() if samples else [0] * len(LABELS)
    print(
        f"[TELME-BOBSL] split={split_name} samples={len(samples)} "
        f"labels={dict(zip(LABELS, counts))} skipped_emotion={skipped_emotion} skipped_score={skipped_score}"
    )
    return samples


def sample_frame_paths(frame_paths: Sequence[Path]) -> List[Path]:
    paths = list(frame_paths)
    if not paths:
        return []
    if len(paths) >= 8:
        idx = np.linspace(0, len(paths) - 1, num=8, dtype=int)
        return [paths[int(i)] for i in idx]
    return paths + [paths[-1]] * (8 - len(paths))


def read_frames_from_mp4(path: Path, max_seconds: float) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if frame_count <= 0:
        cap.release()
        return []
    if max_seconds > 0 and fps > 0 and frame_count / fps > max_seconds:
        cap.release()
        return []

    indices = np.linspace(0, frame_count - 1, num=8, dtype=int)
    frames: List[np.ndarray] = []
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, image = cap.read()
        if ok and image is not None:
            frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    cap.release()
    if frames and len(frames) < 8:
        frames.extend([frames[-1].copy() for _ in range(8 - len(frames))])
    return frames[:8]


def read_frames_from_prefix(frame_dir: Optional[Path], sample_id: str) -> List[np.ndarray]:
    if frame_dir is None or not frame_dir.exists():
        return []
    clip_name = sample_id.split("/", 1)[-1]
    clip_stem = Path(clip_name).stem
    frame_paths = sorted(frame_dir.glob(f"{clip_stem}_*.jpg"))
    if not frame_paths:
        frame_paths = sorted(frame_dir.glob(f"{clip_stem}_*.png"))

    frames: List[np.ndarray] = []
    for path in sample_frame_paths(frame_paths):
        image = cv2.imread(str(path))
        if image is not None:
            frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return frames


def get_video_or_zeros(sample: BobslSample, max_seconds: float) -> torch.Tensor:
    frames: List[np.ndarray] = []
    if sample.video_path is not None and sample.video_path.exists():
        frames = read_frames_from_mp4(sample.video_path, max_seconds)
    if not frames:
        frames = read_frames_from_prefix(sample.frame_dir, sample.sample_id)
    if not frames:
        return torch.zeros(8, 3, 224, 224)
    inputs = video_processor(frames, return_tensors="pt")
    return inputs["pixel_values"][0]


class BobslVideoDataset(Dataset):
    def __init__(self, samples: List[BobslSample], max_seconds: float):
        self.samples = samples
        self.max_seconds = max_seconds

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        return get_video_or_zeros(sample, self.max_seconds), int(sample.label), sample.sample_id


def collate_video(batch):
    videos, labels, sample_ids = zip(*batch)
    return torch.stack(list(videos)), torch.tensor(labels, dtype=torch.long), list(sample_ids)


def evaluate(model: Student_Video, loader: DataLoader, device: torch.device):
    model.eval()
    losses: List[float] = []
    golds: List[int] = []
    preds: List[int] = []
    ids: List[str] = []
    with torch.no_grad():
        for video, labels, sample_ids in loader:
            video = video.to(device)
            labels = labels.to(device)
            _hidden, logits = model(video)
            loss = nn.CrossEntropyLoss()(logits, labels)
            pred = logits.argmax(dim=1)
            losses.append(float(loss.item()) * int(labels.numel()))
            golds.extend(labels.detach().cpu().numpy().astype(int).tolist())
            preds.extend(pred.detach().cpu().numpy().astype(int).tolist())
            ids.extend(sample_ids)
    if not golds:
        return None
    return {
        "loss": float(np.sum(losses) / max(len(golds), 1)),
        "accuracy": float(metrics.accuracy_score(golds, preds)),
        "macro_f1": float(metrics.f1_score(golds, preds, average="macro", zero_division=0)),
        "weighted_f1": float(metrics.f1_score(golds, preds, average="weighted", zero_division=0)),
        "golds": np.asarray(golds, dtype=np.int64),
        "preds": np.asarray(preds, dtype=np.int64),
        "ids": ids,
    }


def save_report(result, out_dir: Path, prefix: str, extra: Dict) -> Optional[Dict]:
    if result is None:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    label_ids = list(range(len(LABELS)))
    golds = result["golds"]
    preds = result["preds"]
    cm = metrics.confusion_matrix(golds, preds, labels=label_ids)
    report = metrics.classification_report(
        golds,
        preds,
        labels=label_ids,
        target_names=LABELS,
        digits=4,
        zero_division=0,
    )
    np.save(out_dir / f"{prefix}_confusion_matrix.npy", cm)
    with (out_dir / f"{prefix}_confusion_matrix.txt").open("w", encoding="utf-8") as f:
        f.write("labels: " + ",".join(LABELS) + "\n")
        for row in cm:
            f.write(" ".join(str(int(x)) for x in row) + "\n")
    (out_dir / f"{prefix}_classification_report.txt").write_text(report, encoding="utf-8")
    with (out_dir / f"{prefix}_predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "gold_id", "gold_label", "pred_id", "pred_label", "correct"])
        for sample_id, gold, pred in zip(result["ids"], golds, preds):
            writer.writerow([sample_id, int(gold), LABELS[int(gold)], int(pred), LABELS[int(pred)], int(gold == pred)])

    per_p, per_r, per_f1, per_support = metrics.precision_recall_fscore_support(golds, preds, labels=label_ids, zero_division=0)
    summary = {
        **extra,
        "n_samples": int(len(golds)),
        "loss": float(result["loss"]),
        "accuracy": float(result["accuracy"]),
        "macro_f1": float(result["macro_f1"]),
        "weighted_f1": float(result["weighted_f1"]),
        "gold_counts": dict(zip(LABELS, np.bincount(golds, minlength=len(LABELS)).astype(int).tolist())),
        "pred_counts": dict(zip(LABELS, np.bincount(preds, minlength=len(LABELS)).astype(int).tolist())),
        "per_class": {
            LABELS[i]: {
                "precision": float(per_p[i]),
                "recall": float(per_r[i]),
                "f1": float(per_f1[i]),
                "support": int(per_support[i]),
            }
            for i in label_ids
        },
    }
    (out_dir / f"{prefix}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain the TELME video student on BOBSL ANJS4.")
    parser.add_argument("--bobsl_root", type=str, default="/raid_zoe/home/lr/wangyi/sign/bobsl")
    parser.add_argument("--train_csv", type=str, default="")
    parser.add_argument("--val_csv", type=str, default="")
    parser.add_argument("--test_csv", type=str, default="")
    parser.add_argument("--video_subdir", type=str, default="clip256")
    parser.add_argument("--frame_subdir", type=str, default="frame")
    parser.add_argument("--save_root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_score", type=float, default=0.0)
    parser.add_argument("--video_max_seconds", type=float, default=30.0)
    parser.add_argument("--limit_train", type=int, default=0)
    parser.add_argument("--limit_val", type=int, default=0)
    parser.add_argument("--limit_test", type=int, default=0)
    parser.add_argument("--init_checkpoint", type=str, default="")
    parser.add_argument("--freeze_video_backbone", action="store_true")
    parser.add_argument("--save_epoch_every", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    bobsl_root = resolve_path(args.bobsl_root, must_exist=True)
    train_csv = resolve_csv(args.train_csv, bobsl_root, "train_clips_balanced_updated.csv")
    val_csv = resolve_csv(args.val_csv, bobsl_root, "val_clips_balanced_updated.csv")
    test_csv = resolve_csv(args.test_csv, bobsl_root, "test_clips_balanced_updated.csv")
    save_root = resolve_path(args.save_root, must_exist=False)
    student_dir = save_root / "student_video"
    student_path = student_dir / "total_student.bin"
    student_dir.mkdir(parents=True, exist_ok=True)

    train_samples = read_split(train_csv, "train", bobsl_root, args.video_subdir, args.frame_subdir, args.min_score, args.limit_train)
    val_samples = read_split(val_csv, "val", bobsl_root, args.video_subdir, args.frame_subdir, args.min_score, args.limit_val)
    test_samples = read_split(test_csv, "test", bobsl_root, args.video_subdir, args.frame_subdir, args.min_score, args.limit_test)
    if not train_samples or not val_samples or not test_samples:
        raise RuntimeError("BOBSL train/val/test samples must all be non-empty.")

    train_loader = DataLoader(
        BobslVideoDataset(train_samples, args.video_max_seconds),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_video,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        BobslVideoDataset(val_samples, args.video_max_seconds),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_video,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        BobslVideoDataset(test_samples, args.video_max_seconds),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_video,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TELME-BOBSL] device={device} save_root={save_root}")
    model = Student_Video("facebook/timesformer-base-finetuned-k400", len(LABELS)).to(device)
    if args.init_checkpoint:
        partially_load_state_dict(model, resolve_path(args.init_checkpoint, must_exist=True), device)
    if args.freeze_video_backbone:
        for param in model.model.parameters():
            param.requires_grad = False
        print("[TELME-BOBSL] freeze TimeSformer backbone; train classifier head only.")

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best_val = -1.0
    best_epoch = 0
    final_val = None
    final_test = None
    epoch_rows: List[Dict] = []

    for epoch in tqdm(range(1, args.epochs + 1), desc="[TELME-BOBSL][video]"):
        model.train()
        losses = []
        golds = []
        preds = []
        for videos, labels, _ids in train_loader:
            videos = videos.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                _hidden, logits = model(videos)
                loss = nn.CrossEntropyLoss()(logits, labels)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.item()) * int(labels.numel()))
            golds.extend(labels.detach().cpu().numpy().astype(int).tolist())
            preds.extend(logits.argmax(dim=1).detach().cpu().numpy().astype(int).tolist())

        train_wf1 = float(metrics.f1_score(golds, preds, average="weighted", zero_division=0)) if golds else 0.0
        train_loss = float(np.sum(losses) / max(len(golds), 1))
        val_result = evaluate(model, val_loader, device)
        test_result = evaluate(model, test_loader, device)
        final_val = val_result
        final_test = test_result
        val_score = float(val_result["weighted_f1"]) if val_result else -1.0
        test_score = float(test_result["weighted_f1"]) if test_result else -1.0
        if val_score > best_val:
            best_val = val_score
            best_epoch = epoch
            torch.save(model.state_dict(), student_path)
            save_report(
                val_result,
                save_root,
                "bobsl_val_best",
                {"script": "MELD/train_telme_bobsl_video.py", "split": "bobsl_val", "selection": "best", "epoch": epoch},
            )
            save_report(
                test_result,
                save_root,
                "bobsl_test_best",
                {"script": "MELD/train_telme_bobsl_video.py", "split": "bobsl_test", "selection": "best", "epoch": epoch},
            )
            print(f"[TELME-BOBSL] new best epoch={epoch} val_wf1={val_score:.4f} test_wf1={test_score:.4f}")

        if args.save_epoch_every > 0 and (epoch % args.save_epoch_every == 0 or epoch == args.epochs):
            ckpt_dir = student_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch:03d}.bin")

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_weighted_f1": train_wf1,
            "val_weighted_f1": val_score,
            "test_weighted_f1": test_score,
            "best_epoch": best_epoch,
            "best_val_weighted_f1": best_val,
        }
        epoch_rows.append(row)
        print(
            f"[TELME-BOBSL] epoch={epoch} train_wf1={train_wf1:.4f} "
            f"val_wf1={val_score:.4f} test_wf1={test_score:.4f}"
        )

    if final_val is not None:
        save_report(
            final_val,
            save_root,
            "bobsl_val_final",
            {"script": "MELD/train_telme_bobsl_video.py", "split": "bobsl_val", "selection": "final", "epoch": args.epochs},
        )
    if final_test is not None:
        save_report(
            final_test,
            save_root,
            "bobsl_test_final",
            {"script": "MELD/train_telme_bobsl_video.py", "split": "bobsl_test", "selection": "final", "epoch": args.epochs},
        )

    metrics_path = save_root / "telme_bobsl_video_epoch_metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(epoch_rows[0]))
        writer.writeheader()
        writer.writerows(epoch_rows)

    manifest = {
        "save_root": str(save_root),
        "student_checkpoint": str(student_path),
        "best_epoch": best_epoch,
        "best_val_weighted_f1": best_val,
        "labels": LABELS,
        "epoch_metrics_csv": str(metrics_path),
        "args": vars(args),
    }
    (save_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
