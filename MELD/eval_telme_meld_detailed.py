import argparse
import csv
import json
import random
from copy import copy
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn import metrics
from torch.utils.data import DataLoader

from dataset import meld_dataset
from model import ASF, Student_Video, Teacher_model
from preprocessing import preprocessing
from train_telme_meld_modalities import MELD_LABELS, make_telme_batch, normalize_modality


ANJS_LABELS = ["A", "N", "J", "S"]
TELME_MELD_TO_ANJS = {
    0: 0,  # anger -> A
    4: 1,  # neutral -> N
    3: 2,  # joy -> J
    5: 3,  # sadness -> S
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_state_dict_compat(path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)

    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {path}")

    state.pop("text_model.embeddings.position_ids", None)
    return state


def resolve_model_root(args) -> Path:
    if args.model_root:
        return Path(args.model_root)
    return Path(args.save_root) / args.modality


def build_split_loader(args, load_video: bool) -> DataLoader:
    split_to_file = {
        "train": "train_meld_emo.csv",
        "dev": "dev_meld_emo.csv",
        "test": "test_meld_emo.csv",
    }
    csv_path = Path(args.data_root) / split_to_file[args.split]
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing MELD {args.split} csv: {csv_path}")

    data = preprocessing(str(csv_path))
    if args.max_samples > 0:
        data = data[: args.max_samples]

    return DataLoader(
        meld_dataset(data),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=partial(make_telme_batch, load_video=load_video),
        pin_memory=torch.cuda.is_available() and not args.no_cuda,
    )


def build_models(args, device: torch.device):
    model_root = resolve_model_root(args)
    cls_num = len(MELD_LABELS)
    need_text = args.modality in {"text", "tv"}
    need_video = args.modality in {"video", "tv"}

    teacher = None
    if need_text:
        teacher_path = model_root / "teacher.bin"
        if not teacher_path.exists():
            raise FileNotFoundError(f"Missing TelME teacher checkpoint: {teacher_path}")
        teacher = Teacher_model("roberta-large", cls_num)
        teacher.load_state_dict(load_state_dict_compat(teacher_path, device), strict=False)
        teacher = teacher.to(device).eval()

    video_student = None
    if need_video:
        video_path = model_root / "student_video" / "total_student.bin"
        if not video_path.exists():
            raise FileNotFoundError(f"Missing TelME video student checkpoint: {video_path}")
        video_student = Student_Video("facebook/timesformer-base-finetuned-k400", cls_num)
        video_student.load_state_dict(load_state_dict_compat(video_path, device), strict=True)
        video_student = video_student.to(device).eval()

    fusion_path = model_root / "total_fusion.bin"
    if not fusion_path.exists():
        raise FileNotFoundError(f"Missing TelME fusion checkpoint: {fusion_path}")
    fusion = ASF(cls_num, hidden_size=768, beta_shift=1e-1, dropout_prob=0.2, num_head=3)
    fusion.load_state_dict(load_state_dict_compat(fusion_path, device), strict=True)
    fusion = fusion.to(device).eval()

    return model_root, teacher, video_student, fusion


def get_modality_hiddens(modality, teacher, video_student, tokens, masks, video):
    text_hidden = None
    video_hidden = None

    if modality in {"text", "tv"}:
        text_hidden, _ = teacher(tokens, masks)
    if modality in {"video", "tv"}:
        video_hidden, _ = video_student(video)

    if text_hidden is None:
        text_hidden = torch.zeros_like(video_hidden)
    if video_hidden is None:
        video_hidden = torch.zeros_like(text_hidden)
    audio_hidden = torch.zeros_like(text_hidden)
    return text_hidden, audio_hidden, video_hidden


def evaluate(args, loader, teacher, video_student, fusion, device):
    golds: List[int] = []
    preds: List[int] = []
    probs_rows: List[np.ndarray] = []

    with torch.no_grad():
        for tokens, masks, _audio, video, labels in loader:
            tokens = tokens.to(device)
            masks = masks.to(device)
            video = video.to(device)
            labels = labels.to(device)

            text_hidden, audio_hidden, video_hidden = get_modality_hiddens(
                args.modality, teacher, video_student, tokens, masks, video
            )
            logits = fusion(text_hidden, audio_hidden, video_hidden)
            probs = torch.softmax(logits, dim=-1)
            pred = probs.argmax(dim=-1)

            golds.extend(labels.cpu().numpy().tolist())
            preds.extend(pred.cpu().numpy().tolist())
            probs_rows.extend(probs.cpu().numpy())

    return np.asarray(golds, dtype=np.int64), np.asarray(preds, dtype=np.int64), np.asarray(probs_rows)


def write_confusion_matrix(cm: np.ndarray, labels: List[str], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("labels: " + ",".join(labels) + "\n")
        for row in cm:
            f.write(" ".join(str(int(x)) for x in row) + "\n")


def build_count_table(values: np.ndarray, n_labels: int) -> List[int]:
    return np.bincount(values.astype(np.int64), minlength=n_labels).tolist()


def save_raw_meld_reports(golds, preds, probs, out_dir: Path, prefix: str) -> Dict:
    label_ids = list(range(len(MELD_LABELS)))
    cm = metrics.confusion_matrix(golds, preds, labels=label_ids)
    report = metrics.classification_report(
        golds,
        preds,
        labels=label_ids,
        target_names=MELD_LABELS,
        digits=4,
        zero_division=0,
    )

    np.save(out_dir / f"{prefix}_meld7_confusion_matrix.npy", cm)
    write_confusion_matrix(cm, MELD_LABELS, out_dir / f"{prefix}_meld7_confusion_matrix.txt")
    (out_dir / f"{prefix}_meld7_classification_report.txt").write_text(report, encoding="utf-8")

    per_p, per_r, per_f1, per_support = metrics.precision_recall_fscore_support(
        golds,
        preds,
        labels=label_ids,
        zero_division=0,
    )
    per_class = {
        MELD_LABELS[i]: {
            "precision": float(per_p[i]),
            "recall": float(per_r[i]),
            "f1": float(per_f1[i]),
            "support": int(per_support[i]),
            "gold_count": int(cm[i].sum()),
            "pred_count": int(cm[:, i].sum()),
            "mean_pred_prob": float(probs[:, i].mean()) if len(probs) else 0.0,
        }
        for i in label_ids
    }

    summary = {
        "n_samples": int(len(golds)),
        "accuracy": float(metrics.accuracy_score(golds, preds)) if len(golds) else 0.0,
        "macro_f1": float(metrics.f1_score(golds, preds, labels=label_ids, average="macro", zero_division=0)),
        "weighted_f1": float(metrics.f1_score(golds, preds, labels=label_ids, average="weighted", zero_division=0)),
        "gold_counts": dict(zip(MELD_LABELS, build_count_table(golds, len(MELD_LABELS)))),
        "pred_counts": dict(zip(MELD_LABELS, build_count_table(preds, len(MELD_LABELS)))),
        "per_class": per_class,
    }
    return summary


def save_anjs_projection_reports(golds, preds, probs, out_dir: Path, prefix: str) -> Dict:
    keep_gold = np.isin(golds, list(TELME_MELD_TO_ANJS))
    if not keep_gold.any():
        return {"n_samples": 0, "note": "no MELD samples in anger/neutral/joy/sadness subset"}

    golds_sub = golds[keep_gold]
    probs_sub = probs[keep_gold]
    raw_preds_sub = preds[keep_gold]
    golds4 = np.asarray([TELME_MELD_TO_ANJS[int(x)] for x in golds_sub], dtype=np.int64)
    probs4 = np.stack(
        [
            probs_sub[:, 0],  # A: anger
            probs_sub[:, 4],  # N: neutral
            probs_sub[:, 3],  # J: joy
            probs_sub[:, 5],  # S: sadness
        ],
        axis=1,
    )
    preds4 = probs4.argmax(axis=1).astype(np.int64)

    label_ids = list(range(len(ANJS_LABELS)))
    cm = metrics.confusion_matrix(golds4, preds4, labels=label_ids)
    report = metrics.classification_report(
        golds4,
        preds4,
        labels=label_ids,
        target_names=ANJS_LABELS,
        digits=4,
        zero_division=0,
    )

    np.save(out_dir / f"{prefix}_anjs_projection_confusion_matrix.npy", cm)
    write_confusion_matrix(cm, ANJS_LABELS, out_dir / f"{prefix}_anjs_projection_confusion_matrix.txt")
    (out_dir / f"{prefix}_anjs_projection_classification_report.txt").write_text(report, encoding="utf-8")

    oos_ids = [1, 2, 6]  # disgust, fear, surprise in TelME/MELD order.
    oos_by_gold = {}
    for meld_id, anjs_id in TELME_MELD_TO_ANJS.items():
        mask = golds_sub == meld_id
        oos_by_gold[ANJS_LABELS[anjs_id]] = {
            MELD_LABELS[oos_id]: int(np.logical_and(mask, raw_preds_sub == oos_id).sum()) for oos_id in oos_ids
        }

    with open(out_dir / f"{prefix}_anjs_projection_predictions.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "gold_label", "pred_label", "correct", "raw_meld7_pred"])
        for idx, (g, p, raw_p) in enumerate(zip(golds4, preds4, raw_preds_sub)):
            writer.writerow([idx, ANJS_LABELS[int(g)], ANJS_LABELS[int(p)], int(g == p), MELD_LABELS[int(raw_p)]])

    return {
        "n_samples": int(len(golds4)),
        "accuracy": float(metrics.accuracy_score(golds4, preds4)) if len(golds4) else 0.0,
        "macro_f1": float(metrics.f1_score(golds4, preds4, labels=label_ids, average="macro", zero_division=0)),
        "weighted_f1": float(metrics.f1_score(golds4, preds4, labels=label_ids, average="weighted", zero_division=0)),
        "gold_counts": dict(zip(ANJS_LABELS, build_count_table(golds4, len(ANJS_LABELS)))),
        "pred_counts": dict(zip(ANJS_LABELS, build_count_table(preds4, len(ANJS_LABELS)))),
        "raw_out_of_scope_predictions_by_gold": oos_by_gold,
    }


def save_prediction_csv(golds, preds, probs, out_dir: Path, prefix: str) -> None:
    with open(out_dir / f"{prefix}_meld7_predictions.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["index", "gold_id", "gold_label", "pred_id", "pred_label", "correct"]
            + [f"prob_{name}" for name in MELD_LABELS]
        )
        for i, (gold, pred, prob) in enumerate(zip(golds, preds, probs)):
            writer.writerow(
                [i, int(gold), MELD_LABELS[int(gold)], int(pred), MELD_LABELS[int(pred)], int(gold == pred)]
                + [float(x) for x in prob]
            )


def parse_modalities(args) -> List[str]:
    raw_values = args.modalities
    if raw_values is None:
        raw_values = [args.modality] if args.modality else ["text", "video", "tv"]

    out: List[str] = []
    for value in raw_values:
        for part in str(value).replace(",", " ").split():
            modality = normalize_modality(part)
            if modality not in out:
                out.append(modality)
    if not out:
        raise ValueError("No modality requested")
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="Load a TelME MELD checkpoint and save detailed MELD reports.")
    parser.add_argument("--data_root", type=str, default="./dataset/MELD.Raw")
    parser.add_argument("--save_root", type=str, default="./MELD/save_model_meld_modalities")
    parser.add_argument("--model_root", type=str, default="", help="Overrides --save_root/--modality when set.")
    parser.add_argument("--modality", type=str, default="", help="Run one modality: text, video, or tv.")
    parser.add_argument("--modalities", nargs="+", default=None, help="Run several modalities. Default: text video tv.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "dev", "test"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="./MELD/outputs_meld_detailed")
    parser.add_argument("--prefix", type=str, default="", help="Defaults to telme_<modality>_<split>.")
    parser.add_argument("--no_cuda", action="store_true")
    return parser.parse_args()


def run_one(args, modality: str, device: torch.device, multi_run: bool) -> None:
    args = copy(args)
    args.modality = modality
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.prefix:
        prefix = f"{args.prefix}_{modality}" if multi_run else args.prefix
    else:
        prefix = f"telme_{modality}_{args.split}"

    model_root, teacher, video_student, fusion = build_models(args, device)
    loader = build_split_loader(args, load_video=args.modality in {"video", "tv"})
    golds, preds, probs = evaluate(args, loader, teacher, video_student, fusion, device)

    raw_summary = save_raw_meld_reports(golds, preds, probs, out_dir, prefix)
    anjs_summary = save_anjs_projection_reports(golds, preds, probs, out_dir, prefix)
    save_prediction_csv(golds, preds, probs, out_dir, prefix)

    summary = {
        "script": "MELD/eval_telme_meld_detailed.py",
        "data_root": str(Path(args.data_root)),
        "model_root": str(model_root),
        "split": args.split,
        "modality": args.modality,
        "device": str(device),
        "meld7": raw_summary,
        "anjs_projection_on_meld_subset": anjs_summary,
    }
    summary_path = out_dir / f"{prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[TelME] model_root={model_root}")
    print(f"[TelME] split={args.split} modality={args.modality} samples={len(golds)}")
    print(
        "[TelME][MELD7] "
        f"acc={raw_summary['accuracy']:.4f} macro_f1={raw_summary['macro_f1']:.4f} "
        f"weighted_f1={raw_summary['weighted_f1']:.4f}"
    )
    print("[TelME][MELD7] gold_counts=" + json.dumps(raw_summary["gold_counts"], ensure_ascii=False))
    print("[TelME][MELD7] pred_counts=" + json.dumps(raw_summary["pred_counts"], ensure_ascii=False))
    if anjs_summary.get("n_samples", 0):
        print(
            "[TelME][ANJS projection] "
            f"acc={anjs_summary['accuracy']:.4f} macro_f1={anjs_summary['macro_f1']:.4f} "
            f"weighted_f1={anjs_summary['weighted_f1']:.4f}"
        )
        print("[TelME][ANJS projection] pred_counts=" + json.dumps(anjs_summary["pred_counts"], ensure_ascii=False))
    print(f"[TelME] reports saved to: {out_dir}")


def main() -> None:
    args = parse_args()
    modalities = parse_modalities(args)
    if args.model_root and len(modalities) != 1:
        raise ValueError("--model_root points to one checkpoint directory, so use it with exactly one --modality")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    for modality in modalities:
        run_one(args, modality, device, multi_run=len(modalities) > 1)


if __name__ == "__main__":
    main()
