#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


LABELS = ["A", "N", "J", "S"]


def resolve_path(raw: str, must_exist: bool = False) -> Path:
    path = Path(raw).expanduser()
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, Path(__file__).resolve().parent.parent / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    if must_exist:
        raise FileNotFoundError(f"Path not found: {raw}. Checked: {', '.join(str(x) for x in candidates)}")
    return candidates[0].resolve()


def load_pkl(path: Path):
    with path.open("rb") as f:
        return pickle.load(f, encoding="latin1")


def collect_split(path: Path, split: str, max_items: int, seed: int) -> Dict[str, np.ndarray]:
    obj = load_pkl(path)
    labels = obj[2]
    text = obj[3]
    audio = obj[4]
    visual = obj[5]
    train_vid = obj[7]
    test_vid = obj[8]
    meta = obj[9] if len(obj) > 9 and isinstance(obj[9], dict) else {}

    if split == "train":
        keys = list(train_vid)
    elif split in {"test", "source", "external", "val"}:
        keys = list(test_vid)
    elif split == "all":
        keys = list(train_vid) + list(test_vid)
    else:
        raise ValueError(f"Unsupported split={split}")

    rows_l: List[int] = []
    rows_t: List[np.ndarray] = []
    rows_a: List[np.ndarray] = []
    rows_v: List[np.ndarray] = []
    ids: List[str] = []
    for vid in keys:
        y_seq = labels[vid]
        for idx, y in enumerate(y_seq):
            y = int(y)
            if y < 0:
                continue
            rows_l.append(y)
            rows_t.append(np.asarray(text[vid][idx], dtype=np.float32))
            rows_a.append(np.asarray(audio[vid][idx], dtype=np.float32))
            rows_v.append(np.asarray(visual[vid][idx], dtype=np.float32))
            ids.append(f"{vid}:utt{idx + 1:02d}")

    if max_items > 0 and len(rows_l) > max_items:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(rows_l), size=max_items, replace=False)
        rows_l = [rows_l[i] for i in indices]
        rows_t = [rows_t[i] for i in indices]
        rows_a = [rows_a[i] for i in indices]
        rows_v = [rows_v[i] for i in indices]
        ids = [ids[i] for i in indices]

    if not rows_l:
        raise RuntimeError(f"No valid labeled rows in {path} split={split}")

    return {
        "ids": np.asarray(ids, dtype=object),
        "labels": np.asarray(rows_l, dtype=np.int64),
        "text": np.stack(rows_t, axis=0).astype(np.float32),
        "audio": np.stack(rows_a, axis=0).astype(np.float32),
        "visual": np.stack(rows_v, axis=0).astype(np.float32),
        "meta": meta,
    }


def modality_matrix(data: Dict[str, np.ndarray], modality: str) -> np.ndarray:
    if modality == "text":
        return data["text"]
    if modality == "visual":
        return data["visual"]
    if modality == "tv":
        return np.concatenate([data["text"], data["visual"]], axis=1)
    raise ValueError(f"Unsupported modality={modality}")


def feature_stats(dataset: str, split: str, modality: str, x: np.ndarray, y: np.ndarray) -> Dict:
    norms = np.linalg.norm(x, axis=1)
    zero_ratio = float(np.mean(norms < 1e-8))
    out = {
        "dataset": dataset,
        "split": split,
        "modality": modality,
        "n": int(len(y)),
        "dim": int(x.shape[1]),
        "zero_ratio": zero_ratio,
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "label_counts": dict(zip(LABELS, np.bincount(y, minlength=len(LABELS)).astype(int).tolist())),
    }
    for label_id, label_name in enumerate(LABELS):
        mask = y == label_id
        if mask.any():
            out[f"norm_mean_{label_name}"] = float(norms[mask].mean())
            out[f"norm_std_{label_name}"] = float(norms[mask].std())
        else:
            out[f"norm_mean_{label_name}"] = None
            out[f"norm_std_{label_name}"] = None
    return out


def accuracy_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def f1_scores_np(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 4) -> Tuple[float, float]:
    per_f1 = []
    support = []
    for label_id in range(n_classes):
        tp = int(np.sum((y_true == label_id) & (y_pred == label_id)))
        fp = int(np.sum((y_true != label_id) & (y_pred == label_id)))
        fn = int(np.sum((y_true == label_id) & (y_pred != label_id)))
        sup = int(np.sum(y_true == label_id))
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        per_f1.append(f1)
        support.append(sup)
    macro = float(np.mean(per_f1))
    total = max(sum(support), 1)
    weighted = float(sum(f * s for f, s in zip(per_f1, support)) / total)
    return macro, weighted


def write_csv(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({k for row in rows for k in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def standardize_train_apply(train_x: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (train_x - mean) / std, (x - mean) / std


def centroid_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    train_x_std, test_x_std = standardize_train_apply(train_x, test_x)
    centroids = []
    for label_id in range(len(LABELS)):
        mask = train_y == label_id
        if mask.any():
            centroids.append(train_x_std[mask].mean(axis=0))
        else:
            centroids.append(np.zeros(train_x_std.shape[1], dtype=np.float32))
    c = np.stack(centroids, axis=0)
    x_norm = test_x_std / np.maximum(np.linalg.norm(test_x_std, axis=1, keepdims=True), 1e-8)
    c_norm = c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-8)
    sims = x_norm @ c_norm.T
    return sims.argmax(axis=1)


def prototype_rows(data: Dict[str, Dict[str, np.ndarray]]) -> List[Dict]:
    rows: List[Dict] = []
    train_sources = ["bobsl_train", "meld_train"]
    eval_targets = ["bobsl_val", "bobsl_test", "meld_test", "ejsl_test"]
    for modality in ["visual", "text", "tv"]:
        for source_name in train_sources:
            if source_name not in data:
                continue
            train = data[source_name]
            train_x = modality_matrix(train, modality)
            train_y = train["labels"]
            for target_name in eval_targets:
                if target_name not in data:
                    continue
                target = data[target_name]
                target_x = modality_matrix(target, modality)
                target_y = target["labels"]
                pred = centroid_predict(train_x, train_y, target_x)
                macro_f1, weighted_f1 = f1_scores_np(target_y, pred)
                rows.append(
                    {
                        "source_centroids": source_name,
                        "target": target_name,
                        "modality": modality,
                        "accuracy": accuracy_score_np(target_y, pred),
                        "macro_f1": macro_f1,
                        "weighted_f1": weighted_f1,
                        "pred_counts": json.dumps(dict(zip(LABELS, np.bincount(pred, minlength=len(LABELS)).astype(int).tolist())), ensure_ascii=False),
                    }
                )
    return rows


def domain_classifier_rows(data: Dict[str, Dict[str, np.ndarray]], seed: int) -> List[Dict]:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        return [{"warning": f"skipped domain classifier because sklearn is unavailable: {exc}"}]

    pairs = [
        ("bobsl_train", "meld_train"),
        ("bobsl_train", "ejsl_test"),
        ("meld_train", "ejsl_test"),
        ("meld_test", "ejsl_test"),
    ]
    rows: List[Dict] = []
    for left, right in pairs:
        if left not in data or right not in data:
            continue
        for modality in ["visual", "text", "tv"]:
            x_left = modality_matrix(data[left], modality)
            x_right = modality_matrix(data[right], modality)
            n = min(len(x_left), len(x_right), 5000)
            if n < 20:
                continue
            rng = np.random.default_rng(seed)
            li = rng.choice(len(x_left), size=n, replace=False)
            ri = rng.choice(len(x_right), size=n, replace=False)
            x = np.concatenate([x_left[li], x_right[ri]], axis=0)
            y = np.asarray([0] * n + [1] * n, dtype=np.int64)
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=seed, stratify=y)
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, solver="saga", n_jobs=1),
            )
            clf.fit(x_train, y_train)
            pred = clf.predict(x_test)
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "modality": modality,
                    "n_per_domain": n,
                    "domain_accuracy": float(accuracy_score(y_test, pred)),
                }
            )
    return rows


def centroid_distance_rows(data: Dict[str, Dict[str, np.ndarray]]) -> List[Dict]:
    pairs = [
        ("bobsl_train", "meld_train"),
        ("bobsl_train", "ejsl_test"),
        ("meld_train", "ejsl_test"),
        ("meld_test", "ejsl_test"),
    ]
    rows: List[Dict] = []
    for left, right in pairs:
        if left not in data or right not in data:
            continue
        for modality in ["visual", "text", "tv"]:
            x_left = modality_matrix(data[left], modality)
            x_right = modality_matrix(data[right], modality)
            y_left = data[left]["labels"]
            y_right = data[right]["labels"]
            for label_id, label_name in enumerate(["ALL"] + LABELS):
                if label_name == "ALL":
                    lx, rx = x_left, x_right
                else:
                    lx = x_left[y_left == (label_id - 1)]
                    rx = x_right[y_right == (label_id - 1)]
                if len(lx) == 0 or len(rx) == 0:
                    continue
                lc = lx.mean(axis=0)
                rc = rx.mean(axis=0)
                l2 = float(np.linalg.norm(lc - rc))
                cos = float(np.dot(lc, rc) / max(np.linalg.norm(lc) * np.linalg.norm(rc), 1e-8))
                rows.append(
                    {
                        "left": left,
                        "right": right,
                        "modality": modality,
                        "label": label_name,
                        "left_n": int(len(lx)),
                        "right_n": int(len(rx)),
                        "centroid_l2": l2,
                        "centroid_cosine": cos,
                    }
                )
    return rows


def read_csv_dicts(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def float_or_nan(value) -> float:
    try:
        if value in (None, ""):
            return math.nan
        return float(value)
    except ValueError:
        return math.nan


def summarize_epoch_metrics(run_roots: Sequence[Path]) -> List[Dict]:
    rows: List[Dict] = []
    for root in run_roots:
        for metrics_path in sorted(root.glob("*/mmgcn_unified_*_epoch_metrics.csv")):
            records = read_csv_dicts(metrics_path)
            if not records:
                continue
            modality = metrics_path.parent.name
            enriched = []
            for record in records:
                enriched.append(
                    {
                        **record,
                        "epoch_int": int(record["epoch"]),
                        "source": float_or_nan(record.get("source_test_weighted_f1")),
                        "external": float_or_nan(record.get("external_test_weighted_f1")),
                        "train": float_or_nan(record.get("train_weighted_f1")),
                    }
                )
            source_records = [x for x in enriched if not math.isnan(x["source"])]
            external_records = [x for x in enriched if not math.isnan(x["external"])]
            best_source = max(source_records, key=lambda x: x["source"]) if source_records else None
            best_external = max(external_records, key=lambda x: x["external"]) if external_records else None
            rows.append(
                {
                    "run_root": str(root),
                    "modality": modality,
                    "metrics_csv": str(metrics_path),
                    "best_source_epoch": best_source["epoch_int"] if best_source else None,
                    "best_source_wf1": best_source["source"] if best_source else None,
                    "external_at_best_source": best_source["external"] if best_source else None,
                    "best_external_epoch": best_external["epoch_int"] if best_external else None,
                    "best_external_wf1": best_external["external"] if best_external else None,
                    "source_at_best_external": best_external["source"] if best_external else None,
                    "source_external_gap_at_source_best": (
                        best_source["source"] - best_source["external"]
                        if best_source and not math.isnan(best_source["external"])
                        else None
                    ),
                    "external_lost_by_source_selection": (
                        best_external["external"] - best_source["external"]
                        if best_source and best_external and not math.isnan(best_source["external"])
                        else None
                    ),
                }
            )
    return rows


def summarize_reports(run_roots: Sequence[Path]) -> List[Dict]:
    rows: List[Dict] = []
    for root in run_roots:
        for summary_path in sorted(root.glob("*/*_summary.json")):
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            row = {
                "run_root": str(root),
                "summary_path": str(summary_path),
                "modality": summary.get("modality", summary_path.parent.name),
                "split": summary.get("split", ""),
                "selection": summary.get("selection", ""),
                "accuracy": summary.get("accuracy"),
                "macro_f1": summary.get("macro_f1"),
                "weighted_f1": summary.get("weighted_f1"),
                "n_samples": summary.get("n_samples"),
                "gold_counts": json.dumps(summary.get("gold_counts", {}), ensure_ascii=False),
                "pred_counts": json.dumps(summary.get("pred_counts", {}), ensure_ascii=False),
            }
            per_class = summary.get("per_class", {})
            for label in LABELS:
                cls = per_class.get(label, {})
                row[f"{label}_f1"] = cls.get("f1")
                row[f"{label}_recall"] = cls.get("recall")
                row[f"{label}_mean_pred_prob"] = cls.get("mean_pred_prob")
            rows.append(row)
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose why MELD and eJSL diverge in unified MMGCN transfer.")
    parser.add_argument("--bobsl_train_val_pkl", type=str, default="")
    parser.add_argument("--bobsl_test_pkl", type=str, default="")
    parser.add_argument("--meld_pkl", type=str, required=True)
    parser.add_argument("--ejsl_pkl", type=str, required=True)
    parser.add_argument("--run_roots", type=str, nargs="*", default=[])
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--max_items", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = resolve_path(args.out_dir, must_exist=False)
    out_dir.mkdir(parents=True, exist_ok=True)

    data: Dict[str, Dict[str, np.ndarray]] = {
        "meld_train": collect_split(resolve_path(args.meld_pkl, must_exist=True), "train", args.max_items, args.seed),
        "meld_test": collect_split(resolve_path(args.meld_pkl, must_exist=True), "test", args.max_items, args.seed + 1),
        "ejsl_test": collect_split(resolve_path(args.ejsl_pkl, must_exist=True), "test", args.max_items, args.seed + 2),
    }
    if args.bobsl_train_val_pkl:
        bobsl_train_val = resolve_path(args.bobsl_train_val_pkl, must_exist=True)
        data["bobsl_train"] = collect_split(bobsl_train_val, "train", args.max_items, args.seed + 3)
        data["bobsl_val"] = collect_split(bobsl_train_val, "test", args.max_items, args.seed + 4)
    if args.bobsl_test_pkl:
        data["bobsl_test"] = collect_split(resolve_path(args.bobsl_test_pkl, must_exist=True), "test", args.max_items, args.seed + 5)

    stats_rows: List[Dict] = []
    for name, split_data in data.items():
        dataset, split = name.split("_", 1)
        for modality in ["visual", "text", "tv"]:
            stats_rows.append(feature_stats(dataset, split, modality, modality_matrix(split_data, modality), split_data["labels"]))
    write_csv(stats_rows, out_dir / "feature_stats.csv")

    proto_rows = prototype_rows(data)
    write_csv(proto_rows, out_dir / "prototype_transfer.csv")

    dist_rows = centroid_distance_rows(data)
    write_csv(dist_rows, out_dir / "centroid_distances.csv")

    domain_rows = domain_classifier_rows(data, args.seed)
    write_csv(domain_rows, out_dir / "domain_classifier.csv")

    run_roots = [resolve_path(x, must_exist=True) for x in args.run_roots]
    epoch_rows = summarize_epoch_metrics(run_roots)
    write_csv(epoch_rows, out_dir / "epoch_gap_summary.csv")

    report_rows = summarize_reports(run_roots)
    write_csv(report_rows, out_dir / "report_summary.csv")

    top = {
        "out_dir": str(out_dir),
        "datasets": {name: {"n": int(len(value["labels"]))} for name, value in data.items()},
        "files": {
            "feature_stats": str(out_dir / "feature_stats.csv"),
            "prototype_transfer": str(out_dir / "prototype_transfer.csv"),
            "centroid_distances": str(out_dir / "centroid_distances.csv"),
            "domain_classifier": str(out_dir / "domain_classifier.csv"),
            "epoch_gap_summary": str(out_dir / "epoch_gap_summary.csv"),
            "report_summary": str(out_dir / "report_summary.csv"),
        },
    }
    (out_dir / "analysis_manifest.json").write_text(json.dumps(top, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(top, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
