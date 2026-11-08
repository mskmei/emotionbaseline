#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


LABELS = ("A", "N", "J", "S")
DEFAULT_PATHS = {
    "BOBSL_train": "/raid_zoe/home/lr/maokeyu/sign/mmgcn_bobsl_meld_ejsl/bobsl_anjs4_train_val.pkl",
    "MELD": "/raid_zoe/home/lr/maokeyu/sign/mmgcn_unified_meld_ejsl/meld_anjs4_unified.pkl",
    "IEMOCAP": "/raid_zoe/home/lr/maokeyu/sign/mmgcn_bobsl_iemocap_ejsl/features/iemocap_mmgcn_video_anjs4.pkl",
    "eJSL_test": "/raid_zoe/home/lr/maokeyu/sign/mmgcn_unified_meld_ejsl/ejsl_anjs4_unified.pkl",
}


def load_tuple(path: Path) -> Tuple[Mapping, Mapping, Sequence[str], Sequence[str], Mapping]:
    with path.open("rb") as handle:
        payload = pickle.load(handle, encoding="latin1")
    if not isinstance(payload, (tuple, list)) or len(payload) < 9:
        raise ValueError(f"Expected MMGCN tuple pickle at {path}")
    labels = payload[2]
    visual = payload[5]
    train_ids = payload[7]
    test_ids = payload[8]
    meta = payload[9] if len(payload) > 9 and isinstance(payload[9], dict) else {}
    return labels, visual, train_ids, test_ids, meta


def flatten(labels: Mapping, visual: Mapping, ids: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[np.ndarray] = []
    ys: List[int] = []
    for dialogue_id in ids:
        dialogue_x = np.asarray(visual[dialogue_id], dtype=np.float32)
        dialogue_y = np.asarray(labels[dialogue_id], dtype=np.int64)
        if dialogue_x.shape[0] != dialogue_y.shape[0]:
            raise ValueError(f"Feature/label length mismatch for {dialogue_id}")
        keep = (dialogue_y >= 0) & (dialogue_y < len(LABELS)) & np.all(np.isfinite(dialogue_x), axis=1)
        xs.append(dialogue_x[keep])
        ys.extend(dialogue_y[keep].tolist())
    if not xs:
        raise ValueError("No valid visual features")
    return np.concatenate(xs, axis=0), np.asarray(ys, dtype=np.int64)


def load_domains(paths: Mapping[str, str]) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, Mapping]]:
    domains: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    metas: Dict[str, Mapping] = {}
    for name, raw_path in paths.items():
        labels, visual, train_ids, test_ids, meta = load_tuple(Path(raw_path))
        metas[name] = meta
        if name == "MELD":
            domains["MELD_train"] = flatten(labels, visual, train_ids)
            domains["MELD_test"] = flatten(labels, visual, test_ids)
        elif name == "IEMOCAP":
            domains["IEMOCAP_train"] = flatten(labels, visual, train_ids)
            domains["IEMOCAP_test"] = flatten(labels, visual, test_ids)
        elif name == "BOBSL_train":
            domains[name] = flatten(labels, visual, train_ids)
        elif name == "eJSL_test":
            domains[name] = flatten(labels, visual, test_ids)
        else:
            raise ValueError(f"Unknown domain key: {name}")
    dimensions = {features.shape[1] for features, _ in domains.values()}
    if len(dimensions) != 1:
        raise ValueError(f"Visual dimensions differ across domains: {dimensions}")
    return domains, metas


def balanced_sample(
    features: np.ndarray,
    labels: np.ndarray,
    max_per_class: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    selected: List[np.ndarray] = []
    for class_id in range(len(LABELS)):
        indices = np.flatnonzero(labels == class_id)
        if len(indices) > max_per_class:
            indices = rng.choice(indices, size=max_per_class, replace=False)
        selected.append(indices)
    all_indices = np.concatenate(selected)
    rng.shuffle(all_indices)
    return features[all_indices], labels[all_indices]


def l2_normalize(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, 1e-12)


def cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    left = left / max(float(np.linalg.norm(left)), 1e-12)
    right = right / max(float(np.linalg.norm(right)), 1e-12)
    return float(1.0 - np.dot(left, right))


def squared_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    result = np.sum(left * left, axis=1, keepdims=True) + np.sum(right * right, axis=1)[None, :] - 2.0 * left @ right.T
    return np.maximum(result, 0.0)


def median_bandwidth(left: np.ndarray, right: np.ndarray, rng: np.random.Generator) -> float:
    merged = np.concatenate([left, right], axis=0)
    if len(merged) > 600:
        merged = merged[rng.choice(len(merged), size=600, replace=False)]
    distances = squared_distances(merged, merged)
    positive = distances[distances > 0]
    return float(np.median(positive)) if len(positive) else 1.0


def rbf_mmd(left: np.ndarray, right: np.ndarray, rng: np.random.Generator, max_samples: int) -> float:
    if len(left) > max_samples:
        left = left[rng.choice(len(left), size=max_samples, replace=False)]
    if len(right) > max_samples:
        right = right[rng.choice(len(right), size=max_samples, replace=False)]
    bandwidth = max(median_bandwidth(left, right, rng), 1e-12)
    k_ll = np.exp(-squared_distances(left, left) / bandwidth)
    k_rr = np.exp(-squared_distances(right, right) / bandwidth)
    k_lr = np.exp(-squared_distances(left, right) / bandwidth)
    return float(k_ll.mean() + k_rr.mean() - 2.0 * k_lr.mean())


def domain_probe(
    left: Tuple[np.ndarray, np.ndarray],
    right: Tuple[np.ndarray, np.ndarray],
    max_per_class: int,
    rng: np.random.Generator,
) -> Tuple[float, float, int]:
    left_x, left_y = balanced_sample(*left, max_per_class=max_per_class, rng=rng)
    right_x, right_y = balanced_sample(*right, max_per_class=max_per_class, rng=rng)
    # Matching class counts prevents the probe from exploiting emotion-prior differences.
    xs: List[np.ndarray] = []
    domain_labels: List[int] = []
    strata: List[int] = []
    for class_id in range(len(LABELS)):
        l = left_x[left_y == class_id]
        r = right_x[right_y == class_id]
        count = min(len(l), len(r))
        xs.extend([l[:count], r[:count]])
        domain_labels.extend([0] * count + [1] * count)
        strata.extend([class_id * 2] * count + [class_id * 2 + 1] * count)
    x = np.concatenate(xs, axis=0)
    y = np.asarray(domain_labels, dtype=np.int64)
    strata_array = np.asarray(strata, dtype=np.int64)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, solver="liblinear", random_state=0))
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(model, x, y, cv=folds.split(x, strata_array), scoring="accuracy")
    return float(scores.mean()), float(scores.std(ddof=0)), len(y)


def prototype_transfer(
    train: Tuple[np.ndarray, np.ndarray],
    test: Tuple[np.ndarray, np.ndarray],
) -> Tuple[float, float]:
    train_x, train_y = train
    test_x, test_y = test
    train_x = l2_normalize(train_x)
    test_x = l2_normalize(test_x)
    prototypes = np.stack([train_x[train_y == class_id].mean(axis=0) for class_id in range(len(LABELS))])
    prototypes = l2_normalize(prototypes)
    predictions = (test_x @ prototypes.T).argmax(axis=1)
    return float(accuracy_score(test_y, predictions)), float(f1_score(test_y, predictions, average="weighted", zero_division=0))


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure same-extractor visual geometry across BOBSL, MELD, IEMOCAP, and eJSL.")
    parser.add_argument("--bobsl_pkl", default=DEFAULT_PATHS["BOBSL_train"])
    parser.add_argument("--meld_pkl", default=DEFAULT_PATHS["MELD"])
    parser.add_argument("--iemocap_pkl", default=DEFAULT_PATHS["IEMOCAP"])
    parser.add_argument("--ejsl_pkl", default=DEFAULT_PATHS["eJSL_test"])
    parser.add_argument("--max_probe_per_class", type=int, default=1000)
    parser.add_argument("--max_mmd_samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--out_dir",
        default="/raid_zoe/home/lr/maokeyu/sign/bobsl_b5k_mechanism_analysis/visual_domain_geometry",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = {
        "BOBSL_train": args.bobsl_pkl,
        "MELD": args.meld_pkl,
        "IEMOCAP": args.iemocap_pkl,
        "eJSL_test": args.ejsl_pkl,
    }
    domains, metas = load_domains(paths)
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir).expanduser().resolve()

    count_rows: List[Dict[str, object]] = []
    for domain, (features, labels) in domains.items():
        counts = Counter(labels.tolist())
        count_rows.append(
            {
                "domain": domain,
                "n": len(labels),
                "dim": features.shape[1],
                **{f"count_{label}": counts[class_id] for class_id, label in enumerate(LABELS)},
            }
        )

    pairs = [
        ("BOBSL_train", "MELD_train"),
        ("BOBSL_train", "IEMOCAP_train"),
        ("BOBSL_train", "eJSL_test"),
        ("MELD_train", "eJSL_test"),
        ("IEMOCAP_train", "eJSL_test"),
    ]
    geometry_rows: List[Dict[str, object]] = []
    for left_name, right_name in pairs:
        left_x, left_y = domains[left_name]
        right_x, right_y = domains[right_name]
        probe_mean, probe_std, probe_n = domain_probe(
            domains[left_name], domains[right_name], args.max_probe_per_class, rng
        )
        for class_id, label in [(None, "ALL"), *enumerate(LABELS)]:
            if class_id is None:
                class_left = left_x
                class_right = right_x
            else:
                class_left = left_x[left_y == class_id]
                class_right = right_x[right_y == class_id]
            geometry_rows.append(
                {
                    "left": left_name,
                    "right": right_name,
                    "label": label,
                    "left_n": len(class_left),
                    "right_n": len(class_right),
                    "centroid_cosine_distance": cosine_distance(class_left.mean(axis=0), class_right.mean(axis=0)),
                    "rbf_mmd": rbf_mmd(class_left, class_right, rng, args.max_mmd_samples),
                    "class_balanced_domain_probe_accuracy": probe_mean if class_id is None else "",
                    "class_balanced_domain_probe_std": probe_std if class_id is None else "",
                    "domain_probe_n": probe_n if class_id is None else "",
                }
            )

    transfer_rows: List[Dict[str, object]] = []
    train_domains = ("BOBSL_train", "MELD_train", "IEMOCAP_train")
    test_domains = ("MELD_test", "IEMOCAP_test", "eJSL_test")
    for train_name in train_domains:
        for test_name in test_domains:
            accuracy, weighted_f1 = prototype_transfer(domains[train_name], domains[test_name])
            transfer_rows.append(
                {"train_domain": train_name, "test_domain": test_name, "accuracy": accuracy, "weighted_f1": weighted_f1}
            )

    model_meta = {
        name: {
            "video_model": meta.get("video_model"),
            "video_processor": meta.get("video_processor"),
            "visual_dim": meta.get("visual_dim"),
            "label_names": meta.get("label_names"),
        }
        for name, meta in metas.items()
    }
    write_csv(out_dir / "domain_counts.csv", count_rows)
    write_csv(out_dir / "pairwise_geometry.csv", geometry_rows)
    write_csv(out_dir / "prototype_transfer.csv", transfer_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "feature_metadata.json").write_text(json.dumps(model_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("[visual-geometry] class-balanced domain probe (0.5=indistinguishable)")
    for row in geometry_rows:
        if row["label"] == "ALL":
            print(
                f"  {row['left']} vs {row['right']}: acc={float(row['class_balanced_domain_probe_accuracy']):.4f} "
                f"mmd={float(row['rbf_mmd']):.4f} centroid_cos={float(row['centroid_cosine_distance']):.4f}"
            )
    print("[visual-geometry] nearest-prototype transfer")
    for row in transfer_rows:
        print(
            f"  {row['train_domain']} -> {row['test_domain']}: acc={float(row['accuracy']):.4f} "
            f"wf1={float(row['weighted_f1']):.4f}"
        )
    print(f"[visual-geometry] out_dir={out_dir}")


if __name__ == "__main__":
    main()
