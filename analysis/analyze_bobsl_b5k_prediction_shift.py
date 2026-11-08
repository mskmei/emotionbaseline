#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


LABELS = ("A", "N", "J", "S")
SEEDS = (35, 36, 37, 41, 42)
BASELINES = ("MMGCN", "MAGTKD", "ECERC", "CMERC", "ConxGNN")
ROOTS = {
    "MELD": {
        "MMGCN": "/raid_zoe/home/lr/maokeyu/sign/mmgcn_bobsl_meld_ejsl/video_joint_bobsl_meld",
        "MAGTKD": "/raid_zoe/home/lr/maokeyu/sign/magtkd_bobsl_meld_ejsl/video_joint_bobsl_meld",
        "ECERC": "/raid_zoe/home/lr/maokeyu/sign/ecerc_bobsl_meld_ejsl/video_joint_bobsl_meld",
        "CMERC": "/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_meld_ejsl/visual_joint_bobsl_meld",
        "ConxGNN": "/raid_zoe/home/lr/maokeyu/sign/conxgnn_bobsl_meld_ejsl/visual_joint_bobsl_meld",
    },
    "IEMOCAP": {
        "MMGCN": "/raid_zoe/home/lr/maokeyu/sign/mmgcn_bobsl_iemocap_ejsl/video_joint_bobsl_iemocap",
        "MAGTKD": "/raid_zoe/home/lr/maokeyu/sign/magtkd_bobsl_iemocap_ejsl/video_joint_bobsl_iemocap",
        "ECERC": "/raid_zoe/home/lr/maokeyu/sign/ecerc_bobsl_iemocap_ejsl/video_joint_bobsl_iemocap",
        "CMERC": "/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_iemocap_ejsl/video_joint_bobsl_iemocap",
        "ConxGNN": "/raid_zoe/home/lr/maokeyu/sign/conxgnn_bobsl_iemocap_ejsl/video_joint_bobsl_iemocap",
    },
}


def safe_log(values: np.ndarray) -> np.ndarray:
    return np.log(np.clip(values, 1e-12, 1.0))


def division(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def confusion(gold: np.ndarray, pred: np.ndarray) -> np.ndarray:
    result = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    np.add.at(result, (gold, pred), 1)
    return result


def classification_metrics(gold: np.ndarray, pred: np.ndarray) -> Dict[str, object]:
    matrix = confusion(gold, pred)
    support = matrix.sum(axis=1)
    f1s: List[float] = []
    recalls: List[float] = []
    precisions: List[float] = []
    for class_id in range(len(LABELS)):
        tp = float(matrix[class_id, class_id])
        fp = float(matrix[:, class_id].sum() - tp)
        fn = float(matrix[class_id, :].sum() - tp)
        precision = division(tp, tp + fp)
        recall = division(tp, tp + fn)
        f1 = division(2.0 * precision * recall, precision + recall)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    return {
        "accuracy": float(np.mean(gold == pred)),
        "macro_f1": float(np.mean(f1s)),
        "weighted_f1": division(float(np.dot(support, f1s)), float(support.sum())),
        "precision": np.asarray(precisions),
        "recall": np.asarray(recalls),
        "f1": np.asarray(f1s),
        "support": support,
        "confusion": matrix,
    }


def expected_calibration_error(gold: np.ndarray, probs: np.ndarray, bins: int = 15) -> float:
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == gold
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        if index == bins - 1:
            keep = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            keep = (confidence >= edges[index]) & (confidence < edges[index + 1])
        if np.any(keep):
            result += float(np.mean(keep)) * abs(float(np.mean(correct[keep])) - float(np.mean(confidence[keep])))
    return result


def probability_metrics(gold: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    finite = np.all(np.isfinite(probs), axis=1)
    gold = gold[finite]
    probs = probs[finite]
    if not len(gold):
        return {metric: float("nan") for metric in ("nll", "brier", "entropy", "confidence", "gold_probability", "ece")}
    one_hot = np.eye(len(LABELS), dtype=np.float64)[gold]
    gold_prob = probs[np.arange(len(gold)), gold]
    return {
        "nll": float(-np.mean(safe_log(gold_prob))),
        "brier": float(np.mean(np.sum((probs - one_hot) ** 2, axis=1))),
        "entropy": float(np.mean(-np.sum(probs * safe_log(probs), axis=1))),
        "confidence": float(np.mean(probs.max(axis=1))),
        "gold_probability": float(np.mean(gold_prob)),
        "ece": expected_calibration_error(gold, probs),
    }


def js_divergence(left: np.ndarray, right: np.ndarray, axis: int = -1) -> np.ndarray:
    middle = 0.5 * (left + right)
    return 0.5 * np.sum(left * (safe_log(left) - safe_log(middle)), axis=axis) + 0.5 * np.sum(
        right * (safe_log(right) - safe_log(middle)), axis=axis
    )


def read_predictions(path: Path) -> Dict[str, Tuple[int, int, np.ndarray]]:
    rows: Dict[str, Tuple[int, int, np.ndarray]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            item_id = row["item_id"]
            rows[item_id] = (
                int(row["gold_id"]),
                int(row["pred_id"]),
                np.asarray([float(row[f"prob_{label}"]) for label in LABELS], dtype=np.float64),
            )
    return rows


def prediction_path(root: Path, baseline: str, config: str, seed: int, split: str) -> Path:
    trial = root / f"{config}_seed{seed}"
    if baseline == "MMGCN":
        trial = trial / "video"
    return trial / f"{split}_test_best_predictions.csv"


def paired_predictions(base_path: Path, joint_path: Path) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base = read_predictions(base_path)
    joint = read_predictions(joint_path)
    if set(base) != set(joint):
        missing_joint = sorted(set(base) - set(joint))[:5]
        missing_base = sorted(set(joint) - set(base))[:5]
        raise ValueError(f"Prediction IDs differ: missing_joint={missing_joint}, missing_base={missing_base}")
    ids = sorted(base)
    base_gold = np.asarray([base[item_id][0] for item_id in ids], dtype=np.int64)
    joint_gold = np.asarray([joint[item_id][0] for item_id in ids], dtype=np.int64)
    if not np.array_equal(base_gold, joint_gold):
        raise ValueError(f"Gold labels differ between {base_path} and {joint_path}")
    base_probs = np.stack([base[item_id][2] for item_id in ids])
    joint_probs = np.stack([joint[item_id][2] for item_id in ids])
    return ids, base_gold, base_probs.argmax(axis=1), base_probs, joint_probs


def mean_std(values: Iterable[float]) -> Tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=0))


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


def paired_rows() -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    overall_rows: List[Dict[str, object]] = []
    class_rows: List[Dict[str, object]] = []
    transition_rows: List[Dict[str, object]] = []
    for corpus, roots in ROOTS.items():
        baseline_config = "meld_only" if corpus == "MELD" else "iemocap_only"
        for baseline in BASELINES:
            root = Path(roots[baseline])
            for seed in SEEDS:
                for split in ("source", "external"):
                    base_path = prediction_path(root, baseline, baseline_config, seed, split)
                    joint_path = prediction_path(root, baseline, "joint_b5k", seed, split)
                    if not base_path.is_file() or not joint_path.is_file():
                        print(f"[prediction-shift] missing pair: {base_path} | {joint_path}")
                        continue
                    _, gold, base_pred, base_probs, joint_probs = paired_predictions(base_path, joint_path)
                    joint_pred = joint_probs.argmax(axis=1)
                    base_cls = classification_metrics(gold, base_pred)
                    joint_cls = classification_metrics(gold, joint_pred)
                    base_prob = probability_metrics(gold, base_probs)
                    joint_prob = probability_metrics(gold, joint_probs)
                    base_correct = base_pred == gold
                    joint_correct = joint_pred == gold
                    common = {
                        "corpus": corpus,
                        "baseline": baseline,
                        "seed": seed,
                        "split": "eJSL" if split == "external" else corpus,
                        "n": len(gold),
                    }
                    row: Dict[str, object] = dict(common)
                    for metric in ("accuracy", "macro_f1", "weighted_f1"):
                        row[f"base_{metric}"] = base_cls[metric]
                        row[f"joint_{metric}"] = joint_cls[metric]
                        row[f"delta_{metric}"] = float(joint_cls[metric]) - float(base_cls[metric])
                    for metric in ("nll", "brier", "entropy", "confidence", "gold_probability", "ece"):
                        row[f"base_{metric}"] = base_prob[metric]
                        row[f"joint_{metric}"] = joint_prob[metric]
                        row[f"delta_{metric}"] = joint_prob[metric] - base_prob[metric]
                    row.update(
                        {
                            "base_invalid_probability_rate": float(np.mean(~np.all(np.isfinite(base_probs), axis=1))),
                            "joint_invalid_probability_rate": float(np.mean(~np.all(np.isfinite(joint_probs), axis=1))),
                            "disagreement_rate": float(np.mean(base_pred != joint_pred)),
                            "fixed_rate": float(np.mean(~base_correct & joint_correct)),
                            "broken_rate": float(np.mean(base_correct & ~joint_correct)),
                            "net_correct_rate": float(np.mean(joint_correct) - np.mean(base_correct)),
                            "mean_output_js": float(np.nanmean(js_divergence(base_probs, joint_probs))),
                        }
                    )
                    overall_rows.append(row)

                    for class_id, label in enumerate(LABELS):
                        keep = gold == class_id
                        finite_pair = keep & np.isfinite(base_probs[:, class_id]) & np.isfinite(joint_probs[:, class_id])
                        class_rows.append(
                            {
                                **common,
                                "label": label,
                                "support": int(np.sum(keep)),
                                "base_f1": float(base_cls["f1"][class_id]),
                                "joint_f1": float(joint_cls["f1"][class_id]),
                                "delta_f1": float(joint_cls["f1"][class_id] - base_cls["f1"][class_id]),
                                "base_recall": float(base_cls["recall"][class_id]),
                                "joint_recall": float(joint_cls["recall"][class_id]),
                                "delta_recall": float(joint_cls["recall"][class_id] - base_cls["recall"][class_id]),
                                "base_pred_prior": float(np.mean(base_pred == class_id)),
                                "joint_pred_prior": float(np.mean(joint_pred == class_id)),
                                "delta_pred_prior": float(np.mean(joint_pred == class_id) - np.mean(base_pred == class_id)),
                                "gold_prior": float(np.mean(keep)),
                                "base_gold_probability": float(np.mean(base_probs[finite_pair, class_id])),
                                "joint_gold_probability": float(np.mean(joint_probs[finite_pair, class_id])),
                                "delta_gold_probability": float(
                                    np.mean(joint_probs[finite_pair, class_id] - base_probs[finite_pair, class_id])
                                ),
                                "fixed_rate": float(np.mean((~base_correct & joint_correct)[keep])),
                                "broken_rate": float(np.mean((base_correct & ~joint_correct)[keep])),
                            }
                        )

                    transition = confusion(base_pred, joint_pred)
                    for old_id, old_label in enumerate(LABELS):
                        for new_id, new_label in enumerate(LABELS):
                            transition_rows.append(
                                {
                                    **common,
                                    "base_prediction": old_label,
                                    "joint_prediction": new_label,
                                    "count": int(transition[old_id, new_id]),
                                    "rate_all": float(transition[old_id, new_id] / len(gold)),
                                    "rate_given_base_prediction": division(
                                        float(transition[old_id, new_id]), float(transition[old_id].sum())
                                    ),
                                }
                            )
    return overall_rows, class_rows, transition_rows


def aggregate(rows: Sequence[Mapping[str, object]], keys: Sequence[str], metrics: Sequence[str]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[object, ...], List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    output: List[Dict[str, object]] = []
    for group_key, group in sorted(groups.items()):
        result: Dict[str, object] = dict(zip(keys, group_key))
        result["n_seeds"] = len(group)
        for metric in metrics:
            mean, std = mean_std(float(row[metric]) for row in group)
            result[f"{metric}_mean"] = mean
            result[f"{metric}_std"] = std
        output.append(result)
    return output


def fmt(value: object) -> str:
    return f"{float(value):+.4f}"


def report(overall: Sequence[Mapping[str, object]], classes: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "# BOBSL b5k prediction-shift audit",
        "",
        "All values are paired joint_b5k minus source-only means over five seeds.",
        "",
        "| corpus | baseline | split | delta wF1 | delta macro F1 | delta NLL | delta ECE | invalid joint | fixed | broken | disagreement | output JS |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overall:
        lines.append(
            f"| {row['corpus']} | {row['baseline']} | {row['split']} | {fmt(row['delta_weighted_f1_mean'])} | "
            f"{fmt(row['delta_macro_f1_mean'])} | {fmt(row['delta_nll_mean'])} | {fmt(row['delta_ece_mean'])} | "
            f"{float(row['joint_invalid_probability_rate_mean']):.4f} | {float(row['fixed_rate_mean']):.4f} | "
            f"{float(row['broken_rate_mean']):.4f} | "
            f"{float(row['disagreement_rate_mean']):.4f} | {float(row['mean_output_js_mean']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Per-class shifts",
            "",
            "| corpus | baseline | split | label | delta F1 | delta recall | delta prediction prior | delta gold probability | fixed | broken |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in classes:
        lines.append(
            f"| {row['corpus']} | {row['baseline']} | {row['split']} | {row['label']} | {fmt(row['delta_f1_mean'])} | "
            f"{fmt(row['delta_recall_mean'])} | {fmt(row['delta_pred_prior_mean'])} | "
            f"{fmt(row['delta_gold_probability_mean'])} | {float(row['fixed_rate_mean']):.4f} | "
            f"{float(row['broken_rate_mean']):.4f} |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit paired prediction changes caused by BOBSL b5k joint training.")
    parser.add_argument(
        "--out_dir",
        default="/raid_zoe/home/lr/maokeyu/sign/bobsl_b5k_mechanism_analysis/prediction_shift",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    overall_rows, class_rows, transition_rows = paired_rows()
    overall_metrics = [
        "delta_accuracy",
        "delta_macro_f1",
        "delta_weighted_f1",
        "delta_nll",
        "delta_brier",
        "delta_entropy",
        "delta_confidence",
        "delta_gold_probability",
        "delta_ece",
        "base_invalid_probability_rate",
        "joint_invalid_probability_rate",
        "disagreement_rate",
        "fixed_rate",
        "broken_rate",
        "net_correct_rate",
        "mean_output_js",
    ]
    class_metrics = [
        "delta_f1",
        "delta_recall",
        "delta_pred_prior",
        "delta_gold_probability",
        "fixed_rate",
        "broken_rate",
    ]
    transition_metrics = ["count", "rate_all", "rate_given_base_prediction"]
    overall_means = aggregate(overall_rows, ("corpus", "baseline", "split"), overall_metrics)
    class_means = aggregate(class_rows, ("corpus", "baseline", "split", "label"), class_metrics)
    transition_means = aggregate(
        transition_rows,
        ("corpus", "baseline", "split", "base_prediction", "joint_prediction"),
        transition_metrics,
    )
    write_csv(out_dir / "paired_seed_overall.csv", overall_rows)
    write_csv(out_dir / "paired_seed_per_class.csv", class_rows)
    write_csv(out_dir / "paired_seed_transitions.csv", transition_rows)
    write_csv(out_dir / "mean_overall.csv", overall_means)
    write_csv(out_dir / "mean_per_class.csv", class_means)
    write_csv(out_dir / "mean_transitions.csv", transition_means)
    report_text = report(overall_means, class_means)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text(report_text, encoding="utf-8")
    print(report_text, end="")
    print(f"[prediction-shift] out_dir={out_dir}")


if __name__ == "__main__":
    main()
