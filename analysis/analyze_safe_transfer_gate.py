#!/usr/bin/env python3
"""Aggregate SafeTransferGate runs and create paper-ready diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


LABELS = ("A", "N", "J", "S")
SPLIT_FILES = {
    "source": ("source_test_best_summary.json", "source_test_best_predictions.csv"),
    "external": ("external_test_best_summary.json", "external_test_best_predictions.csv"),
}


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields: List[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_inputs(values: Sequence[str]) -> Dict[str, Path]:
    output: Dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=PATH, got {value!r}")
        name, path = value.split("=", 1)
        output[name] = Path(path).expanduser().resolve()
    return output


def mean_std(values: Iterable[float]) -> Tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=0))


def bootstrap_ci(values: Sequence[float], seed: int = 2026, repetitions: int = 20000) -> Tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 1:
        return float(array[0]), float(array[0])
    rng = np.random.default_rng(seed)
    means = array[rng.integers(0, len(array), size=(repetitions, len(array)))].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def exact_sign_p(values: Sequence[float]) -> float:
    signs = [value for value in values if abs(value) > 1e-12]
    n = len(signs)
    if n == 0:
        return 1.0
    positives = sum(value > 0 for value in signs)
    tail = min(positives, n - positives)
    probability = 2.0 * sum(math.comb(n, k) for k in range(tail + 1)) / (2**n)
    return min(1.0, probability)


def metric_row(gold: np.ndarray, pred: np.ndarray, prefix: str) -> Dict[str, float]:
    output = {
        f"{prefix}_accuracy": float(metrics.accuracy_score(gold, pred)),
        f"{prefix}_macro_f1": float(metrics.f1_score(gold, pred, labels=range(4), average="macro", zero_division=0)),
        f"{prefix}_weighted_f1": float(
            metrics.f1_score(gold, pred, labels=range(4), average="weighted", zero_division=0)
        ),
    }
    class_f1 = metrics.f1_score(gold, pred, labels=range(4), average=None, zero_division=0)
    for label, value in zip(LABELS, class_f1):
        output[f"{prefix}_{label}_f1"] = float(value)
    return output


def load_prediction(path: Path) -> Dict[str, np.ndarray]:
    rows = read_csv(path)
    output: Dict[str, np.ndarray] = {
        "gold": np.asarray([int(row["gold_id"]) for row in rows], dtype=np.int64),
    }
    for branch in ("source", "joint", "fused"):
        output[f"{branch}_pred"] = np.asarray([int(row[f"{branch}_pred"]) for row in rows], dtype=np.int64)
        output[f"{branch}_prob"] = np.asarray(
            [[float(row[f"{branch}_prob_{label}"]) for label in LABELS] for row in rows], dtype=np.float64
        )
    output["gate"] = np.asarray(
        [[float(row[f"gate_{label}"]) for label in LABELS] for row in rows], dtype=np.float64
    )
    return output


def run_identity(summary_path: Path) -> Tuple[str, str, int]:
    profile, seed = summary_path.parent.name.rsplit("_seed", 1)
    return summary_path.parts[-3], profile, int(seed)


def collect(inputs: Mapping[str, Path], profiles: set[str] | None):
    per_seed: List[Dict[str, object]] = []
    gate_rows: List[Dict[str, object]] = []
    disagreement_rows: List[Dict[str, object]] = []
    confusion: Dict[Tuple[str, str, str, str], np.ndarray] = defaultdict(lambda: np.zeros((4, 4), dtype=np.int64))
    for baseline, root in inputs.items():
        for split, (summary_name, prediction_name) in SPLIT_FILES.items():
            for summary_path in sorted(root.glob(f"*/*_seed*/{summary_name}")):
                corpus, profile, seed = run_identity(summary_path)
                if profiles and profile not in profiles:
                    continue
                metadata = json.loads(summary_path.read_text(encoding="utf-8"))
                prediction_path = summary_path.with_name(prediction_name)
                if not prediction_path.is_file():
                    continue
                data = load_prediction(prediction_path)
                gold = data["gold"]
                row: Dict[str, object] = {
                    "baseline": baseline,
                    "corpus": corpus,
                    "profile": profile,
                    "seed": seed,
                    "split": split,
                    "n": len(gold),
                    "mean_gate": float(data["gate"].mean()),
                    "best_epoch": int(metadata.get("best_epoch", 0)),
                    "source_fallback": float(bool(metadata.get("source_fallback", False))),
                    "boundary_best": float(
                        int(metadata.get("best_epoch", 0)) >= int(metadata.get("args", {}).get("epochs", 0))
                    ),
                }
                for branch in ("source", "joint", "fused"):
                    row.update(metric_row(gold, data[f"{branch}_pred"], branch))
                source_correct = data["source_pred"] == gold
                joint_correct = data["joint_pred"] == gold
                fused_correct = data["fused_pred"] == gold
                expert_oracle_pred = data["source_pred"].copy()
                expert_oracle_pred[~source_correct & joint_correct] = data["joint_pred"][~source_correct & joint_correct]
                row.update(metric_row(gold, expert_oracle_pred, "expert_oracle"))
                disagreement = data["source_pred"] != data["joint_pred"]
                row.update(
                    {
                        "delta_fused_source": row["fused_weighted_f1"] - row["source_weighted_f1"],
                        "delta_fused_joint": row["fused_weighted_f1"] - row["joint_weighted_f1"],
                        "expert_disagreement_rate": float(disagreement.mean()),
                        "joint_oracle_recovery_rate": float((~source_correct & joint_correct).mean()),
                        "source_oracle_recovery_rate": float((source_correct & ~joint_correct).mean()),
                        "fused_recovery_rate": float((~source_correct & fused_correct).sum() / max(1, (~source_correct).sum())),
                        "fused_damage_rate": float((source_correct & ~fused_correct).sum() / max(1, source_correct.sum())),
                    }
                )
                per_seed.append(row)
                for class_id, label in enumerate(LABELS):
                    mask = gold == class_id
                    if not mask.any():
                        continue
                    gate_rows.append(
                        {
                            "baseline": baseline,
                            "corpus": corpus,
                            "profile": profile,
                            "seed": seed,
                            "split": split,
                            "gold_label": label,
                            **{f"gate_{gate_label}": float(data["gate"][mask, index].mean()) for index, gate_label in enumerate(LABELS)},
                        }
                    )
                distance = 0.5 * np.abs(data["source_prob"] - data["joint_prob"]).sum(axis=1)
                edges = np.quantile(distance, np.linspace(0, 1, 6))
                edges[0], edges[-1] = -np.inf, np.inf
                for bin_index in range(5):
                    mask = (distance > edges[bin_index]) & (distance <= edges[bin_index + 1])
                    if not mask.any():
                        continue
                    disagreement_rows.append(
                        {
                            "baseline": baseline,
                            "corpus": corpus,
                            "profile": profile,
                            "seed": seed,
                            "split": split,
                            "distance_bin": bin_index + 1,
                            "n": int(mask.sum()),
                            "mean_tv_distance": float(distance[mask].mean()),
                            "mean_gate": float(data["gate"][mask].mean()),
                            "correctness_gain": float(fused_correct[mask].mean() - source_correct[mask].mean()),
                        }
                    )
                for branch in ("source", "joint", "fused"):
                    confusion[(baseline, corpus, profile, f"{split}_{branch}")] += metrics.confusion_matrix(
                        gold, data[f"{branch}_pred"], labels=range(4)
                    )
    return per_seed, gate_rows, disagreement_rows, confusion


def summarize(per_seed: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, str, str, str], List[Mapping[str, object]]] = defaultdict(list)
    for row in per_seed:
        groups[(str(row["baseline"]), str(row["corpus"]), str(row["profile"]), str(row["split"]))].append(row)
    output: List[Dict[str, object]] = []
    metrics_to_average = [
        "source_weighted_f1",
        "joint_weighted_f1",
        "fused_weighted_f1",
        "source_macro_f1",
        "joint_macro_f1",
        "fused_macro_f1",
        "delta_fused_source",
        "delta_fused_joint",
        "mean_gate",
        "expert_disagreement_rate",
        "joint_oracle_recovery_rate",
        "source_oracle_recovery_rate",
        "fused_recovery_rate",
        "fused_damage_rate",
        "expert_oracle_accuracy",
        "expert_oracle_macro_f1",
        "expert_oracle_weighted_f1",
        "best_epoch",
        "source_fallback",
        "boundary_best",
    ] + [f"{branch}_{label}_f1" for branch in ("source", "joint", "fused", "expert_oracle") for label in LABELS]
    for key, rows in sorted(groups.items()):
        result: Dict[str, object] = dict(zip(("baseline", "corpus", "profile", "split"), key))
        result["n_seeds"] = len(rows)
        for metric_name in metrics_to_average:
            values = [float(row[metric_name]) for row in rows]
            mean, std = mean_std(values)
            result[f"{metric_name}_mean"] = mean
            result[f"{metric_name}_std"] = std
        delta_values = [float(row["delta_fused_source"]) for row in rows]
        ci_low, ci_high = bootstrap_ci(delta_values)
        result["delta_source_ci95_low"] = ci_low
        result["delta_source_ci95_high"] = ci_high
        result["delta_source_positive_seeds"] = sum(value > 0 for value in delta_values)
        result["delta_source_sign_p"] = exact_sign_p(delta_values)
        output.append(result)
    return output


def aggregate_rows(rows: Sequence[Mapping[str, object]], keys: Sequence[str], values: Sequence[str]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, ...], List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row[key]) for key in keys)].append(row)
    output = []
    for key, group in sorted(groups.items()):
        result: Dict[str, object] = dict(zip(keys, key))
        result["n_seeds"] = len({str(row["seed"]) for row in group})
        for value in values:
            result[value] = float(np.mean([float(row[value]) for row in group]))
        output.append(result)
    return output


def save_figure(fig: plt.Figure, out_dir: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_delta_overview(summary: Sequence[Mapping[str, object]], out_dir: Path) -> None:
    rows = list(summary)
    labels = [f"{row['baseline']}\n{row['corpus']}\n{row['profile']}" for row in rows]
    source = [float(row["delta_fused_source_mean"]) for row in rows]
    source_err = [float(row["delta_fused_source_std"]) for row in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 0.72), 4.8))
    colors = ["#2a6fbb" if row["split"] == "source" else "#d1495b" for row in rows]
    ax.bar(x, source, yerr=source_err, color=colors, alpha=0.85, capsize=3)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Weighted-F1 gain over source expert")
    ax.set_xticks(x, labels, rotation=40, ha="right")
    ax.set_title("Safe transfer gain: source domain and eJSL")
    save_figure(fig, out_dir, "delta_overview")


def plot_seed_pairs(per_seed: Sequence[Mapping[str, object]], out_dir: Path) -> None:
    source_rows = [row for row in per_seed if row["split"] == "source"]
    groups: Dict[Tuple[str, str, str], List[Mapping[str, object]]] = defaultdict(list)
    for row in source_rows:
        groups[(str(row["baseline"]), str(row["corpus"]), str(row["profile"]))].append(row)
    fig, axes = plt.subplots(1, len(groups), figsize=(max(6, 3.2 * len(groups)), 4.4), squeeze=False)
    for ax, (key, rows) in zip(axes[0], sorted(groups.items())):
        for row in sorted(rows, key=lambda value: int(value["seed"])):
            ax.plot([0, 1], [row["source_weighted_f1"], row["fused_weighted_f1"]], marker="o", alpha=0.75)
        ax.set_xticks([0, 1], ["Source", "Fused"])
        ax.set_title(" / ".join(key))
        ax.set_ylabel("Weighted F1")
        ax.grid(axis="y", alpha=0.2)
    save_figure(fig, out_dir, "paired_seed_source_f1")


def plot_gate_heatmap(gate_rows: Sequence[Mapping[str, object]], out_dir: Path) -> None:
    aggregated = aggregate_rows(gate_rows, ("baseline", "corpus", "profile", "split", "gold_label"), tuple(f"gate_{x}" for x in LABELS))
    panels = sorted({(row["baseline"], row["corpus"], row["profile"], row["split"]) for row in aggregated})
    fig, axes = plt.subplots(1, len(panels), figsize=(max(6, 3.6 * len(panels)), 4.2), squeeze=False)
    for ax, panel in zip(axes[0], panels):
        selected = {row["gold_label"]: row for row in aggregated if tuple(row[key] for key in ("baseline", "corpus", "profile", "split")) == panel}
        matrix = np.asarray([[selected[label][f"gate_{gate_label}"] for gate_label in LABELS] for label in LABELS])
        image = ax.imshow(matrix, vmin=0, vmax=max(0.5, float(matrix.max())), cmap="YlGnBu")
        ax.set_xticks(range(4), LABELS)
        ax.set_yticks(range(4), LABELS)
        ax.set_xlabel("Transferred logit class")
        ax.set_ylabel("Gold class")
        ax.set_title(" / ".join(panel))
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    save_figure(fig, out_dir, "gate_by_gold_class")


def plot_recovery_damage(summary: Sequence[Mapping[str, object]], out_dir: Path) -> None:
    source_rows = [row for row in summary if row["split"] == "source"]
    labels = [f"{row['baseline']}\n{row['corpus']}\n{row['profile']}" for row in source_rows]
    recovery = [float(row["fused_recovery_rate_mean"]) for row in source_rows]
    damage = [float(row["fused_damage_rate_mean"]) for row in source_rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.9), 4.8))
    ax.bar(x - 0.18, recovery, 0.36, label="Recovers source errors", color="#2a9d8f")
    ax.bar(x + 0.18, damage, 0.36, label="Damages source correct", color="#e76f51")
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylabel("Conditional sample rate")
    ax.legend(frameon=False)
    ax.set_title("Selective transfer: recovery versus damage")
    save_figure(fig, out_dir, "recovery_damage")


def plot_disagreement(rows: Sequence[Mapping[str, object]], out_dir: Path) -> None:
    aggregated = aggregate_rows(rows, ("baseline", "corpus", "profile", "split", "distance_bin"), ("mean_tv_distance", "mean_gate", "correctness_gain"))
    panels = sorted({(row["baseline"], row["corpus"], row["profile"], row["split"]) for row in aggregated})
    fig, axes = plt.subplots(1, len(panels), figsize=(max(7, 3.7 * len(panels)), 4.2), squeeze=False)
    for ax, panel in zip(axes[0], panels):
        selected = sorted(
            [row for row in aggregated if tuple(row[key] for key in ("baseline", "corpus", "profile", "split")) == panel],
            key=lambda row: int(row["distance_bin"]),
        )
        x = [int(row["distance_bin"]) for row in selected]
        ax.plot(x, [row["correctness_gain"] for row in selected], marker="o", label="Accuracy gain")
        ax.plot(x, [row["mean_gate"] for row in selected], marker="s", label="Mean gate")
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_xticks(range(1, 6))
        ax.set_xlabel("Expert disagreement quintile")
        ax.set_title(" / ".join(panel))
        ax.legend(frameon=False, fontsize=8)
    save_figure(fig, out_dir, "gain_by_expert_disagreement")


def write_confusions(out_dir: Path, confusion: Mapping[Tuple[str, str, str, str], np.ndarray]) -> None:
    directory = out_dir / "confusion_matrices"
    directory.mkdir(exist_ok=True)
    for key, matrix in confusion.items():
        name = "__".join(key)
        np.savetxt(directory / f"{name}.csv", matrix, fmt="%d", delimiter=",", header=",".join(LABELS), comments="")


def write_report(path: Path, summary: Sequence[Mapping[str, object]]) -> None:
    lines = [
        "# SafeTransferGate Analysis",
        "",
        "All means are paired over seeds. Check each run's gate_split.json for its checkpoint-selection protocol; external selection is oracle-only.",
        "",
        "| Backbone | Corpus | Profile | Split | n | Source | Joint | Fused | Expert oracle | Delta | 95% CI | Positive seeds | Sign p | Recovery | Damage |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['baseline']} | {row['corpus']} | {row['profile']} | {row['split']} | {row['n_seeds']} | "
            f"{float(row['source_weighted_f1_mean']):.4f} | {float(row['joint_weighted_f1_mean']):.4f} | "
            f"{float(row['fused_weighted_f1_mean']):.4f} | {float(row['expert_oracle_weighted_f1_mean']):.4f} | "
            f"{float(row['delta_fused_source_mean']):+.4f} | "
            f"[{float(row['delta_source_ci95_low']):+.4f}, {float(row['delta_source_ci95_high']):+.4f}] | "
            f"{row['delta_source_positive_seeds']}/{row['n_seeds']} | {float(row['delta_source_sign_p']):.4f} | "
            f"{float(row['fused_recovery_rate_mean']):.3f} | {float(row['fused_damage_rate_mean']):.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, help="NAME=run_root; repeat for each backbone")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--profiles", nargs="*", default=None)
    args = parser.parse_args()
    inputs = parse_inputs(args.input)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_profiles = set(args.profiles) if args.profiles else None
    per_seed, gate_rows, disagreement_rows, confusion = collect(inputs, selected_profiles)
    if not per_seed:
        raise RuntimeError("No completed SafeTransferGate prediction files were found")
    summary = summarize(per_seed)
    gate_summary = aggregate_rows(gate_rows, ("baseline", "corpus", "profile", "split", "gold_label"), tuple(f"gate_{x}" for x in LABELS))
    disagreement_summary = aggregate_rows(
        disagreement_rows,
        ("baseline", "corpus", "profile", "split", "distance_bin"),
        ("mean_tv_distance", "mean_gate", "correctness_gain"),
    )
    write_csv(out_dir / "per_seed_metrics.csv", per_seed)
    write_csv(out_dir / "group_summary.csv", summary)
    write_csv(out_dir / "gate_by_class.csv", gate_summary)
    write_csv(out_dir / "disagreement_bins.csv", disagreement_summary)
    write_confusions(out_dir, confusion)
    write_report(out_dir / "analysis_report.md", summary)
    plot_delta_overview(summary, out_dir)
    plot_seed_pairs(per_seed, out_dir)
    plot_gate_heatmap(gate_rows, out_dir)
    plot_recovery_damage(summary, out_dir)
    plot_disagreement(disagreement_rows, out_dir)
    print(f"[safe-analysis] wrote {out_dir}")


if __name__ == "__main__":
    main()
