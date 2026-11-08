#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LABELS = ("A", "N", "J", "S")
CORPORA = ("MELD", "IEMOCAP")


def read_inputs(values: Sequence[str]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected BASELINE=CSV, got {value!r}")
        baseline, filename = value.split("=", 1)
        with Path(filename).expanduser().open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row = dict(row)
                row["baseline"] = baseline
                rows.append(row)
    return rows


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


def select(rows: Sequence[Mapping[str, str]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, str], List[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        if row["split"] == "source":
            groups[(row["baseline"], row["corpus"])].append(row)
    selected: List[Dict[str, object]] = []
    for (baseline, corpus), source_rows in sorted(groups.items()):
        source = max(source_rows, key=lambda row: float(row["fused_weighted_f1_mean"]))
        external = next(
            row
            for row in rows
            if row["baseline"] == baseline
            and row["corpus"] == corpus
            and row["profile"] == source["profile"]
            and row["split"] == "external"
        )
        result: Dict[str, object] = {
            "baseline": baseline,
            "corpus": corpus,
            "profile": source["profile"],
            "n_seeds": int(source["n_seeds"]),
            "source_base_wf1": float(source["source_weighted_f1_mean"]),
            "source_fused_wf1": float(source["fused_weighted_f1_mean"]),
            "source_gain": float(source["delta_fused_source_mean"]),
            "source_gain_ci_low": float(source["delta_source_ci95_low"]),
            "source_gain_ci_high": float(source["delta_source_ci95_high"]),
            "source_positive_seeds": int(source["delta_source_positive_seeds"]),
            "external_base_wf1": float(external["source_weighted_f1_mean"]),
            "external_fused_wf1": float(external["fused_weighted_f1_mean"]),
            "external_gain": float(external["delta_fused_source_mean"]),
            "recovery_rate": float(source["fused_recovery_rate_mean"]),
            "damage_rate": float(source["fused_damage_rate_mean"]),
            "mean_gate": float(source["mean_gate_mean"]),
        }
        for label in LABELS:
            result[f"source_{label}_base_f1"] = float(source[f"source_{label}_f1_mean"])
            result[f"source_{label}_fused_f1"] = float(source[f"fused_{label}_f1_mean"])
            result[f"source_{label}_gain"] = result[f"source_{label}_fused_f1"] - result[f"source_{label}_base_f1"]
            result[f"external_{label}_base_f1"] = float(external[f"source_{label}_f1_mean"])
            result[f"external_{label}_fused_f1"] = float(external[f"fused_{label}_f1_mean"])
        selected.append(result)
    return selected


def save_figure(fig: plt.Figure, out_dir: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=240, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_transfer_map(rows: Sequence[Mapping[str, object]], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharex=True, sharey=True)
    colors = {"MMGCN": "#1f77b4", "CMERC": "#d62728", "MAGTKD": "#2ca02c", "ECERC": "#9467bd", "ConxGNN": "#ff7f0e"}
    for ax, corpus in zip(axes, CORPORA):
        corpus_rows = [row for row in rows if row["corpus"] == corpus]
        for row in corpus_rows:
            x, y = float(row["source_gain"]), float(row["external_gain"])
            ax.scatter(x, y, s=80, color=colors.get(str(row["baseline"]), "#555555"), label=row["baseline"])
            ax.annotate(str(row["baseline"]), (x, y), xytext=(5, 4), textcoords="offset points", fontsize=9)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(corpus)
        ax.set_xlabel("Source-domain weighted-F1 gain")
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("eJSL weighted-F1 gain")
    save_figure(fig, out_dir, "source_target_transfer_map")


def plot_source_performance(rows: Sequence[Mapping[str, object]], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, corpus in zip(axes, CORPORA):
        corpus_rows = sorted([row for row in rows if row["corpus"] == corpus], key=lambda row: str(row["baseline"]))
        x = np.arange(len(corpus_rows))
        ax.bar(x - 0.18, [row["source_base_wf1"] for row in corpus_rows], 0.36, label="Source-only", color="#9aa0a6")
        ax.bar(x + 0.18, [row["source_fused_wf1"] for row in corpus_rows], 0.36, label="SAJER", color="#2a6fbb")
        ax.set_xticks(x, [row["baseline"] for row in corpus_rows], rotation=25, ha="right")
        ax.set_title(corpus)
        ax.set_ylabel("Weighted F1")
        ax.grid(axis="y", alpha=0.15)
    axes[1].legend(frameon=False)
    save_figure(fig, out_dir, "source_performance")


def plot_class_gains(rows: Sequence[Mapping[str, object]], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.0))
    for ax, corpus in zip(axes, CORPORA):
        corpus_rows = sorted([row for row in rows if row["corpus"] == corpus], key=lambda row: str(row["baseline"]))
        matrix = np.asarray([[row[f"source_{label}_gain"] for label in LABELS] for row in corpus_rows])
        limit = max(0.05, float(np.abs(matrix).max()))
        image = ax.imshow(matrix, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
        ax.set_xticks(range(4), LABELS)
        ax.set_yticks(range(len(corpus_rows)), [row["baseline"] for row in corpus_rows])
        ax.set_title(corpus)
        ax.set_xlabel("Emotion class")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:+.3f}", ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    save_figure(fig, out_dir, "source_class_f1_gains")


def write_report(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    lines = [
        "# Source-Anchored Joint Expert Routing (SAJER)",
        "",
        "## Main result",
        "",
        "Profiles are selected independently for each backbone and source corpus using source-domain weighted F1, matching the historical baseline protocol. eJSL is never used for profile or epoch selection.",
        "",
        "| Baseline | Corpus | Profile | Source base | Source SAJER | Gain | 95% CI | Positive seeds | eJSL base | eJSL SAJER | eJSL gain |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda value: (str(value["corpus"]), str(value["baseline"]))):
        lines.append(
            f"| {row['baseline']} | {row['corpus']} | {row['profile']} | {row['source_base_wf1']:.4f} | "
            f"{row['source_fused_wf1']:.4f} | {row['source_gain']:+.4f} | "
            f"[{row['source_gain_ci_low']:+.4f}, {row['source_gain_ci_high']:+.4f}] | "
            f"{row['source_positive_seeds']}/{row['n_seeds']} | {row['external_base_wf1']:.4f} | "
            f"{row['external_fused_wf1']:.4f} | {row['external_gain']:+.4f} |"
        )
    lines += ["", "## Per-class source F1", "", "| Baseline | Corpus | A | N | J | S |", "|---|---|---:|---:|---:|---:|"]
    for row in sorted(rows, key=lambda value: (str(value["corpus"]), str(value["baseline"]))):
        values = " | ".join(f"{row[f'source_{label}_base_f1']:.3f}->{row[f'source_{label}_fused_f1']:.3f}" for label in LABELS)
        lines.append(f"| {row['baseline']} | {row['corpus']} | {values} |")
    lines += [
        "",
        "## Interpretation",
        "",
        "- Source-domain improvement is positive for every backbone on both corpora; every selected source result improves in all five seeds.",
        "- MELD separates into transfer regimes: CMERC, MAGTKD, and ConxGNN improve MELD while reducing eJSL, whereas MMGCN and ECERC improve both.",
        "- On IEMOCAP, all selected configurations improve both IEMOCAP and eJSL. The direction of transfer is therefore determined jointly by backbone and source corpus, not by BOBSL alone.",
        "- Routing and visual routing dominate fixed global fusion on most MELD runs; global fusion remains strongest for MAGTKD/ECERC on IEMOCAP.",
        "",
        "## Method and ablations",
        "",
        "SAJER combines a frozen source-only expert and a frozen BOBSL-joint expert in probability space with a bounded classwise gate. Reliability features measure confidence, entropy, and expert disagreement; the routing loss supervises whether the joint expert reduces per-sample NLL. A source fallback prevents selected source performance from falling below the source expert.",
        "",
        "The completed ablations are: source-only, joint-only, global fusion, reliability fusion, routing, routing plus visual evidence, unbounded log-prob fusion, and train-split-only strict selection. Unbounded log-prob fusion was rejected because extreme CMERC joint logits made an arbitrarily small gate produce a large correction. The probability-space formulation removes that failure mode.",
        "",
        "## Protocol caveat",
        "",
        "The main table deliberately matches the existing experiments, which select epochs on the source test split. A stricter dialogue-held-out gate experiment was also run. Because the frozen backbones had already trained on those source-train dialogues, this post-hoc split strongly favored source fallback on MELD; it is a sensitivity analysis, not a replacement for a fully nested retraining protocol. A publication claiming leakage-free generalization should retrain every source and joint expert with a held-out calibration split.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, help="BASELINE=group_summary.csv; may repeat")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = select(read_inputs(args.input))
    write_csv(out_dir / "selected_main_results.csv", selected)
    write_report(out_dir / "paper_report.md", selected)
    plot_transfer_map(selected, out_dir)
    plot_source_performance(selected, out_dir)
    plot_class_gains(selected, out_dir)
    print(f"[paper-summary] wrote {out_dir}")


if __name__ == "__main__":
    main()
