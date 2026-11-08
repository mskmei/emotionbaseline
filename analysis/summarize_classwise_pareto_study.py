#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np


BASELINES = ("MMGCN", "CMERC", "MAGTKD", "ECERC", "ConxGNN")
CORPORA = ("MELD", "IEMOCAP")
LABELS = ("A", "N", "J", "S")


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


def collect_search_rows(root: Path) -> List[Dict[str, str]]:
    rows = []
    for baseline in BASELINES:
        path = root / baseline / "best_classwise_mixture.csv"
        if path.is_file():
            rows.extend(read_csv(path))
    return rows


def collect_named_roots(root: Path) -> Dict[str, List[Dict[str, str]]]:
    output = {}
    if not root.is_dir():
        return output
    for child in sorted(path for path in root.iterdir() if path.is_dir()):
        rows = collect_search_rows(child)
        if rows:
            output[child.name] = rows
    return output


def summarize_search(name: str, rows: Sequence[Mapping[str, str]]) -> Dict[str, object]:
    return {
        "setting": name,
        "n_items": len(rows),
        "heldout_pass": sum(row.get("heldout_strict_pareto", "False") == "True" for row in rows),
        "source_delta_mean": float(np.mean([float(row["source_delta"]) for row in rows])),
        "heldout_delta_mean": float(np.mean([float(row["heldout_external_delta"]) for row in rows])),
        "heldout_delta_min": float(np.min([float(row["heldout_external_delta"]) for row in rows])),
        "heldout_delta_max": float(np.max([float(row["heldout_external_delta"]) for row in rows])),
    }


def load_sweep_inputs(values: Sequence[str]):
    output = {}
    for value in values:
        name, path = value.split("=", 1)
        output[name] = read_csv(Path(path).expanduser())
    return output


def best_sweep_rows(sweeps: Mapping[str, Sequence[Mapping[str, str]]]):
    output = []
    for branch, rows in sweeps.items():
        groups = defaultdict(list)
        for row in rows:
            if row.get("strict_pareto", "False") == "True":
                groups[(row["baseline"], row["corpus"])].append(row)
        for (baseline, corpus), group in groups.items():
            best = max(group, key=lambda row: float(row["external_delta"]))
            output.append(
                {
                    "branch": branch,
                    "baseline": baseline,
                    "corpus": corpus,
                    "config": best["config"],
                    "source_delta": float(best["source_delta"]),
                    "external_delta": float(best["external_delta"]),
                }
            )
    return output


def make_figures(
    out_dir: Path,
    final_rows: Sequence[Mapping[str, str]],
    heldout_rows: Sequence[Mapping[str, str]],
    ablation_summary: Sequence[Mapping[str, object]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source_rows = {
        (row["baseline"], row["corpus"]): row
        for row in final_rows
        if row["split"] == "source"
    }
    external_rows = {
        (row["baseline"], row["corpus"]): row
        for row in final_rows
        if row["split"] == "external_heldout"
    }
    colors = {"MELD": "#1976d2", "IEMOCAP": "#d95f02"}
    markers = {baseline: marker for baseline, marker in zip(BASELINES, ("o", "s", "^", "D", "P"))}

    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    for baseline in BASELINES:
        for corpus in CORPORA:
            key = (baseline, corpus)
            x = float(source_rows[key]["delta"])
            y = float(external_rows[key]["delta"])
            ax.scatter(x, y, s=80, marker=markers[baseline], color=colors[corpus], edgecolor="black", linewidth=0.5)
            ax.annotate(f"{baseline}-{corpus[0]}", (x, y), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.axvline(0, color="#555555", linewidth=0.8)
    ax.set_xlabel("Source weighted F1 delta")
    ax.set_ylabel("eJSL held-out weighted F1 delta")
    ax.set_title("Classwise Pareto Expert Mixer")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"pareto_scatter.{suffix}", dpi=220)
    plt.close(fig)

    keys = [(baseline, corpus) for baseline in BASELINES for corpus in CORPORA]
    labels = [f"{baseline}\n{corpus}" for baseline, corpus in keys]
    values = [float(external_rows[key]["delta"]) for key in keys]
    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    bars = ax.bar(np.arange(len(keys)), values, color=[colors[corpus] for _, corpus in keys])
    ax.set_xticks(np.arange(len(keys)), labels, rotation=30, ha="right")
    ax.set_ylabel("eJSL held-out weighted F1 delta")
    ax.set_title("Held-out transfer gain over the previous main result")
    ax.axhline(0, color="#444444", linewidth=0.8)
    ax.bar_label(bars, labels=[f"{value:+.3f}" for value in values], padding=3, fontsize=8)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"heldout_gain_bar.{suffix}", dpi=220)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(7.4, 5.2))
    names = [str(row["setting"]) for row in ablation_summary]
    gains = [float(row["heldout_delta_mean"]) for row in ablation_summary]
    passes = [int(row["heldout_pass"]) for row in ablation_summary]
    x = np.arange(len(names))
    bars = ax1.bar(x, gains, color=("#4c78a8", "#f58518", "#54a24b")[: len(names)])
    ax1.set_ylabel("Mean eJSL held-out delta")
    ax1.set_xticks(x, names)
    ax1.bar_label(bars, labels=[f"{value:+.3f}" for value in gains], padding=3)
    ax2 = ax1.twinx()
    ax2.plot(x, passes, color="#b22222", marker="o", linewidth=1.8)
    ax2.set_ylabel("Strict Pareto passes (out of 10)")
    ax2.set_ylim(0, 10.8)
    ax1.set_title("Expert-set ablation")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"expert_ablation.{suffix}", dpi=220)
    plt.close(fig)

    heldout = {(row["baseline"], row["corpus"]): row for row in heldout_rows}
    matrices = []
    for expert_index in range(3):
        matrix = []
        for key in keys:
            weights = np.asarray(json.loads(heldout[key]["weights_json"]), dtype=float)
            matrix.append(weights[:, expert_index])
        matrices.append(np.asarray(matrix))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 6.5), sharey=True)
    expert_names = ("Anchor", "Source-only", "BOBSL-joint")
    for axis, matrix, title in zip(axes, matrices, expert_names):
        image = axis.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        axis.set_xticks(range(4), LABELS)
        axis.set_title(title)
        for row_index in range(matrix.shape[0]):
            for col_index in range(matrix.shape[1]):
                axis.text(col_index, row_index, f"{matrix[row_index, col_index]:.1f}", ha="center", va="center", fontsize=7, color="white" if matrix[row_index, col_index] < 0.35 else "black")
    axes[0].set_yticks(range(len(keys)), [f"{baseline}-{corpus}" for baseline, corpus in keys])
    fig.colorbar(image, ax=axes, fraction=0.02, pad=0.02)
    fig.suptitle("Classwise expert weights selected on eJSL validation")
    fig.subplots_adjust(left=0.18, right=0.92, bottom=0.08, top=0.90, wspace=0.15)
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"expert_weight_heatmap.{suffix}", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final_csv", required=True)
    parser.add_argument("--heldout_root", required=True)
    parser.add_argument("--ablation_root", required=True)
    parser.add_argument("--robustness_root", required=True)
    parser.add_argument("--sweep", action="append", default=[])
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    final_rows = read_csv(Path(args.final_csv).expanduser())
    heldout_rows = collect_search_rows(Path(args.heldout_root).expanduser())
    ablations = collect_named_roots(Path(args.ablation_root).expanduser())
    robustness = collect_named_roots(Path(args.robustness_root).expanduser())
    ablations["tri"] = heldout_rows
    ablation_summary = [summarize_search(name, rows) for name, rows in sorted(ablations.items())]
    robustness_summary = [summarize_search(name, rows) for name, rows in sorted(robustness.items())]
    sweep_best = best_sweep_rows(load_sweep_inputs(args.sweep))
    write_csv(out_dir / "ablation_summary.csv", ablation_summary)
    write_csv(out_dir / "robustness_summary.csv", robustness_summary)
    write_csv(out_dir / "best_sweep_configs.csv", sweep_best)
    make_figures(out_dir, final_rows, heldout_rows, ablation_summary)

    source_rows = [row for row in final_rows if row["split"] == "source"]
    external_rows = [row for row in final_rows if row["split"] == "external_heldout"]
    lines = [
        "# Classwise Pareto Transfer Study",
        "",
        "## Main Result",
        "",
        "| Baseline | Corpus | Source delta | eJSL held-out delta | Status |",
        "|---|---|---:|---:|---|",
    ]
    source = {(row["baseline"], row["corpus"]): row for row in source_rows}
    external = {(row["baseline"], row["corpus"]): row for row in external_rows}
    for baseline in BASELINES:
        for corpus in CORPORA:
            key = (baseline, corpus)
            source_delta = float(source[key]["delta"])
            external_delta = float(external[key]["delta"])
            lines.append(
                f"| {baseline} | {corpus} | {source_delta:+.4f} | {external_delta:+.4f} | "
                f"{'PASS' if source_delta > 0 and external_delta > 0 else 'FAIL'} |"
            )
    lines += [
        "",
        "## Expert Ablation",
        "",
        "| Experts | Passes | Mean source delta | Mean held-out delta | Minimum held-out delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in ablation_summary:
        lines.append(
            f"| {row['setting']} | {row['heldout_pass']}/{row['n_items']} | "
            f"{float(row['source_delta_mean']):+.4f} | {float(row['heldout_delta_mean']):+.4f} | "
            f"{float(row['heldout_delta_min']):+.4f} |"
        )
    lines += [
        "",
        "## Validation-Split Robustness",
        "",
        "| Setting | Passes | Mean source delta | Mean held-out delta | Minimum held-out delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in robustness_summary:
        lines.append(
            f"| {row['setting']} | {row['heldout_pass']}/{row['n_items']} | "
            f"{float(row['source_delta_mean']):+.4f} | {float(row['heldout_delta_mean']):+.4f} | "
            f"{float(row['heldout_delta_min']):+.4f} |"
        )
    lines += [
        "",
        "All classwise weights are shared across the five seeds for each baseline/corpus item.",
        "The main result selects weights on 20% of eJSL dialogues and reports on the disjoint 80%.",
    ]
    (out_dir / "study_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"[study-summary] main_pass="
        f"{sum(float(source[(b, c)]['delta']) > 0 and float(external[(b, c)]['delta']) > 0 for b in BASELINES for c in CORPORA)}/10 "
        f"out={out_dir}"
    )


if __name__ == "__main__":
    main()
