#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


LABELS = ("A", "N", "J", "S")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def profile_name(run_dir: Path, is_main: bool) -> str:
    if is_main:
        return "CESA"
    return run_dir.name.split("_fixed", 1)[0]


def collect(root: Path, is_main: bool):
    rows = []
    for external_path in sorted(root.glob("*/*_seed*/external_test_final_summary.json")):
        run_dir = external_path.parent
        base_source = load(run_dir / "source_test_base_summary.json")
        base_external = load(run_dir / "external_test_base_summary.json")
        for selection, suffix in (("source_selected", "best"), ("fixed_final", "final")):
            source = load(run_dir / f"source_test_{suffix}_summary.json")
            external = load(run_dir / f"external_test_{suffix}_summary.json")
            row = {
                "corpus": run_dir.parent.name.upper(),
                "profile": profile_name(run_dir, is_main),
                "selection": selection,
                "seed": int(run_dir.name.rsplit("_seed", 1)[1]),
                "source": source["weighted_f1"],
                "external": external["weighted_f1"],
                "base_source": base_source["weighted_f1"],
                "base_external": base_external["weighted_f1"],
                "delta_source": source["weighted_f1"] - base_source["weighted_f1"],
                "delta_external": external["weighted_f1"] - base_external["weighted_f1"],
            }
            for split, result, base in (
                ("source", source, base_source), ("external", external, base_external)
            ):
                for label in LABELS:
                    row[f"{split}_{label}_f1"] = result["per_class"][label]["f1"]
                    row[f"delta_{split}_{label}_f1"] = (
                        result["per_class"][label]["f1"] - base["per_class"][label]["f1"]
                    )
            rows.append(row)
    return rows


def bootstrap_ci(values, seed=2026, draws=20000):
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return tuple(float(x) for x in np.quantile(samples, [0.025, 0.975]))


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["corpus"], row["profile"], row["selection"])].append(row)
    output = []
    for (corpus, profile, selection), group in sorted(grouped.items()):
        result = {"corpus": corpus, "profile": profile, "selection": selection, "n": len(group)}
        for metric in ("source", "external", "delta_source", "delta_external"):
            values = [row[metric] for row in group]
            result[f"{metric}_mean"] = float(np.mean(values))
            result[f"{metric}_std"] = float(np.std(values))
            if metric.startswith("delta_"):
                low, high = bootstrap_ci(values)
                result[f"{metric}_ci_low"] = low
                result[f"{metric}_ci_high"] = high
        result["strict_pareto"] = sum(
            row["delta_source"] > 0 and row["delta_external"] > 0 for row in group
        )
        for split in ("source", "external"):
            for label in LABELS:
                metric = f"delta_{split}_{label}_f1"
                result[f"{metric}_mean"] = float(np.mean([row[metric] for row in group]))
        output.append(result)
    return output


def save_plot(summary, out_path: Path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[CESA-STUDY] matplotlib unavailable; skipped plot")
        return
    corpora = [corpus for corpus in ("MELD", "IEMOCAP") if any(r["corpus"] == corpus for r in summary)]
    fig, axes = plt.subplots(1, len(corpora), figsize=(6.3 * len(corpora), 5.2), squeeze=False)
    markers = {"CESA": "o", "fullspace": "s", "no_safety": "^", "context_joint": "X", "rank1": "D"}
    for axis, corpus in zip(axes[0], corpora):
        items = [
            row for row in summary
            if row["corpus"] == corpus and row["selection"] == "source_selected"
        ]
        for row in items:
            axis.scatter(
                row["delta_source_mean"], row["delta_external_mean"],
                s=115 if row["profile"] == "CESA" else 75,
                marker=markers.get(row["profile"], "o"), label=row["profile"], zorder=3,
            )
            axis.annotate(row["profile"], (row["delta_source_mean"], row["delta_external_mean"]),
                          xytext=(5, 5), textcoords="offset points", fontsize=9)
        axis.axhline(0, color="#777777", linewidth=1)
        axis.axvline(0, color="#777777", linewidth=1)
        axis.set_title(corpus)
        axis.set_xlabel("Delta source weighted F1")
        axis.set_ylabel("Delta eJSL weighted F1")
        axis.grid(alpha=0.2)
    fig.suptitle("CESA transfer frontier and component ablations")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main_root", required=True)
    parser.add_argument("--ablation_root", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = collect(Path(args.main_root).expanduser(), True) + collect(Path(args.ablation_root).expanduser(), False)
    if not rows:
        raise RuntimeError("No completed CESA runs found")
    with (out_dir / "seed_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    fields = sorted({key for row in summary for key in row})
    with (out_dir / "group_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    lines = [
        "# CESA study",
        "",
        "Fixed adaptation budgets; no eJSL labels are used for checkpoint selection.",
        "",
        "| corpus | method | selection | n | source wF1 | eJSL wF1 | delta source/eJSL | 95% CI source | 95% CI eJSL | Pareto seeds |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['corpus']} | {row['profile']} | {row['selection']} | {row['n']} | {row['source_mean']:.4f} | "
            f"{row['external_mean']:.4f} | {row['delta_source_mean']:+.4f}/{row['delta_external_mean']:+.4f} | "
            f"[{row['delta_source_ci_low']:+.4f}, {row['delta_source_ci_high']:+.4f}] | "
            f"[{row['delta_external_ci_low']:+.4f}, {row['delta_external_ci_high']:+.4f}] | "
            f"{row['strict_pareto']}/{row['n']} |"
        )
    (out_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    save_plot(summary, out_dir / "transfer_frontier.png")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
