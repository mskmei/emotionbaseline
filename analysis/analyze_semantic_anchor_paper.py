#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


LABELS = ("A", "N", "J", "S")
MAIN_PROFILES = {
    ("CMERC", "MELD"): "context_joint",
    ("CMERC", "IEMOCAP"): "context_rank1",
    ("ConxGNN", "MELD"): "rank1_src4",
    ("ConxGNN", "IEMOCAP"): "rank1_safe",
}


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def bootstrap_ci(values: Sequence[float], seed: int, draws: int = 20000) -> Tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return tuple(float(value) for value in np.quantile(samples, (0.025, 0.975)))


def exact_sign_flip(values: Sequence[float]) -> float:
    observed = abs(float(np.mean(values)))
    null = [
        abs(float(np.mean([sign * value for sign, value in zip(signs, values)])))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ]
    return float(np.mean(np.asarray(null) >= observed - 1e-15))


def report_row(model: str, corpus: str, profile: str, seed: int, run_dir: Path) -> Dict[str, object]:
    source = load_json(run_dir / "source_test_best_summary.json")
    external = load_json(run_dir / "external_test_best_summary.json")
    base_source = load_json(run_dir / "source_test_base_summary.json")
    base_external = load_json(run_dir / "external_test_base_summary.json")
    row: Dict[str, object] = {
        "model": model,
        "corpus": corpus,
        "profile": profile,
        "seed": seed,
        "best_epoch": int(source.get("best_epoch", source.get("best_epoch_by_selection_metric", 0))),
        "base_source": float(base_source["weighted_f1"]),
        "source": float(source["weighted_f1"]),
        "delta_source": float(source["weighted_f1"] - base_source["weighted_f1"]),
        "base_external": float(base_external["weighted_f1"]),
        "external": float(external["weighted_f1"]),
        "delta_external": float(external["weighted_f1"] - base_external["weighted_f1"]),
        "run_dir": str(run_dir),
    }
    for split, result, base in (
        ("source", source, base_source), ("external", external, base_external)
    ):
        for label in LABELS:
            value = float(result["per_class"][label]["f1"])
            base_value = float(base["per_class"][label]["f1"])
            row[f"{split}_{label}_f1"] = value
            row[f"delta_{split}_{label}_f1"] = value - base_value
    return row


def collect_cmerc(root: Path) -> List[Dict[str, object]]:
    rows = []
    for path in sorted(root.glob("*/*_fixed*_seed*/source_test_best_summary.json")):
        run_dir = path.parent
        corpus = run_dir.parent.name.upper()
        profile = run_dir.name.split("_fixed", 1)[0]
        seed = int(run_dir.name.rsplit("_seed", 1)[1])
        rows.append(report_row("CMERC", corpus, profile, seed, run_dir))
    return rows


def collect_conx(root: Path) -> List[Dict[str, object]]:
    rows = []
    for path in sorted(root.glob("*/*_seed*/source_test_best_summary.json")):
        run_dir = path.parent
        corpus = run_dir.parent.name.upper()
        profile, seed_text = run_dir.name.rsplit("_seed", 1)
        rows.append(report_row("ConxGNN", corpus, profile, int(seed_text), run_dir))
    return rows


def summarize(seed_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, str, str], List[Mapping[str, object]]] = defaultdict(list)
    for row in seed_rows:
        groups[(str(row["model"]), str(row["corpus"]), str(row["profile"]))].append(row)
    output = []
    for index, ((model, corpus, profile), rows) in enumerate(sorted(groups.items())):
        result: Dict[str, object] = {
            "model": model,
            "corpus": corpus,
            "profile": profile,
            "n": len(rows),
            "main": int(MAIN_PROFILES.get((model, corpus)) == profile),
            "strict_pareto": sum(
                float(row["delta_source"]) > 0 and float(row["delta_external"]) > 0
                for row in rows
            ),
        }
        for metric in ("base_source", "source", "delta_source", "base_external", "external", "delta_external"):
            values = [float(row[metric]) for row in rows]
            result[f"{metric}_mean"] = float(np.mean(values))
            result[f"{metric}_std"] = float(np.std(values))
            if metric.startswith("delta_"):
                low, high = bootstrap_ci(values, seed=2026 + index)
                result[f"{metric}_ci_low"] = low
                result[f"{metric}_ci_high"] = high
                result[f"{metric}_sign_flip_p"] = exact_sign_flip(values)
                result[f"{metric}_positive"] = sum(value > 0 for value in values)
        for split in ("source", "external"):
            for label in LABELS:
                metric = f"delta_{split}_{label}_f1"
                result[f"{metric}_mean"] = float(np.mean([float(row[metric]) for row in rows]))
        output.append(result)
    return output


def plot_frontier(summary: Sequence[Mapping[str, object]], out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[SAST-PAPER] matplotlib unavailable; skipped plots")
        return
    colors = {"CMERC": "#0072B2", "ConxGNN": "#D55E00"}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharex=False, sharey=False)
    for axis, corpus in zip(axes, ("MELD", "IEMOCAP")):
        current = [row for row in summary if row["corpus"] == corpus and int(row["n"]) == 5]
        for row in current:
            main = bool(row["main"])
            axis.scatter(
                float(row["delta_source_mean"]), float(row["delta_external_mean"]),
                s=130 if main else 55, marker="*" if main else "o",
                color=colors[str(row["model"])], alpha=1.0 if main else 0.55,
                edgecolor="black" if main else "none", linewidth=0.7, zorder=3,
            )
            if main:
                axis.annotate(
                    f"{row['model']}\n{row['profile']}",
                    (float(row["delta_source_mean"]), float(row["delta_external_mean"])),
                    xytext=(6, 6), textcoords="offset points", fontsize=8,
                )
        axis.axhline(0, color="#777777", linewidth=1)
        axis.axvline(0, color="#777777", linewidth=1)
        axis.set_title(corpus)
        axis.set_xlabel("Delta source weighted F1")
        axis.set_ylabel("Delta eJSL weighted F1")
        axis.grid(alpha=0.2)
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=color, label=model)
        for model, color in colors.items()
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=2, frameon=False)
    fig.suptitle("Source-anchored transfer frontier (five seeds)", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_classwise(summary: Sequence[Mapping[str, object]], out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    selected = [row for row in summary if bool(row["main"])]
    selected.sort(key=lambda row: (str(row["corpus"]), str(row["model"])))
    columns = [f"Src {label}" for label in LABELS] + [f"eJSL {label}" for label in LABELS]
    values = np.asarray([
        [float(row[f"delta_source_{label}_f1_mean"]) for label in LABELS]
        + [float(row[f"delta_external_{label}_f1_mean"]) for label in LABELS]
        for row in selected
    ])
    row_names = [f"{row['corpus']} / {row['model']}" for row in selected]
    limit = max(float(np.max(np.abs(values))), 0.01)
    fig, axis = plt.subplots(figsize=(10.8, 4.5))
    image = axis.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(range(len(columns)), labels=columns)
    axis.set_yticks(range(len(row_names)), labels=row_names)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(column, row, f"{values[row, column]:+.3f}", ha="center", va="center", fontsize=8)
    axis.axvline(3.5, color="black", linewidth=1.2)
    axis.set_title("Per-class F1 change of selected source-anchored adapters")
    fig.colorbar(image, ax=axis, label="Delta F1", fraction=0.025, pad=0.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_seed_deltas(seed_rows: Sequence[Mapping[str, object]], out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    selected = [
        row for row in seed_rows
        if MAIN_PROFILES.get((str(row["model"]), str(row["corpus"]))) == row["profile"]
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 8.6))
    colors = {35: "#0072B2", 36: "#E69F00", 37: "#009E73", 41: "#CC79A7", 42: "#D55E00"}
    for axis, ((model, corpus), profile) in zip(axes.flat, MAIN_PROFILES.items()):
        rows = [row for row in selected if row["model"] == model and row["corpus"] == corpus]
        for row in rows:
            seed = int(row["seed"])
            axis.scatter(float(row["delta_source"]), float(row["delta_external"]), s=70,
                         color=colors.get(seed, "#555555"), label=str(seed))
            axis.annotate(str(seed), (float(row["delta_source"]), float(row["delta_external"])),
                          xytext=(4, 3), textcoords="offset points", fontsize=8)
        axis.axhline(0, color="#777777", linewidth=1)
        axis.axvline(0, color="#777777", linewidth=1)
        axis.grid(alpha=0.2)
        axis.set_title(f"{corpus} / {model} / {profile}")
        axis.set_xlabel("Delta source wF1")
        axis.set_ylabel("Delta eJSL wF1")
    fig.suptitle("Paired seed robustness")
    fig.tight_layout()
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def build_baseline_comparison(
    summary: Sequence[Mapping[str, object]], meld_csv: Path, iemocap_csv: Path
) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    for corpus, path, source_config in (
        ("MELD", meld_csv, "meld_only"),
        ("IEMOCAP", iemocap_csv, "iemocap_only"),
    ):
        rows = read_csv(path)
        for baseline in sorted({row["baseline"] for row in rows}):
            source = next(
                row for row in rows
                if row["baseline"] == baseline and row["config"] == "joint_b5k" and row["split"] == corpus
            )
            external = next(
                row for row in rows
                if row["baseline"] == baseline and row["config"] == "joint_b5k" and row["split"] == "eJSL"
            )
            base_source = next(
                row for row in rows
                if row["baseline"] == baseline and row["config"] == source_config and row["split"] == corpus
            )
            base_external = next(
                row for row in rows
                if row["baseline"] == baseline and row["config"] == source_config and row["split"] == "eJSL"
            )
            output.append({
                "corpus": corpus,
                "method": baseline,
                "kind": "joint_b5k baseline",
                "source_wf1": float(source["weighted_f1_mean"]),
                "external_wf1": float(external["weighted_f1_mean"]),
                "delta_source": float(source["weighted_f1_mean"]) - float(base_source["weighted_f1_mean"]),
                "delta_external": float(external["weighted_f1_mean"]) - float(base_external["weighted_f1_mean"]),
                "n": int(source["n_seeds"]),
            })
        for row in summary:
            if row["corpus"] != corpus or not bool(row["main"]):
                continue
            output.append({
                "corpus": corpus,
                "method": f"SAST-{row['model']}",
                "kind": "proposed",
                "source_wf1": float(row["source_mean"]),
                "external_wf1": float(row["external_mean"]),
                "delta_source": float(row["delta_source_mean"]),
                "delta_external": float(row["delta_external_mean"]),
                "n": int(row["n"]),
            })
    return output


def plot_absolute_comparison(rows: Sequence[Mapping[str, object]], out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    colors = {"joint_b5k baseline": "#777777", "proposed": "#0072B2"}
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2))
    for axis, corpus in zip(axes, ("MELD", "IEMOCAP")):
        current = [row for row in rows if row["corpus"] == corpus]
        for row in current:
            proposed = row["kind"] == "proposed"
            axis.scatter(
                float(row["source_wf1"]), float(row["external_wf1"]),
                s=110 if proposed else 55,
                marker="*" if proposed else "o",
                color=colors[str(row["kind"])],
                edgecolor="black" if proposed else "none", linewidth=0.6,
            )
            axis.annotate(str(row["method"]), (float(row["source_wf1"]), float(row["external_wf1"])),
                          xytext=(4, 4), textcoords="offset points", fontsize=7.5)
        axis.set_title(corpus)
        axis.set_xlabel("Source weighted F1")
        axis.set_ylabel("eJSL weighted F1")
        axis.grid(alpha=0.2)
    fig.suptitle("Absolute performance under the aligned BOBSL b5k budget")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def make_report(summary: Sequence[Mapping[str, object]], comparison: Sequence[Mapping[str, object]]) -> str:
    main = [row for row in summary if bool(row["main"])]
    lines = [
        "# Source-Anchored Safe Transfer (SAST)", "",
        "All results use BOBSL b5k, five seeds (35, 36, 37, 41, 42), and source-only checkpoint selection. "
        "eJSL labels are used only for final evaluation and analysis.", "",
        "## Main results", "",
        "| source | backbone | profile | source base->SAST | eJSL base->SAST | delta source/eJSL | 95% CI source | 95% CI eJSL | positive source/eJSL | strict Pareto |",
        "|---|---|---|---:|---:|---:|---|---|---:|---:|",
    ]
    for row in sorted(main, key=lambda value: (str(value["corpus"]), str(value["model"]))):
        lines.append(
            f"| {row['corpus']} | {row['model']} | {row['profile']} | "
            f"{row['base_source_mean']:.4f}->{row['source_mean']:.4f} | "
            f"{row['base_external_mean']:.4f}->{row['external_mean']:.4f} | "
            f"{row['delta_source_mean']:+.4f}/{row['delta_external_mean']:+.4f} | "
            f"[{row['delta_source_ci_low']:+.4f}, {row['delta_source_ci_high']:+.4f}] | "
            f"[{row['delta_external_ci_low']:+.4f}, {row['delta_external_ci_high']:+.4f}] | "
            f"{row['delta_source_positive']}/{row['n']}; {row['delta_external_positive']}/{row['n']} | "
            f"{row['strict_pareto']}/{row['n']} |"
        )
    lines.extend([
        "", "## Comparison with five joint-b5k baselines", "",
        "| source | method | source wF1 | eJSL wF1 | delta source/eJSL | type |",
        "|---|---|---:|---:|---:|---|",
    ])
    for row in sorted(comparison, key=lambda value: (str(value["corpus"]), -float(value["source_wf1"]))):
        lines.append(
            f"| {row['corpus']} | {row['method']} | {row['source_wf1']:.4f} | {row['external_wf1']:.4f} | "
            f"{row['delta_source']:+.4f}/{row['delta_external']:+.4f} | {row['kind']} |"
        )
    for corpus in ("MELD", "IEMOCAP"):
        current = [row for row in comparison if row["corpus"] == corpus]
        best_source = max(current, key=lambda row: float(row["source_wf1"]))
        best_external = max(current, key=lambda row: float(row["external_wf1"]))
        lines.append(
            f"- {corpus}: best source wF1 is {best_source['source_wf1']:.4f} ({best_source['method']}); "
            f"best eJSL wF1 is {best_external['external_wf1']:.4f} ({best_external['method']})."
        )
    lines.extend([
        "", "## Component and capacity ablations", "",
        "| source | backbone | profile | n | delta source/eJSL | strict Pareto | main |",
        "|---|---|---|---:|---:|---:|---:|",
    ])
    for row in sorted(summary, key=lambda value: (str(value["corpus"]), str(value["model"]), str(value["profile"]))):
        if int(row["n"]) != 5:
            continue
        lines.append(
            f"| {row['corpus']} | {row['model']} | {row['profile']} | {row['n']} | "
            f"{row['delta_source_mean']:+.4f}/{row['delta_external_mean']:+.4f} | "
            f"{row['strict_pareto']}/{row['n']} | {'yes' if row['main'] else 'no'} |"
        )
    lines.extend([
        "", "## Interpretation", "",
        "1. The frozen source classifier is the semantic anchor: BOBSL is optimized through the existing source decision boundary instead of a disconnected local head.",
        "2. The zero-initialized adapter and source safety loss preserve the source solution; epoch 0 is an explicit reject option when adaptation cannot improve the source metric.",
        "3. Capacity is corpus- and backbone-dependent. MELD CMERC uses the full rank-3 emotion subspace, while IEMOCAP CMERC and both ConxGNN settings require rank-1 transfer for stability.",
        "4. The same module improves both metrics on two backbones, while the independently built DLCEN backbone did not reproduce the gain. This supports a plug-in transfer mechanism rather than checkpoint routing.",
        "", "With only five paired seeds, exact two-sided sign-flip tests have minimum p=0.0625. Claims should therefore emphasize paired effect size, bootstrap intervals, and seed consistency rather than conventional significance.",
    ])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper tables and figures for source-anchored transfer.")
    parser.add_argument("--cmerc_root", default="/raid_zoe/home/lr/maokeyu/sign/cesa_cmerc/ablations")
    parser.add_argument("--conx_root", default="/raid_zoe/home/lr/maokeyu/sign/conxgnn_sast/fiveseed")
    parser.add_argument("--meld_csv", default="/raid_zoe/home/lr/maokeyu/sign/five_baseline_joint_per_class_summary/five_baseline_per_class_config_means.csv")
    parser.add_argument("--iemocap_csv", default="/raid_zoe/home/lr/maokeyu/sign/five_baseline_bobsl_iemocap_ejsl_summary/iemocap_joint_config_means.csv")
    parser.add_argument("--out_dir", default="/raid_zoe/home/lr/maokeyu/sign/semantic_anchor_paper")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_rows = collect_cmerc(Path(args.cmerc_root).expanduser()) + collect_conx(Path(args.conx_root).expanduser())
    if not seed_rows:
        raise RuntimeError("No semantic-anchor runs found")
    summary = summarize(seed_rows)
    comparison = build_baseline_comparison(
        summary, Path(args.meld_csv).expanduser(), Path(args.iemocap_csv).expanduser()
    )
    write_csv(out_dir / "seed_results.csv", seed_rows)
    write_csv(out_dir / "group_results.csv", summary)
    write_csv(out_dir / "baseline_comparison.csv", comparison)
    report = make_report(summary, comparison)
    (out_dir / "results.md").write_text(report, encoding="utf-8")
    plot_frontier(summary, out_dir / "transfer_frontier.png")
    plot_classwise(summary, out_dir / "classwise_f1_delta.png")
    plot_seed_deltas(seed_rows, out_dir / "paired_seed_deltas.png")
    plot_absolute_comparison(comparison, out_dir / "absolute_b5k_comparison.png")
    print(report, end="")
    print(f"[SAST-PAPER] out_dir={out_dir}")


if __name__ == "__main__":
    main()
