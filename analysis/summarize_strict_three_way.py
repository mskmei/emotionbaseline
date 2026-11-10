#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


BASELINES = ("MMGCN", "MAGTKD", "ECERC", "CMERC", "CONXGNN")
CORPORA = ("MELD", "IEMOCAP")
NEGATIVE_TRANSFER_MODELS = {"MMGCN", "MAGTKD"}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def normalize_baseline(value: str) -> str:
    return value.upper()


def plain_index(rows, corpus: str):
    indexed = {}
    for row in rows:
        indexed[(normalize_baseline(row["baseline"]), row["config"], row["split"].upper())] = float(
            row["weighted_f1_mean"]
        )
    source_config = "meld_only" if corpus == "MELD" else "iemocap_only"
    output = {}
    for baseline in BASELINES:
        output[(baseline, "source")] = indexed[(baseline, source_config, corpus)]
        output[(baseline, "ejsl")] = indexed[(baseline, source_config, "EJSL")]
    return output


def parse_run_key(root: Path, run_dir: Path):
    parts = run_dir.relative_to(root).parts
    baseline = normalize_baseline(parts[0])
    corpus = parts[1].upper()
    bobsl_size = parts[2].lower()
    profile, seed = parts[3].rsplit("_seed", 1)
    if baseline not in BASELINES or corpus not in CORPORA:
        return None
    return baseline, corpus, bobsl_size, profile, int(seed)


def collect_runs(root: Path):
    rows = []
    for summary_path in sorted(root.glob("*/*/b*k/*_seed*/summary.json")):
        run_dir = summary_path.parent
        key = parse_run_key(root, run_dir)
        if key is None:
            continue
        baseline, corpus, bobsl_size, profile, seed = key
        data = load_json(summary_path)
        rows.append(
            {
                "baseline": baseline,
                "corpus": corpus,
                "bobsl_size": bobsl_size,
                "profile": profile,
                "seed": seed,
                "base_source": float(data["base_source"]),
                "base_ejsl": float(data["base_external"]),
                "ours_source": float(data["best_source"]),
                "ours_ejsl": float(data["best_external"]),
                "best_epoch": int(data["best_epoch"]),
            }
        )
    for best_path in sorted(root.glob("CMERC/*/b*k/*_seed*/external_test_best_summary.json")):
        run_dir = best_path.parent
        key = parse_run_key(root, run_dir)
        if key is None:
            continue
        baseline, corpus, bobsl_size, profile, seed = key
        base_source = load_json(run_dir / "source_test_base_summary.json")
        base_ejsl = load_json(run_dir / "external_test_base_summary.json")
        ours_source = load_json(run_dir / "source_test_best_summary.json")
        ours_ejsl = load_json(best_path)
        rows.append(
            {
                "baseline": baseline,
                "corpus": corpus,
                "bobsl_size": bobsl_size,
                "profile": profile,
                "seed": seed,
                "base_source": float(base_source["weighted_f1"]),
                "base_ejsl": float(base_ejsl["weighted_f1"]),
                "ours_source": float(ours_source["weighted_f1"]),
                "ours_ejsl": float(ours_ejsl["weighted_f1"]),
                "best_epoch": int(ours_source.get("best_epoch", 0)),
            }
        )
    for row in rows:
        row["delta_plain_source"] = row["ours_source"] - row["base_source"]
        row["delta_plain_ejsl"] = row["ours_ejsl"] - row["base_ejsl"]
        row["strict_seed"] = int(row["delta_plain_source"] > 0 and row["delta_plain_ejsl"] > 0)
    return rows


def pareto(delta_source: float, delta_ejsl: float) -> float:
    if delta_source > 0 and delta_ejsl > 0:
        return math.sqrt(delta_source * delta_ejsl)
    return min(delta_source, delta_ejsl)


def group_runs(rows, source_only):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["baseline"], row["corpus"], row["bobsl_size"], row["profile"])].append(row)
    output = []
    for (baseline, corpus, bobsl_size, profile), group in sorted(grouped.items()):
        result = {
            "baseline": baseline,
            "corpus": corpus,
            "bobsl_size": bobsl_size,
            "profile": profile,
            "n": len(group),
            "seeds": " ".join(str(row["seed"]) for row in sorted(group, key=lambda item: item["seed"])),
            "strict_seeds": sum(row["strict_seed"] for row in group),
        }
        for metric in ("base_source", "base_ejsl", "ours_source", "ours_ejsl", "delta_plain_source", "delta_plain_ejsl"):
            values = np.asarray([row[metric] for row in group], dtype=float)
            result[f"{metric}_mean"] = float(values.mean())
            result[f"{metric}_std"] = float(values.std(ddof=0))
        result["source_only_source"] = source_only[corpus][(baseline, "source")]
        result["source_only_ejsl"] = source_only[corpus][(baseline, "ejsl")]
        result["delta_sourceonly_source"] = result["ours_source_mean"] - result["source_only_source"]
        result["delta_sourceonly_ejsl"] = result["ours_ejsl_mean"] - result["source_only_ejsl"]
        result["beats_plain_both"] = int(result["delta_plain_source_mean"] > 0 and result["delta_plain_ejsl_mean"] > 0)
        result["beats_sourceonly_both"] = int(
            result["delta_sourceonly_source"] > 0 and result["delta_sourceonly_ejsl"] > 0
        )
        result["plain_pareto"] = pareto(result["delta_plain_source_mean"], result["delta_plain_ejsl_mean"])
        result["sourceonly_pareto"] = pareto(result["delta_sourceonly_source"], result["delta_sourceonly_ejsl"])
        output.append(result)
    return output


def select_best(rows, min_seeds: int):
    grouped = defaultdict(list)
    for row in rows:
        if int(row["n"]) >= min_seeds:
            grouped[(row["baseline"], row["corpus"], row["bobsl_size"])].append(row)
    selected = []
    for _, candidates in sorted(grouped.items()):
        selected.append(
            max(
                candidates,
                key=lambda row: (
                    row["beats_plain_both"],
                    row["beats_sourceonly_both"],
                    row["plain_pareto"],
                    row["sourceonly_pareto"],
                ),
            )
        )
    return selected


def fmt(value, signed=False):
    return f"{float(value):+.4f}" if signed else f"{float(value):.4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/raid_zoe/home/lr/maokeyu/sign/sast_plain_joint_refine")
    parser.add_argument(
        "--meld_csv",
        default="/raid_zoe/home/lr/maokeyu/sign/five_baseline_joint_per_class_summary/five_baseline_per_class_config_means.csv",
    )
    parser.add_argument(
        "--iemocap_csv",
        default="/raid_zoe/home/lr/maokeyu/sign/five_baseline_bobsl_iemocap_ejsl_summary/iemocap_joint_config_means.csv",
    )
    parser.add_argument("--min_seeds", type=int, default=5)
    parser.add_argument("--out_dir", default="")
    args = parser.parse_args()

    root = Path(args.root).expanduser()
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else root / "three_way"
    source_only = {
        "MELD": plain_index(read_csv(Path(args.meld_csv)), "MELD"),
        "IEMOCAP": plain_index(read_csv(Path(args.iemocap_csv)), "IEMOCAP"),
    }
    seed_rows = collect_runs(root)
    group_rows = group_runs(seed_rows, source_only)
    selected = select_best(group_rows, args.min_seeds)
    write_csv(out_dir / "strict_seed_results.csv", seed_rows)
    write_csv(out_dir / "strict_group_results.csv", group_rows)
    write_csv(out_dir / "strict_three_way_best.csv", selected)

    lines = [
        "# Strict three-way ablation",
        "",
        f"Only profiles with at least {args.min_seeds} paired seeds are eligible for selection.",
        "",
        "| corpus | model | BOBSL | profile | source-only S/E | joint S/E | joint+module S/E | module-joint S/E | module-source S/E | strict seeds | pass A/T |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(selected, key=lambda item: (item["corpus"], item["baseline"], item["bobsl_size"])):
        lines.append(
            f"| {row['corpus']} | {row['baseline']} | {row['bobsl_size']} | {row['profile']} | "
            f"{fmt(row['source_only_source'])}/{fmt(row['source_only_ejsl'])} | "
            f"{fmt(row['base_source_mean'])}/{fmt(row['base_ejsl_mean'])} | "
            f"{fmt(row['ours_source_mean'])}/{fmt(row['ours_ejsl_mean'])} | "
            f"{fmt(row['delta_plain_source_mean'], True)}/{fmt(row['delta_plain_ejsl_mean'], True)} | "
            f"{fmt(row['delta_sourceonly_source'], True)}/{fmt(row['delta_sourceonly_ejsl'], True)} | "
            f"{row['strict_seeds']}/{row['n']} | {row['beats_plain_both']}/{row['beats_sourceonly_both']} |"
        )

    iem_hard = [
        row for row in selected
        if row["corpus"] == "IEMOCAP" and row["baseline"] in NEGATIVE_TRANSFER_MODELS
    ]
    other_rows = [row for row in selected if row["baseline"] not in NEGATIVE_TRANSFER_MODELS]
    lines.extend(
        [
            "",
            "## Requirement audit",
            "",
            f"- IEMOCAP hard models passing both ablation and positive-transfer tests: "
            f"{sum(row['beats_plain_both'] and row['beats_sourceonly_both'] for row in iem_hard)}/{len(iem_hard)}.",
            f"- Remaining-model ablation pass: {sum(row['beats_plain_both'] for row in other_rows)}/{len(other_rows)}.",
            f"- All selected settings positive against source-only: "
            f"{sum(row['beats_sourceonly_both'] for row in selected)}/{len(selected)}.",
        ]
    )
    report = "\n".join(lines) + "\n"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "strict_three_way_best.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
