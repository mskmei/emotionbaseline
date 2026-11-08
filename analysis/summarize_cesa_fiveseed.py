#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root).expanduser()
    rows = []
    for final_external_path in sorted(root.glob("*/*_seed*/external_test_final_summary.json")):
        run_dir = final_external_path.parent
        base_source = load(run_dir / "source_test_base_summary.json")
        base_external = load(run_dir / "external_test_base_summary.json")
        profile, seed = run_dir.name.rsplit("_seed", 1)
        for selection, suffix in (("source_selected", "best"), ("fixed_final", "final")):
            source = load(run_dir / f"source_test_{suffix}_summary.json")
            external = load(run_dir / f"external_test_{suffix}_summary.json")
            rows.append(
                {
                    "corpus": run_dir.parent.name.upper(),
                    "profile": profile,
                    "selection": selection,
                    "seed": int(seed),
                    "base_source": base_source["weighted_f1"],
                    "source": source["weighted_f1"],
                    "delta_source": source["weighted_f1"] - base_source["weighted_f1"],
                    "base_external": base_external["weighted_f1"],
                    "external": external["weighted_f1"],
                    "delta_external": external["weighted_f1"] - base_external["weighted_f1"],
                }
            )
    if not rows:
        print(f"[CESA-5-SUMMARY] no runs under {root}")
        return
    with (root / "seed_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["corpus"], row["profile"], row["selection"])].append(row)
    means = []
    for (corpus, profile, selection), group in sorted(grouped.items()):
        result = {"corpus": corpus, "profile": profile, "selection": selection, "n": len(group)}
        for metric in ("base_source", "source", "delta_source", "base_external", "external", "delta_external"):
            values = np.asarray([row[metric] for row in group], dtype=float)
            result[f"{metric}_mean"] = float(values.mean())
            result[f"{metric}_std"] = float(values.std(ddof=0))
        result["positive_source"] = sum(row["delta_source"] > 0 for row in group)
        result["positive_external"] = sum(row["delta_external"] > 0 for row in group)
        result["strict_pareto"] = sum(row["delta_source"] > 0 and row["delta_external"] > 0 for row in group)
        means.append(result)
    with (root / "group_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(means[0]))
        writer.writeheader()
        writer.writerows(means)
    lines = [
        "# CESA fixed-budget five-seed results",
        "",
        "| corpus | profile | selection | n | source base->CESA | eJSL base->CESA | delta source/eJSL | positive source/eJSL | strict Pareto |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in means:
        lines.append(
            f"| {row['corpus']} | {row['profile']} | {row['selection']} | {row['n']} | "
            f"{row['base_source_mean']:.4f}->{row['source_mean']:.4f} | "
            f"{row['base_external_mean']:.4f}->{row['external_mean']:.4f} | "
            f"{row['delta_source_mean']:+.4f}/{row['delta_external_mean']:+.4f} | "
            f"{row['positive_source']}/{row['positive_external']} | {row['strict_pareto']} |"
        )
    (root / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
