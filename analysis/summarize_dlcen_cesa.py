#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root).expanduser()
    rows = []
    for path in sorted(root.glob("*/*_seed*/external_test_best_summary.json")):
        run_dir = path.parent
        external = load(path)
        source = load(run_dir / "source_test_best_summary.json")
        base_source = load(run_dir / "source_test_base_summary.json")
        base_external = load(run_dir / "external_test_base_summary.json")
        profile, seed = run_dir.name.rsplit("_seed", 1)
        rows.append({
            "corpus": run_dir.parent.name.upper(), "profile": profile, "seed": int(seed),
            "base_source": base_source["weighted_f1"], "source": source["weighted_f1"],
            "delta_source": source["weighted_f1"] - base_source["weighted_f1"],
            "base_external": base_external["weighted_f1"], "external": external["weighted_f1"],
            "delta_external": external["weighted_f1"] - base_external["weighted_f1"],
        })
    if not rows:
        print(f"[DLCEN-CESA-SUMMARY] no completed runs under {root}")
        return
    with (root / "seed_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["corpus"], row["profile"])].append(row)
    lines = [
        "# DLCEN-CESA five-seed results", "",
        "| corpus | profile | n | source base->joint | eJSL base->joint | delta source/eJSL | Pareto seeds |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for (corpus, profile), group in sorted(groups.items()):
        mean = lambda key: float(np.mean([row[key] for row in group]))
        pareto = sum(row["delta_source"] > 0 and row["delta_external"] > 0 for row in group)
        lines.append(
            f"| {corpus} | {profile} | {len(group)} | {mean('base_source'):.4f}->{mean('source'):.4f} | "
            f"{mean('base_external'):.4f}->{mean('external'):.4f} | "
            f"{mean('delta_source'):+.4f}/{mean('delta_external'):+.4f} | {pareto}/{len(group)} |"
        )
    (root / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
