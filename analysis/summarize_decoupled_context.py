#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root).expanduser()
    rows = []
    for path in sorted(root.glob("*/*_seed*/external_test_best_summary.json")):
        run_dir = path.parent
        external = json.loads(path.read_text(encoding="utf-8"))
        source = json.loads((run_dir / "source_test_best_summary.json").read_text(encoding="utf-8"))
        profile, seed = run_dir.name.rsplit("_seed", 1)
        rows.append(
            {
                "corpus": run_dir.parent.name.upper(),
                "profile": profile,
                "seed": int(seed),
                "source": source["weighted_f1"],
                "external": external["weighted_f1"],
                "source_macro": source["macro_f1"],
                "external_macro": external["macro_f1"],
                "best_epoch": source["best_epoch"],
            }
        )
    if not rows:
        print(f"[DLCEN-SUMMARY] no runs under {root}")
        return
    baselines = {(row["corpus"], row["seed"]): row for row in rows if row["profile"] == "source_only"}
    for row in rows:
        base = baselines.get((row["corpus"], row["seed"]))
        row["delta_source"] = row["source"] - base["source"] if base else 0.0
        row["delta_external"] = row["external"] - base["external"] if base else 0.0
    with (root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Decoupled local-context model screen",
        "",
        "| corpus | profile | source | eJSL | delta source/eJSL | epoch |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda value: (value["corpus"], -(value["delta_source"] + value["delta_external"]))):
        lines.append(
            f"| {row['corpus']} | {row['profile']} | {row['source']:.4f} | {row['external']:.4f} | "
            f"{row['delta_source']:+.4f}/{row['delta_external']:+.4f} | {row['best_epoch']} |"
        )
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

