#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root).expanduser()
    rows = []
    for external_path in sorted(root.glob("*/*_seed*/external_test_best_summary.json")):
        run_dir = external_path.parent
        source = load(run_dir / "source_test_best_summary.json")
        external = load(external_path)
        base_source = load(run_dir / "source_test_base_summary.json")
        base_external = load(run_dir / "external_test_base_summary.json")
        oracle_source = load(run_dir / "source_test_oracle_summary.json")
        oracle_external = load(run_dir / "external_test_oracle_summary.json")
        final_source_path = run_dir / "source_test_final_summary.json"
        final_external_path = run_dir / "external_test_final_summary.json"
        final_source = load(final_source_path) if final_source_path.exists() else None
        final_external = load(final_external_path) if final_external_path.exists() else None
        corpus = run_dir.parent.name.upper()
        profile, seed_text = run_dir.name.rsplit("_seed", 1)
        rows.append(
            {
                "corpus": corpus,
                "profile": profile,
                "seed": int(seed_text),
                "base_source": base_source["weighted_f1"],
                "source": source["weighted_f1"],
                "delta_source": source["weighted_f1"] - base_source["weighted_f1"],
                "base_external": base_external["weighted_f1"],
                "external": external["weighted_f1"],
                "delta_external": external["weighted_f1"] - base_external["weighted_f1"],
                "best_epoch": source["best_epoch"],
                "oracle_source": oracle_source["weighted_f1"],
                "oracle_external": oracle_external["weighted_f1"],
                "oracle_delta_source": oracle_source["weighted_f1"] - base_source["weighted_f1"],
                "oracle_delta_external": oracle_external["weighted_f1"] - base_external["weighted_f1"],
                "final_source": final_source["weighted_f1"] if final_source else "",
                "final_external": final_external["weighted_f1"] if final_external else "",
                "final_delta_source": final_source["weighted_f1"] - base_source["weighted_f1"] if final_source else "",
                "final_delta_external": final_external["weighted_f1"] - base_external["weighted_f1"] if final_external else "",
            }
        )
    if not rows:
        print(f"[CESA-SUMMARY] no completed runs under {root}")
        return
    with (root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# CESA screen",
        "",
        "| corpus | profile | source-selected delta | fixed-final delta | oracle delta | selected source/eJSL |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda value: (value["corpus"], -(value["delta_source"] + value["delta_external"]))):
        lines.append(
            f"| {row['corpus']} | {row['profile']} | {row['delta_source']:+.4f}/{row['delta_external']:+.4f} | "
            f"{row['final_delta_source']:+.4f}/{row['final_delta_external']:+.4f} | "
            f"{row['oracle_delta_source']:+.4f}/{row['oracle_delta_external']:+.4f} | "
            f"{row['source']:.4f}/{row['external']:.4f} |"
            if row["final_source"] != ""
            else f"| {row['corpus']} | {row['profile']} | {row['delta_source']:+.4f}/{row['delta_external']:+.4f} | "
            f"NA | {row['oracle_delta_source']:+.4f}/{row['oracle_delta_external']:+.4f} | "
            f"{row['source']:.4f}/{row['external']:.4f} |"
        )
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
