#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def mean(values):
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(clean) / len(clean) if clean else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()
    root = Path(args.root).expanduser()
    seed_rows = []
    for path in sorted(root.rglob("*_seed*/summary.json")):
        match = re.match(r"(.+)_seed(\d+)$", path.parent.name)
        if not match:
            continue
        data = read_json(path)
        if not data:
            continue
        relative_parts = path.relative_to(root).parts
        baseline = next(
            (part for part in relative_parts if part.upper() in {"MMGCN", "MAGTKD", "CMERC", "CONXGNN", "ECERC"}),
            path.parent.parent.name,
        )
        corpus = next((part.upper() for part in relative_parts if part.lower() in {"meld", "iemocap"}), "MELD")
        bobsl_size = next((part.lower() for part in relative_parts if re.fullmatch(r"b\d+k", part.lower())), "b5k")
        seed_rows.append(
            {
                "baseline": baseline,
                "corpus": corpus,
                "bobsl_size": bobsl_size,
                "profile": match.group(1),
                "seed": int(match.group(2)),
                "base_source": data.get("base_source"),
                "base_external": data.get("base_external"),
                "source": data.get("best_source"),
                "external": data.get("best_external"),
                "delta_source": data.get("delta_source"),
                "delta_external": data.get("delta_external"),
                "oracle_source": data.get("oracle_source"),
                "oracle_external": data.get("oracle_external"),
                "best_epoch": data.get("best_epoch"),
                "oracle_epoch": data.get("oracle_epoch"),
                "summary_path": str(path),
            }
        )
    groups = {}
    for row in seed_rows:
        groups.setdefault((row["baseline"], row["corpus"], row["bobsl_size"], row["profile"]), []).append(row)
    group_rows = []
    for (baseline, corpus, bobsl_size, profile), rows in groups.items():
        group_rows.append(
            {
                "baseline": baseline,
                "corpus": corpus,
                "bobsl_size": bobsl_size,
                "profile": profile,
                "n": len(rows),
                "seeds": " ".join(str(row["seed"]) for row in sorted(rows, key=lambda item: item["seed"])),
                "base_source": mean(row["base_source"] for row in rows),
                "base_external": mean(row["base_external"] for row in rows),
                "source": mean(row["source"] for row in rows),
                "external": mean(row["external"] for row in rows),
                "delta_source": mean(row["delta_source"] for row in rows),
                "delta_external": mean(row["delta_external"] for row in rows),
                "oracle_source": mean(row["oracle_source"] for row in rows),
                "oracle_external": mean(row["oracle_external"] for row in rows),
                "strict_double_positive": sum(
                    float(row["delta_source"]) > 0 and float(row["delta_external"]) > 0 for row in rows
                ),
            }
        )
    group_rows.sort(key=lambda row: (row["baseline"], row["corpus"], row["bobsl_size"], -row["delta_external"], -row["delta_source"]))
    out_path = Path(args.out_csv).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(group_rows[0]) if group_rows else []
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(group_rows)
    for row in group_rows:
        print(
            f"[SAST-SUMMARY] {row['baseline']}/{row['corpus']}/{row['bobsl_size']}/{row['profile']} n={row['n']} "
            f"source={row['source']:.4f} ({row['delta_source']:+.4f}) "
            f"external={row['external']:.4f} ({row['delta_external']:+.4f}) "
            f"oracle={row['oracle_source']:.4f}/{row['oracle_external']:.4f} "
            f"strict={row['strict_double_positive']}/{row['n']}"
        )
    print(f"[SAST-SUMMARY] wrote {out_path}")


if __name__ == "__main__":
    main()
