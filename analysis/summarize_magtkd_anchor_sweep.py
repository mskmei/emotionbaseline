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
    rows = []
    for source_path in sorted(root.glob("*_seed*/source_test_best_summary.json")):
        match = re.match(r"(.+)_seed(\d+)$", source_path.parent.name)
        external_path = source_path.with_name("external_test_best_summary.json")
        source = read_json(source_path)
        external = read_json(external_path)
        if not match or not source or not external:
            continue
        rows.append({
            "profile": match.group(1),
            "seed": int(match.group(2)),
            "source": source.get("weighted_f1"),
            "external": external.get("weighted_f1"),
            "source_path": str(source_path),
            "checkpoint": str(source_path.parent / "model_best.pt"),
        })
    groups = {}
    for row in rows:
        groups.setdefault(row["profile"], []).append(row)
    output = []
    for profile, items in groups.items():
        source = mean(item["source"] for item in items)
        external = mean(item["external"] for item in items)
        output.append({
            "profile": profile,
            "n": len(items),
            "seeds": " ".join(str(item["seed"]) for item in sorted(items, key=lambda value: value["seed"])),
            "source": source,
            "external": external,
            "delta_canonical_source": source - 0.4646,
            "delta_canonical_external": external - 0.3330,
            "hard_min_margin": min(source - 0.4646, external - 0.3330),
        })
    output.sort(key=lambda row: (-row["hard_min_margin"], -row["source"], -row["external"]))
    out_path = Path(args.out_csv).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if output:
        with out_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(output[0]))
            writer.writeheader()
            writer.writerows(output)
    for row in output:
        print(
            f"[MAGTKD-ANCHOR] {row['profile']} n={row['n']} source={row['source']:.4f} "
            f"external={row['external']:.4f} margin={row['delta_canonical_source']:+.4f}/"
            f"{row['delta_canonical_external']:+.4f}"
        )


if __name__ == "__main__":
    main()
