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


def finite(value) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def mean(values):
    clean = [float(value) for value in values if finite(value)]
    return sum(clean) / len(clean) if clean else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--seed_csv")
    args = parser.parse_args()
    root = Path(args.root).expanduser()
    seed_rows = []
    for path in sorted(root.rglob("*_seed*/summary.json")):
        match = re.match(r"(.+)_seed(\d+)$", path.parent.name)
        data = read_json(path)
        if not match or not data or not finite(data.get("joint_source")):
            continue
        parts = path.relative_to(root).parts
        baseline = next((part.upper() for part in parts if part.upper() in {"MMGCN", "MAGTKD"}), "UNKNOWN")
        corpus = next((part.upper() for part in parts if part.lower() in {"meld", "iemocap"}), "UNKNOWN")
        bobsl_size = next((part.lower() for part in parts if re.fullmatch(r"b\d+k", part.lower())), "unknown")
        anchor_name = data.get("anchor_name", "source_only")
        if anchor_name == "plain_joint":
            source_only_source = float(data["joint_source"])
            source_only_external = float(data["joint_external"])
            plain_joint_source = float(data["base_source"])
            plain_joint_external = float(data["base_external"])
        else:
            source_only_source = float(data["base_source"])
            source_only_external = float(data["base_external"])
            plain_joint_source = float(data["joint_source"])
            plain_joint_external = float(data["joint_external"])
        ours_source = float(data["best_source"])
        ours_external = float(data["best_external"])
        source_threshold = max(source_only_source, plain_joint_source)
        external_threshold = max(source_only_external, plain_joint_external)
        row = {
            "baseline": baseline,
            "corpus": corpus,
            "bobsl_size": bobsl_size,
            "profile": match.group(1),
            "seed": int(match.group(2)),
            "anchor_name": anchor_name,
            "source_only_source": source_only_source,
            "source_only_external": source_only_external,
            "plain_joint_source": plain_joint_source,
            "plain_joint_external": plain_joint_external,
            "ours_source": ours_source,
            "ours_external": ours_external,
            "delta_source_only_source": ours_source - source_only_source,
            "delta_source_only_external": ours_external - source_only_external,
            "delta_plain_joint_source": ours_source - plain_joint_source,
            "delta_plain_joint_external": ours_external - plain_joint_external,
            "hard_source_margin": ours_source - source_threshold,
            "hard_external_margin": ours_external - external_threshold,
            "hard_min_margin": min(ours_source - source_threshold, ours_external - external_threshold),
            "hard_pass": int(ours_source > source_threshold and ours_external > external_threshold),
            "best_epoch": data.get("best_epoch"),
            "summary_path": str(path),
        }
        seed_rows.append(row)

    grouped = {}
    for row in seed_rows:
        key = (row["baseline"], row["corpus"], row["bobsl_size"], row["anchor_name"], row["profile"])
        grouped.setdefault(key, []).append(row)
    group_rows = []
    metrics = [
        "source_only_source", "source_only_external", "plain_joint_source", "plain_joint_external",
        "ours_source", "ours_external", "delta_source_only_source", "delta_source_only_external",
        "delta_plain_joint_source", "delta_plain_joint_external", "hard_source_margin",
        "hard_external_margin", "hard_min_margin",
    ]
    for (baseline, corpus, bobsl_size, anchor_name, profile), rows in grouped.items():
        result = {
            "baseline": baseline,
            "corpus": corpus,
            "bobsl_size": bobsl_size,
            "anchor_name": anchor_name,
            "profile": profile,
            "n": len(rows),
            "seeds": " ".join(str(row["seed"]) for row in sorted(rows, key=lambda item: item["seed"])),
        }
        result.update({metric: mean(row[metric] for row in rows) for metric in metrics})
        result["hard_pass_count"] = sum(row["hard_pass"] for row in rows)
        result["hard_group_pass"] = int(
            result["hard_source_margin"] > 0 and result["hard_external_margin"] > 0
        )
        group_rows.append(result)
    group_rows.sort(
        key=lambda row: (
            row["baseline"], row["corpus"], -row["hard_group_pass"],
            -row["hard_min_margin"], -row["hard_external_margin"], -row["hard_source_margin"],
        )
    )

    out_path = Path(args.out_csv).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if group_rows:
        with out_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(group_rows[0]))
            writer.writeheader()
            writer.writerows(group_rows)
    seed_path = Path(args.seed_csv).expanduser() if args.seed_csv else out_path.with_name(f"{out_path.stem}_seeds.csv")
    if seed_rows:
        with seed_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(seed_rows[0]))
            writer.writeheader()
            writer.writerows(seed_rows)
    for row in group_rows:
        print(
            f"[DUAL-SUMMARY] {row['baseline']}/{row['corpus']}/{row['bobsl_size']}/"
            f"{row['anchor_name']}/{row['profile']} "
            f"n={row['n']} ours={row['ours_source']:.4f}/{row['ours_external']:.4f} "
            f"vs_source={row['delta_source_only_source']:+.4f}/{row['delta_source_only_external']:+.4f} "
            f"vs_joint={row['delta_plain_joint_source']:+.4f}/{row['delta_plain_joint_external']:+.4f} "
            f"hard={row['hard_source_margin']:+.4f}/{row['hard_external_margin']:+.4f} "
            f"pass={row['hard_pass_count']}/{row['n']}"
        )
    print(f"[DUAL-SUMMARY] wrote {out_path} and {seed_path}")


if __name__ == "__main__":
    main()
