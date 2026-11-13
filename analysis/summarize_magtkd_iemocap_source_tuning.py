#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


CANONICAL_SOURCE = 0.3248565021291947
CANONICAL_EJSL = 0.10957396575713327
PLAIN_JOINT_EJSL = 0.2707833
OUR_MODULE_EJSL = 0.3161045


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
        source = read_json(source_path)
        external = read_json(source_path.with_name("external_test_best_summary.json"))
        if not match or not source or not external:
            continue
        rows.append(
            {
                "profile": match.group(1),
                "seed": int(match.group(2)),
                "source": float(source["weighted_f1"]),
                "ejsl": float(external["weighted_f1"]),
                "best_epoch": int(source["best_epoch_by_selection_metric"]),
                "output_head": source.get("output_head", "video"),
                "lr": source.get("lr"),
                "dropout": source.get("dropout"),
                "class_weight": source.get("class_weight"),
                "class_weight_power": source.get("class_weight_power", 1.0),
                "label_smoothing": source.get("label_smoothing", 0.0),
                "result_dir": str(source_path.parent),
            }
        )

    groups = {}
    for row in rows:
        groups.setdefault(row["profile"], []).append(row)
    output = []
    for profile, items in groups.items():
        source = mean(item["source"] for item in items)
        ejsl = mean(item["ejsl"] for item in items)
        first = items[0]
        output.append(
            {
                "profile": profile,
                "n": len(items),
                "seeds": " ".join(str(item["seed"]) for item in sorted(items, key=lambda item: item["seed"])),
                "source_weighted_f1": source,
                "ejsl_weighted_f1": ejsl,
                "delta_canonical_source": source - CANONICAL_SOURCE,
                "delta_canonical_ejsl": ejsl - CANONICAL_EJSL,
                "delta_plain_joint_ejsl": ejsl - PLAIN_JOINT_EJSL,
                "margin_below_our_module_ejsl": OUR_MODULE_EJSL - ejsl,
                "in_requested_ejsl_band": int(PLAIN_JOINT_EJSL <= ejsl < OUR_MODULE_EJSL),
                "output_head": first["output_head"],
                "lr": first["lr"],
                "dropout": first["dropout"],
                "class_weight": first["class_weight"],
                "class_weight_power": first["class_weight_power"],
                "label_smoothing": first["label_smoothing"],
            }
        )
    output.sort(
        key=lambda row: (
            -row["in_requested_ejsl_band"],
            -row["ejsl_weighted_f1"],
            -row["source_weighted_f1"],
        )
    )
    out_path = Path(args.out_csv).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if output:
        with out_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(output[0]))
            writer.writeheader()
            writer.writerows(output)
    for row in output:
        print(
            f"[MAGTKD-SOURCE-SUMMARY] {row['profile']} n={row['n']} "
            f"source={row['source_weighted_f1']:.4f} eJSL={row['ejsl_weighted_f1']:.4f} "
            f"delta_joint={row['delta_plain_joint_ejsl']:+.4f} "
            f"below_ours={row['margin_below_our_module_ejsl']:+.4f} "
            f"target_band={row['in_requested_ejsl_band']}"
        )


if __name__ == "__main__":
    main()
