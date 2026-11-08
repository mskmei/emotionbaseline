#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


def mean_std(values: Iterable[float]) -> Tuple[float, float]:
    values = list(values)
    mean = sum(values) / len(values)
    return mean, math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields: List[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--baseline", required=True)
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    rows: List[Dict[str, object]] = []
    for source_path in sorted(root.glob("*/*_seed*/source_test_best_summary.json")):
        external_path = source_path.with_name("external_test_best_summary.json")
        source = json.loads(source_path.read_text())
        external = json.loads(external_path.read_text())
        profile, seed = source_path.parent.name.rsplit("_seed", 1)
        rows.append(
            {
                "baseline": args.baseline,
                "corpus": source_path.parts[-3],
                "profile": profile,
                "seed": int(seed),
                "best_epoch": source["best_epoch"],
                "source_base_wf1": source["source"]["weighted_f1"],
                "source_joint_wf1": source["joint"]["weighted_f1"],
                "source_fused_wf1": source["fused"]["weighted_f1"],
                "external_base_wf1": external["source"]["weighted_f1"],
                "external_joint_wf1": external["joint"]["weighted_f1"],
                "external_fused_wf1": external["fused"]["weighted_f1"],
                "source_gate": sum(source["mean_gate"]) / len(source["mean_gate"]),
                "external_gate": sum(external["mean_gate"]) / len(external["mean_gate"]),
            }
        )
    write_csv(root / "all_runs.csv", rows)
    groups: Dict[Tuple[str, str], List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["corpus"]), str(row["profile"]))].append(row)
    means: List[Dict[str, object]] = []
    for (corpus, profile), group in sorted(groups.items()):
        result: Dict[str, object] = {"baseline": args.baseline, "corpus": corpus, "profile": profile, "n_seeds": len(group)}
        for metric in (
            "source_base_wf1",
            "source_joint_wf1",
            "source_fused_wf1",
            "external_base_wf1",
            "external_joint_wf1",
            "external_fused_wf1",
            "source_gate",
            "external_gate",
        ):
            mean, std = mean_std(float(row[metric]) for row in group)
            result[f"{metric}_mean"] = mean
            result[f"{metric}_std"] = std
        result["delta_source_vs_base"] = float(result["source_fused_wf1_mean"]) - float(result["source_base_wf1_mean"])
        result["delta_external_vs_base"] = float(result["external_fused_wf1_mean"]) - float(result["external_base_wf1_mean"])
        means.append(result)
    write_csv(root / "group_means.csv", means)
    lines = [
        f"# Safe transfer gate: {args.baseline}",
        "",
        "| corpus | profile | n | source base/joint/fused | external base/joint/fused | delta source/external | gate source/external |",
        "|---|---|---:|---|---|---|---|",
    ]
    for row in sorted(means, key=lambda value: (str(value["corpus"]), -float(value["source_fused_wf1_mean"]))):
        lines.append(
            f"| {row['corpus']} | {row['profile']} | {row['n_seeds']} | "
            f"{float(row['source_base_wf1_mean']):.4f}/{float(row['source_joint_wf1_mean']):.4f}/{float(row['source_fused_wf1_mean']):.4f} | "
            f"{float(row['external_base_wf1_mean']):.4f}/{float(row['external_joint_wf1_mean']):.4f}/{float(row['external_fused_wf1_mean']):.4f} | "
            f"{float(row['delta_source_vs_base']):+.4f}/{float(row['delta_external_vs_base']):+.4f} | "
            f"{float(row['source_gate_mean']):.3f}/{float(row['external_gate_mean']):.3f} |"
        )
    report = "\n".join(lines) + "\n"
    (root / "report.md").write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
