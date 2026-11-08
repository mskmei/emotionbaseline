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
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def collect(root: Path) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    for source_path in sorted(root.glob("*/*/*_seed*/source_test_best_summary.json")):
        external_path = source_path.with_name("external_test_best_summary.json")
        if not external_path.is_file():
            continue
        source = json.loads(source_path.read_text(encoding="utf-8"))
        external = json.loads(external_path.read_text(encoding="utf-8"))
        profile, seed = source_path.parent.name.rsplit("_seed", 1)
        corpus, condition = source_path.parts[-4:-2]
        output.append(
            {
                "corpus": corpus,
                "condition": condition,
                "profile": profile,
                "seed": int(seed),
                "best_epoch": source["best_epoch"],
                "source_wf1": source["fused"]["weighted_f1"],
                "source_macro_f1": source["fused"]["macro_f1"],
                "source_base_wf1": source["base"]["weighted_f1"],
                "source_adapter_wf1": source["source"]["weighted_f1"],
                "ejsl_wf1": external["fused"]["weighted_f1"],
                "ejsl_macro_f1": external["fused"]["macro_f1"],
                "ejsl_base_wf1": external["base"]["weighted_f1"],
                "ejsl_adapter_wf1": external["source"]["weighted_f1"],
                "source_gate": sum(source["mean_gate"]) / len(source["mean_gate"]),
                "ejsl_gate": sum(external["mean_gate"]) / len(external["mean_gate"]),
            }
        )
    return output


def aggregate(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, str, str], List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["corpus"]), str(row["condition"]), str(row["profile"]))].append(row)
    metrics = ("source_wf1", "source_macro_f1", "ejsl_wf1", "ejsl_macro_f1", "source_gate", "ejsl_gate")
    output: List[Dict[str, object]] = []
    for (corpus, condition, profile), group in sorted(groups.items()):
        result: Dict[str, object] = {
            "corpus": corpus,
            "condition": condition,
            "profile": profile,
            "n_seeds": len(group),
        }
        for metric in metrics:
            mean, std = mean_std(float(row[metric]) for row in group)
            result[f"{metric}_mean"] = mean
            result[f"{metric}_std"] = std
        output.append(result)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    root = Path(parser.parse_args().root).expanduser().resolve()
    rows = collect(root)
    means = aggregate(rows)
    write_csv(root / "all_runs.csv", rows)
    write_csv(root / "group_means.csv", means)
    lines = [
        "# Residual transfer results",
        "",
        "| corpus | condition | profile | n | source wF1 | eJSL wF1 | source/eJSL gate |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in sorted(means, key=lambda value: (str(value["corpus"]), -float(value["source_wf1_mean"]))):
        lines.append(
            f"| {row['corpus']} | {row['condition']} | {row['profile']} | {row['n_seeds']} | "
            f"{float(row['source_wf1_mean']):.4f}+/-{float(row['source_wf1_std']):.4f} | "
            f"{float(row['ejsl_wf1_mean']):.4f}+/-{float(row['ejsl_wf1_std']):.4f} | "
            f"{float(row['source_gate_mean']):.3f}/{float(row['ejsl_gate_mean']):.3f} |"
        )
    report = "\n".join(lines) + "\n"
    (root / "report.md").write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
