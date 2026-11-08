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
    clean = list(values)
    mean = sum(clean) / len(clean)
    return mean, math.sqrt(sum((value - mean) ** 2 for value in clean) / len(clean))


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields: List[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def collect(root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for path in sorted(root.glob("*/*/*_seed*/source_test_best_summary.json")):
        source = json.loads(path.read_text(encoding="utf-8"))
        external_path = path.with_name("external_test_best_summary.json")
        if not external_path.is_file():
            continue
        external = json.loads(external_path.read_text(encoding="utf-8"))
        trial = path.parent.name
        profile, seed = trial.rsplit("_seed", 1)
        corpus, condition = path.parts[-4:-2]
        row: Dict[str, object] = {
            "corpus": corpus,
            "condition": condition,
            "profile": profile,
            "seed": int(seed),
            "best_epoch": source["best_epoch"],
            "source_fused_wf1": source["fused"]["weighted_f1"],
            "source_context_wf1": source["context"]["weighted_f1"],
            "source_local_wf1": source["local"]["weighted_f1"],
            "source_gate": source["mean_gate"],
            "ejsl_fused_wf1": external["fused"]["weighted_f1"],
            "ejsl_context_wf1": external["context"]["weighted_f1"],
            "ejsl_local_wf1": external["local"]["weighted_f1"],
            "ejsl_gate": external["mean_gate"],
            "source_summary": str(path),
            "external_summary": str(external_path),
        }
        rows.append(row)
    return rows


def aggregate(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, str, str], List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["corpus"]), str(row["condition"]), str(row["profile"]))].append(row)
    output: List[Dict[str, object]] = []
    metrics = [
        "source_fused_wf1",
        "source_context_wf1",
        "source_local_wf1",
        "source_gate",
        "ejsl_fused_wf1",
        "ejsl_context_wf1",
        "ejsl_local_wf1",
        "ejsl_gate",
    ]
    for key, group in sorted(groups.items()):
        result: Dict[str, object] = dict(zip(("corpus", "condition", "profile"), key))
        result["n_seeds"] = len(group)
        result["seeds"] = " ".join(str(row["seed"]) for row in sorted(group, key=lambda x: int(x["seed"])))
        for metric in metrics:
            mean, std = mean_std(float(row[metric]) for row in group)
            result[f"{metric}_mean"] = mean
            result[f"{metric}_std"] = std
        output.append(result)
    return output


def paired(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    index = {
        (str(row["corpus"]), str(row["profile"]), int(row["seed"]), str(row["condition"])): row for row in rows
    }
    output: List[Dict[str, object]] = []
    for corpus, profile, seed, condition in sorted(index):
        if condition != "joint_b5k":
            continue
        joint = index[(corpus, profile, seed, condition)]
        source = index.get((corpus, profile, seed, "source_only"))
        if source is None:
            continue
        output.append(
            {
                "corpus": corpus,
                "profile": profile,
                "seed": seed,
                "source_only_source_wf1": source["source_fused_wf1"],
                "joint_source_wf1": joint["source_fused_wf1"],
                "delta_source_wf1": float(joint["source_fused_wf1"]) - float(source["source_fused_wf1"]),
                "source_only_ejsl_wf1": source["ejsl_fused_wf1"],
                "joint_ejsl_wf1": joint["ejsl_fused_wf1"],
                "delta_ejsl_wf1": float(joint["ejsl_fused_wf1"]) - float(source["ejsl_fused_wf1"]),
            }
        )
    return output


def report(means: Sequence[Mapping[str, object]], paired_rows: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "# Domain-Routed CMERC results",
        "",
        "| corpus | condition | profile | n | source fused | source context | source local | eJSL fused | eJSL context | eJSL local | gate source/eJSL |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in means:
        lines.append(
            f"| {row['corpus']} | {row['condition']} | {row['profile']} | {row['n_seeds']} | "
            f"{float(row['source_fused_wf1_mean']):.4f} | {float(row['source_context_wf1_mean']):.4f} | "
            f"{float(row['source_local_wf1_mean']):.4f} | {float(row['ejsl_fused_wf1_mean']):.4f} | "
            f"{float(row['ejsl_context_wf1_mean']):.4f} | {float(row['ejsl_local_wf1_mean']):.4f} | "
            f"{float(row['source_gate_mean']):.3f}/{float(row['ejsl_gate_mean']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Paired BOBSL effect",
            "",
            "| corpus | profile | seed | delta source | delta eJSL |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in paired_rows:
        lines.append(
            f"| {row['corpus']} | {row['profile']} | {row['seed']} | "
            f"{float(row['delta_source_wf1']):+.4f} | {float(row['delta_ejsl_wf1']):+.4f} |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Domain-Routed CMERC runs.")
    parser.add_argument("--root", default="/raid_zoe/home/lr/maokeyu/sign/domain_routed_cmerc/screen")
    return parser.parse_args()


def main() -> None:
    root = Path(parse_args().root).expanduser().resolve()
    rows = collect(root)
    means = aggregate(rows)
    paired_rows = paired(rows)
    write_csv(root / "all_runs.csv", rows)
    write_csv(root / "group_means.csv", means)
    write_csv(root / "paired_deltas.csv", paired_rows)
    text = report(means, paired_rows)
    (root / "report.md").write_text(text, encoding="utf-8")
    print(text, end="")
    print(f"[DR-CMERC-summary] root={root}")


if __name__ == "__main__":
    main()
