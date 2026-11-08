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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def collect(root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for source_path in sorted(root.glob("*/*/*_seed*/source_test_best_summary.json")):
        external_path = source_path.with_name("external_test_best_summary.json")
        if not external_path.is_file():
            continue
        source = json.loads(source_path.read_text(encoding="utf-8"))
        external = json.loads(external_path.read_text(encoding="utf-8"))
        profile, seed = source_path.parent.name.rsplit("_seed", 1)
        corpus, condition = source_path.parts[-4:-2]
        row: Dict[str, object] = {
            "corpus": corpus,
            "condition": condition,
            "profile": profile,
            "seed": int(seed),
            "best_epoch": source["best_epoch"],
        }
        for prefix, payload in (("source", source), ("ejsl", external)):
            for branch in ("fused", "context", "source_local", "bobsl_local"):
                row[f"{prefix}_{branch}_wf1"] = payload[branch]["weighted_f1"]
            for gate_name, value in payload["mean_gate"].items():
                row[f"{prefix}_gate_{gate_name}"] = value
        rows.append(row)
    return rows


def aggregate(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, str, str], List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["corpus"]), str(row["condition"]), str(row["profile"]))].append(row)
    metrics = [field for field in rows[0] if field.startswith(("source_", "ejsl_"))] if rows else []
    output: List[Dict[str, object]] = []
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


def pair(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
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


def report(means: Sequence[Mapping[str, object]], paired: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "# Domain-Expert CMERC results",
        "",
        "| corpus | condition | profile | n | source fused/context/source-local/BOBSL-local | eJSL fused/context/source-local/BOBSL-local | gate source c/s/b | gate eJSL c/s/b |",
        "|---|---|---|---:|---|---|---|---|",
    ]
    for row in means:
        source_scores = "/".join(f"{float(row[f'source_{branch}_wf1_mean']):.4f}" for branch in ("fused", "context", "source_local", "bobsl_local"))
        ejsl_scores = "/".join(f"{float(row[f'ejsl_{branch}_wf1_mean']):.4f}" for branch in ("fused", "context", "source_local", "bobsl_local"))
        source_gate = "/".join(f"{float(row[f'source_gate_{branch}_mean']):.3f}" for branch in ("context", "source_local", "bobsl_local"))
        ejsl_gate = "/".join(f"{float(row[f'ejsl_gate_{branch}_mean']):.3f}" for branch in ("context", "source_local", "bobsl_local"))
        lines.append(
            f"| {row['corpus']} | {row['condition']} | {row['profile']} | {row['n_seeds']} | "
            f"{source_scores} | {ejsl_scores} | {source_gate} | {ejsl_gate} |"
        )
    lines.extend(["", "## Paired BOBSL effect", "", "| corpus | profile | seed | delta source | delta eJSL |", "|---|---|---:|---:|---:|"])
    for row in paired:
        lines.append(
            f"| {row['corpus']} | {row['profile']} | {row['seed']} | "
            f"{float(row['delta_source_wf1']):+.4f} | {float(row['delta_ejsl_wf1']):+.4f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Domain-Expert CMERC runs.")
    parser.add_argument("--root", default="/raid_zoe/home/lr/maokeyu/sign/domain_expert_cmerc/screen")
    root = Path(parser.parse_args().root).expanduser().resolve()
    rows = collect(root)
    means = aggregate(rows)
    paired = pair(rows)
    write_csv(root / "all_runs.csv", rows)
    write_csv(root / "group_means.csv", means)
    write_csv(root / "paired_deltas.csv", paired)
    text = report(means, paired)
    (root / "report.md").write_text(text, encoding="utf-8")
    print(text, end="")
    print(f"[DE-CMERC-summary] root={root}")


if __name__ == "__main__":
    main()
