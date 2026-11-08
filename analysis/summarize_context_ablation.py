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
    std = math.sqrt(sum((value - mean) ** 2 for value in clean) / len(clean))
    return mean, std


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


def load_rows(root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for path in sorted(root.glob("*/*/*/seed*/*/summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload:
            rows.append({**row, "summary_path": str(path)})
    return rows


def aggregate(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, ...], List[Mapping[str, object]]] = defaultdict(list)
    keys = ("baseline", "corpus", "config", "split", "context_mode")
    for row in rows:
        groups[tuple(str(row[key]) for key in keys)].append(row)
    output: List[Dict[str, object]] = []
    for group_key, group in sorted(groups.items()):
        result: Dict[str, object] = dict(zip(keys, group_key))
        result["n_seeds"] = len(group)
        result["seeds"] = " ".join(str(row["seed"]) for row in sorted(group, key=lambda x: int(x["seed"])))
        for metric in ("weighted_f1", "macro_f1", "accuracy"):
            mean, std = mean_std(float(row[metric]) for row in group)
            result[f"{metric}_mean"] = mean
            result[f"{metric}_std"] = std
        output.append(result)
    return output


def interactions(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    index = {
        (str(row["baseline"]), str(row["corpus"]), str(row["split"]), int(row["seed"]), str(row["config"]), str(row["context_mode"])): row
        for row in rows
    }
    seed_rows: List[Dict[str, object]] = []
    keys = sorted({key[:4] for key in index})
    for baseline, corpus, split, seed in keys:
        needed = {
            name: index.get((baseline, corpus, split, seed, config, mode))
            for name, config, mode in (
                ("base_normal", "source_only", "normal"),
                ("joint_normal", "joint_b5k", "normal"),
                ("base_isolate", "source_only", "isolate"),
                ("joint_isolate", "joint_b5k", "isolate"),
            )
        }
        if not all(needed.values()):
            continue
        row: Dict[str, object] = {"baseline": baseline, "corpus": corpus, "split": split, "seed": seed}
        for metric in ("weighted_f1", "macro_f1", "accuracy"):
            values = {name: float(value[metric]) for name, value in needed.items()}
            normal_gain = values["joint_normal"] - values["base_normal"]
            isolate_gain = values["joint_isolate"] - values["base_isolate"]
            row[f"source_only_context_effect_{metric}"] = values["base_normal"] - values["base_isolate"]
            row[f"joint_b5k_context_effect_{metric}"] = values["joint_normal"] - values["joint_isolate"]
            row[f"joint_gain_normal_{metric}"] = normal_gain
            row[f"joint_gain_isolate_{metric}"] = isolate_gain
            row[f"context_interaction_{metric}"] = normal_gain - isolate_gain
        seed_rows.append(row)

    groups: Dict[Tuple[str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in seed_rows:
        groups[(str(row["baseline"]), str(row["corpus"]), str(row["split"]))].append(row)
    means: List[Dict[str, object]] = []
    metric_fields = [field for field in seed_rows[0] if field not in ("baseline", "corpus", "split", "seed")] if seed_rows else []
    for key, group in sorted(groups.items()):
        result: Dict[str, object] = dict(zip(("baseline", "corpus", "split"), key))
        result["n_seeds"] = len(group)
        for field in metric_fields:
            mean, std = mean_std(float(row[field]) for row in group)
            result[f"{field}_mean"] = mean
            result[f"{field}_std"] = std
        means.append(result)
    return seed_rows, means


def report(means: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "# Local-context causal ablation",
        "",
        "Interaction = (joint - source-only under normal context) - (joint - source-only with context=1).",
        "Negative interaction means dialogue context mediates joint-training harm; positive means it mediates benefit.",
        "",
        "| baseline | corpus | split | n | joint gain normal | joint gain context=1 | interaction | source-only context effect | joint context effect |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in means:
        lines.append(
            f"| {row['baseline']} | {row['corpus']} | {row['split']} | {row['n_seeds']} | "
            f"{float(row['joint_gain_normal_weighted_f1_mean']):+.4f} | "
            f"{float(row['joint_gain_isolate_weighted_f1_mean']):+.4f} | "
            f"{float(row['context_interaction_weighted_f1_mean']):+.4f} | "
            f"{float(row['source_only_context_effect_weighted_f1_mean']):+.4f} | "
            f"{float(row['joint_b5k_context_effect_weighted_f1_mean']):+.4f} |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize paired context=1 checkpoint evaluations.")
    parser.add_argument(
        "--root",
        default="/raid_zoe/home/lr/maokeyu/sign/bobsl_b5k_mechanism_analysis/context_ablation",
    )
    return parser.parse_args()


def main() -> None:
    root = Path(parse_args().root).expanduser().resolve()
    rows = load_rows(root)
    group_rows = aggregate(rows)
    seed_interactions, mean_interactions = interactions(rows)
    write_csv(root / "all_results.csv", rows)
    write_csv(root / "group_means.csv", group_rows)
    write_csv(root / "seed_interactions.csv", seed_interactions)
    write_csv(root / "mean_interactions.csv", mean_interactions)
    text = report(mean_interactions)
    (root / "report.md").write_text(text, encoding="utf-8")
    print(text, end="")
    print(f"[context-ablation-summary] root={root}")


if __name__ == "__main__":
    main()
