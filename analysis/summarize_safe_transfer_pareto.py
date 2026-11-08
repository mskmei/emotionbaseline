#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np


def read_csv(path: str | Path) -> List[Dict[str, str]]:
    with Path(path).expanduser().open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_inputs(values: Sequence[str]) -> Dict[str, Path]:
    output: Dict[str, Path] = {}
    for value in values:
        baseline, path = value.split("=", 1)
        output[baseline] = Path(path).expanduser()
    return output


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


def collect(inputs: Mapping[str, Path]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for baseline, root in inputs.items():
        for source_path in sorted(root.glob("*/*_seed*/source_test_best_summary.json")):
            external_path = source_path.with_name("external_test_best_summary.json")
            if not external_path.is_file():
                continue
            corpus = source_path.parts[-3]
            config, seed_text = source_path.parent.name.rsplit("_seed", 1)
            source = json.loads(source_path.read_text(encoding="utf-8"))
            external = json.loads(external_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "baseline": baseline,
                    "corpus": corpus,
                    "config": config,
                    "seed": int(seed_text),
                    "source_wf1": float(source["fused"]["weighted_f1"]),
                    "external_wf1": float(external["fused"]["weighted_f1"]),
                    "source_floor": float(source.get("source_floor", 0.0)),
                    "source_feasible": bool(source.get("source_feasible", False)),
                    "best_epoch": int(source.get("best_epoch", 0)),
                }
            )
    return rows


def summarize(
    rows: Sequence[Mapping[str, object]],
    main_rows: Sequence[Mapping[str, str]],
    required_seeds: int,
    epsilon: float,
) -> List[Dict[str, object]]:
    main = {(row["baseline"], row["corpus"]): row for row in main_rows}
    groups: Dict[Tuple[str, str, str], List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["baseline"]), str(row["corpus"]), str(row["config"]))].append(row)
    output: List[Dict[str, object]] = []
    for (baseline, corpus, config), group in sorted(groups.items()):
        if len(group) < required_seeds:
            continue
        old = main[(baseline, corpus)]
        source = float(np.mean([float(row["source_wf1"]) for row in group]))
        external = float(np.mean([float(row["external_wf1"]) for row in group]))
        source_delta = source - float(old["source_fused_wf1"])
        external_delta = external - float(old["external_fused_wf1"])
        output.append(
            {
                "baseline": baseline,
                "corpus": corpus,
                "config": config,
                "n_seeds": len(group),
                "source_wf1": source,
                "source_delta": source_delta,
                "external_wf1": external,
                "external_delta": external_delta,
                "strict_pareto": source_delta > epsilon and external_delta > epsilon,
                "both_positive_seeds": sum(
                    float(row["source_wf1"]) > float(row["source_floor"])
                    and float(row["external_wf1"]) > float(old["external_fused_wf1"])
                    for row in group
                ),
                "source_feasible_seeds": sum(bool(row["source_feasible"]) for row in group),
                "mean_best_epoch": float(np.mean([int(row["best_epoch"]) for row in group])),
                "pareto_margin": min(source_delta, external_delta),
            }
        )
    return sorted(
        output,
        key=lambda row: (
            bool(row["strict_pareto"]),
            float(row["pareto_margin"]),
            float(row["source_delta"] + row["external_delta"]),
        ),
        reverse=True,
    )


def write_report(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    lines = [
        "# Strict Pareto Sweep",
        "",
        "A row passes only when both mean source and mean eJSL weighted F1 exceed the previous main result.",
        "",
        "| Baseline | Corpus | Config | Source delta | eJSL delta | Both-positive seeds | Status |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        if not row["strict_pareto"]:
            continue
        lines.append(
            f"| {row['baseline']} | {row['corpus']} | {row['config']} | "
            f"{float(row['source_delta']):+.4f} | {float(row['external_delta']):+.4f} | "
            f"{row['both_positive_seeds']}/{row['n_seeds']} | PASS |"
        )
    if len(lines) == 6:
        lines.append("| - | - | - | - | - | - | No completed Pareto candidate yet |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, help="BASELINE=run_root")
    parser.add_argument("--main_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--required_seeds", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    args = parser.parse_args()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = summarize(
        collect(parse_inputs(args.input)),
        read_csv(args.main_csv),
        args.required_seeds,
        args.epsilon,
    )
    write_csv(out_dir / "pareto_ranking.csv", rows)
    write_report(out_dir / "pareto_report.md", rows)
    print(f"[pareto-summary] complete_groups={len(rows)} pass={sum(bool(row['strict_pareto']) for row in rows)}")


if __name__ == "__main__":
    main()
