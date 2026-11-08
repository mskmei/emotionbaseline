#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np


LABELS = ("A", "N", "J", "S")


def read_csv(path: str | Path) -> List[Dict[str, str]]:
    with Path(path).expanduser().open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_oracle(values: Sequence[str]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    groups: List[Dict[str, str]] = []
    seeds: List[Dict[str, str]] = []
    for value in values:
        try:
            baseline, paths = value.split("=", 1)
            group_path, seed_path = paths.split(",", 1)
        except ValueError as error:
            raise ValueError(f"Expected BASELINE=GROUP_CSV,PER_SEED_CSV, got {value!r}") from error
        for row in read_csv(group_path):
            row["baseline"] = baseline
            groups.append(row)
        for row in read_csv(seed_path):
            row["baseline"] = baseline
            seeds.append(row)
    return groups, seeds


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


def indexed(rows: Sequence[Mapping[str, str]], fields: Sequence[str]) -> Dict[Tuple[str, ...], Mapping[str, str]]:
    return {tuple(row[field] for field in fields): row for row in rows}


def group_result(main: Mapping[str, str], source: Mapping[str, str], external: Mapping[str, str], regime: str) -> Dict[str, object]:
    result: Dict[str, object] = {
        "baseline": main["baseline"],
        "corpus": main["corpus"],
        "regime": regime,
        "main_profile": main["profile"],
        "oracle_profile": external["profile"],
        "n_seeds": int(external["n_seeds"]),
        "main_source_wf1": float(main["source_fused_wf1"]),
        "oracle_source_wf1": float(source["fused_weighted_f1_mean"]),
        "delta_source_vs_main": float(source["fused_weighted_f1_mean"]) - float(main["source_fused_wf1"]),
        "main_ejsl_wf1": float(main["external_fused_wf1"]),
        "oracle_ejsl_wf1": float(external["fused_weighted_f1_mean"]),
        "delta_ejsl_vs_main": float(external["fused_weighted_f1_mean"]) - float(main["external_fused_wf1"]),
        "ejsl_source_expert_wf1": float(external["source_weighted_f1_mean"]),
        "oracle_ejsl_gain_vs_source": float(external["delta_fused_source_mean"]),
    }
    for label in LABELS:
        result[f"oracle_ejsl_{label}_f1"] = float(external[f"fused_{label}_f1_mean"])
    return result


def compare_groups(main_rows: Sequence[Mapping[str, str]], oracle_rows: Sequence[Mapping[str, str]]) -> List[Dict[str, object]]:
    lookup = indexed(oracle_rows, ("baseline", "corpus", "profile", "split"))
    by_case: Dict[Tuple[str, str], List[Mapping[str, str]]] = defaultdict(list)
    for row in oracle_rows:
        if row["split"] == "external":
            by_case[(row["baseline"], row["corpus"])].append(row)
    output: List[Dict[str, object]] = []
    for main in main_rows:
        case = (main["baseline"], main["corpus"])
        fixed_external = lookup[(*case, main["profile"], "external")]
        fixed_source = lookup[(*case, main["profile"], "source")]
        output.append(group_result(main, fixed_source, fixed_external, "fixed_main_profile_epoch_oracle"))
        best_external = max(by_case[case], key=lambda row: float(row["fused_weighted_f1_mean"]))
        best_source = lookup[(*case, best_external["profile"], "source")]
        output.append(group_result(main, best_source, best_external, "profile_and_epoch_oracle"))
    return output


def per_seed_upper_bound(main_rows: Sequence[Mapping[str, str]], seed_rows: Sequence[Mapping[str, str]]) -> List[Dict[str, object]]:
    source_lookup = indexed(seed_rows, ("baseline", "corpus", "profile", "seed", "split"))
    groups: Dict[Tuple[str, str, str], List[Mapping[str, str]]] = defaultdict(list)
    for row in seed_rows:
        if row["split"] == "external":
            groups[(row["baseline"], row["corpus"], row["seed"])].append(row)
    main_lookup = indexed(main_rows, ("baseline", "corpus"))
    selected: Dict[Tuple[str, str], List[Tuple[Mapping[str, str], Mapping[str, str]]]] = defaultdict(list)
    for (baseline, corpus, seed), rows in groups.items():
        external = max(rows, key=lambda row: float(row["fused_weighted_f1"]))
        source = source_lookup[(baseline, corpus, external["profile"], seed, "source")]
        selected[(baseline, corpus)].append((source, external))
    output: List[Dict[str, object]] = []
    for case, pairs in sorted(selected.items()):
        main = main_lookup[case]
        source_scores = [float(source["fused_weighted_f1"]) for source, _ in pairs]
        external_scores = [float(external["fused_weighted_f1"]) for _, external in pairs]
        profiles = sorted(external["profile"] for _, external in pairs)
        output.append(
            {
                "baseline": case[0],
                "corpus": case[1],
                "regime": "per_seed_profile_and_epoch_oracle",
                "n_seeds": len(pairs),
                "selected_profiles": ";".join(profiles),
                "main_source_wf1": float(main["source_fused_wf1"]),
                "oracle_source_wf1": float(np.mean(source_scores)),
                "delta_source_vs_main": float(np.mean(source_scores)) - float(main["source_fused_wf1"]),
                "main_ejsl_wf1": float(main["external_fused_wf1"]),
                "oracle_ejsl_wf1": float(np.mean(external_scores)),
                "delta_ejsl_vs_main": float(np.mean(external_scores)) - float(main["external_fused_wf1"]),
            }
        )
    return output


def write_report(path: Path, group_rows: Sequence[Mapping[str, object]], seed_rows: Sequence[Mapping[str, object]]) -> None:
    rows = list(group_rows) + list(seed_rows)
    lines = [
        "# eJSL Oracle Checkpoint Analysis",
        "",
        "These exploratory results use eJSL labels for checkpoint selection. They are an upper-bound diagnostic and must not be reported as label-free transfer.",
        "",
        "| Baseline | Source | Oracle regime | Profile | Source wF1 | eJSL wF1 | eJSL gain over main |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: (str(item["corpus"]), str(item["baseline"]), str(item["regime"]))):
        profile = row.get("oracle_profile", row.get("selected_profiles", ""))
        lines.append(
            f"| {row['baseline']} | {row['corpus']} | {row['regime']} | {profile} | "
            f"{float(row['oracle_source_wf1']):.4f} | {float(row['oracle_ejsl_wf1']):.4f} | "
            f"{float(row['delta_ejsl_vs_main']):+.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main_csv", required=True)
    parser.add_argument("--oracle", action="append", required=True, help="BASELINE=GROUP_CSV,PER_SEED_CSV")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    main_rows = read_csv(args.main_csv)
    oracle_groups, oracle_seeds = parse_oracle(args.oracle)
    group_rows = compare_groups(main_rows, oracle_groups)
    seed_rows = per_seed_upper_bound(main_rows, oracle_seeds)
    write_csv(out_dir / "oracle_group_comparison.csv", group_rows)
    write_csv(out_dir / "oracle_per_seed_upper_bound.csv", seed_rows)
    write_report(out_dir / "oracle_report.md", group_rows, seed_rows)
    print(f"[oracle-comparison] wrote {out_dir}")


if __name__ == "__main__":
    main()
