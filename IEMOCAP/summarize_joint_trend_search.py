#!/usr/bin/env python3
"""Summarize main and sweep results for BOBSL joint-training trends."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List


BASELINES = ("MMGCN", "MAGTKD", "ECERC", "CMERC", "ConxGNN")


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(row: Dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in ("", None) else 0.0


def collect_sweeps(server_root: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for baseline in BASELINES:
        baseline_lower = baseline.lower()
        sweep_root = (
            server_root
            / f"{baseline_lower}_bobsl_iemocap_ejsl"
            / "video_joint_trend_sweep"
        )
        for csv_path in sorted(sweep_root.glob("*/summary/iemocap_joint_config_means.csv")):
            rows = read_csv(csv_path)
            tags = {row["config"] for row in rows if row["config"] != "iemocap_only"}
            for config in sorted(tags):
                source = next(
                    row for row in rows if row["config"] == config and row["split"] == "IEMOCAP"
                )
                external = next(
                    row for row in rows if row["config"] == config and row["split"] == "eJSL"
                )
                source_base = next(
                    row for row in rows if row["config"] == "iemocap_only" and row["split"] == "IEMOCAP"
                )
                external_base = next(
                    row for row in rows if row["config"] == "iemocap_only" and row["split"] == "eJSL"
                )
                delta_source = as_float(source, "delta_weighted_f1_vs_iemocap_only_mean")
                delta_ejsl = as_float(external, "delta_weighted_f1_vs_iemocap_only_mean")
                n_seeds = int(source["n_seeds"])
                records.append(
                    {
                        "baseline": baseline,
                        "experiment": csv_path.parents[1].name,
                        "config": config,
                        "n_seeds": n_seeds,
                        "seeds": source.get("seeds", ""),
                        "source_base_wf1": as_float(source_base, "weighted_f1_mean"),
                        "source_joint_wf1": as_float(source, "weighted_f1_mean"),
                        "delta_source_wf1": delta_source,
                        "ejsl_base_wf1": as_float(external_base, "weighted_f1_mean"),
                        "ejsl_joint_wf1": as_float(external, "weighted_f1_mean"),
                        "delta_ejsl_wf1": delta_ejsl,
                        "target": delta_source > 0.0 and delta_ejsl < 0.0,
                        "lr": source.get("lr", ""),
                        "dropout": source.get("dropout", ""),
                        "class_weight": source.get("class_weight", ""),
                        "temperature": source.get("temperature", ""),
                        "loss": source.get("loss", ""),
                        "source_csv": str(csv_path),
                    }
                )
    return records


def write_records(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    fieldnames = [
        "baseline",
        "experiment",
        "config",
        "n_seeds",
        "seeds",
        "source_base_wf1",
        "source_joint_wf1",
        "delta_source_wf1",
        "ejsl_base_wf1",
        "ejsl_joint_wf1",
        "delta_ejsl_wf1",
        "target",
        "lr",
        "dropout",
        "class_weight",
        "temperature",
        "loss",
        "source_csv",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main_table(rows: List[Dict[str, str]], corpus: str) -> List[str]:
    lines = [
        f"### {corpus}",
        "",
        "| baseline | config | n | source delta | eJSL delta | target seeds | status |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    selected = [row for row in rows if row["corpus"] == corpus]
    for row in selected:
        lines.append(
            f"| {row['baseline']} | {row['config']} | {row['n_seeds']} | "
            f"{float(row['delta_source_wf1']):+.4f} | {float(row['delta_ejsl_wf1']):+.4f} | "
            f"{row['target_seed_count']}/{row['n_seeds']} | {row['status']} |"
        )
    return lines


def sweep_table(rows: List[Dict[str, object]], confirmed: bool) -> List[str]:
    title = "Five-seed sweep results" if confirmed else "Single-seed screening results"
    lines = [
        f"## {title}",
        "",
        "| baseline | experiment | config | n | source delta | eJSL delta | status |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    chosen = [row for row in rows if (int(row["n_seeds"]) >= 5) == confirmed]
    chosen.sort(
        key=lambda row: (
            str(row["baseline"]),
            not bool(row["target"]),
            -float(row["delta_source_wf1"]),
            float(row["delta_ejsl_wf1"]),
        )
    )
    for row in chosen:
        status = "TARGET" if row["target"] else "other"
        lines.append(
            f"| {row['baseline']} | {row['experiment']} | {row['config']} | {row['n_seeds']} | "
            f"{float(row['delta_source_wf1']):+.4f} | {float(row['delta_ejsl_wf1']):+.4f} | {status} |"
        )
    return lines


def conclusion_lines(main_rows: List[Dict[str, str]], sweeps: List[Dict[str, object]]) -> List[str]:
    lines = ["## Confirmed IEMOCAP outcome", ""]
    for baseline in BASELINES:
        candidates: List[Dict[str, object]] = []
        for row in main_rows:
            if row["corpus"] == "IEMOCAP" and row["baseline"] == baseline:
                candidates.append(
                    {
                        "experiment": "main",
                        "config": row["config"],
                        "delta_source_wf1": float(row["delta_source_wf1"]),
                        "delta_ejsl_wf1": float(row["delta_ejsl_wf1"]),
                        "target": row["status"] == "TARGET_SOURCE_UP_EJSL_DOWN",
                    }
                )
        candidates.extend(
            row for row in sweeps if row["baseline"] == baseline and int(row["n_seeds"]) >= 5
        )
        targets = [row for row in candidates if row["target"]]
        if targets:
            best = max(targets, key=lambda row: float(row["delta_source_wf1"]))
            lines.append(
                f"- {baseline}: confirmed target at `{best['experiment']}/{best['config']}` "
                f"(source {float(best['delta_source_wf1']):+.4f}, "
                f"eJSL {float(best['delta_ejsl_wf1']):+.4f})."
            )
        else:
            source_up = [row for row in candidates if float(row["delta_source_wf1"]) > 0.0]
            closest = min(source_up, key=lambda row: float(row["delta_ejsl_wf1"])) if source_up else None
            if closest:
                lines.append(
                    f"- {baseline}: no confirmed target; closest source-improving result is "
                    f"`{closest['experiment']}/{closest['config']}` "
                    f"(source {float(closest['delta_source_wf1']):+.4f}, "
                    f"eJSL {float(closest['delta_ejsl_wf1']):+.4f})."
                )
            else:
                lines.append(f"- {baseline}: no confirmed source-improving result.")
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_root", type=Path, default=Path("/raid_zoe/home/lr/maokeyu/sign"))
    parser.add_argument(
        "--main_comparison_csv",
        type=Path,
        default=Path(
            "/raid_zoe/home/lr/maokeyu/sign/meld_iemocap_joint_trend_comparison/"
            "joint_cross_dataset_config_means.csv"
        ),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("/raid_zoe/home/lr/maokeyu/sign/iemocap_joint_trend_search_summary"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    main_rows = read_csv(args.main_comparison_csv)
    sweeps = collect_sweeps(args.server_root)
    confirmed = [row for row in sweeps if int(row["n_seeds"]) >= 5]
    screens = [row for row in sweeps if int(row["n_seeds"]) < 5]
    write_records(args.out_dir / "five_seed_sweep_results.csv", confirmed)
    write_records(args.out_dir / "single_seed_screening_results.csv", screens)

    report = [
        "# BOBSL joint-training trend summary",
        "",
        "Target means source-corpus weighted F1 increases while eJSL weighted F1 decreases.",
        "Single-seed screens are hypothesis generation only and are not confirmed findings.",
        "",
        "## Main experiments",
        "",
    ]
    report.extend(main_table(main_rows, "MELD"))
    report.append("")
    report.extend(main_table(main_rows, "IEMOCAP"))
    report.append("")
    report.extend(conclusion_lines(main_rows, sweeps))
    report.append("")
    report.extend(sweep_table(sweeps, confirmed=True))
    report.append("")
    report.extend(sweep_table(sweeps, confirmed=False))
    report.append("")
    report_path = args.out_dir / "joint_trend_search_report.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(f"[joint-trend-search-summary] report={report_path}")
    print(f"[joint-trend-search-summary] confirmed_rows={len(confirmed)} screens={len(screens)}")


if __name__ == "__main__":
    main()
