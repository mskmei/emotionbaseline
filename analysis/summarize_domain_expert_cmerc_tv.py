#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize non-oracle TV Domain-Expert CMERC runs.")
    parser.add_argument("--root", default="/raid_zoe/home/lr/maokeyu/sign/domain_expert_cmerc/tv_nonoracle/runs")
    root = Path(parser.parse_args().root).expanduser().resolve()
    rows = []
    for profile_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        pair = {}
        for condition in ("source_only", "joint_b5k"):
            run_dir = profile_dir / condition
            source_path = run_dir / "source_test_best_summary.json"
            external_path = run_dir / "external_test_best_summary.json"
            if not source_path.is_file() or not external_path.is_file():
                continue
            source = json.loads(source_path.read_text(encoding="utf-8"))
            external = json.loads(external_path.read_text(encoding="utf-8"))
            pair[condition] = {
                "source_wf1": source["fused"]["weighted_f1"],
                "ejsl_wf1": external["fused"]["weighted_f1"],
                "context_source_wf1": source["context"]["weighted_f1"],
                "context_ejsl_wf1": external["context"]["weighted_f1"],
                "best_epoch": source["best_epoch"],
            }
        for condition, values in pair.items():
            baseline = pair.get("source_only")
            rows.append(
                {
                    "profile": profile_dir.name,
                    "condition": condition,
                    **values,
                    "delta_source_wf1": "" if baseline is None else values["source_wf1"] - baseline["source_wf1"],
                    "delta_ejsl_wf1": "" if baseline is None else values["ejsl_wf1"] - baseline["ejsl_wf1"],
                }
            )
    fields = list(rows[0]) if rows else []
    with (root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Domain-Expert CMERC non-oracle TV",
        "",
        "| text dropout | condition | MELD wF1 | eJSL wF1 | delta MELD | delta eJSL | context MELD/eJSL | epoch |",
        "|---|---|---:|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        delta_source = float(row["delta_source_wf1"]) if row["delta_source_wf1"] != "" else 0.0
        delta_ejsl = float(row["delta_ejsl_wf1"]) if row["delta_ejsl_wf1"] != "" else 0.0
        lines.append(
            f"| {row['profile']} | {row['condition']} | {float(row['source_wf1']):.4f} | "
            f"{float(row['ejsl_wf1']):.4f} | {delta_source:+.4f} | {delta_ejsl:+.4f} | "
            f"{float(row['context_source_wf1']):.4f}/{float(row['context_ejsl_wf1']):.4f} | {row['best_epoch']} |"
        )
    report = "\n".join(lines) + "\n"
    (root / "report.md").write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"[DE-CMERC-TV-summary] root={root}")


if __name__ == "__main__":
    main()
