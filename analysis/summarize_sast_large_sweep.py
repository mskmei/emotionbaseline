#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


BASELINES = {"MMGCN", "MAGTKD", "CMERC", "CONXGNN", "ECERC"}


def load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def identity(root: Path, run_dir: Path):
    parts = run_dir.relative_to(root).parts
    baseline = next((part.upper() for part in parts if part.upper() in BASELINES), "")
    corpus = next((part.upper() for part in parts if part.lower() in {"meld", "iemocap"}), "")
    bobsl_size = next((part.lower() for part in parts if re.fullmatch(r"b\d+k", part.lower())), "b5k")
    match = re.fullmatch(r"(.+)_seed(\d+)", run_dir.name)
    if not baseline or not corpus or not match:
        return None
    return baseline, corpus, bobsl_size, match.group(1), int(match.group(2))


def report_row(root: Path, run_dir: Path, source, external, base_source, base_external):
    item = identity(root, run_dir)
    if item is None or not all((source, external, base_source, base_external)):
        return None
    baseline, corpus, bobsl_size, profile, seed = item
    source_value = float(source["weighted_f1"] if isinstance(source, dict) else source)
    external_value = float(external["weighted_f1"] if isinstance(external, dict) else external)
    base_source_value = float(base_source["weighted_f1"] if isinstance(base_source, dict) else base_source)
    base_external_value = float(base_external["weighted_f1"] if isinstance(base_external, dict) else base_external)
    return {
        "baseline": baseline,
        "corpus": corpus,
        "bobsl_size": bobsl_size,
        "profile": profile,
        "seed": seed,
        "base_source": base_source_value,
        "source": source_value,
        "delta_source": source_value - base_source_value,
        "base_external": base_external_value,
        "external": external_value,
        "delta_external": external_value - base_external_value,
        "run_dir": str(run_dir),
    }


def collect_adapter_rows(root: Path):
    rows = []
    for path in root.rglob("summary.json"):
        data = load(path)
        if not data or "best_source" not in data or "best_external" not in data:
            continue
        row = report_row(
            root,
            path.parent,
            data["best_source"],
            data["best_external"],
            data["base_source"],
            data["base_external"],
        )
        if row:
            rows.append(row)
    for path in root.rglob("external_test_best_summary.json"):
        run_dir = path.parent
        item = identity(root, run_dir)
        if item is None or item[0] != "CMERC":
            continue
        row = report_row(
            root,
            run_dir,
            load(run_dir / "source_test_best_summary.json"),
            load(path),
            load(run_dir / "source_test_base_summary.json"),
            load(run_dir / "external_test_base_summary.json"),
        )
        if row:
            rows.append(row)
    return rows


def ecerc_base_dir(corpus: str, seed: int):
    if corpus == "MELD":
        root = Path("/raid_zoe/home/lr/maokeyu/sign/ecerc_bobsl_meld_ejsl/video_joint_bobsl_meld")
        prefix = "meld_only"
    else:
        root = Path("/raid_zoe/home/lr/maokeyu/sign/ecerc_bobsl_iemocap_ejsl/video_joint_bobsl_iemocap")
        prefix = "iemocap_only"
    return root / f"{prefix}_seed{seed}"


def collect_ecerc_rows(root: Path):
    rows = []
    ecerc_root = root / "ECERC"
    if not ecerc_root.exists():
        return rows
    for external_path in ecerc_root.rglob("external_test_best_summary.json"):
        run_dir = external_path.parent
        item = identity(root, run_dir)
        if item is None:
            continue
        _, corpus, _, _, seed = item
        base_dir = ecerc_base_dir(corpus, seed)
        row = report_row(
            root,
            run_dir,
            load(run_dir / "source_test_best_summary.json"),
            load(external_path),
            load(base_dir / "source_test_best_summary.json"),
            load(base_dir / "external_test_best_summary.json"),
        )
        if row:
            rows.append(row)
    return rows


def write_csv(path: Path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["baseline"], row["corpus"], row["bobsl_size"], row["profile"])].append(row)
    results = []
    metrics = ("base_source", "source", "delta_source", "base_external", "external", "delta_external")
    for key, group in sorted(grouped.items()):
        result = dict(zip(("baseline", "corpus", "bobsl_size", "profile"), key))
        result["n"] = len(group)
        result["seeds"] = " ".join(str(row["seed"]) for row in sorted(group, key=lambda item: item["seed"]))
        for metric in metrics:
            values = [float(row[metric]) for row in group]
            result[f"{metric}_mean"] = mean(values)
            result[f"{metric}_std"] = pstdev(values)
        ds = result["delta_source_mean"]
        de = result["delta_external_mean"]
        result["double_positive"] = int(ds > 0 and de > 0)
        result["strict_seeds"] = sum(row["delta_source"] > 0 and row["delta_external"] > 0 for row in group)
        result["pareto_score"] = math.sqrt(ds * de) if ds > 0 and de > 0 else min(ds, de)
        results.append(result)
    return results


def select_best(groups):
    buckets = defaultdict(list)
    for row in groups:
        buckets[(row["baseline"], row["corpus"], row["bobsl_size"])].append(row)
    selected = []
    for _, rows in sorted(buckets.items()):
        rows.sort(
            key=lambda row: (
                row["double_positive"],
                row["pareto_score"],
                row["delta_source_mean"] + row["delta_external_mean"],
            ),
            reverse=True,
        )
        selected.append(rows[0])
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/raid_zoe/home/lr/maokeyu/sign/sast_large_sweep")
    args = parser.parse_args()
    root = Path(args.root).expanduser()
    rows = collect_adapter_rows(root) + collect_ecerc_rows(root)
    rows.sort(key=lambda row: (row["baseline"], row["corpus"], row["bobsl_size"], row["profile"], row["seed"]))
    groups = summarize(rows)
    best = select_best(groups)
    write_csv(root / "all_seed_results.csv", rows)
    write_csv(root / "all_group_results.csv", groups)
    write_csv(root / "best_results.csv", best)
    lines = [
        "# SAST large sweep best results",
        "",
        "Selection maximizes the geometric mean of positive source/eJSL deltas.",
        "",
        "| baseline | corpus | BOBSL | profile | n | source base->method | eJSL base->method | delta source/eJSL | strict seeds |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in best:
        lines.append(
            f"| {row['baseline']} | {row['corpus']} | {row['bobsl_size']} | {row['profile']} | {row['n']} | "
            f"{row['base_source_mean']:.4f}->{row['source_mean']:.4f} | "
            f"{row['base_external_mean']:.4f}->{row['external_mean']:.4f} | "
            f"{row['delta_source_mean']:+.4f}/{row['delta_external_mean']:+.4f} | "
            f"{row['strict_seeds']}/{row['n']} |"
        )
    report = "\n".join(lines) + "\n"
    (root / "best_results.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
