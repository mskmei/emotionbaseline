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
from typing import Dict, Iterable, List, Mapping, Sequence


LABELS = ("A", "N", "J", "S")
BASELINE_NAMES = {
    "MMGCN": "MMGCN",
    "MAGTKD": "MAGTKD",
    "CMERC": "CMERC",
    "CONXGNN": "ConxGNN",
    "ECERC": "ECERC",
}


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def weighted_f1(report: Mapping) -> float:
    return float(report["weighted_f1"])


def identity(root: Path, run_dir: Path):
    parts = run_dir.relative_to(root).parts
    baseline = next((BASELINE_NAMES[p.upper()] for p in parts if p.upper() in BASELINE_NAMES), None)
    corpus = next((p.upper() for p in parts if p.lower() in {"meld", "iemocap"}), None)
    bobsl = next((p.lower() for p in parts if re.fullmatch(r"b\d+k", p.lower())), None)
    match = re.fullmatch(r"(.+)_seed(\d+)", run_dir.name)
    if baseline is None or corpus is None or bobsl is None or match is None:
        return None
    return baseline, corpus, bobsl, match.group(1), int(match.group(2))


def collect(root: Path) -> List[Dict]:
    rows: List[Dict] = []
    for source_path in sorted(root.rglob("source_test_best_summary.json")):
        run_dir = source_path.parent
        item = identity(root, run_dir)
        paths = {
            "source": source_path,
            "external": run_dir / "external_test_best_summary.json",
            "base_source": run_dir / "source_test_base_summary.json",
            "base_external": run_dir / "external_test_base_summary.json",
        }
        if item is None or not all(path.is_file() for path in paths.values()):
            continue
        reports = {name: load_json(path) for name, path in paths.items()}
        baseline, corpus, bobsl, profile, seed = item
        row: Dict[str, object] = {
            "baseline": baseline,
            "corpus": corpus,
            "bobsl": bobsl,
            "profile": profile,
            "seed": seed,
            "base_source": weighted_f1(reports["base_source"]),
            "source": weighted_f1(reports["source"]),
            "base_ejsl": weighted_f1(reports["base_external"]),
            "ejsl": weighted_f1(reports["external"]),
            "run_dir": str(run_dir),
        }
        row["delta_source"] = float(row["source"]) - float(row["base_source"])
        row["delta_ejsl"] = float(row["ejsl"]) - float(row["base_ejsl"])
        for split, report in (("source", reports["source"]), ("ejsl", reports["external"])):
            per_class = report.get("per_class", {})
            for label in LABELS:
                row[f"{split}_{label}_f1"] = float(per_class.get(label, {}).get("f1", 0.0))
        rows.append(row)
    return rows


def summarize(rows: Iterable[Mapping]) -> List[Dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["baseline"], row["corpus"], row["bobsl"], row["profile"])].append(row)
    output: List[Dict] = []
    metrics = ("base_source", "source", "delta_source", "base_ejsl", "ejsl", "delta_ejsl")
    for key, group in sorted(groups.items()):
        result: Dict[str, object] = dict(zip(("baseline", "corpus", "bobsl", "profile"), key))
        result["n"] = len(group)
        result["seeds"] = " ".join(str(row["seed"]) for row in sorted(group, key=lambda x: x["seed"]))
        for metric in metrics:
            values = [float(row[metric]) for row in group]
            result[f"{metric}_mean"] = mean(values)
            result[f"{metric}_std"] = pstdev(values)
        result["strict_positive_seeds"] = sum(
            float(row["delta_source"]) > 0 and float(row["delta_ejsl"]) > 0 for row in group
        )
        ds = float(result["delta_source_mean"])
        de = float(result["delta_ejsl_mean"])
        result["both_positive"] = int(ds > 0 and de > 0)
        result["balanced_gain"] = math.sqrt(ds * de) if ds > 0 and de > 0 else min(ds, de)
        for split in ("source", "ejsl"):
            for label in LABELS:
                result[f"{split}_{label}_f1_mean"] = mean(float(row[f"{split}_{label}_f1"]) for row in group)
        output.append(result)
    return output


def select_best(groups: Iterable[Mapping], min_seeds: int) -> List[Mapping]:
    buckets = defaultdict(list)
    for row in groups:
        if int(row["n"]) >= min_seeds and int(row["both_positive"]):
            buckets[(row["baseline"], row["corpus"], row["bobsl"])].append(row)
    selected = []
    for rows in buckets.values():
        rows.sort(
            key=lambda row: (
                float(row["balanced_gain"]),
                float(row["delta_source_mean"]) + float(row["delta_ejsl_mean"]),
                int(row["strict_positive_seeds"]),
            ),
            reverse=True,
        )
        selected.append(rows[0])
    return sorted(selected, key=lambda row: (row["corpus"], row["baseline"], row["bobsl"]))


def write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def report(best: Sequence[Mapping], min_seeds: int) -> str:
    lines = [
        "# Proposal 2: source-anchored single-checkpoint adapter",
        "",
        f"Only groups with n >= {min_seeds} and positive mean Source/eJSL deltas are eligible.",
        "Checkpoint selection uses the source metric; eJSL is evaluation only.",
        "",
        "| corpus | backbone | BOBSL | profile | n | source-only -> ours | eJSL base -> ours | delta S/E | strict seeds |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in best:
        lines.append(
            f"| {row['corpus']} | {row['baseline']} | {row['bobsl']} | {row['profile']} | {row['n']} | "
            f"{row['base_source_mean']:.4f} -> {row['source_mean']:.4f} | "
            f"{row['base_ejsl_mean']:.4f} -> {row['ejsl_mean']:.4f} | "
            f"{row['delta_source_mean']:+.4f} / {row['delta_ejsl_mean']:+.4f} | "
            f"{row['strict_positive_seeds']}/{row['n']} |"
        )
    covered = {(row["baseline"], row["corpus"]) for row in best}
    expected = {(baseline, corpus) for baseline in BASELINE_NAMES.values() for corpus in ("MELD", "IEMOCAP")}
    missing = sorted(expected - covered)
    lines.extend(["", f"Coverage: {len(covered)}/10 model-corpus pairs."])
    if missing:
        lines.append("Missing double-positive groups: " + ", ".join(f"{corpus}/{model}" for model, corpus in missing))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/raid_zoe/home/lr/maokeyu/sign/proposal2_source_anchor")
    parser.add_argument("--min_seeds", type=int, default=5)
    args = parser.parse_args()
    root = Path(args.root).expanduser()
    rows = collect(root)
    groups = summarize(rows)
    best = select_best(groups, args.min_seeds)
    write_csv(root / "proposal2_seed_results.csv", rows)
    write_csv(root / "proposal2_group_results.csv", groups)
    write_csv(root / f"proposal2_best_n{args.min_seeds}.csv", best)
    markdown = report(best, args.min_seeds)
    (root / f"proposal2_main_table_n{args.min_seeds}.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
