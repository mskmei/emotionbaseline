#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


BASELINE_ORDER = ["MMGCN", "MAGTKD", "ECERC", "CMERC", "ConxGNN"]
CORPUS_SPECS = {
    "MELD": {"source_split": "MELD", "baseline_config": "meld_only"},
    "IEMOCAP": {"source_split": "IEMOCAP", "baseline_config": "iemocap_only"},
}


def as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def fmt(value: object) -> str:
    number = as_float(value)
    return "NA" if math.isnan(number) else f"{number:.4f}"


def mean_std(values: Iterable[float]) -> Tuple[float, float]:
    clean = [value for value in values if not math.isnan(value)]
    if not clean:
        return float("nan"), float("nan")
    mean = sum(clean) / len(clean)
    std = math.sqrt(sum((value - mean) ** 2 for value in clean) / len(clean))
    return mean, std


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
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


def config_sort_key(config: str) -> Tuple[int, int, str]:
    if config.endswith("_only"):
        return (0, 0, config)
    for token in config.split("_"):
        if token.startswith("b") and token.endswith("k") and token[1:-1].isdigit():
            return (1, int(token[1:-1]) * 1000, config)
    return (2, 0, config)


def baseline_sort_key(name: str) -> Tuple[int, str]:
    return (BASELINE_ORDER.index(name), name) if name in BASELINE_ORDER else (len(BASELINE_ORDER), name)


def normalize_rows(corpus: str, raw_rows: Sequence[Dict[str, str]]) -> List[Dict[str, object]]:
    spec = CORPUS_SPECS[corpus]
    source_split = spec["source_split"]
    output: List[Dict[str, object]] = []
    for row in raw_rows:
        split = str(row.get("split", ""))
        if split not in (source_split, "eJSL"):
            continue
        output.append(
            {
                "corpus": corpus,
                "baseline": str(row.get("baseline", "")),
                "config": str(row.get("config", "")),
                "seed": str(row.get("seed", "")),
                "split": "source" if split == source_split else "eJSL",
                "weighted_f1": as_float(row.get("weighted_f1")),
                "bobsl_max": row.get("bobsl_max", ""),
                "epochs": row.get("epochs", ""),
                "lr": row.get("lr", ""),
                "dropout": row.get("dropout", ""),
                "class_weight": row.get("class_weight", ""),
                "temperature": row.get("temperature", ""),
                "loss": row.get("loss", ""),
            }
        )
    return output


def paired_seed_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    by_key = {
        (str(row["corpus"]), str(row["baseline"]), str(row["config"]), str(row["seed"]), str(row["split"])): row
        for row in rows
    }
    output: List[Dict[str, object]] = []
    configs = sorted(
        {(str(row["corpus"]), str(row["baseline"]), str(row["config"]), str(row["seed"])) for row in rows},
        key=lambda item: (item[0], baseline_sort_key(item[1]), config_sort_key(item[2]), item[3]),
    )
    for corpus, baseline, config, seed in configs:
        baseline_config = CORPUS_SPECS[corpus]["baseline_config"]
        if config == baseline_config:
            continue
        source = by_key.get((corpus, baseline, config, seed, "source"))
        external = by_key.get((corpus, baseline, config, seed, "eJSL"))
        base_source = by_key.get((corpus, baseline, baseline_config, seed, "source"))
        base_external = by_key.get((corpus, baseline, baseline_config, seed, "eJSL"))
        if not all((source, external, base_source, base_external)):
            continue
        source_score = as_float(source["weighted_f1"])
        external_score = as_float(external["weighted_f1"])
        base_source_score = as_float(base_source["weighted_f1"])
        base_external_score = as_float(base_external["weighted_f1"])
        if any(math.isnan(value) for value in (source_score, external_score, base_source_score, base_external_score)):
            continue
        output.append(
            {
                "corpus": corpus,
                "baseline": baseline,
                "config": config,
                "seed": seed,
                "bobsl_max": source.get("bobsl_max", ""),
                "epochs": source.get("epochs", ""),
                "lr": source.get("lr", ""),
                "dropout": source.get("dropout", ""),
                "class_weight": source.get("class_weight", ""),
                "temperature": source.get("temperature", ""),
                "loss": source.get("loss", ""),
                "baseline_source_wf1": base_source_score,
                "joint_source_wf1": source_score,
                "delta_source_wf1": source_score - base_source_score,
                "baseline_ejsl_wf1": base_external_score,
                "joint_ejsl_wf1": external_score,
                "delta_ejsl_wf1": external_score - base_external_score,
            }
        )
    return output


def trend_status(delta_source: float, delta_ejsl: float, complete: bool, min_delta: float) -> str:
    if not complete:
        return "INCOMPLETE"
    source_up = delta_source > min_delta
    ejsl_down = delta_ejsl < -min_delta
    if source_up and ejsl_down:
        return "TARGET_SOURCE_UP_EJSL_DOWN"
    if source_up:
        return "BOTH_UP"
    if ejsl_down:
        return "BOTH_DOWN"
    return "SOURCE_DOWN_EJSL_UP"


def aggregate(seed_rows: Sequence[Dict[str, object]], required_seeds: int, min_delta: float) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in seed_rows:
        grouped[(str(row["corpus"]), str(row["baseline"]), str(row["config"]))].append(row)
    output: List[Dict[str, object]] = []
    for (corpus, baseline, config), group in grouped.items():
        group = sorted(group, key=lambda row: str(row["seed"]))
        source_mean, source_std = mean_std(as_float(row["joint_source_wf1"]) for row in group)
        base_source_mean, _ = mean_std(as_float(row["baseline_source_wf1"]) for row in group)
        delta_source_mean, delta_source_std = mean_std(as_float(row["delta_source_wf1"]) for row in group)
        ejsl_mean, ejsl_std = mean_std(as_float(row["joint_ejsl_wf1"]) for row in group)
        base_ejsl_mean, _ = mean_std(as_float(row["baseline_ejsl_wf1"]) for row in group)
        delta_ejsl_mean, delta_ejsl_std = mean_std(as_float(row["delta_ejsl_wf1"]) for row in group)
        complete = len(group) >= required_seeds
        output.append(
            {
                "corpus": corpus,
                "baseline": baseline,
                "config": config,
                "n_seeds": len(group),
                "seeds": " ".join(str(row["seed"]) for row in group),
                "bobsl_max": group[0].get("bobsl_max", ""),
                "epochs": group[0].get("epochs", ""),
                "lr": group[0].get("lr", ""),
                "dropout": group[0].get("dropout", ""),
                "class_weight": group[0].get("class_weight", ""),
                "temperature": group[0].get("temperature", ""),
                "loss": group[0].get("loss", ""),
                "baseline_source_wf1": base_source_mean,
                "joint_source_wf1": source_mean,
                "joint_source_wf1_std": source_std,
                "delta_source_wf1": delta_source_mean,
                "delta_source_wf1_std": delta_source_std,
                "baseline_ejsl_wf1": base_ejsl_mean,
                "joint_ejsl_wf1": ejsl_mean,
                "joint_ejsl_wf1_std": ejsl_std,
                "delta_ejsl_wf1": delta_ejsl_mean,
                "delta_ejsl_wf1_std": delta_ejsl_std,
                "target_seed_count": sum(
                    as_float(row["delta_source_wf1"]) > min_delta and as_float(row["delta_ejsl_wf1"]) < -min_delta
                    for row in group
                ),
                "status": trend_status(delta_source_mean, delta_ejsl_mean, complete, min_delta),
            }
        )
    output.sort(key=lambda row: (row["corpus"], baseline_sort_key(str(row["baseline"])), config_sort_key(str(row["config"]))))
    return output


def build_report(rows: Sequence[Dict[str, object]], required_seeds: int, min_delta: float) -> str:
    lines = [
        "[cross-dataset-joint-trends] paired by baseline, seed, and corpus-only reference",
        f"[cross-dataset-joint-trends] required_seeds={required_seeds} min_delta={min_delta}",
        "",
        "| corpus | baseline | config | n | source base | source joint | delta source | eJSL base | eJSL joint | delta eJSL | target seeds | status |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['corpus']} | {row['baseline']} | {row['config']} | {row['n_seeds']} | "
            f"{fmt(row['baseline_source_wf1'])} | {fmt(row['joint_source_wf1'])} | {fmt(row['delta_source_wf1'])} | "
            f"{fmt(row['baseline_ejsl_wf1'])} | {fmt(row['joint_ejsl_wf1'])} | {fmt(row['delta_ejsl_wf1'])} | "
            f"{row['target_seed_count']}/{row['n_seeds']} | {row['status']} |"
        )
    lines.extend(["", "[cross-dataset-joint-trends] target coverage"])
    for corpus in CORPUS_SPECS:
        for baseline in BASELINE_ORDER:
            subset = [row for row in rows if row["corpus"] == corpus and row["baseline"] == baseline]
            targets = [str(row["config"]) for row in subset if row["status"] == "TARGET_SOURCE_UP_EJSL_DOWN"]
            lines.append(f"- {corpus}/{baseline}: {', '.join(targets) if targets else 'NONE'}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare paired BOBSL joint-training trends on MELD and IEMOCAP.")
    parser.add_argument(
        "--meld_seed_csv",
        default="/raid_zoe/home/lr/maokeyu/sign/five_baseline_joint_per_class_summary/five_baseline_per_class_seed_rows.csv",
    )
    parser.add_argument(
        "--iemocap_seed_csv",
        default="/raid_zoe/home/lr/maokeyu/sign/five_baseline_bobsl_iemocap_ejsl_summary/iemocap_joint_seed_metrics.csv",
    )
    parser.add_argument(
        "--out_dir",
        default="/raid_zoe/home/lr/maokeyu/sign/meld_iemocap_joint_trend_comparison",
    )
    parser.add_argument("--required_seeds", type=int, default=5)
    parser.add_argument("--min_delta", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meld_rows = normalize_rows("MELD", read_csv(Path(args.meld_seed_csv).expanduser()))
    iemocap_rows = normalize_rows("IEMOCAP", read_csv(Path(args.iemocap_seed_csv).expanduser()))
    seed_rows = paired_seed_rows(meld_rows + iemocap_rows)
    means = aggregate(seed_rows, args.required_seeds, args.min_delta)
    out_dir = Path(args.out_dir).expanduser().resolve()
    write_csv(out_dir / "joint_cross_dataset_seed_deltas.csv", seed_rows)
    write_csv(out_dir / "joint_cross_dataset_config_means.csv", means)
    report = build_report(means, args.required_seeds, args.min_delta)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "joint_cross_dataset_report.md").write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"[cross-dataset-joint-trends] out_dir={out_dir}")


if __name__ == "__main__":
    main()
