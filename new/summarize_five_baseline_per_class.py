#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


LABELS = ["A", "N", "J", "S"]
SPLITS = [("MELD", "source_test_best"), ("eJSL", "external_test_best")]
BASELINE_ORDER = ["MMGCN", "MAGTKD", "ECERC", "CMERC", "ConxGNN"]
DEFAULT_ROOTS = {
    "MMGCN": "/raid_zoe/home/lr/maokeyu/sign/mmgcn_bobsl_meld_ejsl/video_joint_bobsl_meld",
    "MAGTKD": "/raid_zoe/home/lr/maokeyu/sign/magtkd_bobsl_meld_ejsl/video_joint_bobsl_meld",
    "ECERC": "/raid_zoe/home/lr/maokeyu/sign/ecerc_bobsl_meld_ejsl/video_joint_bobsl_meld",
    "CMERC": "/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_meld_ejsl/visual_joint_bobsl_meld",
    "ConxGNN": "/raid_zoe/home/lr/maokeyu/sign/conxgnn_bobsl_meld_ejsl/visual_joint_bobsl_meld",
}


def as_float(value: Any) -> float:
    try:
        if value == "" or value is None:
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def fmt(value: Any) -> str:
    number = as_float(value)
    if math.isnan(number):
        return "NA"
    return f"{number:.4f}"


def numeric(values: Iterable[Any]) -> Dict[str, float]:
    clean = [as_float(x) for x in values]
    clean = [x for x in clean if not math.isnan(x)]
    if not clean:
        return {"n": 0, "mean": float("nan"), "std": float("nan")}
    mean = sum(clean) / len(clean)
    var = sum((x - mean) ** 2 for x in clean) / len(clean)
    return {"n": len(clean), "mean": mean, "std": math.sqrt(var)}


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def parse_trial_dir(path: Path) -> Dict[str, str]:
    name = path.name
    if "_seed" in name:
        config, seed = name.rsplit("_seed", 1)
    else:
        config, seed = name, ""
    return {"config": config, "seed": seed}


def config_sort_key(config: str) -> Tuple[int, int, str]:
    if config == "meld_only":
        return (0, 0, config)
    match = re.search(r"b(\d+)k", config)
    if match:
        return (1, int(match.group(1)) * 1000, config)
    match = re.search(r"b(\d+)", config)
    if match:
        return (1, int(match.group(1)), config)
    return (2, 0, config)


def baseline_sort_key(baseline: str) -> Tuple[int, str]:
    if baseline in BASELINE_ORDER:
        return (BASELINE_ORDER.index(baseline), baseline)
    return (len(BASELINE_ORDER), baseline)


def parse_classification_report(path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {"per_class": {}, "weighted_f1": float("nan")}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.strip().split()
        if len(parts) >= 5 and parts[0] in LABELS:
            out["per_class"].setdefault(parts[0], {})["precision"] = as_float(parts[1])
            out["per_class"].setdefault(parts[0], {})["recall"] = as_float(parts[2])
            out["per_class"].setdefault(parts[0], {})["f1"] = as_float(parts[3])
            out["per_class"].setdefault(parts[0], {})["support"] = as_float(parts[4])
        elif len(parts) >= 6 and parts[0] == "weighted" and parts[1] == "avg":
            out["weighted_f1"] = as_float(parts[4])
    return out


def get_summary_metrics(result_dir: Path, prefix: str) -> Dict[str, Any]:
    summary_path = result_dir / f"{prefix}_summary.json"
    report_path = result_dir / f"{prefix}_classification_report.txt"
    summary = read_json(summary_path)
    fallback = parse_classification_report(report_path)
    metrics: Dict[str, Any] = {
        "summary_path": str(summary_path) if summary_path.exists() else "",
        "report_path": str(report_path) if report_path.exists() else "",
        "weighted_f1": summary.get("weighted_f1", fallback.get("weighted_f1", "")),
        "macro_f1": summary.get("macro_f1", ""),
        "accuracy": summary.get("accuracy", ""),
        "per_class": {},
    }
    per_class = summary.get("per_class") or fallback.get("per_class") or {}
    for label in LABELS:
        item = per_class.get(label, {}) if isinstance(per_class, dict) else {}
        metrics["per_class"][label] = {
            "f1": item.get("f1", ""),
            "precision": item.get("precision", ""),
            "recall": item.get("recall", ""),
            "support": item.get("support", ""),
        }
    return metrics


def trial_dirs_for_root(baseline: str, root: Path, nested: str = "") -> List[Tuple[Path, Path]]:
    root = root.expanduser().resolve()
    if nested:
        pattern = f"*_seed*/{nested}/external_test_best_summary.json"
        return [(path.parent.parent, path.parent) for path in sorted(root.glob(pattern))]
    if baseline == "MMGCN":
        nested_paths = [(path.parent.parent, path.parent) for path in sorted(root.glob("*_seed*/video/external_test_best_summary.json"))]
        if nested_paths:
            return nested_paths
    return [(path.parent, path.parent) for path in sorted(root.glob("*_seed*/external_test_best_summary.json"))]


def collect_root(baseline: str, root: Path, nested: str = "") -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for trial_dir, result_dir in trial_dirs_for_root(baseline, root, nested=nested):
        parsed = parse_trial_dir(trial_dir)
        config_data = read_json(trial_dir / "config.json")
        config = str(config_data.get("config", parsed["config"]))
        seed = str(config_data.get("seed", parsed["seed"]))
        common = {
            "baseline": baseline,
            "config": config,
            "seed": seed,
            "trial": trial_dir.name,
            "result_dir": str(result_dir),
            "root": str(root),
            "bobsl_max": config_data.get("bobsl_max", ""),
            "epochs": config_data.get("epochs", ""),
            "lr": config_data.get("lr", ""),
            "dropout": config_data.get("dropout", config_data.get("drop_rate", "")),
            "loss": config_data.get("loss", config_data.get("loss_gamma", "")),
        }
        for split_name, prefix in SPLITS:
            metrics = get_summary_metrics(result_dir, prefix)
            if not metrics["summary_path"] and not metrics["report_path"]:
                continue
            row = dict(common)
            row.update(
                {
                    "split": split_name,
                    "weighted_f1": metrics.get("weighted_f1", ""),
                    "macro_f1": metrics.get("macro_f1", ""),
                    "accuracy": metrics.get("accuracy", ""),
                    "summary_path": metrics.get("summary_path", ""),
                    "report_path": metrics.get("report_path", ""),
                }
            )
            for label in LABELS:
                item = metrics["per_class"].get(label, {})
                row[f"{label}_f1"] = item.get("f1", "")
                row[f"{label}_precision"] = item.get("precision", "")
                row[f"{label}_recall"] = item.get("recall", "")
                row[f"{label}_support"] = item.get("support", "")
            rows.append(row)
    return rows


def parse_root_specs(root_specs: Sequence[str], use_defaults: bool) -> List[Tuple[str, Path, str]]:
    specs: List[Tuple[str, Path, str]] = []
    if use_defaults or not root_specs:
        specs.extend((baseline, Path(path), "video" if baseline == "MMGCN" else "") for baseline, path in DEFAULT_ROOTS.items())
    for raw in root_specs:
        if "=" not in raw:
            raise ValueError(f"--root must be NAME=PATH or NAME=PATH|NESTED, got: {raw}")
        name, value = raw.split("=", 1)
        if "|" in value:
            path_text, nested = value.rsplit("|", 1)
        else:
            path_text, nested = value, "video" if name == "MMGCN" else ""
        specs.append((name.strip(), Path(path_text.strip()), nested.strip()))
    return specs


def collect_all(specs: Sequence[Tuple[str, Path, str]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for baseline, root, nested in specs:
        rows.extend(collect_root(baseline, root, nested=nested))
    return rows


def summarize_long(seed_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for row in seed_rows:
        grouped.setdefault((str(row["baseline"]), str(row["config"]), str(row["split"])), []).append(row)
    out: List[Dict[str, Any]] = []
    for (baseline, config, split), rows in grouped.items():
        rows_sorted = sorted(rows, key=lambda row: str(row.get("seed", "")))
        summary: Dict[str, Any] = {
            "baseline": baseline,
            "config": config,
            "split": split,
            "n_seeds": len(rows_sorted),
            "seeds": " ".join(str(row.get("seed", "")) for row in rows_sorted),
            "bobsl_max": rows_sorted[0].get("bobsl_max", ""),
            "epochs": rows_sorted[0].get("epochs", ""),
            "lr": rows_sorted[0].get("lr", ""),
            "dropout": rows_sorted[0].get("dropout", ""),
            "loss": rows_sorted[0].get("loss", ""),
        }
        for metric in ["weighted_f1", "macro_f1", "accuracy"]:
            stats = numeric(row.get(metric) for row in rows_sorted)
            summary[f"{metric}_mean"] = stats["mean"]
            summary[f"{metric}_std"] = stats["std"]
        for label in LABELS:
            stats = numeric(row.get(f"{label}_f1") for row in rows_sorted)
            summary[f"{label}_f1_mean"] = stats["mean"]
            summary[f"{label}_f1_std"] = stats["std"]
            support_stats = numeric(row.get(f"{label}_support") for row in rows_sorted)
            summary[f"{label}_support_mean"] = support_stats["mean"]
        out.append(summary)
    out.sort(key=lambda row: (baseline_sort_key(row["baseline"]), config_sort_key(row["config"]), row["split"]))
    return out


def summarize_wide(long_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = {}
    for row in long_rows:
        grouped.setdefault((str(row["baseline"]), str(row["config"])), {})[str(row["split"])] = row
    wide_rows: List[Dict[str, Any]] = []
    for (baseline, config), split_rows in grouped.items():
        first = next(iter(split_rows.values()))
        row: Dict[str, Any] = {
            "baseline": baseline,
            "config": config,
            "n_seeds": first.get("n_seeds", ""),
            "seeds": first.get("seeds", ""),
            "bobsl_max": first.get("bobsl_max", ""),
            "epochs": first.get("epochs", ""),
            "lr": first.get("lr", ""),
            "dropout": first.get("dropout", ""),
            "loss": first.get("loss", ""),
        }
        for split in ["MELD", "eJSL"]:
            source = split_rows.get(split, {})
            row[f"{split}_weighted_f1_mean"] = source.get("weighted_f1_mean", "")
            row[f"{split}_weighted_f1_std"] = source.get("weighted_f1_std", "")
            for label in LABELS:
                row[f"{split}_{label}_f1_mean"] = source.get(f"{label}_f1_mean", "")
                row[f"{split}_{label}_f1_std"] = source.get(f"{label}_f1_std", "")
        wide_rows.append(row)
    wide_rows.sort(key=lambda row: (baseline_sort_key(row["baseline"]), config_sort_key(row["config"])))
    return wide_rows


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_report(wide_rows: Sequence[Dict[str, Any]]) -> str:
    lines = [
        "[five-baseline-per-class] metric: mean over seeds of best-by-MELD/source checkpoints",
        "[five-baseline-per-class] columns: weighted F1 plus per-class F1 for A/N/J/S",
        "",
        "| baseline | config | seeds | MELD_wF1 | MELD_A | MELD_N | MELD_J | MELD_S | eJSL_wF1 | eJSL_A | eJSL_N | eJSL_J | eJSL_S |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in wide_rows:
        lines.append(
            f"| {row['baseline']} | {row['config']} | {row.get('n_seeds', '')} | "
            f"{fmt(row.get('MELD_weighted_f1_mean'))} | "
            f"{fmt(row.get('MELD_A_f1_mean'))} | {fmt(row.get('MELD_N_f1_mean'))} | "
            f"{fmt(row.get('MELD_J_f1_mean'))} | {fmt(row.get('MELD_S_f1_mean'))} | "
            f"{fmt(row.get('eJSL_weighted_f1_mean'))} | "
            f"{fmt(row.get('eJSL_A_f1_mean'))} | {fmt(row.get('eJSL_N_f1_mean'))} | "
            f"{fmt(row.get('eJSL_J_f1_mean'))} | {fmt(row.get('eJSL_S_f1_mean'))} |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize five visual baselines by config: per-class F1 and weighted F1 on MELD/eJSL.")
    parser.add_argument("--root", action="append", default=[], help="NAME=PATH or NAME=PATH|NESTED. Repeatable.")
    parser.add_argument("--default_roots", action="store_true", help="Include default server roots. Used automatically when --root is omitted.")
    parser.add_argument("--out_dir", type=str, default="/raid_zoe/home/lr/maokeyu/sign/five_baseline_joint_per_class_summary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = parse_root_specs(args.root, args.default_roots)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_rows = collect_all(specs)
    long_rows = summarize_long(seed_rows)
    wide_rows = summarize_wide(long_rows)

    write_csv(out_dir / "five_baseline_per_class_seed_rows.csv", seed_rows)
    write_csv(out_dir / "five_baseline_per_class_config_means.csv", long_rows)
    write_csv(out_dir / "five_baseline_per_class_wide_means.csv", wide_rows)
    report = build_report(wide_rows)
    (out_dir / "five_baseline_per_class_report.md").write_text(report, encoding="utf-8")

    print(report, end="")
    print(f"[five-baseline-per-class] seed_rows={out_dir / 'five_baseline_per_class_seed_rows.csv'}")
    print(f"[five-baseline-per-class] config_means={out_dir / 'five_baseline_per_class_config_means.csv'}")
    print(f"[five-baseline-per-class] wide_means={out_dir / 'five_baseline_per_class_wide_means.csv'}")
    print(f"[five-baseline-per-class] report={out_dir / 'five_baseline_per_class_report.md'}")


if __name__ == "__main__":
    main()
