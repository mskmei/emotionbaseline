#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


LABELS = ["A", "N", "J", "S"]
BASELINE_ORDER = ["MMGCN", "MAGTKD", "ECERC", "CMERC", "ConxGNN"]
DEFAULT_ROOTS = {
    "MMGCN": "/raid_zoe/home/lr/maokeyu/sign/mmgcn_bobsl_iemocap_ejsl/video_joint_bobsl_iemocap",
    "MAGTKD": "/raid_zoe/home/lr/maokeyu/sign/magtkd_bobsl_iemocap_ejsl/video_joint_bobsl_iemocap",
    "ECERC": "/raid_zoe/home/lr/maokeyu/sign/ecerc_bobsl_iemocap_ejsl/video_joint_bobsl_iemocap",
    "CMERC": "/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_iemocap_ejsl/video_joint_bobsl_iemocap",
    "ConxGNN": "/raid_zoe/home/lr/maokeyu/sign/conxgnn_bobsl_iemocap_ejsl/video_joint_bobsl_iemocap",
}


def as_float(value: Any) -> float:
    try:
        if value in ("", None):
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def fmt(value: Any) -> str:
    number = as_float(value)
    return "NA" if math.isnan(number) else f"{number:.4f}"


def stats(values: Iterable[Any]) -> Dict[str, float]:
    clean = [as_float(value) for value in values]
    clean = [value for value in clean if not math.isnan(value)]
    if not clean:
        return {"mean": float("nan"), "std": float("nan")}
    mean = sum(clean) / len(clean)
    return {"mean": mean, "std": math.sqrt(sum((value - mean) ** 2 for value in clean) / len(clean))}


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def read_report(path: Path) -> Dict[str, Any]:
    output: Dict[str, Any] = {"per_class": {}, "weighted_f1": float("nan")}
    if not path.exists():
        return output
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.strip().split()
        if len(parts) >= 5 and parts[0] in LABELS:
            output["per_class"][parts[0]] = {
                "precision": as_float(parts[1]),
                "recall": as_float(parts[2]),
                "f1": as_float(parts[3]),
                "support": as_float(parts[4]),
            }
        elif len(parts) >= 6 and parts[:2] == ["weighted", "avg"]:
            output["weighted_f1"] = as_float(parts[4])
    return output


def result_metrics(result_dir: Path, prefix: str) -> Dict[str, Any]:
    summary_path = result_dir / f"{prefix}_summary.json"
    report_path = result_dir / f"{prefix}_classification_report.txt"
    summary = read_json(summary_path)
    report = read_report(report_path)
    per_class = summary.get("per_class") if isinstance(summary.get("per_class"), dict) else report["per_class"]
    output: Dict[str, Any] = {
        "weighted_f1": summary.get("weighted_f1", report["weighted_f1"]),
        "macro_f1": summary.get("macro_f1", ""),
        "accuracy": summary.get("accuracy", ""),
        "summary_path": str(summary_path) if summary_path.exists() else "",
        "report_path": str(report_path) if report_path.exists() else "",
    }
    for label in LABELS:
        item = per_class.get(label, {}) if isinstance(per_class, dict) else {}
        output[f"{label}_f1"] = item.get("f1", "")
        output[f"{label}_precision"] = item.get("precision", "")
        output[f"{label}_recall"] = item.get("recall", "")
        output[f"{label}_support"] = item.get("support", "")
    return output


def trial_results(baseline: str, root: Path) -> List[Tuple[Path, Path]]:
    if baseline == "MMGCN":
        return [(path.parent.parent, path.parent) for path in sorted(root.glob("*_seed*/video/external_test_best_summary.json"))]
    return [(path.parent, path.parent) for path in sorted(root.glob("*_seed*/external_test_best_summary.json"))]


def collect(baseline: str, root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for trial_dir, result_dir in trial_results(baseline, root):
        config_data = read_json(trial_dir / "config.json")
        match = re.match(r"^(.*)_seed(\d+)$", trial_dir.name)
        fallback_config, fallback_seed = match.groups() if match else (trial_dir.name, "")
        common = {
            "baseline": baseline,
            "config": config_data.get("config", fallback_config),
            "seed": str(config_data.get("seed", fallback_seed)),
            "bobsl_max": config_data.get("bobsl_max", ""),
            "epochs": config_data.get("epochs", ""),
            "lr": config_data.get("lr", ""),
            "dropout": config_data.get("dropout", ""),
            "result_dir": str(result_dir),
        }
        for split, prefix in [("IEMOCAP", "source_test_best"), ("eJSL", "external_test_best")]:
            metrics = result_metrics(result_dir, prefix)
            if not metrics["summary_path"] and not metrics["report_path"]:
                continue
            rows.append({**common, "split": split, **metrics})
    return rows


def add_paired_deltas(rows: List[Dict[str, Any]]) -> None:
    baseline_values = {
        (row["baseline"], row["seed"], row["split"]): as_float(row["weighted_f1"])
        for row in rows
        if row["config"] == "iemocap_only"
    }
    for row in rows:
        baseline = baseline_values.get((row["baseline"], row["seed"], row["split"]), float("nan"))
        value = as_float(row["weighted_f1"])
        row["delta_weighted_f1_vs_iemocap_only"] = "" if math.isnan(baseline) or math.isnan(value) else value - baseline


def summarize(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["baseline"], row["config"], row["split"]), []).append(row)
    output: List[Dict[str, Any]] = []
    for (baseline, config, split), group in grouped.items():
        row: Dict[str, Any] = {
            "baseline": baseline,
            "config": config,
            "split": split,
            "n_seeds": len(group),
            "seeds": " ".join(sorted(item["seed"] for item in group)),
            "bobsl_max": group[0].get("bobsl_max", ""),
            "epochs": group[0].get("epochs", ""),
            "lr": group[0].get("lr", ""),
            "dropout": group[0].get("dropout", ""),
        }
        for metric in ["weighted_f1", "macro_f1", "accuracy", "delta_weighted_f1_vs_iemocap_only"]:
            value_stats = stats(item.get(metric) for item in group)
            row[f"{metric}_mean"] = value_stats["mean"]
            row[f"{metric}_std"] = value_stats["std"]
        for label in LABELS:
            value_stats = stats(item.get(f"{label}_f1") for item in group)
            row[f"{label}_f1_mean"] = value_stats["mean"]
            row[f"{label}_f1_std"] = value_stats["std"]
        output.append(row)
    output.sort(key=lambda row: (BASELINE_ORDER.index(row["baseline"]), row["config"], row["split"]))
    return output


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def report(group_rows: Sequence[Dict[str, Any]]) -> str:
    lines = [
        "[IEMOCAP-joint-summary] means over seeds; source=IEMOCAP Session5, external=eJSL",
        "",
        "| baseline | config | split | n | wF1 | delta vs IEMOCAP-only | A F1 | N F1 | J F1 | S F1 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in group_rows:
        lines.append(
            f"| {row['baseline']} | {row['config']} | {row['split']} | {row['n_seeds']} | "
            f"{fmt(row['weighted_f1_mean'])} | {fmt(row['delta_weighted_f1_vs_iemocap_only_mean'])} | "
            f"{fmt(row['A_f1_mean'])} | {fmt(row['N_f1_mean'])} | {fmt(row['J_f1_mean'])} | {fmt(row['S_f1_mean'])} |"
        )
    return "\n".join(lines) + "\n"


def parse_roots(raw_roots: Sequence[str]) -> List[Tuple[str, Path]]:
    if not raw_roots:
        return [(name, Path(path)) for name, path in DEFAULT_ROOTS.items()]
    output: List[Tuple[str, Path]] = []
    for raw in raw_roots:
        if "=" not in raw:
            raise ValueError(f"--root must be NAME=PATH, got {raw!r}")
        name, path = raw.split("=", 1)
        if name not in BASELINE_ORDER:
            raise ValueError(f"Unknown baseline {name!r}")
        output.append((name, Path(path)))
    return output


def main() -> None:
    args = parse_args()
    roots = parse_roots(args.root)
    rows: List[Dict[str, Any]] = []
    for baseline, root in roots:
        rows.extend(collect(baseline, root.expanduser().resolve()))
    if not rows:
        raise RuntimeError(f"No completed runs found under: {roots}")
    add_paired_deltas(rows)
    groups = summarize(rows)
    out_dir = Path(args.out_dir).expanduser().resolve()
    write_csv(out_dir / "iemocap_joint_seed_metrics.csv", rows)
    write_csv(out_dir / "iemocap_joint_config_means.csv", groups)
    text = report(groups)
    (out_dir / "iemocap_joint_report.md").write_text(text, encoding="utf-8")
    print(text, end="")
    print(f"[IEMOCAP-joint-summary] out_dir={out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize five BOBSL+IEMOCAP video-only baselines on IEMOCAP and eJSL.")
    parser.add_argument("--root", action="append", default=[], help="Repeat NAME=PATH; defaults to all five server output roots.")
    parser.add_argument("--out_dir", default="/raid_zoe/home/lr/maokeyu/sign/five_baseline_bobsl_iemocap_ejsl_summary")
    return parser.parse_args()


if __name__ == "__main__":
    main()
