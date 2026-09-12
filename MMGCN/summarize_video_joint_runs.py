#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_ROOT = "/raid_zoe/home/lr/maokeyu/sign/mmgcn_bobsl_meld_ejsl/video_joint_bobsl_meld"


def as_float(value: Any, default: float = 0.0) -> float:
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt(value: Any, digits: int = 4) -> str:
    if value in ("", None):
        return "NA"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
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


def numeric(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": mean(values),
        "median": median(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def sign_counts(values: Iterable[float], eps: float = 1e-12) -> Tuple[int, int, int]:
    wins = ties = losses = 0
    for value in values:
        if value > eps:
            wins += 1
        elif value < -eps:
            losses += 1
        else:
            ties += 1
    return wins, ties, losses


def safe_corr(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx = mean(xs)
    my = mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def parse_trial_name(name: str) -> Tuple[str, str]:
    match = re.match(r"^(.*)_seed(\d+)$", name)
    if match:
        return match.group(1), match.group(2)
    return name, ""


def load_rows(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(root.glob("*/video/external_test_best_summary.json")):
        trial = path.parent.parent.name
        config, seed = parse_trial_name(trial)
        external = read_json(path)
        source_path = path.parent / "source_test_best_summary.json"
        source = read_json(source_path) if source_path.exists() else {}
        config_path = path.parent.parent / "config.json"
        config_data = read_json(config_path) if config_path.exists() else {}
        rows.append(
            {
                "trial": trial,
                "config": config_data.get("config", config),
                "seed": config_data.get("seed", seed or external.get("seed", "")),
                "train_pkl": external.get("train_pkl", ""),
                "joint_pkl": config_data.get("joint_pkl", ""),
                "bobsl_max": config_data.get("bobsl_max", ""),
                "bobsl_include_val": config_data.get("bobsl_include_val", ""),
                "lr": external.get("lr", config_data.get("lr", "")),
                "l2": external.get("l2", config_data.get("l2", "")),
                "dropout": external.get("dropout", config_data.get("dropout", "")),
                "loss": external.get("loss", config_data.get("loss", "")),
                "focal_gamma": external.get("focal_gamma", config_data.get("focal_gamma", "")),
                "extra_args": config_data.get("extra_args", ""),
                "source_wf1": source.get("weighted_f1", ""),
                "source_macro_f1": source.get("macro_f1", ""),
                "source_acc": source.get("accuracy", ""),
                "external_wf1": external.get("weighted_f1", ""),
                "external_macro_f1": external.get("macro_f1", ""),
                "external_acc": external.get("accuracy", ""),
                "selected_epoch": external.get("best_epoch_by_selection_metric", ""),
                "external_pred_counts": json.dumps(external.get("pred_counts", {}), ensure_ascii=False, sort_keys=True),
                "summary_path": str(path),
            }
        )
    return rows


def add_baseline_deltas(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    baseline_by_seed = {str(row["seed"]): row for row in rows if str(row.get("config")) == "meld_only"}
    out: List[Dict[str, Any]] = []
    for row in rows:
        enriched = dict(row)
        baseline = baseline_by_seed.get(str(row.get("seed", "")))
        if baseline and str(row.get("config")) != "meld_only":
            enriched["baseline_source_wf1"] = baseline.get("source_wf1", "")
            enriched["baseline_external_wf1"] = baseline.get("external_wf1", "")
            enriched["delta_source_wf1_vs_meld_only"] = as_float(row.get("source_wf1")) - as_float(baseline.get("source_wf1"))
            enriched["delta_external_wf1_vs_meld_only"] = as_float(row.get("external_wf1")) - as_float(baseline.get("external_wf1"))
        else:
            enriched["baseline_source_wf1"] = ""
            enriched["baseline_external_wf1"] = ""
            enriched["delta_source_wf1_vs_meld_only"] = ""
            enriched["delta_external_wf1_vs_meld_only"] = ""
        out.append(enriched)
    return out


def summarize_by_config(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("config", "")), []).append(row)

    summaries: List[Dict[str, Any]] = []
    for config, config_rows in grouped.items():
        source = [as_float(row.get("source_wf1")) for row in config_rows]
        external = [as_float(row.get("external_wf1")) for row in config_rows]
        delta_source = [as_float(row.get("delta_source_wf1_vs_meld_only")) for row in config_rows if row.get("delta_source_wf1_vs_meld_only") != ""]
        delta_external = [as_float(row.get("delta_external_wf1_vs_meld_only")) for row in config_rows if row.get("delta_external_wf1_vs_meld_only") != ""]
        ds = numeric(delta_source)
        de = numeric(delta_external)
        source_stats = numeric(source)
        external_stats = numeric(external)
        wins_source, ties_source, losses_source = sign_counts(delta_source)
        wins_external, ties_external, losses_external = sign_counts(delta_external)
        target_count = sum(1 for m, e in zip(delta_source, delta_external) if m > 0 and e < 0)
        summaries.append(
            {
                "config": config,
                "n_seeds": len(config_rows),
                "seeds": " ".join(str(row.get("seed", "")) for row in sorted(config_rows, key=lambda x: str(x.get("seed", "")))),
                "bobsl_max": config_rows[0].get("bobsl_max", ""),
                "lr": config_rows[0].get("lr", ""),
                "dropout": config_rows[0].get("dropout", ""),
                "loss": config_rows[0].get("loss", ""),
                "focal_gamma": config_rows[0].get("focal_gamma", ""),
                "extra_args": config_rows[0].get("extra_args", ""),
                "source_wf1_mean": source_stats["mean"],
                "source_wf1_std": source_stats["std"],
                "external_wf1_mean": external_stats["mean"],
                "external_wf1_std": external_stats["std"],
                "delta_source_mean_vs_meld_only": "" if not delta_source else ds["mean"],
                "delta_source_median_vs_meld_only": "" if not delta_source else ds["median"],
                "delta_external_mean_vs_meld_only": "" if not delta_external else de["mean"],
                "delta_external_median_vs_meld_only": "" if not delta_external else de["median"],
                "source_wins_ties_losses_vs_meld_only": "" if not delta_source else f"{wins_source}/{ties_source}/{losses_source}",
                "external_wins_ties_losses_vs_meld_only": "" if not delta_external else f"{wins_external}/{ties_external}/{losses_external}",
                "target_meld_up_ejsl_down": target_count,
                "delta_corr_meld_ejsl": "" if not delta_source else safe_corr(delta_source, delta_external),
            }
        )
    summaries.sort(
        key=lambda row: (
            as_float(row.get("target_meld_up_ejsl_down")),
            as_float(row.get("delta_source_mean_vs_meld_only")) - as_float(row.get("delta_external_mean_vs_meld_only")),
        ),
        reverse=True,
    )
    return summaries


def build_report(summaries: Sequence[Dict[str, Any]], top_k: int) -> str:
    lines = [
        "[joint-summary] metric: source=MELD test weighted F1, external=eJSL translated test weighted F1",
        f"[joint-summary] configs={len(summaries)}",
        "",
        "[joint-summary] config means",
    ]
    for row in summaries[:top_k]:
        lines.append(
            "  "
            f"{row['config']}: n={row['n_seeds']} "
            f"MELD={fmt(row['source_wf1_mean'])}+/-{fmt(row['source_wf1_std'])} "
            f"eJSL={fmt(row['external_wf1_mean'])}+/-{fmt(row['external_wf1_std'])} "
            f"delta_MELD={fmt(row['delta_source_mean_vs_meld_only'])} "
            f"delta_eJSL={fmt(row['delta_external_mean_vs_meld_only'])} "
            f"target_count={row['target_meld_up_ejsl_down']} "
            f"bobsl_max={row['bobsl_max']} lr={row['lr']} drop={row['dropout']} loss={row['loss']} extra={row['extra_args']}"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize video-only joint MELD+BOBSL runs.")
    parser.add_argument("--root", type=str, default=DEFAULT_ROOT)
    parser.add_argument("--out_runs_csv", type=str, default="")
    parser.add_argument("--out_group_csv", type=str, default="")
    parser.add_argument("--out_txt", type=str, default="")
    parser.add_argument("--top_k", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    rows = add_baseline_deltas(load_rows(root))
    if not rows:
        raise RuntimeError(f"No video summaries found under {root}")

    summaries = summarize_by_config(rows)
    out_runs = Path(args.out_runs_csv).expanduser() if args.out_runs_csv else root / "video_joint_runs.csv"
    out_group = Path(args.out_group_csv).expanduser() if args.out_group_csv else root / "video_joint_group_averages.csv"
    out_txt = Path(args.out_txt).expanduser() if args.out_txt else root / "video_joint_group_averages.txt"
    write_csv(rows, out_runs)
    write_csv(summaries, out_group)
    report = build_report(summaries, args.top_k)
    out_txt.write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"[joint-summary] runs_csv={out_runs}")
    print(f"[joint-summary] group_csv={out_group}")
    print(f"[joint-summary] report={out_txt}")


if __name__ == "__main__":
    main()
