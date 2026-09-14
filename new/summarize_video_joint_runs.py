#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence


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


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def numeric(values: Sequence[float]) -> Dict[str, float]:
    clean = [x for x in values if not math.isnan(x)]
    if not clean:
        return {"mean": float("nan"), "std": float("nan"), "median": float("nan")}
    mean = sum(clean) / len(clean)
    var = sum((x - mean) ** 2 for x in clean) / len(clean)
    clean_sorted = sorted(clean)
    mid = len(clean_sorted) // 2
    if len(clean_sorted) % 2:
        median = clean_sorted[mid]
    else:
        median = 0.5 * (clean_sorted[mid - 1] + clean_sorted[mid])
    return {"mean": mean, "std": math.sqrt(var), "median": median}


def sign_counts(values: Sequence[float]) -> tuple[int, int, int]:
    wins = ties = losses = 0
    for value in values:
        if math.isnan(value):
            continue
        if value > 1e-12:
            wins += 1
        elif value < -1e-12:
            losses += 1
        else:
            ties += 1
    return wins, ties, losses


def parse_trial_dir(path: Path) -> Dict[str, str]:
    name = path.name
    if "_seed" in name:
        config, seed = name.rsplit("_seed", 1)
    else:
        config, seed = name, ""
    return {"config": config, "seed": seed}


def collect_rows(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for external_path in sorted(root.glob("*_seed*/external_test_best_summary.json")):
        trial_dir = external_path.parent
        source_path = trial_dir / "source_test_best_summary.json"
        config_path = trial_dir / "config.json"
        parsed = parse_trial_dir(trial_dir)
        config_data = read_json(config_path)
        source = read_json(source_path)
        external = read_json(external_path)
        row = {
            "trial": trial_dir.name,
            "config": config_data.get("config", parsed["config"]),
            "seed": str(config_data.get("seed", parsed["seed"])),
            "baseline": external.get("baseline", config_data.get("baseline", "")),
            "bobsl_max": config_data.get("bobsl_max", ""),
            "lr": external.get("lr", config_data.get("lr", "")),
            "dropout": external.get("dropout", config_data.get("dropout", "")),
            "loss": external.get("loss", config_data.get("loss", external.get("loss_gamma", config_data.get("loss_gamma", "")))),
            "extra_args": config_data.get("extra_args", ""),
            "source_wf1": source.get("weighted_f1", ""),
            "source_macro_f1": source.get("macro_f1", ""),
            "source_acc": source.get("accuracy", ""),
            "external_wf1": external.get("weighted_f1", ""),
            "external_macro_f1": external.get("macro_f1", ""),
            "external_acc": external.get("accuracy", ""),
            "selected_epoch": external.get("best_epoch_by_selection_metric", ""),
            "external_pred_counts": json.dumps(external.get("pred_counts", {}), ensure_ascii=False, sort_keys=True),
            "summary_path": str(external_path),
        }
        rows.append(row)
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
        first = config_rows[0]
        summaries.append(
            {
                "config": config,
                "n_seeds": len(config_rows),
                "seeds": " ".join(str(row.get("seed", "")) for row in sorted(config_rows, key=lambda x: str(x.get("seed", "")))),
                "baseline": first.get("baseline", ""),
                "bobsl_max": first.get("bobsl_max", ""),
                "lr": first.get("lr", ""),
                "dropout": first.get("dropout", ""),
                "loss": first.get("loss", ""),
                "extra_args": first.get("extra_args", ""),
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


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_report(baseline: str, summaries: Sequence[Dict[str, Any]], top_k: int) -> str:
    lines = [
        f"[{baseline}-joint-summary] metric: source=MELD test weighted F1, external=eJSL translated test weighted F1",
        f"[{baseline}-joint-summary] configs={len(summaries)}",
        "",
        f"[{baseline}-joint-summary] config means",
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
    parser = argparse.ArgumentParser(description="Summarize video-only joint BOBSL+MELD baseline runs.")
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--baseline", type=str, default="baseline")
    parser.add_argument("--out_runs_csv", type=str, default="")
    parser.add_argument("--out_group_csv", type=str, default="")
    parser.add_argument("--out_txt", type=str, default="")
    parser.add_argument("--top_k", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    rows = add_baseline_deltas(collect_rows(root))
    summaries = summarize_by_config(rows)
    out_runs = Path(args.out_runs_csv).expanduser() if args.out_runs_csv else root / "video_joint_runs.csv"
    out_group = Path(args.out_group_csv).expanduser() if args.out_group_csv else root / "video_joint_group_averages.csv"
    out_txt = Path(args.out_txt).expanduser() if args.out_txt else root / "video_joint_group_averages.txt"
    write_csv(out_runs, rows)
    write_csv(out_group, summaries)
    report = build_report(args.baseline, summaries, args.top_k)
    out_txt.write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"[{args.baseline}-joint-summary] runs_csv={out_runs}")
    print(f"[{args.baseline}-joint-summary] group_csv={out_group}")
    print(f"[{args.baseline}-joint-summary] report={out_txt}")


if __name__ == "__main__":
    main()
