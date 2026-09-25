#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


LABELS = ["A", "N", "J", "S"]
DEFAULT_ROOTS = {
    "CMERC": "/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_meld_ejsl/visual_joint_bobsl_meld",
    "ConxGNN": "/raid_zoe/home/lr/maokeyu/sign/conxgnn_bobsl_meld_ejsl/visual_joint_bobsl_meld",
    "ECERC": "/raid_zoe/home/lr/maokeyu/sign/ecerc_bobsl_meld_ejsl/video_joint_bobsl_meld",
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


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def numeric(values: Iterable[float]) -> Dict[str, float]:
    clean = [x for x in values if not math.isnan(x)]
    if not clean:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "median": float("nan")}
    mean = sum(clean) / len(clean)
    var = sum((x - mean) ** 2 for x in clean) / len(clean)
    sorted_values = sorted(clean)
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2:
        median = sorted_values[mid]
    else:
        median = 0.5 * (sorted_values[mid - 1] + sorted_values[mid])
    return {"n": len(clean), "mean": mean, "std": math.sqrt(var), "median": median}


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if not math.isnan(x) and not math.isnan(y)]
    if len(pairs) < 2:
        return float("nan")
    x_vals = [p[0] for p in pairs]
    y_vals = [p[1] for p in pairs]
    x_mean = sum(x_vals) / len(x_vals)
    y_mean = sum(y_vals) / len(y_vals)
    x_var = sum((x - x_mean) ** 2 for x in x_vals)
    y_var = sum((y - y_mean) ** 2 for y in y_vals)
    if x_var <= 0 or y_var <= 0:
        return float("nan")
    cov = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
    return cov / math.sqrt(x_var * y_var)


def parse_trial_dir(path: Path) -> Dict[str, str]:
    name = path.name
    if "_seed" in name:
        config, seed = name.rsplit("_seed", 1)
    else:
        config, seed = name, ""
    return {"config": config, "seed": seed}


def per_class(summary: Dict[str, Any], label: str, metric: str) -> float:
    value = summary.get("per_class", {}).get(label, {}).get(metric, "")
    return as_float(value)


def collect_one_root(baseline_hint: str, root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for external_path in sorted(root.glob("*_seed*/external_test_best_summary.json")):
        trial_dir = external_path.parent
        source_path = trial_dir / "source_test_best_summary.json"
        config_path = trial_dir / "config.json"
        parsed = parse_trial_dir(trial_dir)
        config = read_json(config_path)
        source = read_json(source_path)
        external = read_json(external_path)
        if not source or not external:
            continue
        baseline = str(external.get("baseline") or source.get("baseline") or config.get("baseline") or baseline_hint)
        row: Dict[str, Any] = {
            "baseline": baseline,
            "root": str(root),
            "trial": trial_dir.name,
            "config": str(config.get("config", parsed["config"])),
            "seed": str(config.get("seed", parsed["seed"])),
            "bobsl_max": config.get("bobsl_max", ""),
            "epochs": external.get("epochs", config.get("epochs", "")),
            "lr": external.get("lr", config.get("lr", "")),
            "dropout": external.get("dropout", config.get("dropout", "")),
            "loss": external.get("loss", config.get("loss", external.get("loss_gamma", config.get("loss_gamma", "")))),
            "source_wf1": source.get("weighted_f1", ""),
            "source_macro_f1": source.get("macro_f1", ""),
            "source_acc": source.get("accuracy", ""),
            "external_wf1": external.get("weighted_f1", ""),
            "external_macro_f1": external.get("macro_f1", ""),
            "external_acc": external.get("accuracy", ""),
            "source_pred_counts": json.dumps(source.get("pred_counts", {}), ensure_ascii=False, sort_keys=True),
            "external_pred_counts": json.dumps(external.get("pred_counts", {}), ensure_ascii=False, sort_keys=True),
            "source_summary_path": str(source_path),
            "external_summary_path": str(external_path),
        }
        for label in LABELS:
            row[f"source_{label}_f1"] = per_class(source, label, "f1")
            row[f"external_{label}_f1"] = per_class(external, label, "f1")
            row[f"source_{label}_recall"] = per_class(source, label, "recall")
            row[f"external_{label}_recall"] = per_class(external, label, "recall")
        rows.append(row)
    return rows


def collect_rows(roots: Sequence[Tuple[str, Path]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for baseline, root in roots:
        rows.extend(collect_one_root(baseline, root.expanduser().resolve()))
    return rows


def classify_quadrant(delta_meld: float, delta_ejsl: float, eps: float) -> str:
    if math.isnan(delta_meld) or math.isnan(delta_ejsl):
        return "unpaired"
    meld_up = delta_meld > eps
    meld_down = delta_meld < -eps
    ejsl_up = delta_ejsl > eps
    ejsl_down = delta_ejsl < -eps
    if meld_up and ejsl_down:
        return "target_meld_up_ejsl_down"
    if meld_up and ejsl_up:
        return "both_up"
    if meld_down and ejsl_up:
        return "meld_down_ejsl_up"
    if meld_down and ejsl_down:
        return "both_down"
    return "near_zero_or_mixed"


def needed_count(fraction: float, paired_runs: int) -> int:
    if paired_runs <= 0:
        return 0
    return max(1, int(math.ceil(float(fraction) * paired_runs)))


def target_status(summary: Dict[str, Any], eps: float, target_fraction: float, strong_fraction: float) -> str:
    paired_runs = int(summary.get("paired_runs") or 0)
    if paired_runs <= 0:
        return "BASELINE_OR_UNPAIRED"
    target_count = int(summary.get("target_count") or 0)
    delta_meld = as_float(summary.get("delta_MELD_mean"))
    delta_ejsl = as_float(summary.get("delta_eJSL_mean"))
    mean_is_target = delta_meld > eps and delta_ejsl < -eps
    if mean_is_target and target_count >= needed_count(strong_fraction, paired_runs):
        return "STRONG_TARGET"
    if mean_is_target and target_count >= needed_count(target_fraction, paired_runs):
        return "TARGET"
    if mean_is_target:
        return "MEAN_TARGET_WEAK_SEEDS"
    if target_count > 0:
        return "MIXED_TARGET_SEEDS"
    if delta_meld > eps and delta_ejsl > eps:
        return "BOTH_UP"
    if delta_meld < -eps and delta_ejsl < -eps:
        return "BOTH_DOWN"
    if delta_meld < -eps and delta_ejsl > eps:
        return "MELD_DOWN_EJSL_UP"
    return "NOT_TARGET"


def status_rank(status: str) -> int:
    ranks = {
        "STRONG_TARGET": 6,
        "TARGET": 5,
        "MEAN_TARGET_WEAK_SEEDS": 4,
        "MIXED_TARGET_SEEDS": 3,
        "BOTH_DOWN": 2,
        "NOT_TARGET": 1,
        "BOTH_UP": 0,
        "MELD_DOWN_EJSL_UP": 0,
        "BASELINE_OR_UNPAIRED": -1,
    }
    return ranks.get(status, -1)


def add_paired_deltas(rows: List[Dict[str, Any]], eps: float) -> List[Dict[str, Any]]:
    baseline_by_key = {
        (str(row.get("baseline")), str(row.get("seed"))): row
        for row in rows
        if str(row.get("config")) == "meld_only"
    }
    out: List[Dict[str, Any]] = []
    for row in rows:
        enriched = dict(row)
        key = (str(row.get("baseline")), str(row.get("seed")))
        base = baseline_by_key.get(key)
        if base and str(row.get("config")) != "meld_only":
            enriched["baseline_source_wf1"] = base.get("source_wf1", "")
            enriched["baseline_external_wf1"] = base.get("external_wf1", "")
            delta_meld = as_float(row.get("source_wf1")) - as_float(base.get("source_wf1"))
            delta_ejsl = as_float(row.get("external_wf1")) - as_float(base.get("external_wf1"))
            enriched["delta_MELD_wf1"] = delta_meld
            enriched["delta_eJSL_wf1"] = delta_ejsl
            enriched["quadrant"] = classify_quadrant(delta_meld, delta_ejsl, eps)
            for label in LABELS:
                for metric in ["f1", "recall"]:
                    src_key = f"source_{label}_{metric}"
                    ext_key = f"external_{label}_{metric}"
                    enriched[f"delta_MELD_{label}_{metric}"] = as_float(row.get(src_key)) - as_float(base.get(src_key))
                    enriched[f"delta_eJSL_{label}_{metric}"] = as_float(row.get(ext_key)) - as_float(base.get(ext_key))
        else:
            enriched["baseline_source_wf1"] = ""
            enriched["baseline_external_wf1"] = ""
            enriched["delta_MELD_wf1"] = ""
            enriched["delta_eJSL_wf1"] = ""
            enriched["quadrant"] = "baseline" if str(row.get("config")) == "meld_only" else "unpaired"
            for label in LABELS:
                for metric in ["f1", "recall"]:
                    enriched[f"delta_MELD_{label}_{metric}"] = ""
                    enriched[f"delta_eJSL_{label}_{metric}"] = ""
        out.append(enriched)
    return out


def summarize_groups(
    rows: Sequence[Dict[str, Any]],
    eps: float,
    target_fraction: float,
    strong_fraction: float,
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("baseline", "")), str(row.get("config", "")))].append(row)

    summaries: List[Dict[str, Any]] = []
    for (baseline, config), group_rows in sorted(grouped.items()):
        d_meld = [as_float(row.get("delta_MELD_wf1")) for row in group_rows]
        d_ejsl = [as_float(row.get("delta_eJSL_wf1")) for row in group_rows]
        source = numeric(as_float(row.get("source_wf1")) for row in group_rows)
        external = numeric(as_float(row.get("external_wf1")) for row in group_rows)
        meld_delta = numeric(d_meld)
        ejsl_delta = numeric(d_ejsl)
        quadrant_counts = Counter(str(row.get("quadrant", "")) for row in group_rows)
        first = group_rows[0]
        summary: Dict[str, Any] = {
            "baseline": baseline,
            "config": config,
            "n_runs": len(group_rows),
            "paired_runs": meld_delta["n"],
            "seeds": " ".join(sorted(str(row.get("seed", "")) for row in group_rows)),
            "bobsl_max": first.get("bobsl_max", ""),
            "epochs": first.get("epochs", ""),
            "lr": first.get("lr", ""),
            "dropout": first.get("dropout", ""),
            "loss": first.get("loss", ""),
            "MELD_wf1_mean": source["mean"],
            "MELD_wf1_std": source["std"],
            "eJSL_wf1_mean": external["mean"],
            "eJSL_wf1_std": external["std"],
            "delta_MELD_mean": "" if meld_delta["n"] == 0 else meld_delta["mean"],
            "delta_MELD_median": "" if meld_delta["n"] == 0 else meld_delta["median"],
            "delta_eJSL_mean": "" if ejsl_delta["n"] == 0 else ejsl_delta["mean"],
            "delta_eJSL_median": "" if ejsl_delta["n"] == 0 else ejsl_delta["median"],
            "delta_corr": pearson(d_meld, d_ejsl),
            "target_count": quadrant_counts.get("target_meld_up_ejsl_down", 0),
            "both_up_count": quadrant_counts.get("both_up", 0),
            "meld_down_ejsl_up_count": quadrant_counts.get("meld_down_ejsl_up", 0),
            "both_down_count": quadrant_counts.get("both_down", 0),
            "near_zero_or_mixed_count": quadrant_counts.get("near_zero_or_mixed", 0),
        }
        paired_runs = int(summary["paired_runs"])
        target_count = int(summary["target_count"])
        summary["target_fraction"] = "" if paired_runs == 0 else target_count / paired_runs
        summary["target_needed"] = "" if paired_runs == 0 else needed_count(target_fraction, paired_runs)
        summary["strong_target_needed"] = "" if paired_runs == 0 else needed_count(strong_fraction, paired_runs)
        summary["target_status"] = target_status(summary, eps, target_fraction, strong_fraction)
        delta_meld_mean = as_float(summary["delta_MELD_mean"])
        delta_ejsl_mean = as_float(summary["delta_eJSL_mean"])
        delta_gap = delta_meld_mean - delta_ejsl_mean
        summary["delta_gap_MELD_minus_eJSL"] = "" if math.isnan(delta_gap) else delta_gap
        summary["target_rank_score"] = (
            status_rank(str(summary["target_status"])) * 100.0
            + (0.0 if paired_runs == 0 else 10.0 * target_count / paired_runs)
            + (0.0 if math.isnan(delta_gap) else delta_gap)
        )
        for label in LABELS:
            summary[f"delta_MELD_{label}_f1_mean"] = numeric(as_float(row.get(f"delta_MELD_{label}_f1")) for row in group_rows)["mean"]
            summary[f"delta_eJSL_{label}_f1_mean"] = numeric(as_float(row.get(f"delta_eJSL_{label}_f1")) for row in group_rows)["mean"]
        summaries.append(summary)

    summaries.sort(
        key=lambda row: (
            as_float(row.get("target_rank_score")),
            as_float(row.get("target_count")),
            as_float(row.get("delta_gap_MELD_minus_eJSL")),
        ),
        reverse=True,
    )
    return summaries


def target_config_ranking(groups: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = [row for row in groups if str(row.get("target_status")) != "BASELINE_OR_UNPAIRED"]
    return sorted(
        rows,
        key=lambda row: (
            as_float(row.get("target_rank_score")),
            as_float(row.get("target_count")),
            as_float(row.get("delta_gap_MELD_minus_eJSL")),
        ),
        reverse=True,
    )


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


def build_report(rows: Sequence[Dict[str, Any]], groups: Sequence[Dict[str, Any]], top_k: int) -> str:
    paired_rows = [row for row in rows if row.get("quadrant") not in {"baseline", "unpaired"}]
    target_rows = [row for row in paired_rows if row.get("quadrant") == "target_meld_up_ejsl_down"]
    d_meld = [as_float(row.get("delta_MELD_wf1")) for row in paired_rows]
    d_ejsl = [as_float(row.get("delta_eJSL_wf1")) for row in paired_rows]
    quadrant_counts = Counter(str(row.get("quadrant", "")) for row in paired_rows)

    lines = [
        "[joint-tradeoff] metric: source=MELD test weighted F1, external=translated eJSL weighted F1",
        f"[joint-tradeoff] runs={len(rows)} paired_joint_runs={len(paired_rows)} target_cases={len(target_rows)}",
        f"[joint-tradeoff] overall corr(delta_MELD, delta_eJSL)={fmt(pearson(d_meld, d_ejsl))}",
        f"[joint-tradeoff] quadrants={dict(sorted(quadrant_counts.items()))}",
        "",
        "[joint-tradeoff] target-config ranking",
    ]
    for idx, row in enumerate(target_config_ranking(groups)[:top_k], start=1):
        lines.append(
            f"  {idx:02d}. {row['baseline']}/{row['config']} status={row['target_status']} "
            f"target={row['target_count']}/{row['paired_runs']} "
            f"mean_delta_MELD={fmt(row['delta_MELD_mean'])} "
            f"mean_delta_eJSL={fmt(row['delta_eJSL_mean'])} "
            f"gap={fmt(row['delta_gap_MELD_minus_eJSL'])} "
            f"MELD={fmt(row['MELD_wf1_mean'])} eJSL={fmt(row['eJSL_wf1_mean'])} "
            f"bobsl_max={row['bobsl_max']} lr={row['lr']} drop={row['dropout']} loss={row['loss']}"
        )
    lines.extend(["", "[joint-tradeoff] group means"])
    for row in groups:
        lines.append(
            "  "
            f"{row['baseline']}/{row['config']}: status={row['target_status']} "
            f"n={row['n_runs']} paired={row['paired_runs']} "
            f"MELD={fmt(row['MELD_wf1_mean'])}+/-{fmt(row['MELD_wf1_std'])} "
            f"eJSL={fmt(row['eJSL_wf1_mean'])}+/-{fmt(row['eJSL_wf1_std'])} "
            f"delta_MELD={fmt(row['delta_MELD_mean'])} "
            f"delta_eJSL={fmt(row['delta_eJSL_mean'])} "
            f"corr={fmt(row['delta_corr'])} target={row['target_count']} "
            f"both_up={row['both_up_count']} both_down={row['both_down_count']} "
            f"bobsl_max={row['bobsl_max']} lr={row['lr']} drop={row['dropout']} loss={row['loss']}"
        )
    lines.extend(["", f"[joint-tradeoff] top target-like natural cases"])
    ranked = sorted(
        [row for row in paired_rows if not math.isnan(as_float(row.get("delta_MELD_wf1"))) and not math.isnan(as_float(row.get("delta_eJSL_wf1")))],
        key=lambda row: (as_float(row.get("delta_MELD_wf1")) - as_float(row.get("delta_eJSL_wf1")), as_float(row.get("delta_MELD_wf1"))),
        reverse=True,
    )
    for idx, row in enumerate(ranked[:top_k], start=1):
        lines.append(
            f"  {idx:02d}. {row['baseline']}/{row['config']} seed={row['seed']} quadrant={row['quadrant']} "
            f"MELD={fmt(row['source_wf1'])} eJSL={fmt(row['external_wf1'])} "
            f"delta_MELD={fmt(row['delta_MELD_wf1'])} delta_eJSL={fmt(row['delta_eJSL_wf1'])}"
        )
    return "\n".join(lines) + "\n"


def parse_root_args(root_args: Sequence[str], use_defaults: bool) -> List[Tuple[str, Path]]:
    roots: List[Tuple[str, Path]] = []
    if use_defaults or not root_args:
        roots.extend((name, Path(path)) for name, path in DEFAULT_ROOTS.items())
    for item in root_args:
        if "=" not in item:
            raise ValueError(f"--root must be NAME=PATH, got: {item}")
        name, path = item.split("=", 1)
        roots.append((name.strip(), Path(path.strip())))
    return roots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize natural MELD/eJSL trade-off for joint BOBSL+MELD visual baselines.")
    parser.add_argument("--root", action="append", default=[], help="Baseline root as NAME=PATH. Can be repeated.")
    parser.add_argument("--default_roots", action="store_true", help="Include default server roots for CMERC, ConxGNN, and ECERC.")
    parser.add_argument("--out_dir", type=str, default="/raid_zoe/home/lr/maokeyu/sign/joint_tradeoff_natural_summary")
    parser.add_argument("--eps", type=float, default=0.0, help="Tolerance for up/down quadrant decisions.")
    parser.add_argument("--target_fraction", type=float, default=0.6, help="Minimum target seed fraction for TARGET status.")
    parser.add_argument("--strong_target_fraction", type=float, default=0.8, help="Minimum target seed fraction for STRONG_TARGET status.")
    parser.add_argument("--top_k", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = parse_root_args(args.root, args.default_roots)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = add_paired_deltas(collect_rows(roots), eps=args.eps)
    groups = summarize_groups(
        rows,
        eps=args.eps,
        target_fraction=args.target_fraction,
        strong_fraction=args.strong_target_fraction,
    )
    target_rows = [row for row in rows if row.get("quadrant") == "target_meld_up_ejsl_down"]
    target_configs = target_config_ranking(groups)

    write_csv(out_dir / "joint_tradeoff_rows.csv", rows)
    write_csv(out_dir / "joint_tradeoff_group_summary.csv", groups)
    write_csv(out_dir / "joint_tradeoff_target_cases.csv", target_rows)
    write_csv(out_dir / "joint_tradeoff_target_config_ranking.csv", target_configs)
    report = build_report(rows, groups, top_k=args.top_k)
    (out_dir / "joint_tradeoff_report.txt").write_text(report, encoding="utf-8")

    print(report, end="")
    print(f"[joint-tradeoff] rows_csv={out_dir / 'joint_tradeoff_rows.csv'}")
    print(f"[joint-tradeoff] group_csv={out_dir / 'joint_tradeoff_group_summary.csv'}")
    print(f"[joint-tradeoff] target_csv={out_dir / 'joint_tradeoff_target_cases.csv'}")
    print(f"[joint-tradeoff] target_config_csv={out_dir / 'joint_tradeoff_target_config_ranking.csv'}")
    print(f"[joint-tradeoff] report={out_dir / 'joint_tradeoff_report.txt'}")


if __name__ == "__main__":
    main()
