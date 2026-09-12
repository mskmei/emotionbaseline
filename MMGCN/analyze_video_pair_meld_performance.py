#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_DELTA_CSV = (
    "/raid_zoe/home/lr/maokeyu/sign/mmgcn_bobsl_meld_ejsl/"
    "video_bobsl_seed_param_sweep/video_pair_sweep_deltas.csv"
)
LABELS = ["A", "N", "J", "S"]


def as_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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


def fmt(value: Any, digits: int = 4) -> str:
    if value in ("", None):
        return "NA"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def trial_group(pair: str) -> str:
    if pair.startswith("pre_"):
        return "bobsl_pretrain"
    if pair.startswith("seed") or pair.startswith("base_seed"):
        return "finetune_seed"
    if pair.startswith("lr"):
        return "finetune_lr"
    if pair.startswith("drop"):
        return "finetune_dropout"
    if pair.startswith("gamma") or pair in {"nll", "nocw"}:
        return "loss"
    if pair.startswith("batch"):
        return "batch_size"
    if pair.startswith("layer") or pair == "nores":
        return "deepgcn_arch"
    return "other"


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


def numeric_summary(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": mean(values),
        "median": median(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def load_source_class_deltas(runs_csv: Path) -> Dict[str, Dict[str, float]]:
    if not runs_csv.exists():
        return {}
    runs = read_csv(runs_csv)
    by_pair: Dict[str, Dict[str, Dict[str, str]]] = {}
    for row in runs:
        by_pair.setdefault(row.get("pair", ""), {})[row.get("variant", "")] = row

    out: Dict[str, Dict[str, float]] = {}
    for pair, variants in by_pair.items():
        scratch = variants.get("scratch")
        bobsl = variants.get("bobsl")
        if not scratch or not bobsl:
            continue
        out[pair] = {}
        for label in LABELS:
            out[pair][f"scratch_source_{label}_f1"] = as_float(scratch.get(f"source_{label}_f1"))
            out[pair][f"bobsl_source_{label}_f1"] = as_float(bobsl.get(f"source_{label}_f1"))
            out[pair][f"delta_source_{label}_f1"] = (
                out[pair][f"bobsl_source_{label}_f1"] - out[pair][f"scratch_source_{label}_f1"]
            )
    return out


def enrich_rows(delta_rows: List[Dict[str, str]], class_deltas: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in delta_rows:
        enriched: Dict[str, Any] = dict(row)
        pair = row.get("pair", "")
        enriched["group"] = trial_group(pair)
        enriched["scratch_source_wf1_num"] = as_float(row.get("scratch_source_wf1"))
        enriched["bobsl_source_wf1_num"] = as_float(row.get("bobsl_source_wf1"))
        enriched["delta_source_wf1_num"] = as_float(row.get("delta_source_wf1"))
        enriched["delta_external_at_source_wf1_num"] = as_float(row.get("delta_external_at_source_wf1"))
        enriched.update(class_deltas.get(pair, {}))
        out.append(enriched)
    return out


def group_summary(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("group", "other")), []).append(row)

    summaries: List[Dict[str, Any]] = []
    for group, group_rows in sorted(grouped.items()):
        deltas = [as_float(row.get("delta_source_wf1_num")) for row in group_rows]
        wins, ties, losses = sign_counts(deltas)
        stats = numeric_summary(deltas)
        best = max(group_rows, key=lambda row: as_float(row.get("delta_source_wf1_num")))
        worst = min(group_rows, key=lambda row: as_float(row.get("delta_source_wf1_num")))
        summaries.append(
            {
                "group": group,
                "n": len(group_rows),
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "mean_delta_source_wf1": stats["mean"],
                "median_delta_source_wf1": stats["median"],
                "min_delta_source_wf1": stats["min"],
                "max_delta_source_wf1": stats["max"],
                "best_pair": best.get("pair", ""),
                "best_delta_source_wf1": best.get("delta_source_wf1_num", 0.0),
                "worst_pair": worst.get("pair", ""),
                "worst_delta_source_wf1": worst.get("delta_source_wf1_num", 0.0),
            }
        )
    summaries.sort(key=lambda row: as_float(row["mean_delta_source_wf1"]), reverse=True)
    return summaries


def top_lines(title: str, rows: Sequence[Dict[str, Any]], top_k: int, reverse: bool = True) -> List[str]:
    ranked = sorted(rows, key=lambda row: as_float(row.get("delta_source_wf1_num")), reverse=reverse)
    lines = [title]
    for rank, row in enumerate(ranked[:top_k], start=1):
        lines.append(
            "  "
            f"{rank:02d}. {row.get('pair', '')} "
            f"group={row.get('group', '')} "
            f"scratch_MELD={fmt(row.get('scratch_source_wf1_num'))} "
            f"bobsl_MELD={fmt(row.get('bobsl_source_wf1_num'))} "
            f"delta_MELD={fmt(row.get('delta_source_wf1_num'), 4)} "
            f"delta_eJSL={fmt(row.get('delta_external_at_source_wf1_num'), 4)} "
            f"pretrain={row.get('pretrain_key', '')} "
            f"seed={row.get('ft_seed', '')} "
            f"lr={row.get('ft_lr', '')} "
            f"drop={row.get('ft_dropout', '')} "
            f"loss={row.get('ft_loss', '')} "
            f"extra={row.get('ft_extra_args', '')}"
        )
    return lines


def build_report(rows: Sequence[Dict[str, Any]], group_rows: Sequence[Dict[str, Any]], top_k: int, has_class: bool) -> str:
    deltas = [as_float(row.get("delta_source_wf1_num")) for row in rows]
    scratch_values = [as_float(row.get("scratch_source_wf1_num")) for row in rows]
    bobsl_values = [as_float(row.get("bobsl_source_wf1_num")) for row in rows]
    ext_deltas = [as_float(row.get("delta_external_at_source_wf1_num")) for row in rows]
    wins, ties, losses = sign_counts(deltas)
    both_up = sum(1 for row in rows if as_float(row.get("delta_source_wf1_num")) > 0 and as_float(row.get("delta_external_at_source_wf1_num")) > 0)
    meld_up_ejsl_down = sum(1 for row in rows if as_float(row.get("delta_source_wf1_num")) > 0 and as_float(row.get("delta_external_at_source_wf1_num")) < 0)
    meld_down_ejsl_up = sum(1 for row in rows if as_float(row.get("delta_source_wf1_num")) < 0 and as_float(row.get("delta_external_at_source_wf1_num")) > 0)
    both_down = sum(1 for row in rows if as_float(row.get("delta_source_wf1_num")) < 0 and as_float(row.get("delta_external_at_source_wf1_num")) < 0)
    corr = safe_corr(deltas, ext_deltas)
    stats = numeric_summary(deltas)

    lines = [
        "[paired-meld] metric: MELD/source_test weighted F1, delta = BOBSL-init - scratch",
        f"[paired-meld] completed_pairs={len(rows)}",
        (
            f"[paired-meld] scratch_mean={fmt(mean(scratch_values) if scratch_values else 0)} "
            f"bobsl_mean={fmt(mean(bobsl_values) if bobsl_values else 0)} "
            f"mean_delta={fmt(stats['mean'])} median_delta={fmt(stats['median'])} "
            f"std_delta={fmt(stats['std'])} min_delta={fmt(stats['min'])} max_delta={fmt(stats['max'])}"
        ),
        f"[paired-meld] BOBSL wins/ties/losses on MELD = {wins}/{ties}/{losses}",
        (
            "[paired-meld] source-vs-eJSL delta consistency: "
            f"both_up={both_up} MELD_up_eJSL_down={meld_up_ejsl_down} "
            f"MELD_down_eJSL_up={meld_down_ejsl_up} both_down={both_down} "
            f"corr={fmt(corr) if corr is not None else 'NA'}"
        ),
        "",
        "[paired-meld] group summary",
    ]

    for row in group_rows:
        lines.append(
            "  "
            f"{row['group']}: n={row['n']} wins/ties/losses={row['wins']}/{row['ties']}/{row['losses']} "
            f"mean_delta={fmt(row['mean_delta_source_wf1'])} "
            f"median_delta={fmt(row['median_delta_source_wf1'])} "
            f"best={row['best_pair']}({fmt(row['best_delta_source_wf1'])}) "
            f"worst={row['worst_pair']}({fmt(row['worst_delta_source_wf1'])})"
        )

    if has_class:
        lines += ["", "[paired-meld] MELD per-class F1 delta, averaged across pairs"]
        for label in LABELS:
            values = [as_float(row.get(f"delta_source_{label}_f1")) for row in rows if f"delta_source_{label}_f1" in row]
            class_wins, class_ties, class_losses = sign_counts(values)
            class_stats = numeric_summary(values)
            lines.append(
                "  "
                f"{label}: mean_delta={fmt(class_stats['mean'])} "
                f"median_delta={fmt(class_stats['median'])} "
                f"wins/ties/losses={class_wins}/{class_ties}/{class_losses} "
                f"min={fmt(class_stats['min'])} max={fmt(class_stats['max'])}"
            )

    lines += [""] + top_lines("[paired-meld] top BOBSL helps MELD", rows, top_k, reverse=True)
    lines += [""] + top_lines("[paired-meld] top BOBSL hurts MELD", rows, top_k, reverse=False)

    best_scratch = max(rows, key=lambda row: as_float(row.get("scratch_source_wf1_num")), default=None)
    best_bobsl = max(rows, key=lambda row: as_float(row.get("bobsl_source_wf1_num")), default=None)
    if best_scratch and best_bobsl:
        lines += [
            "",
            "[paired-meld] absolute best MELD",
            (
                "  "
                f"scratch: pair={best_scratch.get('pair', '')} "
                f"MELD={fmt(best_scratch.get('scratch_source_wf1_num'))} "
                f"paired_bobsl={fmt(best_scratch.get('bobsl_source_wf1_num'))}"
            ),
            (
                "  "
                f"bobsl: pair={best_bobsl.get('pair', '')} "
                f"MELD={fmt(best_bobsl.get('bobsl_source_wf1_num'))} "
                f"paired_scratch={fmt(best_bobsl.get('scratch_source_wf1_num'))}"
            ),
        ]

    return "\n".join(lines) + "\n"


def ranked_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    keys = [
        "pair",
        "group",
        "pretrain_key",
        "ft_seed",
        "ft_epochs",
        "ft_batch_size",
        "ft_lr",
        "ft_l2",
        "ft_dropout",
        "ft_loss",
        "ft_focal_gamma",
        "ft_extra_args",
        "scratch_source_wf1",
        "bobsl_source_wf1",
        "delta_source_wf1",
        "scratch_external_at_source_wf1",
        "bobsl_external_at_source_wf1",
        "delta_external_at_source_wf1",
        "scratch_selected_epoch",
        "bobsl_selected_epoch",
        "scratch_external_oracle_wf1",
        "bobsl_external_oracle_wf1",
        "delta_external_oracle_wf1",
    ]
    for label in LABELS:
        keys += [f"scratch_source_{label}_f1", f"bobsl_source_{label}_f1", f"delta_source_{label}_f1"]

    ordered = sorted(rows, key=lambda row: as_float(row.get("delta_source_wf1_num")), reverse=True)
    return [{key: row.get(key, "") for key in keys if key in row} for row in ordered]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze paired MELD/source performance for video-only MMGCN sweeps.")
    parser.add_argument("--delta_csv", type=str, default=DEFAULT_DELTA_CSV)
    parser.add_argument("--runs_csv", type=str, default="")
    parser.add_argument("--out_txt", type=str, default="")
    parser.add_argument("--out_ranked_csv", type=str, default="")
    parser.add_argument("--top_k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    delta_csv = Path(args.delta_csv).expanduser()
    if not delta_csv.exists():
        raise FileNotFoundError(f"delta_csv not found: {delta_csv}")

    runs_csv = Path(args.runs_csv).expanduser() if args.runs_csv else delta_csv.with_name("video_pair_sweep_runs.csv")
    out_txt = Path(args.out_txt).expanduser() if args.out_txt else delta_csv.with_name("video_pair_meld_analysis.txt")
    out_ranked = (
        Path(args.out_ranked_csv).expanduser()
        if args.out_ranked_csv
        else delta_csv.with_name("video_pair_meld_ranked.csv")
    )

    class_deltas = load_source_class_deltas(runs_csv)
    rows = enrich_rows(read_csv(delta_csv), class_deltas)
    if not rows:
        raise RuntimeError(f"No rows found in {delta_csv}")

    groups = group_summary(rows)
    report = build_report(rows, groups, args.top_k, has_class=bool(class_deltas))
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text(report, encoding="utf-8")
    write_csv(ranked_rows(rows), out_ranked)

    print(report, end="")
    print(f"[paired-meld] report_txt={out_txt}")
    print(f"[paired-meld] ranked_csv={out_ranked}")
    if class_deltas:
        print(f"[paired-meld] per-class MELD deltas loaded from {runs_csv}")
    else:
        print(f"[paired-meld] runs_csv not found or incomplete, skipped per-class MELD deltas: {runs_csv}")


if __name__ == "__main__":
    main()
