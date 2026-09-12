#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_ROOT = "/raid_zoe/home/lr/maokeyu/sign/mmgcn_bobsl_meld_ejsl/video_seed5_bobsl_vs_scratch"


def as_float(value: Any, default: float = 0.0) -> float:
    if value in ("", None):
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


def config_from_pair(pair: str) -> str:
    match = re.match(r"^(.*)_seed\d+$", pair)
    return match.group(1) if match else pair


def seed_from_pair(pair: str, row: Dict[str, str]) -> str:
    match = re.match(r"^.*_seed(\d+)$", pair)
    return match.group(1) if match else row.get("ft_seed", "")


def summarize_config(config: str, rows: Sequence[Dict[str, str]]) -> Dict[str, Any]:
    meld_delta = [as_float(row.get("delta_source_wf1")) for row in rows]
    ejsl_delta = [as_float(row.get("delta_external_at_source_wf1")) for row in rows]
    scratch_meld = [as_float(row.get("scratch_source_wf1")) for row in rows]
    bobsl_meld = [as_float(row.get("bobsl_source_wf1")) for row in rows]
    scratch_ejsl = [as_float(row.get("scratch_external_at_source_wf1")) for row in rows]
    bobsl_ejsl = [as_float(row.get("bobsl_external_at_source_wf1")) for row in rows]
    meld_wins, meld_ties, meld_losses = sign_counts(meld_delta)
    ejsl_wins, ejsl_ties, ejsl_losses = sign_counts(ejsl_delta)
    target_count = sum(1 for m, e in zip(meld_delta, ejsl_delta) if m > 0 and e < 0)
    both_up = sum(1 for m, e in zip(meld_delta, ejsl_delta) if m > 0 and e > 0)
    both_down = sum(1 for m, e in zip(meld_delta, ejsl_delta) if m < 0 and e < 0)
    meld_down_ejsl_up = sum(1 for m, e in zip(meld_delta, ejsl_delta) if m < 0 and e > 0)
    meld_stats = numeric(meld_delta)
    ejsl_stats = numeric(ejsl_delta)
    example = max(rows, key=lambda row: as_float(row.get("delta_source_wf1")) - as_float(row.get("delta_external_at_source_wf1")))

    return {
        "config": config,
        "n_seeds": len(rows),
        "seeds": " ".join(seed_from_pair(row.get("pair", ""), row) for row in rows),
        "pretrain_key": rows[0].get("pretrain_key", "") if rows else "",
        "ft_lr": rows[0].get("ft_lr", "") if rows else "",
        "ft_dropout": rows[0].get("ft_dropout", "") if rows else "",
        "ft_loss": rows[0].get("ft_loss", "") if rows else "",
        "ft_focal_gamma": rows[0].get("ft_focal_gamma", "") if rows else "",
        "ft_extra_args": rows[0].get("ft_extra_args", "") if rows else "",
        "scratch_meld_mean": mean(scratch_meld) if scratch_meld else 0.0,
        "bobsl_meld_mean": mean(bobsl_meld) if bobsl_meld else 0.0,
        "delta_meld_mean": meld_stats["mean"],
        "delta_meld_median": meld_stats["median"],
        "delta_meld_std": meld_stats["std"],
        "delta_meld_min": meld_stats["min"],
        "delta_meld_max": meld_stats["max"],
        "meld_wins": meld_wins,
        "meld_ties": meld_ties,
        "meld_losses": meld_losses,
        "scratch_ejsl_mean": mean(scratch_ejsl) if scratch_ejsl else 0.0,
        "bobsl_ejsl_mean": mean(bobsl_ejsl) if bobsl_ejsl else 0.0,
        "delta_ejsl_mean": ejsl_stats["mean"],
        "delta_ejsl_median": ejsl_stats["median"],
        "delta_ejsl_std": ejsl_stats["std"],
        "delta_ejsl_min": ejsl_stats["min"],
        "delta_ejsl_max": ejsl_stats["max"],
        "ejsl_wins": ejsl_wins,
        "ejsl_ties": ejsl_ties,
        "ejsl_losses": ejsl_losses,
        "target_meld_up_ejsl_down": target_count,
        "both_up": both_up,
        "both_down": both_down,
        "meld_down_ejsl_up": meld_down_ejsl_up,
        "delta_corr_meld_ejsl": safe_corr(meld_delta, ejsl_delta),
        "best_target_like_pair": example.get("pair", ""),
        "best_target_like_delta_meld": example.get("delta_source_wf1", ""),
        "best_target_like_delta_ejsl": example.get("delta_external_at_source_wf1", ""),
    }


def build_report(rows: Sequence[Dict[str, Any]], top_k: int) -> str:
    ranked_target = sorted(
        rows,
        key=lambda row: (as_float(row.get("delta_meld_mean")) - as_float(row.get("delta_ejsl_mean"))),
        reverse=True,
    )
    ranked_meld = sorted(rows, key=lambda row: as_float(row.get("delta_meld_mean")), reverse=True)
    ranked_ejsl_hurt = sorted(rows, key=lambda row: as_float(row.get("delta_ejsl_mean")))

    lines = [
        "[seed5-summary] metric: weighted F1; delta = BOBSL-init - scratch",
        f"[seed5-summary] configs={len(rows)}",
        "",
        "[seed5-summary] most target-like configs: MELD up, eJSL down",
    ]
    for rank, row in enumerate(ranked_target[:top_k], start=1):
        lines.append(
            "  "
            f"{rank:02d}. {row['config']} "
            f"n={row['n_seeds']} pretrain={row['pretrain_key']} "
            f"MELD_delta_mean={fmt(row['delta_meld_mean'])} "
            f"eJSL_delta_mean={fmt(row['delta_ejsl_mean'])} "
            f"target_count={row['target_meld_up_ejsl_down']}/{row['n_seeds']} "
            f"MELD_w/t/l={row['meld_wins']}/{row['meld_ties']}/{row['meld_losses']} "
            f"eJSL_w/t/l={row['ejsl_wins']}/{row['ejsl_ties']}/{row['ejsl_losses']} "
            f"example={row['best_target_like_pair']}({fmt(row['best_target_like_delta_meld'])}, {fmt(row['best_target_like_delta_ejsl'])})"
        )

    lines += ["", "[seed5-summary] top mean MELD gains"]
    for rank, row in enumerate(ranked_meld[:top_k], start=1):
        lines.append(
            "  "
            f"{rank:02d}. {row['config']} "
            f"scratch_MELD={fmt(row['scratch_meld_mean'])} "
            f"bobsl_MELD={fmt(row['bobsl_meld_mean'])} "
            f"delta={fmt(row['delta_meld_mean'])} "
            f"eJSL_delta={fmt(row['delta_ejsl_mean'])}"
        )

    lines += ["", "[seed5-summary] strongest mean eJSL drops"]
    for rank, row in enumerate(ranked_ejsl_hurt[:top_k], start=1):
        lines.append(
            "  "
            f"{rank:02d}. {row['config']} "
            f"delta_eJSL={fmt(row['delta_ejsl_mean'])} "
            f"delta_MELD={fmt(row['delta_meld_mean'])} "
            f"target_count={row['target_meld_up_ejsl_down']}/{row['n_seeds']}"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Average video-only BOBSL-vs-scratch paired sweep results over seed groups.")
    parser.add_argument("--root", type=str, default=DEFAULT_ROOT)
    parser.add_argument("--delta_csv", type=str, default="")
    parser.add_argument("--out_csv", type=str, default="")
    parser.add_argument("--out_txt", type=str, default="")
    parser.add_argument("--top_k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    delta_csv = Path(args.delta_csv).expanduser() if args.delta_csv else root / "video_pair_sweep_deltas.csv"
    if not delta_csv.exists():
        raise FileNotFoundError(f"delta_csv not found: {delta_csv}")

    rows = read_csv(delta_csv)
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(config_from_pair(row.get("pair", "")), []).append(row)

    summaries = [summarize_config(config, sorted(config_rows, key=lambda row: seed_from_pair(row.get("pair", ""), row))) for config, config_rows in grouped.items()]
    summaries.sort(
        key=lambda row: (
            as_float(row.get("target_meld_up_ejsl_down")),
            as_float(row.get("delta_meld_mean")) - as_float(row.get("delta_ejsl_mean")),
        ),
        reverse=True,
    )

    out_csv = Path(args.out_csv).expanduser() if args.out_csv else root / "video_seed5_group_averages.csv"
    out_txt = Path(args.out_txt).expanduser() if args.out_txt else root / "video_seed5_group_averages.txt"
    write_csv(summaries, out_csv)
    report = build_report(summaries, args.top_k)
    out_txt.write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"[seed5-summary] csv={out_csv}")
    print(f"[seed5-summary] report={out_txt}")


if __name__ == "__main__":
    main()
