#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Dict, Iterable, List, Sequence, Tuple


DEFAULT_ROOT = "/raid_zoe/home/lr/maokeyu/sign/mmgcn_bobsl_meld_ejsl/video_meld_only_epoch_grid"


def as_float(value: Any, default: float = 0.0) -> float:
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    if value in ("", None):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def fmt(value: Any, digits: int = 4) -> str:
    if value in ("", None):
        return "NA"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


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


def parse_grid(raw: str) -> List[int]:
    out: List[int] = []
    for part in str(raw).replace(",", " ").split():
        value = int(part)
        if value not in out:
            out.append(value)
    return sorted(out)


def seed_from_trial(name: str) -> str:
    marker = "_seed"
    if marker in name:
        return name.rsplit(marker, 1)[1]
    return ""


def metric_row(seed: str, target_epoch: int, mode: str, selected: Dict[str, str]) -> Dict[str, Any]:
    return {
        "seed": seed,
        "target_epoch": target_epoch,
        "mode": mode,
        "selected_epoch": as_int(selected.get("epoch")),
        "train_wf1": selected.get("train_weighted_f1", ""),
        "meld_source_acc": selected.get("source_test_acc", ""),
        "meld_source_macro_f1": selected.get("source_test_macro_f1", ""),
        "meld_source_wf1": selected.get("source_test_weighted_f1", ""),
        "ejsl_external_acc": selected.get("external_test_acc", ""),
        "ejsl_external_macro_f1": selected.get("external_test_macro_f1", ""),
        "ejsl_external_wf1": selected.get("external_test_weighted_f1", ""),
    }


def load_seed_rows(root: Path, grid: Sequence[int]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for metrics_csv in sorted(root.glob("meld_only_seed*/video/mmgcn_unified_video_epoch_metrics.csv")):
        trial = metrics_csv.parent.parent.name
        seed = seed_from_trial(trial)
        rows = read_csv(metrics_csv)
        by_epoch = {as_int(row.get("epoch")): row for row in rows}

        for target_epoch in grid:
            exact = by_epoch.get(target_epoch)
            if exact is not None:
                out.append(metric_row(seed, target_epoch, "exact_epoch", exact))

            upto = [row for row in rows if as_int(row.get("epoch")) <= target_epoch and row.get("source_test_weighted_f1") not in ("", None)]
            if upto:
                selected = max(upto, key=lambda row: as_float(row.get("source_test_weighted_f1")))
                out.append(metric_row(seed, target_epoch, "best_upto_source", selected))

            upto_external = [row for row in rows if as_int(row.get("epoch")) <= target_epoch and row.get("external_test_weighted_f1") not in ("", None)]
            if upto_external:
                selected_external = max(upto_external, key=lambda row: as_float(row.get("external_test_weighted_f1")))
                out.append(metric_row(seed, target_epoch, "best_upto_external_oracle", selected_external))
    return out


def group_summary(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["mode"]), int(row["target_epoch"])), []).append(row)

    summaries: List[Dict[str, Any]] = []
    for (mode, target_epoch), group_rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        source = [as_float(row.get("meld_source_wf1")) for row in group_rows]
        external = [as_float(row.get("ejsl_external_wf1")) for row in group_rows]
        selected_epochs = [as_float(row.get("selected_epoch")) for row in group_rows]
        source_stats = numeric(source)
        external_stats = numeric(external)
        selected_stats = numeric(selected_epochs)
        summaries.append(
            {
                "mode": mode,
                "target_epoch": target_epoch,
                "n_seeds": len(group_rows),
                "seeds": " ".join(str(row.get("seed", "")) for row in sorted(group_rows, key=lambda row: str(row.get("seed", "")))),
                "selected_epoch_mean": selected_stats["mean"],
                "selected_epoch_min": selected_stats["min"],
                "selected_epoch_max": selected_stats["max"],
                "meld_source_wf1_mean": source_stats["mean"],
                "meld_source_wf1_std": source_stats["std"],
                "meld_source_wf1_median": source_stats["median"],
                "meld_source_wf1_min": source_stats["min"],
                "meld_source_wf1_max": source_stats["max"],
                "ejsl_external_wf1_mean": external_stats["mean"],
                "ejsl_external_wf1_std": external_stats["std"],
                "ejsl_external_wf1_median": external_stats["median"],
                "ejsl_external_wf1_min": external_stats["min"],
                "ejsl_external_wf1_max": external_stats["max"],
            }
        )
    return summaries


def build_report(summaries: Sequence[Dict[str, Any]]) -> str:
    order = {"best_upto_source": 0, "exact_epoch": 1, "best_upto_external_oracle": 2}
    rows = sorted(summaries, key=lambda row: (order.get(str(row["mode"]), 99), int(row["target_epoch"])))
    lines = [
        "[meld-only-epoch-grid] source=MELD test, external=eJSL translated test",
        "[meld-only-epoch-grid] best_upto_source matches the normal source-selected checkpoint if training stops at target_epoch.",
        "",
    ]
    current_mode = None
    for index, row in enumerate(rows):
        if row["mode"] != current_mode:
            current_mode = row["mode"]
            lines.append(f"[meld-only-epoch-grid] mode={current_mode}")
        lines.append(
            "  "
            f"epoch={row['target_epoch']:>3} n={row['n_seeds']} "
            f"selected_epoch_mean={fmt(row['selected_epoch_mean'], 1)} "
            f"MELD={fmt(row['meld_source_wf1_mean'])}+/-{fmt(row['meld_source_wf1_std'])} "
            f"eJSL={fmt(row['ejsl_external_wf1_mean'])}+/-{fmt(row['ejsl_external_wf1_std'])}"
        )
        next_index = index + 1
        if next_index < len(rows) and rows[next_index]["mode"] != current_mode:
            lines.append("")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize MELD-only video baseline over an epoch grid.")
    parser.add_argument("--root", type=str, default=DEFAULT_ROOT)
    parser.add_argument("--epochs", type=str, default="1 5 10 15 20 25 30 35 40 45 50")
    parser.add_argument("--out_seed_csv", type=str, default="")
    parser.add_argument("--out_group_csv", type=str, default="")
    parser.add_argument("--out_txt", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    grid = parse_grid(args.epochs)
    seed_rows = load_seed_rows(root, grid)
    if not seed_rows:
        raise RuntimeError(f"No epoch metric files found under {root}/meld_only_seed*/video")

    summaries = group_summary(seed_rows)
    out_seed = Path(args.out_seed_csv).expanduser() if args.out_seed_csv else root / "meld_only_epoch_grid_seed_rows.csv"
    out_group = Path(args.out_group_csv).expanduser() if args.out_group_csv else root / "meld_only_epoch_grid_summary.csv"
    out_txt = Path(args.out_txt).expanduser() if args.out_txt else root / "meld_only_epoch_grid_summary.txt"
    write_csv(seed_rows, out_seed)
    write_csv(summaries, out_group)
    report = build_report(summaries)
    out_txt.write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"[meld-only-epoch-grid] seed_csv={out_seed}")
    print(f"[meld-only-epoch-grid] summary_csv={out_group}")
    print(f"[meld-only-epoch-grid] report={out_txt}")


if __name__ == "__main__":
    main()
