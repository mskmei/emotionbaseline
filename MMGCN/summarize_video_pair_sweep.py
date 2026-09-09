#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


LABELS = ["A", "N", "J", "S"]


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    if value in (None, ""):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def get_class(summary: Dict[str, Any], label: str, metric: str) -> Any:
    return (summary.get("per_class") or {}).get(label, {}).get(metric, "")


def read_epoch_metrics(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def best_epoch(rows: List[Dict[str, str]], metric: str) -> Tuple[int, float, Dict[str, str]]:
    best_row: Dict[str, str] = {}
    best_score = float("-inf")
    best_ep = 0
    for row in rows:
        score = as_float(row.get(metric), float("-inf"))
        if score > best_score:
            best_score = score
            best_ep = as_int(row.get("epoch"))
            best_row = row
    if best_score == float("-inf"):
        return 0, 0.0, {}
    return best_ep, best_score, best_row


def find_metrics_csv(video_dir: Path, summary: Dict[str, Any]) -> Path:
    explicit = summary.get("epoch_metrics_csv")
    if explicit:
        path = Path(str(explicit)).expanduser()
        if path.exists():
            return path
    return video_dir / "mmgcn_unified_video_epoch_metrics.csv"


def load_run(root: Path, summary_path: Path) -> Dict[str, Any]:
    video_dir = summary_path.parent
    variant = video_dir.parent.name
    pair = video_dir.parent.parent.name
    config = load_json(video_dir.parent.parent / "config.json")
    external_best = load_json(summary_path)
    source_best = load_json(video_dir / "source_test_best_summary.json")
    external_final = load_json(video_dir / "external_test_final_summary.json")
    source_final = load_json(video_dir / "source_test_final_summary.json")
    metrics_csv = find_metrics_csv(video_dir, external_best)
    metrics_rows = read_epoch_metrics(metrics_csv)

    source_epoch, source_wf1, source_row = best_epoch(metrics_rows, "source_test_weighted_f1")
    external_epoch, external_wf1, external_row = best_epoch(metrics_rows, "external_test_weighted_f1")

    row: Dict[str, Any] = {
        "pair": pair,
        "variant": variant,
        "summary_path": str(summary_path),
        "metrics_csv": str(metrics_csv),
        "pretrain_key": config.get("pretrain_key", ""),
        "pretrain_checkpoint": config.get("pretrain_checkpoint", ""),
        "ft_seed": external_best.get("seed", config.get("ft_seed", "")),
        "ft_epochs": external_best.get("epochs", config.get("ft_epochs", "")),
        "ft_batch_size": external_best.get("batch_size", config.get("ft_batch_size", "")),
        "ft_lr": external_best.get("lr", config.get("ft_lr", "")),
        "ft_l2": external_best.get("l2", config.get("ft_l2", "")),
        "ft_dropout": external_best.get("dropout", config.get("ft_dropout", "")),
        "ft_loss": external_best.get("loss", config.get("ft_loss", "")),
        "ft_focal_gamma": external_best.get("focal_gamma", config.get("ft_focal_gamma", "")),
        "ft_class_weight": external_best.get("class_weight", ""),
        "ft_max_grad_norm": external_best.get("max_grad_norm", config.get("ft_max_grad_norm", "")),
        "ft_extra_args": config.get("ft_extra_args", ""),
        "selection_split": external_best.get("selection_split", ""),
        "selected_epoch": external_best.get("best_epoch_by_selection_metric", ""),
        "source_best_epoch": source_epoch,
        "source_best_wf1": source_wf1,
        "external_at_source_best_wf1": external_best.get("weighted_f1", ""),
        "external_at_source_best_macro_f1": external_best.get("macro_f1", ""),
        "external_at_source_best_acc": external_best.get("accuracy", ""),
        "external_oracle_epoch": external_epoch,
        "external_oracle_wf1": external_wf1,
        "source_at_external_oracle_wf1": as_float(external_row.get("source_test_weighted_f1")),
        "source_final_wf1": source_final.get("weighted_f1", ""),
        "external_final_wf1": external_final.get("weighted_f1", ""),
        "external_pred_counts": json.dumps(external_best.get("pred_counts", {}), ensure_ascii=False, sort_keys=True),
        "external_gold_counts": json.dumps(external_best.get("gold_counts", {}), ensure_ascii=False, sort_keys=True),
    }
    for label in LABELS:
        row[f"{label}_f1"] = get_class(external_best, label, "f1")
        row[f"{label}_recall"] = get_class(external_best, label, "recall")
        row[f"{label}_precision"] = get_class(external_best, label, "precision")
        row[f"{label}_mean_prob"] = get_class(external_best, label, "mean_pred_prob")
        row[f"source_{label}_f1"] = get_class(source_best, label, "f1")
    return row


def load_rows(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(root.glob("*/scratch/video/external_test_best_summary.json")):
        rows.append(load_run(root, path))
    for path in sorted(root.glob("*/bobsl/video/external_test_best_summary.json")):
        rows.append(load_run(root, path))
    return rows


def pair_delta_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_pair: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in rows:
        by_pair.setdefault(str(row["pair"]), {})[str(row["variant"])] = row

    out: List[Dict[str, Any]] = []
    for pair, variants in sorted(by_pair.items()):
        scratch = variants.get("scratch")
        bobsl = variants.get("bobsl")
        if not scratch or not bobsl:
            continue

        delta: Dict[str, Any] = {
            "pair": pair,
            "pretrain_key": bobsl.get("pretrain_key", ""),
            "ft_seed": bobsl.get("ft_seed", ""),
            "ft_epochs": bobsl.get("ft_epochs", ""),
            "ft_batch_size": bobsl.get("ft_batch_size", ""),
            "ft_lr": bobsl.get("ft_lr", ""),
            "ft_l2": bobsl.get("ft_l2", ""),
            "ft_dropout": bobsl.get("ft_dropout", ""),
            "ft_loss": bobsl.get("ft_loss", ""),
            "ft_focal_gamma": bobsl.get("ft_focal_gamma", ""),
            "ft_extra_args": bobsl.get("ft_extra_args", ""),
            "scratch_source_wf1": scratch.get("source_best_wf1", ""),
            "bobsl_source_wf1": bobsl.get("source_best_wf1", ""),
            "delta_source_wf1": as_float(bobsl.get("source_best_wf1")) - as_float(scratch.get("source_best_wf1")),
            "scratch_external_at_source_wf1": scratch.get("external_at_source_best_wf1", ""),
            "bobsl_external_at_source_wf1": bobsl.get("external_at_source_best_wf1", ""),
            "delta_external_at_source_wf1": as_float(bobsl.get("external_at_source_best_wf1")) - as_float(scratch.get("external_at_source_best_wf1")),
            "scratch_external_oracle_epoch": scratch.get("external_oracle_epoch", ""),
            "bobsl_external_oracle_epoch": bobsl.get("external_oracle_epoch", ""),
            "scratch_external_oracle_wf1": scratch.get("external_oracle_wf1", ""),
            "bobsl_external_oracle_wf1": bobsl.get("external_oracle_wf1", ""),
            "delta_external_oracle_wf1": as_float(bobsl.get("external_oracle_wf1")) - as_float(scratch.get("external_oracle_wf1")),
            "scratch_selected_epoch": scratch.get("selected_epoch", ""),
            "bobsl_selected_epoch": bobsl.get("selected_epoch", ""),
            "scratch_pred_counts": scratch.get("external_pred_counts", ""),
            "bobsl_pred_counts": bobsl.get("external_pred_counts", ""),
        }
        for label in LABELS:
            delta[f"scratch_{label}_f1"] = scratch.get(f"{label}_f1", "")
            delta[f"bobsl_{label}_f1"] = bobsl.get(f"{label}_f1", "")
            delta[f"delta_{label}_f1"] = as_float(bobsl.get(f"{label}_f1")) - as_float(scratch.get(f"{label}_f1"))
        out.append(delta)

    out.sort(key=lambda row: as_float(row["delta_external_at_source_wf1"]), reverse=True)
    return out


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize paired video-only MMGCN BOBSL-vs-scratch sweeps.")
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--out_csv", type=str, default="")
    parser.add_argument("--out_delta_csv", type=str, default="")
    parser.add_argument("--top_k", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    rows = load_rows(root)
    rows.sort(key=lambda row: (row["pair"], row["variant"]))

    out_csv = Path(args.out_csv).expanduser() if args.out_csv else root / "video_pair_sweep_runs.csv"
    write_csv(rows, out_csv)

    deltas = pair_delta_rows(rows)
    out_delta = Path(args.out_delta_csv).expanduser() if args.out_delta_csv else root / "video_pair_sweep_deltas.csv"
    write_csv(deltas, out_delta)

    print(f"[video-pair-summary] root={root}")
    print(f"[video-pair-summary] completed_runs={len(rows)} completed_pairs={len(deltas)}")
    print(f"[video-pair-summary] runs_csv={out_csv}")
    print(f"[video-pair-summary] deltas_csv={out_delta}")
    for rank, row in enumerate(deltas[: args.top_k], start=1):
        print(
            f"[video-pair-summary][top{rank:02d}] pair={row['pair']} "
            f"pretrain={row['pretrain_key']} seed={row['ft_seed']} "
            f"lr={row['ft_lr']} drop={row['ft_dropout']} loss={row['ft_loss']} "
            f"scratch_ext={as_float(row['scratch_external_at_source_wf1']):.4f} "
            f"bobsl_ext={as_float(row['bobsl_external_at_source_wf1']):.4f} "
            f"delta={as_float(row['delta_external_at_source_wf1']):+.4f} "
            f"oracle_delta={as_float(row['delta_external_oracle_wf1']):+.4f} "
            f"A={as_float(row['delta_A_f1']):+.3f} "
            f"N={as_float(row['delta_N_f1']):+.3f} "
            f"J={as_float(row['delta_J_f1']):+.3f} "
            f"S={as_float(row['delta_S_f1']):+.3f}"
        )


if __name__ == "__main__":
    main()
