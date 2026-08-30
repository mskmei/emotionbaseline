#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def get_class(summary: Dict[str, Any], label: str, metric: str) -> Any:
    return (summary.get("per_class") or {}).get(label, {}).get(metric, "")


def load_rows(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(root.glob("*/tv/external_test_best_summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        trial_dir = path.parent.parent
        rows.append(
            {
                "trial": trial_dir.name,
                "summary_path": str(path),
                "accuracy": summary.get("accuracy", 0.0),
                "macro_f1": summary.get("macro_f1", 0.0),
                "weighted_f1": summary.get("weighted_f1", 0.0),
                "n_samples": summary.get("n_samples", 0),
                "seed": summary.get("seed", ""),
                "epochs": summary.get("epochs", ""),
                "batch_size": summary.get("batch_size", ""),
                "lr": summary.get("lr", ""),
                "l2": summary.get("l2", ""),
                "dropout": summary.get("dropout", ""),
                "loss": summary.get("loss", ""),
                "focal_gamma": summary.get("focal_gamma", ""),
                "class_weight": summary.get("class_weight", ""),
                "max_grad_norm": summary.get("max_grad_norm", ""),
                "use_speaker": summary.get("use_speaker", ""),
                "use_modal": summary.get("use_modal", ""),
                "av_using_lstm": summary.get("av_using_lstm", ""),
                "use_residue": summary.get("use_residue", ""),
                "pred_counts": json.dumps(summary.get("pred_counts", {}), ensure_ascii=False, sort_keys=True),
                "A_f1": get_class(summary, "A", "f1"),
                "N_f1": get_class(summary, "N", "f1"),
                "J_f1": get_class(summary, "J", "f1"),
                "S_f1": get_class(summary, "S", "f1"),
                "A_recall": get_class(summary, "A", "recall"),
                "N_recall": get_class(summary, "N", "recall"),
                "J_recall": get_class(summary, "J", "recall"),
                "S_recall": get_class(summary, "S", "recall"),
            }
        )
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize MMGCN unified trial search results.")
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--out_csv", type=str, default="")
    parser.add_argument("--metric", choices=["weighted_f1", "macro_f1", "accuracy"], default="weighted_f1")
    parser.add_argument("--top_k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser()
    rows = load_rows(root)
    rows.sort(key=lambda row: float(row.get(args.metric) or 0.0), reverse=True)

    out_csv = Path(args.out_csv).expanduser() if args.out_csv else root / "search_summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"[summary] root={root}")
    print(f"[summary] found={len(rows)} out_csv={out_csv}")
    for rank, row in enumerate(rows[: args.top_k], start=1):
        print(
            f"[summary][top{rank:02d}] trial={row['trial']} "
            f"wf1={float(row['weighted_f1']):.4f} "
            f"mf1={float(row['macro_f1']):.4f} "
            f"acc={float(row['accuracy']):.4f} "
            f"seed={row['seed']} lr={row['lr']} dropout={row['dropout']} "
            f"loss={row['loss']} gamma={row['focal_gamma']} "
            f"A={row['A_f1']} N={row['N_f1']} J={row['J_f1']} S={row['S_f1']}"
        )


if __name__ == "__main__":
    main()
