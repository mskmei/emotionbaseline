#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", value).strip("_")


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize OpenRouter zero-shot eJSL BLEU/cost metrics.")
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--out_json", type=str, required=True)
    return parser.parse_args()


def load_cost_summary(out_root: Path) -> Dict[str, object]:
    path = out_root / "cost_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    args = parse_args()
    out_root = Path(args.out_root)
    cost_summary = load_cost_summary(out_root)
    model_costs = cost_summary.get("models") if isinstance(cost_summary.get("models"), dict) else {}

    rows: List[Dict[str, object]] = []
    for model in args.models:
        model_name = safe_name(model)
        metrics_path = out_root / "metrics" / f"{model_name}.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        costs = model_costs.get(model, {}) if isinstance(model_costs, dict) else {}
        rows.append(
            {
                "model": model,
                "model_name": model_name,
                "n_scored": metrics.get("n_scored", ""),
                "n_missing": metrics.get("n_missing", ""),
                "corpus_bleu4_char": metrics.get("corpus_bleu4_char", ""),
                "mean_sentence_bleu4_char": metrics.get("mean_sentence_bleu4_char", ""),
                "mean_char_f1": metrics.get("mean_char_f1", ""),
                "mean_edit_similarity": metrics.get("mean_edit_similarity", ""),
                "mean_gt_len": metrics.get("mean_gt_len", ""),
                "mean_pred_len": metrics.get("mean_pred_len", ""),
                "total_cost_usd": costs.get("total_cost_usd", ""),
                "mean_cost_usd": costs.get("mean_cost_usd", ""),
                "review_csv": costs.get("review_csv", ""),
                "predictions_jsonl": costs.get("predictions_jsonl", ""),
                "metrics_json": str(metrics_path),
            }
        )

    def sort_key(row: Dict[str, object]):
        value = row.get("corpus_bleu4_char")
        try:
            return float(value)
        except Exception:
            return -1.0

    rows = sorted(rows, key=sort_key, reverse=True)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "model_name",
        "n_scored",
        "n_missing",
        "corpus_bleu4_char",
        "mean_sentence_bleu4_char",
        "mean_char_f1",
        "mean_edit_similarity",
        "mean_gt_len",
        "mean_pred_len",
        "total_cost_usd",
        "mean_cost_usd",
        "review_csv",
        "predictions_jsonl",
        "metrics_json",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    for idx, row in enumerate(rows, start=1):
        print(
            f"{idx:02d}. {row['model']} corpus_bleu4_char={row['corpus_bleu4_char']} "
            f"char_f1={row['mean_char_f1']} cost={row['total_cost_usd']}"
        )


if __name__ == "__main__":
    main()
