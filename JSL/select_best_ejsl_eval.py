#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def parse_args():
    parser = argparse.ArgumentParser(description="Select the best JSL checkpoint by eJSL oracle text metrics.")
    parser.add_argument("--eval_root", type=str, required=True)
    parser.add_argument("--metric", type=str, default="corpus_bleu4_char")
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--out_json", type=str, required=True)
    return parser.parse_args()


def load_metrics(eval_root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for path in sorted(eval_root.glob("*/metrics.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        checkpoint = path.parent.name
        data["checkpoint"] = checkpoint
        data["checkpoint_eval_dir"] = str(path.parent)
        rows.append(data)
    return rows


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "checkpoint",
        "corpus_bleu4_char",
        "mean_sentence_bleu4_char",
        "mean_char_f1",
        "mean_edit_similarity",
        "mean_gt_len",
        "mean_pred_len",
        "n_scored",
        "n_missing",
        "checkpoint_eval_dir",
        "predictions_jsonl",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main():
    args = parse_args()
    rows = load_metrics(Path(args.eval_root))
    if not rows:
        raise RuntimeError(f"No metrics.json files found under {args.eval_root}")
    rows = sorted(rows, key=lambda row: float(row.get(args.metric, float("-inf"))), reverse=True)
    best = rows[0]
    out = {
        "metric": args.metric,
        "best_checkpoint": best["checkpoint"],
        "best_checkpoint_eval_dir": best["checkpoint_eval_dir"],
        "best_metrics": best,
        "ranked_count": len(rows),
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(Path(args.out_csv), rows)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
