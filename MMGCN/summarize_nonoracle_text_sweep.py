#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize MMGCN non-oracle text root sweep results.")
    parser.add_argument("--sweep_root", type=str, required=True)
    parser.add_argument("--out_csv", type=str, default="")
    parser.add_argument("--out_json", type=str, default="")
    parser.add_argument("--rank_metric", type=str, default="weighted_f1")
    return parser.parse_args()


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def row_for(candidate_dir: Path, modality: str) -> Dict[str, object] | None:
    meta_path = candidate_dir / "candidate_meta.json"
    summary_path = candidate_dir / "runs" / modality / "external_test_best_summary.json"
    if not summary_path.exists():
        return None
    meta = load_json(meta_path) if meta_path.exists() else {}
    summary = load_json(summary_path)
    per_class = summary.get("per_class", {}) or {}
    row: Dict[str, object] = {
        "candidate": meta.get("candidate", candidate_dir.name),
        "modality": modality,
        "accuracy": summary.get("accuracy"),
        "macro_f1": summary.get("macro_f1"),
        "weighted_f1": summary.get("weighted_f1"),
        "n_samples": summary.get("n_samples"),
        "best_epoch": summary.get("best_epoch_by_selection_metric") or summary.get("best_epoch_by_external_or_source_weighted_f1"),
        "best_selection_split": summary.get("best_selection_split"),
        "best_selection_score": summary.get("best_selection_score"),
        "txt_root": meta.get("txt_root", ""),
        "translated_txt_root": meta.get("translated_txt_root", ""),
        "ejsl_pkl": meta.get("ejsl_pkl", ""),
        "run_dir": str(candidate_dir / "runs" / modality),
        "summary_json": str(summary_path),
    }
    for label in ["A", "N", "J", "S"]:
        values = per_class.get(label, {}) or {}
        row[f"{label}_precision"] = values.get("precision")
        row[f"{label}_recall"] = values.get("recall")
        row[f"{label}_f1"] = values.get("f1")
        row[f"{label}_support"] = values.get("support")
    return row


def main():
    args = parse_args()
    sweep_root = Path(args.sweep_root)
    rows: List[Dict[str, object]] = []
    for candidate_dir in sorted(p for p in sweep_root.iterdir() if p.is_dir()):
        if candidate_dir.name in {"features", "translated_txt", "feature_cache"}:
            continue
        for modality in ["text", "tv"]:
            row = row_for(candidate_dir, modality)
            if row is not None:
                rows.append(row)

    if not rows:
        raise RuntimeError(f"No external_test_best_summary.json files found under {sweep_root}")

    rows.sort(
        key=lambda row: (
            str(row.get("modality", "")),
            -float(row.get(args.rank_metric) or -1),
            str(row.get("candidate", "")),
        )
    )
    ranked: List[Dict[str, object]] = []
    current_modality = None
    rank = 0
    for row in rows:
        if row["modality"] != current_modality:
            current_modality = row["modality"]
            rank = 1
        else:
            rank += 1
        ranked.append({"rank": rank, **row})

    fieldnames = [
        "rank",
        "candidate",
        "modality",
        "weighted_f1",
        "macro_f1",
        "accuracy",
        "n_samples",
        "best_epoch",
        "best_selection_split",
        "best_selection_score",
        "A_f1",
        "N_f1",
        "J_f1",
        "S_f1",
        "A_recall",
        "N_recall",
        "J_recall",
        "S_recall",
        "txt_root",
        "translated_txt_root",
        "ejsl_pkl",
        "run_dir",
        "summary_json",
    ]
    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in ranked:
                writer.writerow({k: row.get(k, "") for k in fieldnames})
    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(ranked, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for modality in ["text", "tv"]:
        print(f"===== {modality} ranking by {args.rank_metric} =====")
        for row in [x for x in ranked if x["modality"] == modality]:
            print(
                f"{int(row['rank']):02d}. {row['candidate']} "
                f"wf1={float(row['weighted_f1']):.4f} "
                f"macro={float(row['macro_f1']):.4f} "
                f"acc={float(row['accuracy']):.4f} "
                f"txt={row['txt_root']}"
            )


if __name__ == "__main__":
    main()
