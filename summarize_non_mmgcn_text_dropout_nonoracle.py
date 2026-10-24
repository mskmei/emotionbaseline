#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, Optional


LABELS = ["A", "N", "J", "S"]


def load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_config(path: Path, root: Path) -> Dict:
    cur = path if path.is_dir() else path.parent
    while True:
        candidate = cur / "config.json"
        data = load_json(candidate)
        if data is not None:
            return data
        if cur == root or cur.parent == cur:
            return {}
        cur = cur.parent


def per_class_f1(summary: Optional[Dict], label: str) -> str:
    if not summary:
        return ""
    per_class = summary.get("per_class") or {}
    value = per_class.get(label, {}).get("f1")
    return "" if value is None else f"{float(value):.6f}"


def metric(summary: Optional[Dict], name: str) -> str:
    if not summary:
        return ""
    value = summary.get(name)
    return "" if value is None else f"{float(value):.6f}"


def iter_run_dirs(root: Path) -> Iterable[Path]:
    seen = set()
    for path in root.rglob("external_test_best_summary.json"):
        run_dir = path.parent
        if run_dir not in seen:
            seen.add(run_dir)
            yield run_dir
    for path in root.rglob("*_ejsl_summary.json"):
        if "telme" not in str(path).lower():
            continue
        run_dir = path.parent
        if run_dir.name == "eval":
            run_dir = run_dir.parent
        if run_dir not in seen:
            seen.add(run_dir)
            yield run_dir


def read_run(run_dir: Path, root: Path) -> Optional[Dict]:
    config = find_config(run_dir, root)
    baseline = config.get("baseline", "")
    if not baseline:
        if (run_dir / "external_test_best_summary.json").exists():
            external = load_json(run_dir / "external_test_best_summary.json") or {}
            baseline = external.get("baseline", "")
        else:
            baseline = "TelME"

    if baseline == "TelME":
        eval_dir = run_dir / "eval"
        modality = str(config.get("modality", ""))
        external_path = eval_dir / f"telme_meld4_{modality}_ejsl_summary.json"
        external = load_json(external_path)
        source = load_json(run_dir / "save_model" / modality / "meld_test_report.summary.json")
    else:
        source = load_json(run_dir / "source_test_best_summary.json")
        external = load_json(run_dir / "external_test_best_summary.json")

    if source is None and external is None:
        return None

    row = {
        "baseline": baseline,
        "run_name": config.get("run_name", run_dir.name),
        "modality": config.get("modality", ""),
        "train_text_dropout": config.get("train_text_dropout", ""),
        "dropout_scope": config.get("train_text_dropout_scope", ""),
        "source_weighted_f1": metric(source, "weighted_f1"),
        "source_macro_f1": metric(source, "macro_f1"),
        "source_acc": metric(source, "accuracy"),
        "ejsl_weighted_f1": metric(external, "weighted_f1"),
        "ejsl_macro_f1": metric(external, "macro_f1"),
        "ejsl_acc": metric(external, "accuracy"),
        "run_dir": str(run_dir),
    }
    for label in LABELS:
        row[f"source_{label}_f1"] = per_class_f1(source, label)
        row[f"ejsl_{label}_f1"] = per_class_f1(external, label)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize non-MMGCN non-oracle text-dropout runs.")
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--out_json", type=str, required=True)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    rows = []
    for run_dir in sorted(iter_run_dirs(root), key=lambda p: str(p)):
        row = read_run(run_dir, root)
        if row is not None:
            rows.append(row)

    order = {"text": 0, "tv": 1, "video": 2, "visual": 2}
    rows.sort(key=lambda r: (r["baseline"], order.get(str(r["modality"]), 9), str(r["train_text_dropout"]), r["run_name"]))

    fieldnames = [
        "baseline",
        "run_name",
        "modality",
        "train_text_dropout",
        "dropout_scope",
        "source_weighted_f1",
        "source_macro_f1",
        "source_acc",
        "ejsl_weighted_f1",
        "ejsl_macro_f1",
        "ejsl_acc",
    ]
    for label in LABELS:
        fieldnames.extend([f"source_{label}_f1", f"ejsl_{label}_f1"])
    fieldnames.append("run_dir")

    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[summary] rows={len(rows)}")
    print(f"[summary] csv={out_csv}")
    print(f"[summary] json={out_json}")
    for idx, row in enumerate(sorted(rows, key=lambda r: float(r['ejsl_weighted_f1'] or -1), reverse=True), 1):
        print(
            f"{idx:02d}. {row['baseline']} {row['run_name']} "
            f"mod={row['modality']} td={row['train_text_dropout']} "
            f"MELD={row['source_weighted_f1']} eJSL={row['ejsl_weighted_f1']}"
        )


if __name__ == "__main__":
    main()
