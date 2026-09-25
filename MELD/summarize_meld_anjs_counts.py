#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


ANJS_MAP = {
    "anger": "A",
    "neutral": "N",
    "joy": "J",
    "sadness": "S",
}
ANJS_ORDER = ["A", "N", "J", "S"]
MELD_ORDER = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
SPLITS = [
    ("train", "Train"),
    ("dev", "Val"),
    ("test", "Test"),
]


def resolve_split_csv(root: Path, split: str) -> Path:
    candidates = [
        root / f"{split}_meld_emo.csv",
        root / f"{split}_sent_emo.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    checked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Missing MELD csv for split={split}. Checked: {checked}")


def pct(part: int, total: int) -> float:
    return 100.0 * float(part) / float(total) if total else 0.0


def ordered_dict(keys: Iterable[str], counts: Counter) -> Dict[str, int]:
    return {key: int(counts.get(key, 0)) for key in keys}


def summarize_split(path: Path) -> Dict:
    emotion_counts: Counter = Counter()
    anjs_counts: Counter = Counter()
    dialogue_ids = set()
    rows = 0
    anjs_rows = 0
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        required = {"Emotion", "Dialogue_ID", "Utterance_ID"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        for row in reader:
            rows += 1
            emotion = str(row.get("Emotion", "")).strip().lower()
            emotion_counts[emotion] += 1
            dialogue_ids.add(str(row.get("Dialogue_ID", "")).strip())
            if emotion in ANJS_MAP:
                anjs_rows += 1
                anjs_counts[ANJS_MAP[emotion]] += 1
    other_counts = {
        emotion: int(count)
        for emotion, count in sorted(emotion_counts.items())
        if emotion not in ANJS_MAP
    }
    return {
        "csv": str(path),
        "utterances_total": int(rows),
        "dialogues": int(len(dialogue_ids)),
        "anjs_utterances": int(anjs_rows),
        "non_anjs_utterances": int(rows - anjs_rows),
        "anjs_ratio_percent": pct(anjs_rows, rows),
        "anjs_counts": ordered_dict(ANJS_ORDER, anjs_counts),
        "meld7_counts": ordered_dict(MELD_ORDER, emotion_counts),
        "non_anjs_counts": other_counts,
    }


def summarize(root: Path) -> Dict:
    split_summaries = {}
    total_anjs = Counter()
    total_meld = Counter()
    total_dialogues = 0
    total_rows = 0
    total_anjs_rows = 0
    for split, display_name in SPLITS:
        path = resolve_split_csv(root, split)
        summary = summarize_split(path)
        split_summaries[display_name] = summary
        total_dialogues += summary["dialogues"]
        total_rows += summary["utterances_total"]
        total_anjs_rows += summary["anjs_utterances"]
        total_anjs.update(summary["anjs_counts"])
        total_meld.update(summary["meld7_counts"])
    return {
        "meld_root": str(root),
        "label_mapping": {
            "A": "anger",
            "N": "neutral",
            "J": "joy",
            "S": "sadness",
        },
        "splits": split_summaries,
        "overall": {
            "utterances_total": int(total_rows),
            "dialogues": int(total_dialogues),
            "anjs_utterances": int(total_anjs_rows),
            "non_anjs_utterances": int(total_rows - total_anjs_rows),
            "anjs_ratio_percent": pct(total_anjs_rows, total_rows),
            "anjs_counts": ordered_dict(ANJS_ORDER, total_anjs),
            "meld7_counts": ordered_dict(MELD_ORDER, total_meld),
        },
    }


def print_markdown(summary: Dict) -> None:
    print(f"# MELD ANJS4 count summary")
    print()
    print(f"- meld_root: `{summary['meld_root']}`")
    print("- mapping: A=anger, N=neutral, J=joy, S=sadness")
    print()
    print("| split | dialogues | total utt | ANJS utt | ANJS % | A | N | J | S | non-ANJS |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for split_name in ["Train", "Val", "Test"]:
        item = summary["splits"][split_name]
        counts = item["anjs_counts"]
        print(
            f"| {split_name} | {item['dialogues']} | {item['utterances_total']} | "
            f"{item['anjs_utterances']} | {item['anjs_ratio_percent']:.2f} | "
            f"{counts['A']} | {counts['N']} | {counts['J']} | {counts['S']} | "
            f"{item['non_anjs_utterances']} |"
        )
    overall = summary["overall"]
    counts = overall["anjs_counts"]
    print(
        f"| Overall | {overall['dialogues']} | {overall['utterances_total']} | "
        f"{overall['anjs_utterances']} | {overall['anjs_ratio_percent']:.2f} | "
        f"{counts['A']} | {counts['N']} | {counts['J']} | {counts['S']} | "
        f"{overall['non_anjs_utterances']} |"
    )
    print()
    print("## MELD7 counts")
    print()
    print("| split | anger | disgust | fear | joy | neutral | sadness | surprise |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for split_name in ["Train", "Val", "Test"]:
        counts7 = summary["splits"][split_name]["meld7_counts"]
        print(
            f"| {split_name} | {counts7['anger']} | {counts7['disgust']} | "
            f"{counts7['fear']} | {counts7['joy']} | {counts7['neutral']} | "
            f"{counts7['sadness']} | {counts7['surprise']} |"
        )
    counts7 = overall["meld7_counts"]
    print(
        f"| Overall | {counts7['anger']} | {counts7['disgust']} | "
        f"{counts7['fear']} | {counts7['joy']} | {counts7['neutral']} | "
        f"{counts7['sadness']} | {counts7['surprise']} |"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize MELD Train/Val/Test emotion counts for ANJS4.")
    parser.add_argument("--meld_root", type=str, default="./dataset/MELD.Raw")
    parser.add_argument("--json_out", type=str, default="", help="Optional path to save the full summary as JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.meld_root).expanduser().resolve()
    summary = summarize(root)
    print_markdown(summary)
    if args.json_out:
        out_path = Path(args.json_out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print()
        print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
