#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

from manifest_utils import parse_ejsl_sample_id


def parse_args():
    parser = argparse.ArgumentParser(description="Print OpenRouter eJSL predictions beside oracle GT text.")
    parser.add_argument(
        "--review_csv",
        type=str,
        default="/raid_zoe/home/lr/maokeyu/sign/jsl_nonoracle/openrouter_ejsl_20/reviews/review_google__gemini-3.5-flash.csv",
    )
    parser.add_argument(
        "--structure_txt_root",
        type=str,
        default="/raid_elmo/home/lr/wangyi/PTR/STUDIES-Japanese/Short_dialogue",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--only_ok", action="store_true")
    parser.add_argument("--show_meta", action="store_true")
    return parser.parse_args()


def read_review(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def read_gt_line(structure_txt_root: Path, sample_id: str) -> Dict[str, str]:
    sd_id, dialogue_idx, utterance_idx, label = parse_ejsl_sample_id(sample_id)
    txt_file = structure_txt_root / sd_id / "txt" / f"{sd_id}-Dialogue-{dialogue_idx:02d}.txt"
    lines = [line.strip() for line in txt_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if utterance_idx < 1 or utterance_idx > len(lines):
        raise RuntimeError(f"{sample_id}: turn {utterance_idx} outside {txt_file} with {len(lines)} lines")
    parts = lines[utterance_idx - 1].split("|", 2)
    if len(parts) < 3:
        raise RuntimeError(f"{sample_id}: expected speaker|emotion|text in {txt_file}:{utterance_idx}")
    return {
        "speaker": parts[0].strip(),
        "emotion": parts[1].strip(),
        "text": parts[2].strip(),
        "label": label,
        "txt_file": str(txt_file),
    }


def main():
    args = parse_args()
    rows = read_review(Path(args.review_csv))
    if args.only_ok:
        rows = [row for row in rows if row.get("status") == "ok"]
    if args.limit > 0:
        rows = rows[: args.limit]

    print(f"[compare] review_csv={args.review_csv}")
    print(f"[compare] structure_txt_root={args.structure_txt_root}")
    print(f"[compare] rows={len(rows)}")
    print()

    for i, row in enumerate(rows, start=1):
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id:
            continue
        try:
            gt = read_gt_line(Path(args.structure_txt_root), sample_id)
            gt_text = gt["text"]
            speaker = gt["speaker"]
            emotion = gt["emotion"]
        except Exception as exc:
            gt_text = f"[GT-ERROR] {exc}"
            speaker = ""
            emotion = ""

        pred = str(row.get("text_ja", "")).strip()
        status = str(row.get("status", "")).strip()
        confidence = str(row.get("confidence", "")).strip()
        uncertain = str(row.get("uncertain", "")).strip()
        cost = str(row.get("cost_usd", "")).strip()

        print(f"===== {i:02d}. {sample_id} =====")
        if args.show_meta:
            print(f"status={status} speaker={speaker} emotion={emotion} confidence={confidence} uncertain={uncertain} cost_usd={cost}")
        else:
            print(f"status={status} confidence={confidence} uncertain={uncertain}")
        print(f"GT  : {gt_text}")
        print(f"PRED: {pred}")
        error = str(row.get("error", "")).strip()
        if error:
            print(f"ERR : {error}")
        print()


if __name__ == "__main__":
    main()
