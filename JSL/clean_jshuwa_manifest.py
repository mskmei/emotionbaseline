#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from manifest_utils import read_csv_rows, write_csv_rows


# Japanese characters: Hiragana, Katakana, CJK
JP_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")

ONLY_PUNCT_RE = re.compile(
    r"""^[\s\u3000、。，．！？!?…・･「」『』（）()［\]\[\]【】<>〈〉《》"'“”‘’\-ー〜~]+$"""
)

URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)

REPEAT_RE = re.compile(r"(.)\1{5,}")

SPACES_RE = re.compile(r"\s+")

QUOTE_EDGE_RE = re.compile(
    r"""^[「『"'“”‘’\s]+|[」』"'“”‘’\s]+$"""
)


# Strong metadata / channel-promotion terms only.
# Do NOT include domain terms such as 日本手話 / ろう者 / 聴者 / 指文字.
DEFAULT_META_TERMS = [
    "チャンネル登録",
    "チャンネルを登録",
    "高評価お願いします",
    "高評価よろしく",
    "コメントお願いします",
    "コメント欄",
    "概要欄",
    "字幕提供",
    "字幕制作",
    "ご視聴ありがとうございました",
    "ご視聴ありがとうございます",
]


# Match annotation-like noise rather than any sentence containing 音楽/拍手.
NOISE_ANNOTATION_PATTERNS = [
    re.compile(r"^\s*[\[\【（(]\s*(音楽|BGM|bgm|拍手|効果音|雑音|ノイズ)\s*[\]\】）)]\s*$"),
    re.compile(r"^\s*(音楽|BGM|bgm|拍手|効果音|雑音|ノイズ)\s*$"),
    re.compile(r"^\s*♪+\s*$"),
    re.compile(r"^\s*♪.*♪\s*$"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clean a J-Shuwa CC manifest for keypoint-to-Japanese training."
    )

    parser.add_argument("--in_csv", type=str, required=True)
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--report_json", type=str, required=True)

    # Duration
    parser.add_argument("--min_duration", type=float, default=0.5)
    parser.add_argument("--max_duration", type=float, default=20.0)

    # Text length
    parser.add_argument("--min_text_chars", type=int, default=2)
    parser.add_argument("--max_text_chars", type=int, default=120)
    parser.add_argument("--min_japanese_chars", type=int, default=1)

    # Alignment proxy
    parser.add_argument("--min_chars_per_sec", type=float, default=0.2)
    parser.add_argument("--max_chars_per_sec", type=float, default=15.0)

    # Duplicate control:
    # same video + same text is treated more strictly
    parser.add_argument("--max_same_video_duplicate", type=int, default=1)

    # Same transcript across DIFFERENT videos/signers can be useful.
    parser.add_argument("--max_duplicate_text", type=int, default=30)
    parser.add_argument("--max_short_duplicate_text", type=int, default=10)
    parser.add_argument("--short_text_chars", type=int, default=6)

    # Optional aggressive metadata cleaning
    parser.add_argument("--drop_meta_terms", action="store_true")
    parser.add_argument(
        "--meta_terms",
        type=str,
        default=",".join(DEFAULT_META_TERMS),
    )

    return parser.parse_args()


def split_terms(text: str) -> List[str]:
    return [
        x.strip()
        for x in str(text or "").split(",")
        if x.strip()
    ]


def parse_float(value) -> float:
    try:
        number = float(str(value).strip())
    except Exception:
        return float("nan")

    return number if math.isfinite(number) else float("nan")


def normalize_text(text: str) -> str:
    text = str(text or "")

    text = (
        text.replace("\r", " ")
        .replace("\n", " ")
        .replace("|", " ")
    )

    text = SPACES_RE.sub(" ", text).strip()

    # Remove quotation marks only at outer edges.
    text = QUOTE_EDGE_RE.sub("", text).strip()

    return text


def duplicate_key(text: str) -> str:
    """
    Normalize punctuation/spacing away when detecting duplicates.
    Do not normalize lexical content itself.
    """
    text = normalize_text(text)

    text = re.sub(
        r"""[、。，．！？!?…・･「」『』（）()［\]\[\]【】<>〈〉《》"'“”‘’\s]""",
        "",
        text,
    )

    return text


def text_stats(text: str) -> Tuple[int, int]:
    compact = re.sub(r"\s+", "", text)

    jp_count = len(JP_RE.findall(compact))

    return len(compact), jp_count


def is_noise_annotation(text: str) -> bool:
    """
    Reject things like:
      [音楽]
      （拍手）
      BGM
      ♪♪♪

    But keep valid sentences such as:
      音楽が好きです。
      拍手をもらいました。
    """
    return any(pattern.match(text) for pattern in NOISE_ANNOTATION_PATTERNS)


def has_meta_term(text: str, meta_terms: List[str]) -> bool:
    """
    This is intentionally conservative.
    Only strong channel/subtitle metadata phrases should be supplied.
    """
    return any(term in text for term in meta_terms)


def base_reasons(
    row: Dict[str, str],
    args,
    meta_terms: List[str],
) -> List[str]:

    reasons: List[str] = []

    text = normalize_text(row.get("text", ""))

    start = parse_float(row.get("start"))
    end = parse_float(row.get("end"))

    duration = (
        end - start
        if math.isfinite(start) and math.isfinite(end)
        else float("nan")
    )

    text_len, jp_count = text_stats(text)

    # ------------------------------------------------------------
    # Basic text checks
    # ------------------------------------------------------------

    if not text:
        reasons.append("empty_text")

    if text and ONLY_PUNCT_RE.match(text):
        reasons.append("punct_only")

    if URL_RE.search(text):
        reasons.append("url")

    if REPEAT_RE.search(text):
        reasons.append("char_repetition")

    if is_noise_annotation(text):
        reasons.append("noise_annotation")

    if args.drop_meta_terms and has_meta_term(text, meta_terms):
        reasons.append("meta_term")

    # ------------------------------------------------------------
    # Duration checks
    # ------------------------------------------------------------

    if not math.isfinite(duration) or duration <= 0:
        reasons.append("bad_duration")

    else:
        if duration < args.min_duration:
            reasons.append("too_short_duration")

        if duration > args.max_duration:
            reasons.append("too_long_duration")

        # Character-per-second is only meaningful when text exists.
        if text_len > 0:
            cps = text_len / duration

            if cps < args.min_chars_per_sec:
                reasons.append("too_few_chars_per_sec")

            if cps > args.max_chars_per_sec:
                reasons.append("too_many_chars_per_sec")

    # ------------------------------------------------------------
    # Text length/language checks
    # ------------------------------------------------------------

    if text_len < args.min_text_chars:
        reasons.append("too_short_text")

    if text_len > args.max_text_chars:
        reasons.append("too_long_text")

    if jp_count < args.min_japanese_chars:
        reasons.append("too_few_japanese_chars")

    return reasons


def append_example(
    container: Dict[str, List[Dict[str, str]]],
    reason: str,
    row: Dict[str, str],
    text: str,
    limit: int = 5,
):
    container.setdefault(reason, [])

    if len(container[reason]) >= limit:
        return

    container[reason].append(
        {
            "sample_id": str(row.get("sample_id", "")),
            "yid": str(row.get("yid", "")),
            "text": text,
        }
    )


def main():
    args = parse_args()

    rows = read_csv_rows(Path(args.in_csv))

    if not rows:
        raise RuntimeError(
            f"Empty manifest: {args.in_csv}"
        )

    meta_terms = split_terms(args.meta_terms)

    reason_counts: Counter[str] = Counter()

    dropped_examples: Dict[str, List[Dict[str, str]]] = {}

    candidates: List[Dict[str, str]] = []

    # ============================================================
    # Stage 1: base quality filtering
    # ============================================================

    for row in rows:
        text = normalize_text(row.get("text", ""))

        reasons = base_reasons(
            row,
            args,
            meta_terms,
        )

        if reasons:
            for reason in reasons:
                reason_counts[reason] += 1

                append_example(
                    dropped_examples,
                    reason,
                    row,
                    text,
                )

            continue

        out = dict(row)

        out["text"] = text
        out["_dup_key"] = duplicate_key(text)

        candidates.append(out)

    # ============================================================
    # Stage 2: duplicate filtering
    # ============================================================

    #
    # Level 1:
    # Same video + same normalized transcript.
    #
    # This is usually subtitle duplication / overlapping segments,
    # so we apply a strict cap.
    #
    same_video_seen: Counter[Tuple[str, str]] = Counter()

    after_video_dedup: List[Dict[str, str]] = []

    same_video_duplicate_examples: List[Dict[str, str]] = []

    for row in candidates:
        yid = str(row.get("yid", "")).strip()
        text_key = row["_dup_key"]

        video_key = (yid, text_key)

        same_video_seen[video_key] += 1

        if (
            same_video_seen[video_key]
            > args.max_same_video_duplicate
        ):
            reason_counts["same_video_duplicate"] += 1

            if len(same_video_duplicate_examples) < 20:
                same_video_duplicate_examples.append(
                    {
                        "sample_id": str(
                            row.get("sample_id", "")
                        ),
                        "yid": yid,
                        "text": row["text"],
                        "count": str(
                            same_video_seen[video_key]
                        ),
                        "cap": str(
                            args.max_same_video_duplicate
                        ),
                    }
                )

            continue

        after_video_dedup.append(row)

    #
    # Level 2:
    # Same transcript across the whole corpus.
    #
    # Different videos / signers saying the same sentence can be
    # useful for keypoint-to-text training, so this cap is loose.
    #
    global_text_seen: Counter[str] = Counter()

    kept: List[Dict[str, str]] = []

    global_duplicate_examples: List[Dict[str, str]] = []

    for row in after_video_dedup:
        key = row["_dup_key"]

        text_len, _ = text_stats(row["text"])

        if text_len <= args.short_text_chars:
            max_dup = args.max_short_duplicate_text
        else:
            max_dup = args.max_duplicate_text

        global_text_seen[key] += 1

        if global_text_seen[key] > max_dup:
            reason_counts["global_duplicate_text_cap"] += 1

            if len(global_duplicate_examples) < 20:
                global_duplicate_examples.append(
                    {
                        "sample_id": str(
                            row.get("sample_id", "")
                        ),
                        "yid": str(
                            row.get("yid", "")
                        ),
                        "text": row["text"],
                        "count": str(
                            global_text_seen[key]
                        ),
                        "cap": str(max_dup),
                    }
                )

            continue

        out = {
            k: v
            for k, v in row.items()
            if k != "_dup_key"
        }

        kept.append(out)

    # ============================================================
    # Sanity checks
    # ============================================================

    if not kept:
        raise RuntimeError(
            f"Cleaning dropped every row from {args.in_csv}; "
            f"reasons={dict(reason_counts)}"
        )

    # ============================================================
    # Write cleaned manifest
    # ============================================================

    fieldnames = list(rows[0].keys())

    for name in kept[0].keys():
        if name not in fieldnames:
            fieldnames.append(name)

    write_csv_rows(
        Path(args.out_csv),
        kept,
        fieldnames,
    )

    # ============================================================
    # Diagnostics
    # ============================================================

    top_texts = Counter(
        row["text"]
        for row in kept
    ).most_common(30)

    # Some useful descriptive statistics
    durations: List[float] = []
    text_lengths: List[int] = []
    cps_values: List[float] = []

    for row in kept:
        start = parse_float(row.get("start"))
        end = parse_float(row.get("end"))

        if not (
            math.isfinite(start)
            and math.isfinite(end)
        ):
            continue

        duration = end - start

        if duration <= 0:
            continue

        text_len, _ = text_stats(
            row.get("text", "")
        )

        durations.append(duration)
        text_lengths.append(text_len)

        if text_len > 0:
            cps_values.append(
                text_len / duration
            )

    def summarize(values: List[float]):
        if not values:
            return {}

        values = sorted(values)

        def percentile(p: float):
            if len(values) == 1:
                return values[0]

            idx = (len(values) - 1) * p

            lo = math.floor(idx)
            hi = math.ceil(idx)

            if lo == hi:
                return values[lo]

            frac = idx - lo

            return (
                values[lo] * (1 - frac)
                + values[hi] * frac
            )

        return {
            "min": values[0],
            "p01": percentile(0.01),
            "p05": percentile(0.05),
            "p50": percentile(0.50),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
            "max": values[-1],
            "mean": sum(values) / len(values),
        }

    report = {
        "in_csv": args.in_csv,
        "out_csv": args.out_csv,

        "input_rows": len(rows),

        "candidate_rows_after_base_filter": len(candidates),

        "rows_after_same_video_dedup": len(after_video_dedup),

        "kept_rows": len(kept),

        "dropped_rows": len(rows) - len(kept),

        "keep_ratio": len(kept) / len(rows),

        "reason_counts": dict(reason_counts),

        "statistics": {
            "duration_sec": summarize(durations),
            "text_chars": summarize(
                [float(x) for x in text_lengths]
            ),
            "chars_per_sec": summarize(cps_values),
        },

        "top_kept_texts": top_texts,

        "dropped_examples": dropped_examples,

        "same_video_duplicate_examples":
            same_video_duplicate_examples,

        "global_duplicate_examples":
            global_duplicate_examples,

        "settings": vars(args),
    }

    out_report = Path(args.report_json)

    out_report.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_report.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()