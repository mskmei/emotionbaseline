#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from manifest_utils import parse_ejsl_sample_id, read_ejsl_names, read_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate generated eJSL non-oracle text against oracle Japanese text.")
    parser.add_argument("--predictions_jsonl", type=str, required=True)
    parser.add_argument(
        "--dial_list",
        type=str,
        default="/home/lr/wangyi/Sign/RO-MAN/eJSL_dial_dataset/ejsldial_filenames.csv",
    )
    parser.add_argument(
        "--structure_txt_root",
        type=str,
        default="/raid_elmo/home/lr/wangyi/PTR/STUDIES-Japanese/Short_dialogue",
    )
    parser.add_argument("--out_json", type=str, required=True)
    parser.add_argument("--out_csv", type=str, default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--sample_id_csv",
        type=str,
        default="",
        help="Optional CSV with a sample_id column; when set, evaluate exactly these samples.",
    )
    return parser.parse_args()


def compact(text: str) -> str:
    return "".join(str(text or "").replace("|", " ").split())


def char_tokens(text: str) -> List[str]:
    return list(compact(text))


def ngrams(tokens: List[str], n: int) -> Counter[Tuple[str, ...]]:
    return Counter(tuple(tokens[i : i + n]) for i in range(0, max(0, len(tokens) - n + 1)))


def f1_from_precision_recall(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def sentence_bleu_char(pred: str, ref: str, max_order: int = 4, smooth: float = 1.0) -> float:
    pred_toks = char_tokens(pred)
    ref_toks = char_tokens(ref)
    if not pred_toks or not ref_toks:
        return 0.0
    precisions = []
    for n in range(1, max_order + 1):
        pred_ng = ngrams(pred_toks, n)
        ref_ng = ngrams(ref_toks, n)
        overlap = sum(min(count, ref_ng.get(ng, 0)) for ng, count in pred_ng.items())
        total = sum(pred_ng.values())
        precisions.append((overlap + smooth) / (total + smooth))
    geo = math.exp(sum(math.log(p) for p in precisions) / max_order)
    ref_len = len(ref_toks)
    pred_len = len(pred_toks)
    bp = 1.0 if pred_len > ref_len else math.exp(1.0 - ref_len / max(pred_len, 1))
    return bp * geo


def corpus_bleu_char(pairs: Iterable[Tuple[str, str]], max_order: int = 4, smooth: float = 1.0) -> float:
    pairs = list(pairs)
    pred_len = 0
    ref_len = 0
    precisions = []
    for n in range(1, max_order + 1):
        overlap_total = 0.0
        pred_total = 0.0
        for pred, ref in pairs:
            pred_toks = char_tokens(pred)
            ref_toks = char_tokens(ref)
            if n == 1:
                pred_len += len(pred_toks)
                ref_len += len(ref_toks)
            pred_ng = ngrams(pred_toks, n)
            ref_ng = ngrams(ref_toks, n)
            overlap_total += sum(min(count, ref_ng.get(ng, 0)) for ng, count in pred_ng.items())
            pred_total += sum(pred_ng.values())
        precisions.append((overlap_total + smooth) / (pred_total + smooth))
    geo = math.exp(sum(math.log(p) for p in precisions) / max_order)
    bp = 1.0 if pred_len > ref_len else math.exp(1.0 - ref_len / max(pred_len, 1))
    return bp * geo


def rouge_n_f1_char(pred: str, ref: str, n: int) -> float:
    pred_ng = ngrams(char_tokens(pred), n)
    ref_ng = ngrams(char_tokens(ref), n)
    if not pred_ng or not ref_ng:
        return 0.0
    overlap = sum(min(count, ref_ng.get(ng, 0)) for ng, count in pred_ng.items())
    precision = overlap / max(sum(pred_ng.values()), 1)
    recall = overlap / max(sum(ref_ng.values()), 1)
    return f1_from_precision_recall(precision, recall)


def lcs_length(xs: List[str], ys: List[str]) -> int:
    if not xs or not ys:
        return 0
    prev = [0] * (len(ys) + 1)
    for x in xs:
        cur = [0]
        for j, y in enumerate(ys, start=1):
            if x == y:
                cur.append(prev[j - 1] + 1)
            else:
                cur.append(max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_l_f1_char(pred: str, ref: str) -> float:
    pred_toks = char_tokens(pred)
    ref_toks = char_tokens(ref)
    if not pred_toks or not ref_toks:
        return 0.0
    overlap = lcs_length(pred_toks, ref_toks)
    precision = overlap / max(len(pred_toks), 1)
    recall = overlap / max(len(ref_toks), 1)
    return f1_from_precision_recall(precision, recall)


def char_f1(pred: str, ref: str) -> float:
    pred_counts = Counter(char_tokens(pred))
    ref_counts = Counter(char_tokens(ref))
    if not pred_counts or not ref_counts:
        return 0.0
    overlap = sum(min(count, ref_counts.get(ch, 0)) for ch, count in pred_counts.items())
    precision = overlap / max(sum(pred_counts.values()), 1)
    recall = overlap / max(sum(ref_counts.values()), 1)
    return f1_from_precision_recall(precision, recall)


def edit_similarity(pred: str, ref: str) -> float:
    return SequenceMatcher(None, compact(pred), compact(ref)).ratio()


def load_predictions(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for row in read_jsonl(path):
        sample_id = str(row.get("sample_id", "")).strip()
        text = str(row.get("text", "")).strip()
        if sample_id and text:
            out[sample_id] = text
    return out


def load_sample_ids_from_csv(path: Path) -> List[str]:
    rows: List[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "sample_id" not in reader.fieldnames:
            raise RuntimeError(f"{path} must contain a sample_id column")
        for row in reader:
            sample_id = str(row.get("sample_id", "")).strip()
            if sample_id:
                rows.append(sample_id)
    return rows


def gt_text(structure_txt_root: Path, sample_id: str) -> Tuple[str, str, str]:
    sd_id, dialogue_idx, utterance_idx, label = parse_ejsl_sample_id(sample_id)
    txt_file = structure_txt_root / sd_id / "txt" / f"{sd_id}-Dialogue-{dialogue_idx:02d}.txt"
    lines = [line.strip() for line in txt_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if utterance_idx < 1 or utterance_idx > len(lines):
        raise RuntimeError(f"{sample_id}: turn {utterance_idx} outside {txt_file}")
    parts = lines[utterance_idx - 1].split("|", 2)
    if len(parts) < 3:
        raise RuntimeError(f"{sample_id}: expected speaker|emotion|text at {txt_file}:{utterance_idx}")
    return parts[2].strip(), parts[1].strip(), label


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "label",
        "emotion",
        "gt",
        "pred",
        "bleu1_char",
        "bleu2_char",
        "bleu4_char",
        "rouge1_f1_char",
        "rouge2_f1_char",
        "rougeL_f1_char",
        "char_f1",
        "edit_similarity",
        "gt_len",
        "pred_len",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main():
    args = parse_args()
    if args.sample_id_csv:
        sample_ids = load_sample_ids_from_csv(Path(args.sample_id_csv))
    else:
        sample_ids = read_ejsl_names(Path(args.dial_list))
    if args.limit > 0 and not args.sample_id_csv:
        sample_ids = sample_ids[: args.limit]
    preds = load_predictions(Path(args.predictions_jsonl))
    rows: List[Dict[str, object]] = []
    pairs: List[Tuple[str, str]] = []
    missing = []

    for sample_id in sample_ids:
        pred = preds.get(sample_id)
        if pred is None:
            missing.append(sample_id)
            continue
        ref, emotion, label = gt_text(Path(args.structure_txt_root), sample_id)
        pairs.append((pred, ref))
        rows.append(
            {
                "sample_id": sample_id,
                "label": label,
                "emotion": emotion,
                "gt": ref,
                "pred": pred,
                "bleu1_char": sentence_bleu_char(pred, ref, max_order=1),
                "bleu2_char": sentence_bleu_char(pred, ref, max_order=2),
                "bleu4_char": sentence_bleu_char(pred, ref),
                "rouge1_f1_char": rouge_n_f1_char(pred, ref, n=1),
                "rouge2_f1_char": rouge_n_f1_char(pred, ref, n=2),
                "rougeL_f1_char": rouge_l_f1_char(pred, ref),
                "char_f1": char_f1(pred, ref),
                "edit_similarity": edit_similarity(pred, ref),
                "gt_len": len(char_tokens(ref)),
                "pred_len": len(char_tokens(pred)),
            }
        )

    if not rows:
        raise RuntimeError(f"No comparable predictions found in {args.predictions_jsonl}")

    metrics = {
        "predictions_jsonl": args.predictions_jsonl,
        "dial_list": args.dial_list,
        "structure_txt_root": args.structure_txt_root,
        "n_expected": len(sample_ids),
        "n_scored": len(rows),
        "n_missing": len(missing),
        "missing_first": missing[:20],
        "corpus_bleu1_char": corpus_bleu_char(pairs, max_order=1),
        "corpus_bleu2_char": corpus_bleu_char(pairs, max_order=2),
        "corpus_bleu4_char": corpus_bleu_char(pairs),
        "mean_sentence_bleu1_char": sum(float(r["bleu1_char"]) for r in rows) / len(rows),
        "mean_sentence_bleu2_char": sum(float(r["bleu2_char"]) for r in rows) / len(rows),
        "mean_sentence_bleu4_char": sum(float(r["bleu4_char"]) for r in rows) / len(rows),
        "mean_rouge1_f1_char": sum(float(r["rouge1_f1_char"]) for r in rows) / len(rows),
        "mean_rouge2_f1_char": sum(float(r["rouge2_f1_char"]) for r in rows) / len(rows),
        "mean_rougeL_f1_char": sum(float(r["rougeL_f1_char"]) for r in rows) / len(rows),
        "mean_char_f1": sum(float(r["char_f1"]) for r in rows) / len(rows),
        "mean_edit_similarity": sum(float(r["edit_similarity"]) for r in rows) / len(rows),
        "mean_gt_len": sum(int(r["gt_len"]) for r in rows) / len(rows),
        "mean_pred_len": sum(int(r["pred_len"]) for r in rows) / len(rows),
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.out_csv:
        write_csv(Path(args.out_csv), rows)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
