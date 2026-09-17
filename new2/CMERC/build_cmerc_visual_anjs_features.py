#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "new"))

from anjs_video_common import (  # noqa: E402
    ANJS_LABELS,
    FeatureCache,
    UtteranceItem,
    VideoEmbeddingEncoder,
    label_counts,
    make_speaker_onehot,
    read_bobsl_dialogues,
    read_ejsl_dialogues,
    read_meld_dialogues,
    resolve_path,
)


def zeros(n_items: int, dim: int) -> np.ndarray:
    return np.zeros((n_items, dim), dtype=np.float32)


def encode_dialogues(
    dialogues: Dict[str, List[UtteranceItem]],
    train_ids: Sequence[str],
    test_ids: Sequence[str],
    valid_ids: Sequence[str],
    encoder: VideoEmbeddingEncoder,
    text_dim: int,
    audio_dim: int,
    n_speakers: int,
    meta: Dict,
    out_pkl: Path,
) -> Dict:
    data = {
        "roberta1": {},
        "roberta2": {},
        "roberta3": {},
        "roberta4": {},
        "video_visual": {},
        "video_audio": {},
        "speakers": {},
        "labels": {},
        "sentences": {},
        "train_ids": list(train_ids),
        "test_ids": list(test_ids),
        "valid_ids": list(valid_ids),
        "meta": {
            **meta,
            "format": "CMERC_ANJS4_visual_only_dict",
            "label_space": "ANJS4",
            "label_names": ANJS_LABELS,
            "unknown_context_label": -100,
            "text_dim": text_dim,
            "audio_dim": audio_dim,
            "n_speakers": n_speakers,
            "feature_policy": "cmerc_visual_only_zero_text_audio",
        },
    }

    keys = list(train_ids) + list(test_ids) + list(valid_ids)
    total_items = sum(len(dialogues[key]) for key in keys)
    seen = 0
    status_counts: Dict[str, int] = {}

    for key in keys:
        items = dialogues[key]
        visual_vectors: List[np.ndarray] = []
        for item in items:
            vec, status = encoder.encode_item(item)
            visual_vectors.append(vec)
            status_counts[status] = status_counts.get(status, 0) + 1
            seen += 1
            if seen == 1 or seen % 250 == 0 or seen == total_items:
                print(f"[CMERC-FEATURES] video encode {seen}/{total_items} status={status_counts}")

        n_items = len(items)
        text_zero = zeros(n_items, text_dim)
        audio_zero = zeros(n_items, audio_dim)
        visual = np.stack(visual_vectors, axis=0).astype(np.float32) if visual_vectors else np.zeros((0, encoder.dim), dtype=np.float32)
        data["roberta1"][key] = text_zero.copy()
        data["roberta2"][key] = text_zero.copy()
        data["roberta3"][key] = text_zero.copy()
        data["roberta4"][key] = text_zero.copy()
        data["video_visual"][key] = visual
        data["video_audio"][key] = audio_zero
        data["speakers"][key] = make_speaker_onehot([item.speaker for item in items], n_speakers)
        data["labels"][key] = [int(item.label) for item in items]
        data["sentences"][key] = [item.text for item in items]

    data["meta"]["video_dim"] = encoder.dim
    data["meta"]["video_status"] = status_counts
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with out_pkl.open("wb") as f:
        pickle.dump(data, f)

    summary = {
        "out_pkl": str(out_pkl),
        "format": "CMERC_ANJS4_visual_only_dict",
        "label_space": "ANJS4",
        "label_names": ANJS_LABELS,
        "unknown_context_label": -100,
        "train_dialogues": len(train_ids),
        "test_dialogues": len(test_ids),
        "valid_dialogues": len(valid_ids),
        "utterances": total_items,
        "train_label_counts": label_counts(dialogues, train_ids),
        "test_label_counts": label_counts(dialogues, test_ids),
        "valid_label_counts": label_counts(dialogues, valid_ids),
        "dims": {
            "text": text_dim,
            "audio": audio_dim,
            "visual": encoder.dim,
            "n_speakers": n_speakers,
            "n_classes": len(ANJS_LABELS),
        },
        "video_status": status_counts,
        "meta": data["meta"],
    }
    out_pkl.with_suffix(out_pkl.suffix + ".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[CMERC-FEATURES] wrote {out_pkl}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CMERC ANJS4 visual-only feature pkls for BOBSL, MELD, and translated eJSL.")
    parser.add_argument("--bobsl_root", type=str, default="/raid_zoe/home/lr/wangyi/sign/bobsl")
    parser.add_argument("--bobsl_train_csv", type=str, default="")
    parser.add_argument("--bobsl_val_csv", type=str, default="")
    parser.add_argument("--bobsl_test_csv", type=str, default="")
    parser.add_argument("--bobsl_video_subdir", type=str, default="clip256")
    parser.add_argument("--bobsl_frame_subdir", type=str, default="frame")
    parser.add_argument("--bobsl_min_score", type=float, default=0.0)
    parser.add_argument("--limit_bobsl_train", type=int, default=0)
    parser.add_argument("--limit_bobsl_val", type=int, default=0)
    parser.add_argument("--limit_bobsl_test", type=int, default=0)
    parser.add_argument("--meld_root", type=str, default="/raid_zoe/home/lr/maokeyu/sign/emotionbaseline/dataset/MELD.Raw")
    parser.add_argument("--ejsl_txt_root", type=str, required=True)
    parser.add_argument("--ejsl_dial_list", type=str, default="/home/lr/wangyi/Sign/RO-MAN/eJSL_dial_dataset/ejsldial_filenames.csv")
    parser.add_argument("--ejsl_frame_root", type=str, default="/raid_zoe/home/lr/wangyi/sign/eJSL_dial/frame")
    parser.add_argument("--ejsl_mp4_root", type=str, default="/raid_zoe/home/lr/wangyi/sign/eJSL_dial/video")
    parser.add_argument("--limit_meld_dialogues", type=int, default=0)
    parser.add_argument("--limit_ejsl_dialogues", type=int, default=0)
    parser.add_argument("--keep_non_anjs_context", action="store_true", default=True)
    parser.add_argument("--drop_non_anjs_context", dest="keep_non_anjs_context", action="store_false")
    parser.add_argument("--out_meld_pkl", type=str, required=True)
    parser.add_argument("--out_ejsl_pkl", type=str, required=True)
    parser.add_argument("--out_bobsl_train_val_pkl", type=str, required=True)
    parser.add_argument("--out_bobsl_test_pkl", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--video_model", type=str, default="facebook/timesformer-base-finetuned-k400")
    parser.add_argument("--video_processor", type=str, default="MCG-NJU/videomae-base")
    parser.add_argument("--local_model_root", type=str, default="")
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--video_max_seconds", type=float, default=30.0)
    parser.add_argument("--text_dim", type=int, default=1024)
    parser.add_argument("--audio_dim", type=int, default=300)
    parser.add_argument("--n_speakers", type=int, default=9)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--skip_bobsl", action="store_true")
    parser.add_argument("--skip_meld", action="store_true")
    parser.add_argument("--skip_ejsl", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.skip_bobsl and args.skip_meld and args.skip_ejsl:
        raise RuntimeError("Nothing to build: all sources were skipped.")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    cache = FeatureCache(resolve_path(args.cache_dir, must_exist=False))
    encoder = VideoEmbeddingEncoder(
        video_model=args.video_model,
        video_processor=args.video_processor,
        cache=cache,
        device=device,
        local_model_root=args.local_model_root,
        num_frames=args.num_frames,
        max_seconds=args.video_max_seconds,
        fp16=args.fp16,
    )
    print(f"[CMERC-FEATURES] device={device}")
    print(f"[CMERC-FEATURES] cache_dir={cache.root}")
    print(f"[CMERC-FEATURES] video_model={encoder.video_model}")
    print(f"[CMERC-FEATURES] video_processor={encoder.video_processor}")

    if not args.skip_meld:
        meld_dialogues, meld_train_ids, meld_test_ids, meld_meta = read_meld_dialogues(
            args.meld_root,
            keep_non_anjs_context=args.keep_non_anjs_context,
            limit_meld_dialogues=args.limit_meld_dialogues,
        )
        encode_dialogues(
            meld_dialogues,
            meld_train_ids,
            meld_test_ids,
            [],
            encoder,
            args.text_dim,
            args.audio_dim,
            args.n_speakers,
            {**meld_meta, "dataset": "MELD"},
            resolve_path(args.out_meld_pkl, must_exist=False),
        )

    if not args.skip_ejsl:
        ejsl_dialogues, _, ejsl_test_ids, ejsl_meta = read_ejsl_dialogues(
            args.ejsl_txt_root,
            args.ejsl_dial_list,
            args.ejsl_frame_root,
            args.ejsl_mp4_root,
            limit_ejsl_dialogues=args.limit_ejsl_dialogues,
        )
        encode_dialogues(
            ejsl_dialogues,
            [],
            ejsl_test_ids,
            [],
            encoder,
            args.text_dim,
            args.audio_dim,
            args.n_speakers,
            {**ejsl_meta, "dataset": "eJSL"},
            resolve_path(args.out_ejsl_pkl, must_exist=False),
        )

    if not args.skip_bobsl:
        bobsl_dialogues, bobsl_train_ids, bobsl_val_ids, bobsl_test_ids, bobsl_meta = read_bobsl_dialogues(
            args.bobsl_root,
            train_csv=args.bobsl_train_csv,
            val_csv=args.bobsl_val_csv,
            test_csv=args.bobsl_test_csv,
            video_subdir=args.bobsl_video_subdir,
            frame_subdir=args.bobsl_frame_subdir,
            min_score=args.bobsl_min_score,
            limit_train=args.limit_bobsl_train,
            limit_val=args.limit_bobsl_val,
            limit_test=args.limit_bobsl_test,
        )
        encode_dialogues(
            bobsl_dialogues,
            bobsl_train_ids,
            bobsl_val_ids,
            [],
            encoder,
            args.text_dim,
            args.audio_dim,
            args.n_speakers,
            {**bobsl_meta, "dataset": "BOBSL", "pkl_role": "train_val"},
            resolve_path(args.out_bobsl_train_val_pkl, must_exist=False),
        )
        encode_dialogues(
            bobsl_dialogues,
            [],
            bobsl_test_ids,
            [],
            encoder,
            args.text_dim,
            args.audio_dim,
            args.n_speakers,
            {**bobsl_meta, "dataset": "BOBSL", "pkl_role": "test"},
            resolve_path(args.out_bobsl_test_pkl, must_exist=False),
        )


if __name__ == "__main__":
    main()
