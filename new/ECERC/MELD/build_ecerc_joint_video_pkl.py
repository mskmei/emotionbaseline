#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import pickle
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
NEW_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(NEW_ROOT))

from anjs_video_common import ANJS_LABELS, resolve_path  # noqa: E402


ECERC_FIELDS = [
    "speakers",
    "emotion_labels",
    "sentiment_labels",
    "eroberta1",
    "eroberta2",
    "eroberta3",
    "eroberta4",
    "sroberta1",
    "sroberta2",
    "sroberta3",
    "sroberta4",
    "video_audio",
    "video_visual",
    "sentences",
    "train_ids",
    "test_ids",
    "valid_ids",
]


def load_payload(path: Path) -> Dict:
    obj = pickle.load(path.open("rb"), encoding="latin1")
    if not isinstance(obj, (tuple, list)) or len(obj) != len(ECERC_FIELDS):
        raise ValueError(f"Expected ECERC feature tuple with {len(ECERC_FIELDS)} fields: {path}")
    return dict(zip(ECERC_FIELDS, obj))


def payload_dims(payload: Dict) -> Dict[str, int]:
    keys = list(payload["train_ids"]) + list(payload["test_ids"]) + list(payload["valid_ids"])
    if not keys:
        raise ValueError("Feature pkl has no dialogue keys.")
    first = keys[0]
    return {
        "text_dim": int(np.asarray(payload["eroberta1"][first]).shape[-1]),
        "audio_dim": int(np.asarray(payload["video_audio"][first]).shape[-1]),
        "visual_dim": int(np.asarray(payload["video_visual"][first]).shape[-1]),
        "n_speakers": int(np.asarray(payload["speakers"][first]).shape[-1]),
        "n_classes": len(ANJS_LABELS),
    }


def check_compatible(meld: Dict, bobsl: Dict) -> Dict[str, int]:
    meld_dims = payload_dims(meld)
    bobsl_dims = payload_dims(bobsl)
    for key in ["text_dim", "audio_dim", "visual_dim", "n_speakers"]:
        if int(meld_dims[key]) != int(bobsl_dims[key]):
            raise RuntimeError(f"Feature mismatch for {key}: MELD={meld_dims[key]} BOBSL={bobsl_dims[key]}")
    return meld_dims


def choose_bobsl_keys(payload: Dict, include_val: bool, max_count: int, seed: int) -> List[str]:
    keys = list(payload["train_ids"])
    if include_val:
        keys.extend(list(payload["test_ids"]))
    rng = random.Random(seed)
    rng.shuffle(keys)
    if max_count > 0:
        keys = keys[:max_count]
    return keys


def label_counts(emotion_labels: Dict[str, Sequence[int]], keys: Iterable[str]) -> Dict[str, int]:
    labels: List[int] = []
    for key in keys:
        labels.extend(int(x) for x in emotion_labels[key] if int(x) >= 0)
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=len(ANJS_LABELS)).astype(int).tolist() if labels else [0] * len(ANJS_LABELS)
    return dict(zip(ANJS_LABELS, counts))


def add_key(source: Dict, source_key: str, target: Dict, target_key: str) -> None:
    for field in ECERC_FIELDS[:14]:
        value = source[field][source_key]
        if isinstance(value, np.ndarray):
            target[field][target_key] = value.copy()
        elif isinstance(value, list):
            target[field][target_key] = list(value)
        else:
            target[field][target_key] = copy.deepcopy(value)


def build_joint(args: argparse.Namespace) -> Dict:
    meld_path = resolve_path(args.meld_pkl, must_exist=True)
    bobsl_path = resolve_path(args.bobsl_train_val_pkl, must_exist=True)
    out_path = resolve_path(args.out_pkl, must_exist=False)
    meld = load_payload(meld_path)
    bobsl = load_payload(bobsl_path)
    dims = check_compatible(meld, bobsl)

    target = {field: {} for field in ECERC_FIELDS[:14]}
    train_ids: List[str] = []
    test_ids: List[str] = []
    valid_ids: List[str] = []

    for key in meld["train_ids"]:
        target_key = str(key)
        add_key(meld, key, target, target_key)
        train_ids.append(target_key)
    for key in meld["test_ids"]:
        target_key = str(key)
        add_key(meld, key, target, target_key)
        test_ids.append(target_key)

    bobsl_keys = choose_bobsl_keys(
        bobsl,
        include_val=args.include_bobsl_val,
        max_count=args.max_bobsl_train_dialogues,
        seed=args.bobsl_sample_seed,
    )
    for key in bobsl_keys:
        target_key = f"joint_{key}"
        suffix = 2
        while target_key in target["emotion_labels"]:
            target_key = f"joint_{key}_{suffix}"
            suffix += 1
        add_key(bobsl, key, target, target_key)
        train_ids.append(target_key)

    payload = tuple(target[field] for field in ECERC_FIELDS[:14]) + (train_ids, test_ids, valid_ids)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(payload, f)

    bobsl_train_keys = [key for key in train_ids if str(key).startswith("joint_bobsl")]
    meld_train_keys = [key for key in train_ids if not str(key).startswith("joint_bobsl")]
    summary = {
        "out_pkl": str(out_path),
        "format": "ECERC_MELD_roberta_tuple_ANJS4_video_only_joint",
        "meld_pkl": str(meld_path),
        "bobsl_train_val_pkl": str(bobsl_path),
        "label_space": "ANJS4",
        "label_names": ANJS_LABELS,
        "unknown_context_label": -100,
        "dims": dims,
        "train_dialogues": len(train_ids),
        "test_dialogues": len(test_ids),
        "valid_dialogues": len(valid_ids),
        "meld_train_dialogues": len(meld_train_keys),
        "bobsl_train_dialogues": len(bobsl_train_keys),
        "source_test_meaning": "MELD test only",
        "train_label_counts_all": label_counts(target["emotion_labels"], train_ids),
        "train_label_counts_meld": label_counts(target["emotion_labels"], meld_train_keys),
        "train_label_counts_bobsl": label_counts(target["emotion_labels"], bobsl_train_keys),
        "source_test_label_counts_meld": label_counts(target["emotion_labels"], test_ids),
        "joint_train_policy": {
            "bobsl_include_val": bool(args.include_bobsl_val),
            "bobsl_requested_max_train_dialogues": int(args.max_bobsl_train_dialogues),
            "bobsl_selected_source_dialogues": len(bobsl_keys),
            "bobsl_sample_seed": int(args.bobsl_sample_seed),
            "bobsl_context_policy": "context1_as_prebuilt_single_utterance_dialogues",
        },
    }
    out_path.with_suffix(out_path.suffix + ".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[ECERC-JOINT] wrote {out_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ECERC ANJS4 video-only joint pkl: MELD train/dev + sampled BOBSL, MELD test as source.")
    parser.add_argument("--meld_pkl", type=str, required=True)
    parser.add_argument("--bobsl_train_val_pkl", type=str, required=True)
    parser.add_argument("--out_pkl", type=str, required=True)
    parser.add_argument("--max_bobsl_train_dialogues", type=int, default=5000)
    parser.add_argument("--bobsl_sample_seed", type=int, default=123)
    parser.add_argument("--include_bobsl_val", action="store_true")
    return parser.parse_args()


def main() -> None:
    build_joint(parse_args())


if __name__ == "__main__":
    main()

