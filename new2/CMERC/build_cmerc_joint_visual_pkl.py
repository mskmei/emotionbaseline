#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import pickle
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "new"))

from anjs_video_common import ANJS_LABELS, resolve_path  # noqa: E402


DICT_FIELDS = ["roberta1", "roberta2", "roberta3", "roberta4", "video_visual", "video_audio", "speakers", "labels", "sentences"]


def load_payload(path: Path) -> Dict:
    obj = pickle.load(path.open("rb"), encoding="latin1")
    if not isinstance(obj, dict):
        raise ValueError(f"Expected CMERC visual dict: {path}")
    missing = [field for field in DICT_FIELDS + ["train_ids", "test_ids"] if field not in obj]
    if missing:
        raise ValueError(f"Missing fields in {path}: {missing}")
    obj.setdefault("valid_ids", [])
    obj.setdefault("meta", {})
    return obj


def payload_dims(payload: Dict) -> Dict[str, int]:
    keys = list(payload["train_ids"]) + list(payload["test_ids"]) + list(payload["valid_ids"])
    if not keys:
        raise ValueError("Feature pkl has no dialogue keys.")
    first = keys[0]
    return {
        "text_dim": int(np.asarray(payload["roberta1"][first]).shape[-1]),
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


def label_counts(labels: Dict[str, Sequence[int]], keys: Iterable[str]) -> Dict[str, int]:
    flat: List[int] = []
    for key in keys:
        flat.extend(int(x) for x in labels[key] if int(x) >= 0)
    counts = np.bincount(np.asarray(flat, dtype=np.int64), minlength=len(ANJS_LABELS)).astype(int).tolist() if flat else [0] * len(ANJS_LABELS)
    return dict(zip(ANJS_LABELS, counts))


def add_key(source: Dict, source_key: str, target: Dict, target_key: str) -> None:
    for field in DICT_FIELDS:
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

    target = {field: {} for field in DICT_FIELDS}
    train_ids: List[str] = []
    test_ids: List[str] = []
    valid_ids: List[str] = []

    for key in meld["train_ids"]:
        add_key(meld, key, target, key)
        train_ids.append(str(key))
    for key in meld["test_ids"]:
        add_key(meld, key, target, key)
        test_ids.append(str(key))

    bobsl_keys = choose_bobsl_keys(
        bobsl,
        include_val=args.include_bobsl_val,
        max_count=args.max_bobsl_train_dialogues,
        seed=args.bobsl_sample_seed,
    )
    for key in bobsl_keys:
        target_key = f"joint_{key}"
        suffix = 2
        while target_key in target["labels"]:
            target_key = f"joint_{key}_{suffix}"
            suffix += 1
        add_key(bobsl, key, target, target_key)
        train_ids.append(target_key)

    payload = {
        **target,
        "train_ids": train_ids,
        "test_ids": test_ids,
        "valid_ids": valid_ids,
        "meta": {
            "format": "CMERC_ANJS4_visual_only_joint_dict",
            "meld_pkl": str(meld_path),
            "bobsl_train_val_pkl": str(bobsl_path),
            "label_space": "ANJS4",
            "label_names": ANJS_LABELS,
            "unknown_context_label": -100,
            "dims": dims,
            "source_test_meaning": "MELD test only",
            "joint_train_policy": {
                "bobsl_include_val": bool(args.include_bobsl_val),
                "bobsl_requested_max_train_dialogues": int(args.max_bobsl_train_dialogues),
                "bobsl_selected_source_dialogues": len(bobsl_keys),
                "bobsl_sample_seed": int(args.bobsl_sample_seed),
                "bobsl_context_policy": "context1_as_prebuilt_single_utterance_dialogues",
            },
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(payload, f)

    bobsl_train_keys = [key for key in train_ids if str(key).startswith("joint_bobsl")]
    meld_train_keys = [key for key in train_ids if not str(key).startswith("joint_bobsl")]
    summary = {
        "out_pkl": str(out_path),
        "format": payload["meta"]["format"],
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
        "train_label_counts_all": label_counts(target["labels"], train_ids),
        "train_label_counts_meld": label_counts(target["labels"], meld_train_keys),
        "train_label_counts_bobsl": label_counts(target["labels"], bobsl_train_keys),
        "source_test_label_counts_meld": label_counts(target["labels"], test_ids),
        "joint_train_policy": payload["meta"]["joint_train_policy"],
    }
    out_path.with_suffix(out_path.suffix + ".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[CMERC-JOINT] wrote {out_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CMERC visual-only joint pkl: MELD train/dev + sampled BOBSL, MELD test as source.")
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
