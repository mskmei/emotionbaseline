#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import pickle
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


LABELS = ["A", "N", "J", "S"]


def resolve_path(raw: str, must_exist: bool = False) -> Path:
    path = Path(raw).expanduser()
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, Path(__file__).resolve().parent.parent / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    if must_exist:
        raise FileNotFoundError(f"Path not found: {raw}. Checked: {', '.join(str(x) for x in candidates)}")
    return candidates[0].resolve()


def load_payload(path: Path):
    obj = pickle.load(path.open("rb"), encoding="latin1")
    if not isinstance(obj, (tuple, list)) or len(obj) < 10:
        raise ValueError(f"Expected unified tuple pkl with >=10 fields: {path}")
    return obj


def payload_dims(obj) -> Dict[str, int]:
    meta = obj[9] if isinstance(obj[9], dict) else {}
    keys = list(obj[7]) + list(obj[8])
    if not keys:
        raise ValueError("Unified pkl has no dialogue keys.")
    first = keys[0]
    return {
        "text_dim": int(meta.get("text_dim") or np.asarray(obj[3][first]).shape[-1]),
        "audio_dim": int(meta.get("audio_dim") or np.asarray(obj[4][first]).shape[-1]),
        "visual_dim": int(meta.get("visual_dim") or np.asarray(obj[5][first]).shape[-1]),
        "n_speakers": int(meta.get("n_speakers") or np.asarray(obj[1][first]).shape[-1]),
        "n_classes": int(meta.get("n_classes") or len(meta.get("label_names", LABELS))),
    }


def check_compatible(meld_obj, bobsl_obj) -> Dict[str, int]:
    meld_dims = payload_dims(meld_obj)
    bobsl_dims = payload_dims(bobsl_obj)
    for key in ["text_dim", "audio_dim", "visual_dim", "n_classes"]:
        if int(meld_dims[key]) != int(bobsl_dims[key]):
            raise RuntimeError(f"Feature mismatch for {key}: MELD={meld_dims[key]} BOBSL={bobsl_dims[key]}")
    if int(meld_dims["n_speakers"]) != int(bobsl_dims["n_speakers"]):
        raise RuntimeError(
            f"Speaker dim mismatch: MELD={meld_dims['n_speakers']} BOBSL={bobsl_dims['n_speakers']}. "
            "Rebuild both pkl files with the same --n_speakers."
        )
    return meld_dims


def label_counts(video_labels: Dict[str, Sequence[int]], keys: Iterable[str]) -> Dict[str, int]:
    labels: List[int] = []
    for key in keys:
        labels.extend(int(x) for x in video_labels[key] if int(x) >= 0)
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=len(LABELS)).astype(int).tolist() if labels else [0] * len(LABELS)
    return dict(zip(LABELS, counts))


def choose_bobsl_keys(
    bobsl_obj,
    include_val: bool,
    max_bobsl_train_dialogues: int,
    seed: int,
) -> List[str]:
    keys = list(bobsl_obj[7])
    if include_val:
        keys.extend(list(bobsl_obj[8]))
    rng = random.Random(seed)
    rng.shuffle(keys)
    if max_bobsl_train_dialogues > 0:
        keys = keys[:max_bobsl_train_dialogues]
    return keys


def add_key(
    source_obj,
    source_key: str,
    target: Tuple[Dict, Dict, Dict, Dict, Dict, Dict, Dict],
    target_key: str,
    item_slice: slice | None = None,
) -> None:
    video_ids, video_speakers, video_labels, video_text, video_audio, video_visual, video_sentence = target
    sl = item_slice if item_slice is not None else slice(None)
    video_ids[target_key] = target_key
    video_speakers[target_key] = np.asarray(source_obj[1][source_key], dtype=np.float32)[sl].copy()
    video_labels[target_key] = [int(x) for x in list(source_obj[2][source_key])[sl]]
    video_text[target_key] = np.asarray(source_obj[3][source_key], dtype=np.float32)[sl].copy()
    video_audio[target_key] = np.asarray(source_obj[4][source_key], dtype=np.float32)[sl].copy()
    video_visual[target_key] = np.asarray(source_obj[5][source_key], dtype=np.float32)[sl].copy()
    sentences = list(source_obj[6].get(source_key, [""] * len(source_obj[2][source_key])))
    video_sentence[target_key] = [str(x) for x in sentences[sl]]


def add_bobsl_context1(
    bobsl_obj,
    source_key: str,
    target: Tuple[Dict, Dict, Dict, Dict, Dict, Dict, Dict],
    train_vid: List[str],
) -> None:
    labels = list(bobsl_obj[2][source_key])
    if len(labels) <= 1:
        target_key = f"joint_{source_key}"
        add_key(bobsl_obj, source_key, target, target_key)
        train_vid.append(target_key)
        return
    for idx in range(len(labels)):
        target_key = f"joint_{source_key}_utt{idx + 1:02d}"
        add_key(bobsl_obj, source_key, target, target_key, slice(idx, idx + 1))
        train_vid.append(target_key)


def build_joint(args: argparse.Namespace) -> Dict:
    meld_path = resolve_path(args.meld_pkl, must_exist=True)
    bobsl_path = resolve_path(args.bobsl_train_val_pkl, must_exist=True)
    out_path = resolve_path(args.out_pkl, must_exist=False)
    meld_obj = load_payload(meld_path)
    bobsl_obj = load_payload(bobsl_path)
    dims = check_compatible(meld_obj, bobsl_obj)

    video_ids: Dict = {}
    video_speakers: Dict = {}
    video_labels: Dict = {}
    video_text: Dict = {}
    video_audio: Dict = {}
    video_visual: Dict = {}
    video_sentence: Dict = {}
    target = (video_ids, video_speakers, video_labels, video_text, video_audio, video_visual, video_sentence)

    meld_train_vid = [str(x) for x in list(meld_obj[7])]
    meld_test_vid = [str(x) for x in list(meld_obj[8])]
    train_vid: List[str] = []
    test_vid: List[str] = []

    for key in meld_train_vid:
        target_key = f"joint_{key}" if key in video_ids else key
        add_key(meld_obj, key, target, target_key)
        train_vid.append(target_key)
    for key in meld_test_vid:
        target_key = f"joint_{key}" if key in video_ids else key
        add_key(meld_obj, key, target, target_key)
        test_vid.append(target_key)

    bobsl_keys = choose_bobsl_keys(
        bobsl_obj,
        include_val=args.include_bobsl_val,
        max_bobsl_train_dialogues=args.max_bobsl_train_dialogues,
        seed=args.bobsl_sample_seed,
    )
    for key in bobsl_keys:
        if args.force_bobsl_context1:
            add_bobsl_context1(bobsl_obj, key, target, train_vid)
        else:
            target_key = f"joint_{key}"
            add_key(bobsl_obj, key, target, target_key)
            train_vid.append(target_key)

    meta = {
        **copy.deepcopy(meld_obj[9] if isinstance(meld_obj[9], dict) else {}),
        "dataset": "MELD+BOBSL",
        "label_space": "ANJS4",
        "label_names": LABELS,
        "n_classes": len(LABELS),
        "text_dim": dims["text_dim"],
        "audio_dim": dims["audio_dim"],
        "visual_dim": dims["visual_dim"],
        "n_speakers": dims["n_speakers"],
        "source_test_meaning": "MELD test only",
        "joint_train_policy": {
            "meld_train_dialogues": len(meld_train_vid),
            "meld_test_dialogues": len(meld_test_vid),
            "bobsl_source_pkl": str(bobsl_path),
            "bobsl_include_val": bool(args.include_bobsl_val),
            "bobsl_requested_max_train_dialogues": int(args.max_bobsl_train_dialogues),
            "bobsl_selected_source_dialogues": len(bobsl_keys),
            "bobsl_force_context1": bool(args.force_bobsl_context1),
            "bobsl_sample_seed": int(args.bobsl_sample_seed),
        },
    }

    payload = (
        video_ids,
        video_speakers,
        video_labels,
        video_text,
        video_audio,
        video_visual,
        video_sentence,
        train_vid,
        test_vid,
        meta,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(payload, f)

    bobsl_train_keys = [key for key in train_vid if key.startswith("joint_bobsl")]
    meld_train_keys = [key for key in train_vid if not key.startswith("joint_bobsl")]
    summary = {
        "out_pkl": str(out_path),
        "meld_pkl": str(meld_path),
        "bobsl_train_val_pkl": str(bobsl_path),
        "train_dialogues": len(train_vid),
        "test_dialogues": len(test_vid),
        "meld_train_dialogues": len(meld_train_keys),
        "bobsl_train_dialogues": len(bobsl_train_keys),
        "source_test_dialogues": len(test_vid),
        "train_label_counts_all": label_counts(video_labels, train_vid),
        "train_label_counts_meld": label_counts(video_labels, meld_train_keys),
        "train_label_counts_bobsl": label_counts(video_labels, bobsl_train_keys),
        "source_test_label_counts_meld": label_counts(video_labels, test_vid),
        "meta": meta,
    }
    out_path.with_suffix(out_path.suffix + ".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[joint-pkl] wrote {out_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a video-only joint-train pkl: MELD train/dev + BOBSL context=1, MELD test as source_test.")
    parser.add_argument("--meld_pkl", type=str, required=True)
    parser.add_argument("--bobsl_train_val_pkl", type=str, required=True)
    parser.add_argument("--out_pkl", type=str, required=True)
    parser.add_argument("--max_bobsl_train_dialogues", type=int, default=5000)
    parser.add_argument("--bobsl_sample_seed", type=int, default=123)
    parser.add_argument("--include_bobsl_val", action="store_true")
    parser.add_argument("--force_bobsl_context1", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    build_joint(parse_args())


if __name__ == "__main__":
    main()
