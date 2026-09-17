#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import pickle
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "new"))

from anjs_video_common import ANJS_LABELS, resolve_path  # noqa: E402


def load_payload(path: Path) -> Dict:
    obj = pickle.load(path.open("rb"), encoding="latin1")
    if not isinstance(obj, dict):
        raise ValueError(f"Expected ConxGNN visual dict: {path}")
    for key in ["train", "test"]:
        if key not in obj:
            raise ValueError(f"Missing split {key} in {path}")
    obj.setdefault("dev", [])
    obj.setdefault("meta", {})
    return obj


def sample_ids(samples: List[Dict]) -> List[str]:
    return [str(sample.get("id", f"sample_{idx}")) for idx, sample in enumerate(samples)]


def dims(payload: Dict) -> Dict[str, int]:
    samples = list(payload.get("train", [])) + list(payload.get("test", [])) + list(payload.get("dev", []))
    if not samples:
        raise ValueError("Feature pkl has no samples.")
    sample = samples[0]
    return {
        "text_dim": int(np.asarray(sample["text"]).shape[-1]),
        "audio_dim": int(np.asarray(sample["audio"]).shape[-1]),
        "visual_dim": int(np.asarray(sample["visual"]).shape[-1]),
        "n_speakers_for_model": 2,
        "n_classes": len(ANJS_LABELS),
    }


def check_compatible(meld: Dict, bobsl: Dict) -> Dict[str, int]:
    meld_dims = dims(meld)
    bobsl_dims = dims(bobsl)
    for key in ["text_dim", "audio_dim", "visual_dim"]:
        if int(meld_dims[key]) != int(bobsl_dims[key]):
            raise RuntimeError(f"Feature mismatch for {key}: MELD={meld_dims[key]} BOBSL={bobsl_dims[key]}")
    return meld_dims


def choose_bobsl_samples(payload: Dict, include_val: bool, max_count: int, seed: int) -> List[Dict]:
    samples = list(payload.get("train", []))
    if include_val:
        samples.extend(list(payload.get("test", [])))
    rng = random.Random(seed)
    rng.shuffle(samples)
    if max_count > 0:
        samples = samples[:max_count]
    return [copy.deepcopy(sample) for sample in samples]


def label_counts(samples: List[Dict]) -> Dict[str, int]:
    labels: List[int] = []
    for sample in samples:
        labels.extend(int(x) for x in sample["labels"] if int(x) >= 0)
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=len(ANJS_LABELS)).astype(int).tolist() if labels else [0] * len(ANJS_LABELS)
    return dict(zip(ANJS_LABELS, counts))


def build_joint(args: argparse.Namespace) -> Dict:
    meld_path = resolve_path(args.meld_pkl, must_exist=True)
    bobsl_path = resolve_path(args.bobsl_train_val_pkl, must_exist=True)
    out_path = resolve_path(args.out_pkl, must_exist=False)
    meld = load_payload(meld_path)
    bobsl = load_payload(bobsl_path)
    feature_dims = check_compatible(meld, bobsl)

    meld_train = [copy.deepcopy(sample) for sample in meld["train"]]
    meld_test = [copy.deepcopy(sample) for sample in meld["test"]]
    meld_dev = [copy.deepcopy(sample) for sample in meld.get("dev", [])]
    bobsl_selected = choose_bobsl_samples(
        bobsl,
        include_val=args.include_bobsl_val,
        max_count=args.max_bobsl_train_dialogues,
        seed=args.bobsl_sample_seed,
    )
    for sample in bobsl_selected:
        sample["id"] = f"joint_{sample.get('id', 'bobsl')}"

    payload = {
        "train": meld_train + bobsl_selected,
        "dev": meld_dev,
        "test": meld_test,
        "meta": {
            "format": "ConxGNN_ANJS4_visual_only_joint_dict",
            "meld_pkl": str(meld_path),
            "bobsl_train_val_pkl": str(bobsl_path),
            "label_space": "ANJS4",
            "label_names": ANJS_LABELS,
            "unknown_context_label": -100,
            "dataset_for_model": "iemocap_4",
            "dims": feature_dims,
            "source_test_meaning": "MELD test only",
            "joint_train_policy": {
                "bobsl_include_val": bool(args.include_bobsl_val),
                "bobsl_requested_max_train_dialogues": int(args.max_bobsl_train_dialogues),
                "bobsl_selected_source_dialogues": len(bobsl_selected),
                "bobsl_sample_seed": int(args.bobsl_sample_seed),
                "bobsl_context_policy": "context1_as_prebuilt_single_utterance_dialogues",
            },
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(payload, f)

    summary = {
        "out_pkl": str(out_path),
        "format": payload["meta"]["format"],
        "meld_pkl": str(meld_path),
        "bobsl_train_val_pkl": str(bobsl_path),
        "label_space": "ANJS4",
        "label_names": ANJS_LABELS,
        "unknown_context_label": -100,
        "dims": feature_dims,
        "train_dialogues": len(payload["train"]),
        "test_dialogues": len(payload["test"]),
        "dev_dialogues": len(payload["dev"]),
        "meld_train_dialogues": len(meld_train),
        "bobsl_train_dialogues": len(bobsl_selected),
        "source_test_meaning": "MELD test only",
        "train_label_counts_all": label_counts(payload["train"]),
        "train_label_counts_meld": label_counts(meld_train),
        "train_label_counts_bobsl": label_counts(bobsl_selected),
        "source_test_label_counts_meld": label_counts(meld_test),
        "joint_train_policy": payload["meta"]["joint_train_policy"],
        "meld_train_ids": sample_ids(meld_train)[:5],
        "bobsl_train_ids": sample_ids(bobsl_selected)[:5],
    }
    out_path.with_suffix(out_path.suffix + ".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[CONXGNN-JOINT] wrote {out_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ConxGNN visual-only joint pkl: MELD train/dev + sampled BOBSL, MELD test as source.")
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
