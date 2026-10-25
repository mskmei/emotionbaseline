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


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "new"))

from anjs_video_common import (  # noqa: E402
    ANJS_LABELS,
    FeatureCache,
    UtteranceItem,
    VideoEmbeddingEncoder,
    label_counts,
    make_speaker_onehot,
    read_iemocap_dialogues,
    resolve_path,
)
from new.ECERC.MELD.build_ecerc_video_anjs_features import encode_dialogue_features as encode_ecerc  # noqa: E402
from new.MAGTKD.MELD.build_magtkd_video_anjs_features import encode_dialogue_features as encode_magtkd  # noqa: E402
from new2.CMERC.build_cmerc_visual_anjs_features import encode_dialogues as encode_cmerc  # noqa: E402
from new2.ConxGNN.build_conxgnn_visual_anjs_features import encode_splits as encode_conxgnn  # noqa: E402


BASELINES = ["mmgcn", "ecerc", "magtkd", "cmerc", "conxgnn"]


def load_pickle(path: str) -> object:
    resolved = resolve_path(path, must_exist=True)
    with resolved.open("rb") as f:
        return pickle.load(f, encoding="latin1")


def first_key(keys: Sequence[str]) -> str:
    if not keys:
        raise ValueError("Reference feature pkl has no samples/dialogues")
    return str(keys[0])


def mmgcn_dims(reference_path: str) -> Dict[str, int]:
    obj = load_pickle(reference_path)
    if not isinstance(obj, (tuple, list)) or len(obj) < 10:
        raise ValueError(f"Expected MMGCN unified tuple: {reference_path}")
    key = first_key(list(obj[7]) + list(obj[8]))
    meta = obj[9] if isinstance(obj[9], dict) else {}
    return {
        "text": int(meta.get("text_dim") or np.asarray(obj[3][key]).shape[-1]),
        "audio": int(meta.get("audio_dim") or np.asarray(obj[4][key]).shape[-1]),
        "visual": int(meta.get("visual_dim") or np.asarray(obj[5][key]).shape[-1]),
        "speakers": int(meta.get("n_speakers") or np.asarray(obj[1][key]).shape[-1]),
    }


def ecerc_dims(reference_path: str) -> Dict[str, int]:
    obj = load_pickle(reference_path)
    if not isinstance(obj, (tuple, list)) or len(obj) != 17:
        raise ValueError(f"Expected ECERC 17-field tuple: {reference_path}")
    key = first_key(list(obj[14]) + list(obj[15]) + list(obj[16]))
    return {
        "text": int(np.asarray(obj[3][key]).shape[-1]),
        "audio": int(np.asarray(obj[11][key]).shape[-1]),
        "visual": int(np.asarray(obj[12][key]).shape[-1]),
        "speakers": int(np.asarray(obj[0][key]).shape[-1]),
    }


def dict_dims(reference_path: str, kind: str) -> Dict[str, int]:
    obj = load_pickle(reference_path)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected {kind} feature dict: {reference_path}")
    if kind == "magtkd":
        key = first_key(list(obj.get("train_vids", [])) + list(obj.get("test_vids", [])) + list(obj.get("valid_vids", [])) + list(obj.get("vids", [])))
        return {
            "text": int(np.asarray(obj["text"][key]).shape[-1]),
            "audio": int(np.asarray(obj["audio"][key]).shape[-1]),
            "visual": int(np.asarray(obj["video"][key]).shape[-1]),
            "speakers": int((obj.get("meta") or {}).get("max_speakers", 304)),
        }
    if kind == "cmerc":
        key = first_key(list(obj.get("train_ids", [])) + list(obj.get("test_ids", [])) + list(obj.get("valid_ids", [])))
        return {
            "text": int(np.asarray(obj["roberta1"][key]).shape[-1]),
            "audio": int(np.asarray(obj["video_audio"][key]).shape[-1]),
            "visual": int(np.asarray(obj["video_visual"][key]).shape[-1]),
            "speakers": int(np.asarray(obj["speakers"][key]).shape[-1]),
        }
    samples = list(obj.get("train", [])) + list(obj.get("test", [])) + list(obj.get("dev", []))
    if not samples:
        raise ValueError(f"ConxGNN reference pkl has no samples: {reference_path}")
    return {
        "text": int(np.asarray(samples[0]["text"]).shape[-1]),
        "audio": int(np.asarray(samples[0]["audio"]).shape[-1]),
        "visual": int(np.asarray(samples[0]["visual"]).shape[-1]),
        "speakers": 2,
    }


def ensure_visual_dim(name: str, expected: int, actual: int) -> None:
    if expected != actual:
        raise RuntimeError(
            f"{name} reference visual dim is {expected}, but {actual} was produced by the configured video encoder. "
            "Use the same VIDEO_MODEL as the existing BOBSL/eJSL features."
        )


def encode_visual_once(
    dialogues: Dict[str, List[UtteranceItem]],
    keys: Sequence[str],
    encoder: VideoEmbeddingEncoder,
) -> tuple[Dict[str, np.ndarray], Dict[str, int]]:
    output: Dict[str, np.ndarray] = {}
    status_counts: Dict[str, int] = {}
    total = sum(len(dialogues[key]) for key in keys)
    seen = 0
    for key in keys:
        vectors: List[np.ndarray] = []
        for item in dialogues[key]:
            vector, status = encoder.encode_item(item)
            vectors.append(vector)
            status_counts[status] = status_counts.get(status, 0) + 1
            seen += 1
            if seen == 1 or seen % 250 == 0 or seen == total:
                print(f"[IEMOCAP-FEATURES] visual encode {seen}/{total} status={status_counts}")
        output[key] = np.stack(vectors, axis=0).astype(np.float32)
    return output, status_counts


class PrecomputedVideoEncoder:
    def __init__(self, dialogues: Dict[str, List[UtteranceItem]], vectors: Dict[str, np.ndarray], dim: int):
        self.vectors = {
            (key, item.sample_id): vector
            for key, items in dialogues.items()
            for item, vector in zip(items, vectors[key])
        }
        self.dim = dim

    def encode_item(self, item: UtteranceItem):
        return self.vectors[(item.dialogue_id, item.sample_id)], "shared_precomputed"


def write_mmgcn(
    dialogues: Dict[str, List[UtteranceItem]],
    train_ids: Sequence[str],
    test_ids: Sequence[str],
    visual: Dict[str, np.ndarray],
    dims: Dict[str, int],
    meta: Dict,
    out_path: Path,
    status_counts: Dict[str, int],
) -> None:
    video_ids: Dict[str, str] = {}
    speakers: Dict[str, np.ndarray] = {}
    labels: Dict[str, List[int]] = {}
    text: Dict[str, np.ndarray] = {}
    audio: Dict[str, np.ndarray] = {}
    sentences: Dict[str, List[str]] = {}
    keys = list(train_ids) + list(test_ids)
    for key in keys:
        items = dialogues[key]
        video_ids[key] = key
        speakers[key] = make_speaker_onehot([item.speaker for item in items], dims["speakers"])
        labels[key] = [int(item.label) for item in items]
        text[key] = np.zeros((len(items), dims["text"]), dtype=np.float32)
        audio[key] = np.zeros((len(items), dims["audio"]), dtype=np.float32)
        sentences[key] = [item.text for item in items]
    payload_meta = {
        **meta,
        "dataset": "IEMOCAP",
        "label_space": "ANJS4",
        "label_names": ANJS_LABELS,
        "n_classes": 4,
        "text_dim": dims["text"],
        "audio_dim": dims["audio"],
        "visual_dim": dims["visual"],
        "n_speakers": dims["speakers"],
        "feature_policy": "mmgcn_video_only_zero_text_audio",
        "video_status": status_counts,
    }
    payload = (video_ids, speakers, labels, text, audio, visual, sentences, list(train_ids), list(test_ids), payload_meta)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(payload, f)
    summary = {
        "out_pkl": str(out_path),
        "format": "MMGCN_unified_ANJS4",
        "train_dialogues": len(train_ids),
        "test_dialogues": len(test_ids),
        "utterances": sum(len(dialogues[key]) for key in keys),
        "train_label_counts": label_counts(dialogues, train_ids),
        "test_label_counts": label_counts(dialogues, test_ids),
        "dims": dims,
        "video_status": status_counts,
        "meta": payload_meta,
    }
    out_path.with_suffix(out_path.suffix + ".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[IEMOCAP-FEATURES][MMGCN] wrote {out_path}")


def parse_baselines(raw: str) -> List[str]:
    values = BASELINES if raw.strip().lower() == "all" else raw.lower().replace(",", " ").split()
    unknown = sorted(set(values).difference(BASELINES))
    if unknown:
        raise ValueError(f"Unknown baselines: {unknown}; choices={BASELINES}")
    return [name for name in BASELINES if name in values]


def main() -> None:
    args = parse_args()
    baselines = parse_baselines(args.baselines)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    dialogues, train_ids, test_ids, valid_ids, source_meta = read_iemocap_dialogues(
        args.iemocap_root,
        args.metadata_root,
        args.frame_root,
        num_frames=args.num_frames,
        fold_dev_into_train=True,
        require_frames=True,
        limit_dialogues=args.limit_dialogues,
    )
    if valid_ids:
        raise RuntimeError("IEMOCAP dev must be folded into train to match the MELD experiment protocol")
    keys = list(train_ids) + list(test_ids)
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
    print(f"[IEMOCAP-FEATURES] baselines={baselines} device={device} cache={cache.root}")
    visual, status_counts = encode_visual_once(dialogues, keys, encoder)
    precomputed = PrecomputedVideoEncoder(dialogues, visual, encoder.dim)
    common_meta = {
        **source_meta,
        "dataset": "IEMOCAP",
        "label_space": "ANJS4",
        "label_mapping_description": "ang+fru=A, neu=N, hap+exc=J, sad=S",
        "split_policy": "repository train+dev folded into train; repository Session5 test kept as source test",
        "video_model": encoder.video_model,
        "video_processor": encoder.video_processor,
        "feature_extraction_shared_across_baselines": True,
    }

    if "mmgcn" in baselines:
        dims = mmgcn_dims(args.reference_mmgcn_pkl)
        ensure_visual_dim("MMGCN", dims["visual"], encoder.dim)
        write_mmgcn(dialogues, train_ids, test_ids, visual, dims, common_meta, resolve_path(args.out_mmgcn_pkl), status_counts)
    if "ecerc" in baselines:
        dims = ecerc_dims(args.reference_ecerc_pkl)
        ensure_visual_dim("ECERC", dims["visual"], encoder.dim)
        encode_ecerc(
            dialogues, train_ids, test_ids, [], precomputed, None, dims["text"], dims["audio"], dims["speakers"],
            {**common_meta, "feature_policy": "ecerc_video_only_zero_text_audio_semantic"}, resolve_path(args.out_ecerc_pkl),
        )
    if "magtkd" in baselines:
        dims = dict_dims(args.reference_magtkd_pkl, "magtkd")
        ensure_visual_dim("MAGTKD", dims["visual"], encoder.dim)
        encode_magtkd(
            dialogues, train_ids, test_ids, [], precomputed, None, dims["visual"], dims["speakers"], common_meta,
            resolve_path(args.out_magtkd_pkl),
        )
    if "cmerc" in baselines:
        dims = dict_dims(args.reference_cmerc_pkl, "cmerc")
        ensure_visual_dim("CMERC", dims["visual"], encoder.dim)
        encode_cmerc(
            dialogues, train_ids, test_ids, [], precomputed, None, dims["text"], dims["audio"], dims["speakers"],
            common_meta, resolve_path(args.out_cmerc_pkl),
        )
    if "conxgnn" in baselines:
        dims = dict_dims(args.reference_conxgnn_pkl, "conxgnn")
        ensure_visual_dim("ConxGNN", dims["visual"], encoder.dim)
        encode_conxgnn(
            dialogues, train_ids, test_ids, [], precomputed, None, dims["text"], dims["audio"], common_meta,
            resolve_path(args.out_conxgnn_pkl),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build IEMOCAP ANJS4 video-only feature pkls for all five existing baseline schemas.")
    parser.add_argument("--baselines", default="all", help="all or a space/comma-separated subset of mmgcn ecerc magtkd cmerc conxgnn")
    parser.add_argument("--iemocap_root", default="/raid_zoe/home/lr/maokeyu/IEMOCAP_full_release")
    parser.add_argument("--metadata_root", default=str(REPO_ROOT / "new" / "MAGTKD" / "datasets" / "IEMOCAP"))
    parser.add_argument("--frame_root", default="/raid_zoe/home/lr/maokeyu/sign/iemocap_bobsl_ejsl/frames_8_margin1")
    parser.add_argument("--cache_dir", default="/raid_zoe/home/lr/maokeyu/sign/iemocap_bobsl_ejsl/timesformer_feature_cache")
    parser.add_argument("--video_model", default="facebook/timesformer-base-finetuned-k400")
    parser.add_argument("--video_processor", default="MCG-NJU/videomae-base")
    parser.add_argument("--local_model_root", default="")
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--video_max_seconds", type=float, default=30.0)
    parser.add_argument("--limit_dialogues", type=int, default=0)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--no_cuda", action="store_true")

    parser.add_argument("--reference_mmgcn_pkl", default="/raid_zoe/home/lr/maokeyu/sign/mmgcn_bobsl_meld_ejsl/bobsl_anjs4_train_val.pkl")
    parser.add_argument("--reference_ecerc_pkl", default="/raid_zoe/home/lr/maokeyu/sign/ecerc_bobsl_meld_ejsl/features/bobsl_ecerc_video_anjs4_train_val.pkl")
    parser.add_argument("--reference_magtkd_pkl", default="/raid_zoe/home/lr/maokeyu/sign/magtkd_bobsl_meld_ejsl/features/bobsl_magtkd_video_anjs4_train_val.pkl")
    parser.add_argument("--reference_cmerc_pkl", default="/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_meld_ejsl/features/bobsl_cmerc_visual_anjs4_train_val.pkl")
    parser.add_argument("--reference_conxgnn_pkl", default="/raid_zoe/home/lr/maokeyu/sign/conxgnn_bobsl_meld_ejsl/features/bobsl_conxgnn_visual_anjs4_train_val.pkl")

    parser.add_argument("--out_mmgcn_pkl", default="/raid_zoe/home/lr/maokeyu/sign/mmgcn_bobsl_iemocap_ejsl/features/iemocap_mmgcn_video_anjs4.pkl")
    parser.add_argument("--out_ecerc_pkl", default="/raid_zoe/home/lr/maokeyu/sign/ecerc_bobsl_iemocap_ejsl/features/iemocap_ecerc_video_anjs4.pkl")
    parser.add_argument("--out_magtkd_pkl", default="/raid_zoe/home/lr/maokeyu/sign/magtkd_bobsl_iemocap_ejsl/features/iemocap_magtkd_video_anjs4.pkl")
    parser.add_argument("--out_cmerc_pkl", default="/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_iemocap_ejsl/features/iemocap_cmerc_visual_anjs4.pkl")
    parser.add_argument("--out_conxgnn_pkl", default="/raid_zoe/home/lr/maokeyu/sign/conxgnn_bobsl_iemocap_ejsl/features/iemocap_conxgnn_visual_anjs4.pkl")
    return parser.parse_args()


if __name__ == "__main__":
    main()
