#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from build_unified_meld_ejsl_pkl import (
    ANJS_LABELS,
    FeatureCache,
    MELD_TO_ANJS,
    UtteranceItem,
    VideoEncoder,
    make_speaker_mask,
    resolve_path,
)


def normalize_text(value: str) -> str:
    return " ".join(str(value or "").replace("\t", " ").split())


def resolve_under_root(raw: str, root: Path, default_name: str, must_exist: bool = True) -> Path:
    value = raw or default_name
    path = Path(value).expanduser()
    candidates = [path] if path.is_absolute() else [root / path, Path.cwd() / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    if must_exist:
        raise FileNotFoundError(f"Path not found: {value}. Checked: {', '.join(str(x) for x in candidates)}")
    return candidates[0].resolve()


def resolve_video_path(bobsl_root: Path, video_subdir: str, stem: str, clip_name: str) -> Optional[Path]:
    candidates = [
        bobsl_root / video_subdir / stem / clip_name,
        bobsl_root / "video" / stem / clip_name,
        bobsl_root / stem / clip_name,
        bobsl_root / clip_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0]


def frame_stem_dir(bobsl_root: Path, frame_subdir: str, stem: str) -> Optional[Path]:
    path = bobsl_root / frame_subdir / stem
    return path.resolve() if path.exists() else path


def unique_vid(base: str, dialogues: Dict[str, List[UtteranceItem]]) -> str:
    if base not in dialogues:
        return base
    index = 2
    while f"{base}_{index}" in dialogues:
        index += 1
    return f"{base}_{index}"


def read_bobsl_split(
    csv_path: Path,
    split_name: str,
    bobsl_root: Path,
    video_subdir: str,
    frame_subdir: str,
    min_score: float,
    limit: int,
) -> Tuple[Dict[str, List[UtteranceItem]], List[str], Dict]:
    dialogues: Dict[str, List[UtteranceItem]] = {}
    keys: List[str] = []
    skipped_emotion = 0
    skipped_score = 0

    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            emotion = str(row.get("emotion", "")).strip().lower()
            label = MELD_TO_ANJS.get(emotion, -100)
            if label < 0:
                skipped_emotion += 1
                continue

            try:
                score = float(row.get("score", 0.0) or 0.0)
            except ValueError:
                score = 0.0
            if score < min_score:
                skipped_score += 1
                continue

            stem = str(row.get("stem", "")).strip()
            clip_name = str(row.get("clip_name", "")).strip()
            if not stem or not clip_name:
                skipped_emotion += 1
                continue

            clip_stem = Path(clip_name).stem
            vid = unique_vid(f"bobsl_{split_name}_{stem}_{clip_stem}", dialogues)
            item = UtteranceItem(
                text=normalize_text(str(row.get("text", ""))),
                speaker="bobsl",
                label=label,
                video_path=resolve_video_path(bobsl_root, video_subdir, stem, clip_name),
                frame_dir=frame_stem_dir(bobsl_root, frame_subdir, stem),
                sample_id=f"{stem}/{clip_name}",
            )
            dialogues[vid] = [item]
            keys.append(vid)
            if limit > 0 and len(keys) >= limit:
                break

    labels = [items[0].label for items in dialogues.values()]
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=len(ANJS_LABELS)).tolist() if labels else [0] * len(ANJS_LABELS)
    summary = {
        "csv_path": str(csv_path),
        "split": split_name,
        "dialogues": len(keys),
        "label_counts": dict(zip(ANJS_LABELS, counts)),
        "skipped_emotion": skipped_emotion,
        "skipped_score": skipped_score,
        "min_score": min_score,
    }
    print(f"[BOBSL] split={split_name} dialogues={len(keys)} labels={summary['label_counts']}")
    return dialogues, keys, summary


def sample_frame_paths(frame_paths: Sequence[Path]) -> List[Path]:
    paths = list(frame_paths)
    if not paths:
        return []
    if len(paths) >= 8:
        indices = np.linspace(0, len(paths) - 1, num=8, dtype=int)
        return [paths[int(i)] for i in indices]
    return paths + [paths[-1]] * (8 - len(paths))


def find_prefixed_frames(frame_dir: Optional[Path], sample_id: str) -> List[Path]:
    if frame_dir is None or not frame_dir.exists():
        return []
    clip_name = sample_id.split("/", 1)[-1]
    clip_stem = Path(clip_name).stem
    frame_paths = sorted(frame_dir.glob(f"{clip_stem}_*.jpg"))
    if not frame_paths:
        frame_paths = sorted(frame_dir.glob(f"{clip_stem}_*.png"))
    return sample_frame_paths(frame_paths)


def frame_cache_key(video_encoder: VideoEncoder, frame_paths: Sequence[Path]) -> str:
    if frame_paths:
        first = frame_paths[0]
        last = frame_paths[-1]
        source = {
            "count": len(frame_paths),
            "first": str(first.resolve()),
            "last": str(last.resolve()),
            "first_mtime": first.stat().st_mtime_ns,
            "last_mtime": last.stat().st_mtime_ns,
        }
    else:
        source = {"missing": True}
    return json.dumps(
        {
            "model": video_encoder.model_name,
            "processor": video_encoder.processor_name,
            "source": source,
        },
        sort_keys=True,
    )


def encode_prefixed_frames(video_encoder: VideoEncoder, frame_paths: Sequence[Path]) -> Tuple[np.ndarray, str]:
    key = frame_cache_key(video_encoder, frame_paths)
    cached = video_encoder.cache.get("video", key)
    if cached is not None:
        return cached, "cache"

    frames = []
    for path in frame_paths:
        image = cv2.imread(str(path))
        if image is not None:
            frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not frames:
        vec = np.zeros(video_encoder.dim, dtype=np.float32)
        video_encoder.cache.put("video", key, vec)
        return vec, "frames_empty"

    inputs = video_encoder.processor(frames, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(video_encoder.device)
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=video_encoder.device.type == "cuda" and video_encoder.fp16):
            hidden = video_encoder.model(pixel_values).last_hidden_state[:, 0, :]
    vec = hidden[0].detach().cpu().numpy().astype(np.float32)
    video_encoder.cache.put("video", key, vec)
    return vec, "frames"


def encode_bobsl_video(item: UtteranceItem, video_encoder: VideoEncoder, use_frame_fallback: bool) -> Tuple[np.ndarray, str]:
    if item.video_path is not None and item.video_path.exists():
        return video_encoder.encode_one(item.video_path, None)
    if use_frame_fallback:
        frame_paths = find_prefixed_frames(item.frame_dir, item.sample_id)
        if frame_paths:
            return encode_prefixed_frames(video_encoder, frame_paths)
    return video_encoder.encode_one(item.video_path, None)


def write_bobsl_feature_pkl(
    dialogues: Dict[str, List[UtteranceItem]],
    train_vid: List[str],
    test_vid: List[str],
    out_pkl: Path,
    video_encoder: VideoEncoder,
    text_dim: int,
    audio_dim: int,
    n_speakers: int,
    use_frame_fallback: bool,
    meta: Dict,
) -> Dict:
    all_items: List[UtteranceItem] = []
    for vid in train_vid + test_vid:
        all_items.extend(dialogues[vid])

    video_vectors: List[np.ndarray] = []
    video_status: Dict[str, int] = {}
    for idx, item in enumerate(all_items, start=1):
        vec, status = encode_bobsl_video(item, video_encoder, use_frame_fallback)
        video_vectors.append(vec)
        video_status[status] = video_status.get(status, 0) + 1
        if idx == 1 or idx % 250 == 0 or idx == len(all_items):
            print(f"[BOBSL] video encode items={idx}/{len(all_items)} status={video_status}")

    cursor = 0
    video_ids: Dict[str, str] = {}
    video_speakers: Dict[str, np.ndarray] = {}
    video_labels: Dict[str, List[int]] = {}
    video_text: Dict[str, np.ndarray] = {}
    video_audio: Dict[str, np.ndarray] = {}
    video_visual: Dict[str, np.ndarray] = {}
    video_sentence: Dict[str, List[str]] = {}

    for vid in train_vid + test_vid:
        items = dialogues[vid]
        n_items = len(items)
        visual_seq = np.stack(video_vectors[cursor : cursor + n_items], axis=0).astype(np.float32)
        cursor += n_items

        video_ids[vid] = vid
        video_speakers[vid] = make_speaker_mask([item.speaker for item in items], n_speakers)
        video_labels[vid] = [int(item.label) for item in items]
        video_text[vid] = np.zeros((n_items, text_dim), dtype=np.float32)
        video_audio[vid] = np.zeros((n_items, audio_dim), dtype=np.float32)
        video_visual[vid] = visual_seq
        video_sentence[vid] = [item.text for item in items]

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
        {
            **meta,
            "label_names": ANJS_LABELS,
            "n_classes": len(ANJS_LABELS),
            "text_dim": text_dim,
            "audio_dim": audio_dim,
            "visual_dim": video_encoder.dim,
            "n_speakers": n_speakers,
            "video_status": video_status,
        },
    )
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with out_pkl.open("wb") as f:
        pickle.dump(payload, f)

    valid_labels = [item.label for item in all_items if item.label >= 0]
    counts = np.bincount(np.asarray(valid_labels, dtype=np.int64), minlength=len(ANJS_LABELS)).tolist()
    summary = {
        "out_pkl": str(out_pkl),
        "dialogues": len(train_vid) + len(test_vid),
        "train_dialogues": len(train_vid),
        "test_dialogues": len(test_vid),
        "utterances": len(all_items),
        "valid_labeled_utterances": len(valid_labels),
        "label_counts": dict(zip(ANJS_LABELS, counts)),
        "video_status": video_status,
    }
    out_pkl.with_suffix(out_pkl.suffix + ".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[BOBSL] wrote {out_pkl}")
    print(f"[BOBSL] label_counts={summary['label_counts']}")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Build BOBSL ANJS4 feature pkl files aligned with the unified MMGCN pipeline.")
    parser.add_argument("--bobsl_root", type=str, default="/raid_zoe/home/lr/wangyi/sign/bobsl")
    parser.add_argument("--train_csv", type=str, default="")
    parser.add_argument("--val_csv", type=str, default="")
    parser.add_argument("--test_csv", type=str, default="")
    parser.add_argument("--video_subdir", type=str, default="clip256")
    parser.add_argument("--frame_subdir", type=str, default="frame")
    parser.add_argument("--out_train_val_pkl", type=str, required=True)
    parser.add_argument("--out_test_pkl", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="./MMGCN/unified_features/bobsl_cache")
    parser.add_argument("--video_model", type=str, default="facebook/timesformer-base-finetuned-k400")
    parser.add_argument("--video_processor", type=str, default="MCG-NJU/videomae-base")
    parser.add_argument("--video_max_seconds", type=float, default=30.0)
    parser.add_argument("--text_dim", type=int, default=1024)
    parser.add_argument("--audio_dim", type=int, default=768)
    parser.add_argument("--n_speakers", type=int, default=9)
    parser.add_argument("--min_score", type=float, default=0.0)
    parser.add_argument("--limit_train", type=int, default=0)
    parser.add_argument("--limit_val", type=int, default=0)
    parser.add_argument("--limit_test", type=int, default=0)
    parser.add_argument("--no_frame_fallback", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--no_cuda", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bobsl_root = resolve_path(args.bobsl_root, must_exist=True)
    train_csv = resolve_under_root(args.train_csv, bobsl_root, "train_clips_balanced_updated.csv")
    val_csv = resolve_under_root(args.val_csv, bobsl_root, "val_clips_balanced_updated.csv")
    test_csv = resolve_under_root(args.test_csv, bobsl_root, "test_clips_balanced_updated.csv")
    out_train_val_pkl = resolve_path(args.out_train_val_pkl, must_exist=False)
    out_test_pkl = resolve_path(args.out_test_pkl, must_exist=False)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    cache = FeatureCache(resolve_path(args.cache_dir, must_exist=False) if args.cache_dir else None)
    print(f"[BOBSL] root={bobsl_root}")
    print(f"[BOBSL] device={device}")
    print(f"[BOBSL] cache_dir={cache.root}")
    print(f"[BOBSL] text_dim={args.text_dim} audio_dim={args.audio_dim}")

    video_encoder = VideoEncoder(
        args.video_model,
        args.video_processor,
        device,
        cache,
        args.video_max_seconds,
        args.fp16,
    )

    train_dialogues, train_vid, train_summary = read_bobsl_split(
        train_csv,
        "train",
        bobsl_root,
        args.video_subdir,
        args.frame_subdir,
        args.min_score,
        args.limit_train,
    )
    val_dialogues, val_vid, val_summary = read_bobsl_split(
        val_csv,
        "val",
        bobsl_root,
        args.video_subdir,
        args.frame_subdir,
        args.min_score,
        args.limit_val,
    )
    test_dialogues, test_vid, test_summary = read_bobsl_split(
        test_csv,
        "test",
        bobsl_root,
        args.video_subdir,
        args.frame_subdir,
        args.min_score,
        args.limit_test,
    )

    train_val_dialogues = {**train_dialogues, **val_dialogues}
    common_meta = {
        "dataset": "BOBSL",
        "label_space": "ANJS4",
        "unknown_context_label": -100,
        "text_feature_policy": "zeros_for_video_pretrain",
        "video_model": args.video_model,
        "video_processor": args.video_processor,
        "video_subdir": args.video_subdir,
        "frame_subdir": args.frame_subdir,
        "source_root": str(bobsl_root),
        "split_summaries": {
            "train": train_summary,
            "val": val_summary,
            "test": test_summary,
        },
    }
    summaries = {
        "train_val": write_bobsl_feature_pkl(
            train_val_dialogues,
            train_vid,
            val_vid,
            out_train_val_pkl,
            video_encoder,
            args.text_dim,
            args.audio_dim,
            args.n_speakers,
            not args.no_frame_fallback,
            {**common_meta, "pkl_role": "train_val", "source_test_meaning": "bobsl_val"},
        ),
        "test": write_bobsl_feature_pkl(
            test_dialogues,
            [],
            test_vid,
            out_test_pkl,
            video_encoder,
            args.text_dim,
            args.audio_dim,
            args.n_speakers,
            not args.no_frame_fallback,
            {**common_meta, "pkl_role": "test", "source_test_meaning": "bobsl_test"},
        ),
    }
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
