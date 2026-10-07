#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from keypoints import (
    extract_holistic_keypoints_from_frame_dir,
    extract_holistic_keypoints_from_video,
)
from manifest_utils import parse_ejsl_sample_id, read_ejsl_names


VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v")


def parse_args():
    parser = argparse.ArgumentParser(description="Build an oracle-text eJSL keypoint manifest for JSL fine-tuning.")
    parser.add_argument("--dial_list", type=str, default="/home/lr/wangyi/Sign/RO-MAN/eJSL_dial_dataset/ejsldial_filenames.csv")
    parser.add_argument("--video_root", type=str, default="/raid_zoe/home/lr/wangyi/sign/eJSL_dial/video")
    parser.add_argument("--frame_root", type=str, default="/raid_zoe/home/lr/wangyi/sign/eJSL_dial/frame")
    parser.add_argument("--structure_txt_root", type=str, default="/raid_elmo/home/lr/wangyi/PTR/STUDIES-Japanese/Short_dialogue")
    parser.add_argument("--keypoint_cache_dir", type=str, required=True)
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--split_csv_dir", type=str, default="")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_fps", type=float, default=10.0)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--model_complexity", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def find_video(video_root: Path, sample_id: str) -> Path | None:
    for ext in VIDEO_EXTS:
        path = video_root / f"{sample_id}{ext}"
        if path.exists():
            return path
    matches = [p for p in video_root.glob(f"{sample_id}.*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    return matches[0] if matches else None


def gt_text(structure_txt_root: Path, sample_id: str) -> Tuple[str, str, str]:
    sd_id, dialogue_idx, utterance_idx, label = parse_ejsl_sample_id(sample_id)
    txt_file = structure_txt_root / sd_id / "txt" / f"{sd_id}-Dialogue-{dialogue_idx:02d}.txt"
    lines = [line.strip() for line in txt_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    parts = lines[utterance_idx - 1].split("|", 2)
    if len(parts) < 3:
        raise RuntimeError(f"Expected speaker|emotion|text at {txt_file}:{utterance_idx}")
    return parts[2].strip(), parts[0].strip(), parts[1].strip()


def valid_npz(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        data = np.load(path)
        keypoints = data["keypoints"]
        return keypoints.ndim == 2 and keypoints.shape[0] > 0
    except Exception:
        return False


def extract_or_load(args, sample_id: str, video_path: Path | None, frame_dir: Path | None) -> Tuple[Path, str, str]:
    cache_dir = Path(args.keypoint_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{sample_id}.npz"
    if args.resume and valid_npz(cache_path):
        if video_path is not None:
            return cache_path, "video", str(video_path)
        return cache_path, "frame_dir", str(frame_dir)

    if video_path is not None:
        keypoints, timestamps = extract_holistic_keypoints_from_video(
            video_path,
            sample_fps=args.sample_fps,
            max_frames=args.max_frames,
            model_complexity=args.model_complexity,
        )
        media_kind = "video"
        media_path = str(video_path)
    elif frame_dir is not None and frame_dir.exists():
        keypoints, timestamps = extract_holistic_keypoints_from_frame_dir(
            frame_dir,
            sample_fps=0.0,
            max_frames=args.max_frames,
            model_complexity=args.model_complexity,
        )
        media_kind = "frame_dir"
        media_path = str(frame_dir)
    else:
        raise FileNotFoundError(f"Missing video/frame media for {sample_id}")

    tmp_path = cache_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp_path,
        keypoints=keypoints.astype(np.float32),
        timestamps=timestamps.astype(np.float32),
        sample_id=sample_id,
        media_kind=media_kind,
        media_path=media_path,
    )
    tmp_path.replace(cache_path)
    return cache_path, media_kind, media_path


def split_map(sample_ids: List[str], train_ratio: float, seed: int) -> Dict[str, str]:
    ids = list(sample_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_train = int(round(len(ids) * float(train_ratio)))
    train = set(ids[:n_train])
    return {sample_id: ("train" if sample_id in train else "test") for sample_id in sample_ids}


def write_split_csvs(split_dir: Path, rows: List[Dict[str, object]]) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    for split in sorted({str(row["split"]) for row in rows}):
        path = split_dir / f"{split}.csv"
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["sample_id"])
            writer.writeheader()
            for row in rows:
                if row["split"] == split:
                    writer.writerow({"sample_id": row["sample_id"]})


def main():
    args = parse_args()
    sample_ids = read_ejsl_names(Path(args.dial_list))
    if args.limit > 0:
        sample_ids = sample_ids[: args.limit]
    splits = split_map(sample_ids, args.train_ratio, args.seed)
    video_root = Path(args.video_root)
    frame_root = Path(args.frame_root)
    structure_txt_root = Path(args.structure_txt_root)

    rows: List[Dict[str, object]] = []
    for sample_id in tqdm(sample_ids, desc="eJSL keypoints"):
        video_path = find_video(video_root, sample_id)
        frame_dir = frame_root / sample_id
        text, speaker, emotion = gt_text(structure_txt_root, sample_id)
        keypoints_path, media_kind, media_path = extract_or_load(args, sample_id, video_path, frame_dir)
        rows.append(
            {
                "sample_id": sample_id,
                "keypoints_path": str(keypoints_path),
                "text": text,
                "split": splits[sample_id],
                "speaker": speaker,
                "emotion": emotion,
                "media_kind": media_kind,
                "media_path": media_path,
            }
        )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sample_id", "keypoints_path", "text", "split", "speaker", "emotion", "media_kind", "media_path"]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    if args.split_csv_dir:
        write_split_csvs(Path(args.split_csv_dir), rows)
    counts = {split: sum(1 for row in rows if row["split"] == split) for split in sorted(set(splits.values()))}
    print(f"[eJSL-manifest] rows={len(rows)} splits={counts} out={out_csv}")


if __name__ == "__main__":
    main()
