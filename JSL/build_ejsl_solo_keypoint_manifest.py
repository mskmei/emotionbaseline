#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from keypoints import extract_holistic_keypoints_from_frame_dir, extract_holistic_keypoints_from_video


VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v")
SOLO_RE = re.compile(r"^s(?P<speaker>\d+)_(?P<idx>\d{1,3})(?P<emotion>.*)$", re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(description="Build an eJSL-solo keypoint manifest for JSL fine-tuning.")
    parser.add_argument("--script_txt", type=str, default="JSL/script-solo-78.txt")
    parser.add_argument("--frame_root", type=str, default="/raid_zoe/home/lr/wangyi/sign/eJSL_solo/frame")
    parser.add_argument("--video_root", type=str, default="/raid_zoe/home/lr/wangyi/sign/eJSL_solo/video")
    parser.add_argument("--keypoint_cache_dir", type=str, required=True)
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--sample_fps", type=float, default=0.0)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--model_complexity", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_script(path: Path) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise RuntimeError(f"Expected id + text at {path}:{line_no}: {raw!r}")
        match = re.search(r"(\d+)$", parts[0])
        if match is None:
            raise RuntimeError(f"Cannot parse solo id at {path}:{line_no}: {parts[0]!r}")
        out[int(match.group(1))] = parts[1].strip()
    if not out:
        raise RuntimeError(f"No solo text loaded from {path}")
    return out


def parse_sample_name(name: str) -> Tuple[int, str, str]:
    match = SOLO_RE.match(name)
    if match is None:
        raise RuntimeError(f"Unexpected eJSL-solo media name: {name}")
    return int(match.group("idx")), match.group("speaker"), match.group("emotion")


def find_video(video_root: Path, stem: str) -> Path | None:
    for ext in VIDEO_EXTS:
        path = video_root / f"{stem}{ext}"
        if path.exists():
            return path
    matches = [p for p in video_root.glob(f"{stem}.*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    return matches[0] if matches else None


def collect_samples(frame_root: Path, video_root: Path) -> List[Tuple[str, Path | None, Path | None]]:
    names = set()
    for path in frame_root.iterdir() if frame_root.exists() else []:
        if path.is_dir():
            names.add(path.name)
    for path in video_root.iterdir() if video_root.exists() else []:
        if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
            names.add(path.stem)
    out = []
    for name in sorted(names):
        if SOLO_RE.match(name) is None:
            continue
        frame_dir = frame_root / name
        video_path = find_video(video_root, name)
        out.append((name, frame_dir if frame_dir.exists() else None, video_path))
    return out


def valid_npz(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        data = np.load(path)
        keypoints = data["keypoints"]
        return keypoints.ndim == 2 and keypoints.shape[0] > 0
    except Exception:
        return False


def extract_or_load(args, sample_id: str, frame_dir: Path | None, video_path: Path | None) -> Tuple[Path, str, str]:
    cache_dir = Path(args.keypoint_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{sample_id}.npz"
    if args.resume and valid_npz(cache_path):
        if frame_dir is not None:
            return cache_path, "frame_dir", str(frame_dir)
        return cache_path, "video", str(video_path)

    if frame_dir is not None:
        keypoints, timestamps = extract_holistic_keypoints_from_frame_dir(
            frame_dir,
            sample_fps=args.sample_fps,
            max_frames=args.max_frames,
            model_complexity=args.model_complexity,
        )
        media_kind = "frame_dir"
        media_path = str(frame_dir)
    elif video_path is not None:
        keypoints, timestamps = extract_holistic_keypoints_from_video(
            video_path,
            sample_fps=args.sample_fps if args.sample_fps > 0 else 10.0,
            max_frames=args.max_frames,
            model_complexity=args.model_complexity,
        )
        media_kind = "video"
        media_path = str(video_path)
    else:
        raise FileNotFoundError(f"Missing frame/video for solo sample {sample_id}")

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


def main():
    args = parse_args()
    script = load_script(Path(args.script_txt))
    samples = collect_samples(Path(args.frame_root), Path(args.video_root))
    if args.limit > 0:
        samples = samples[: args.limit]
    rows: List[Dict[str, object]] = []
    for sample_id, frame_dir, video_path in tqdm(samples, desc="eJSL-solo keypoints"):
        idx, speaker, emotion = parse_sample_name(sample_id)
        text = script.get(idx)
        if not text:
            raise RuntimeError(f"No script text for solo index {idx} sample={sample_id}")
        keypoints_path, media_kind, media_path = extract_or_load(args, sample_id, frame_dir, video_path)
        rows.append(
            {
                "sample_id": sample_id,
                "keypoints_path": str(keypoints_path),
                "text": text,
                "split": "train",
                "solo_idx": idx,
                "speaker": speaker,
                "emotion": emotion,
                "media_kind": media_kind,
                "media_path": media_path,
            }
        )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "keypoints_path",
        "text",
        "split",
        "solo_idx",
        "speaker",
        "emotion",
        "media_kind",
        "media_path",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[eJSL-solo-manifest] rows={len(rows)} out={out_csv}")


if __name__ == "__main__":
    main()
