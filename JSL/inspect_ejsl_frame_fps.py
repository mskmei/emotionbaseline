#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from manifest_utils import read_ejsl_names


VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Inspect eJSL extracted frame folders and estimate their effective FPS "
            "by comparing frame counts with the matching video durations."
        )
    )
    parser.add_argument(
        "--dial_list",
        type=str,
        default="/home/lr/wangyi/Sign/RO-MAN/eJSL_dial_dataset/ejsldial_filenames.csv",
    )
    parser.add_argument(
        "--frame_root",
        type=str,
        default="/raid_zoe/home/lr/wangyi/sign/eJSL_dial/frame",
    )
    parser.add_argument(
        "--video_root",
        type=str,
        default="/raid_zoe/home/lr/wangyi/sign/eJSL_dial/video",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out_csv", type=str, default="")
    parser.add_argument("--summary_json", type=str, default="")
    return parser.parse_args()


def find_video(video_root: Path, sample_id: str) -> Optional[Path]:
    for ext in VIDEO_EXTS:
        path = video_root / f"{sample_id}{ext}"
        if path.exists():
            return path
    matches = [p for p in video_root.glob(f"{sample_id}.*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    return matches[0] if matches else None


def list_frames(frame_dir: Path) -> List[Path]:
    paths: List[Path] = []
    for ext in IMAGE_EXTS:
        paths.extend(frame_dir.glob(f"*{ext}"))
        paths.extend(frame_dir.glob(f"*{ext.upper()}"))
    return sorted(set(paths))


def cv2_video_info(path: Path) -> Dict[str, Optional[float]]:
    try:
        import cv2  # type: ignore
    except Exception:
        return {"video_fps": None, "video_frame_count": None, "duration_sec": None}

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"video_fps": None, "video_frame_count": None, "duration_sec": None}
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    cap.release()
    duration = frames / fps if fps > 0 and frames > 0 else None
    return {
        "video_fps": fps if fps > 0 else None,
        "video_frame_count": frames if frames > 0 else None,
        "duration_sec": duration,
    }


def ffprobe_video_info(path: Path) -> Dict[str, Optional[float]]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except FileNotFoundError:
        return {"video_fps": None, "video_frame_count": None, "duration_sec": None}
    if result.returncode != 0:
        return {"video_fps": None, "video_frame_count": None, "duration_sec": None}
    try:
        data = json.loads(result.stdout or "{}")
        stream = (data.get("streams") or [{}])[0]
    except Exception:
        return {"video_fps": None, "video_frame_count": None, "duration_sec": None}

    fps = parse_rate(stream.get("avg_frame_rate"))
    frames = parse_float(stream.get("nb_frames"))
    duration = parse_float(stream.get("duration"))
    if duration is None and fps and frames:
        duration = frames / fps
    return {"video_fps": fps, "video_frame_count": frames, "duration_sec": duration}


def parse_float(value) -> Optional[float]:
    if value in (None, "", "N/A"):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def parse_rate(value) -> Optional[float]:
    if not value or value == "0/0":
        return None
    text = str(value)
    if "/" in text:
        num, den = text.split("/", 1)
        num_f, den_f = parse_float(num), parse_float(den)
        if num_f and den_f:
            return num_f / den_f
        return None
    return parse_float(text)


def video_info(path: Optional[Path]) -> Dict[str, Optional[float]]:
    if path is None:
        return {"video_fps": None, "video_frame_count": None, "duration_sec": None}
    info = cv2_video_info(path)
    if info.get("duration_sec"):
        return info
    ff_info = ffprobe_video_info(path)
    return {
        "video_fps": info.get("video_fps") or ff_info.get("video_fps"),
        "video_frame_count": info.get("video_frame_count") or ff_info.get("video_frame_count"),
        "duration_sec": info.get("duration_sec") or ff_info.get("duration_sec"),
    }


def summarize(values: Iterable[Optional[float]]) -> Dict[str, Optional[float]]:
    nums = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    if not nums:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": len(nums),
        "mean": statistics.fmean(nums),
        "median": statistics.median(nums),
        "min": min(nums),
        "max": max(nums),
    }


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "frame_dir",
        "frame_count",
        "video_path",
        "video_fps",
        "video_frame_count",
        "duration_sec",
        "effective_frame_fps",
        "status",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main():
    args = parse_args()
    sample_ids = read_ejsl_names(Path(args.dial_list))
    if args.limit > 0:
        sample_ids = sample_ids[: args.limit]
    if not sample_ids:
        raise RuntimeError(f"No eJSL sample IDs loaded from {args.dial_list}")

    frame_root = Path(args.frame_root)
    video_root = Path(args.video_root)

    rows: List[Dict[str, object]] = []
    for sample_id in sample_ids:
        frame_dir = frame_root / sample_id
        frames = list_frames(frame_dir) if frame_dir.exists() else []
        video_path = find_video(video_root, sample_id)
        info = video_info(video_path)
        duration = info.get("duration_sec")
        effective_fps = (len(frames) / duration) if frames and duration and duration > 0 else None
        if not frame_dir.exists():
            status = "missing_frame_dir"
        elif not frames:
            status = "empty_frame_dir"
        elif video_path is None:
            status = "missing_video"
        elif not duration:
            status = "missing_duration"
        else:
            status = "ok"
        rows.append(
            {
                "sample_id": sample_id,
                "frame_dir": str(frame_dir),
                "frame_count": len(frames),
                "video_path": str(video_path) if video_path else "",
                "video_fps": info.get("video_fps"),
                "video_frame_count": info.get("video_frame_count"),
                "duration_sec": duration,
                "effective_frame_fps": effective_fps,
                "status": status,
            }
        )

    status_counts: Dict[str, int] = {}
    for row in rows:
        status_counts[str(row["status"])] = status_counts.get(str(row["status"]), 0) + 1

    summary = {
        "dial_list": args.dial_list,
        "frame_root": args.frame_root,
        "video_root": args.video_root,
        "samples": len(rows),
        "status_counts": status_counts,
        "frame_count": summarize(row["frame_count"] for row in rows),
        "video_fps": summarize(row.get("video_fps") for row in rows),
        "duration_sec": summarize(row.get("duration_sec") for row in rows),
        "effective_frame_fps": summarize(row.get("effective_frame_fps") for row in rows),
        "note": (
            "effective_frame_fps = extracted_frame_count / video_duration_sec. "
            "Existing visual feature scripts sample 8 frames uniformly by default, independent of this FPS."
        ),
    }

    if args.out_csv:
        write_csv(Path(args.out_csv), rows)
    if args.summary_json:
        out = Path(args.summary_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
