#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
ANJS_LABELS = ["A", "N", "J", "S"]
IEMOCAP_TO_ANJS = {"ang": 0, "fru": 0, "neu": 1, "hap": 2, "exc": 2, "sad": 3}


REQUIRED_COLUMNS = {
    "Utterance",
    "Speaker",
    "Dialogue_ID",
    "Start_Time",
    "End_Time",
    "Video_Path",
    "Emotion",
    "Utterance_ID",
}


def resolve_iemocap_video_path(raw_video_path: str, iemocap_root: Path) -> Path:
    raw = Path(str(raw_video_path or "").strip())
    candidates: List[Path] = [raw] if raw.is_absolute() else []
    for idx, part in enumerate(raw.parts):
        if re.fullmatch(r"Session[1-5]", part):
            candidates.append(iemocap_root.joinpath(*raw.parts[idx:]))
            break
    candidates.extend([iemocap_root / raw, iemocap_root / raw.name])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0] if candidates else iemocap_root / raw.name


def import_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("opencv-python is required: pip install opencv-python") from exc
    return cv2


def read_rows(metadata_root: Path, iemocap_root: Path, frame_root: Path) -> Dict[str, List[Dict[str, str]]]:
    output: Dict[str, List[Dict[str, str]]] = {}
    for split in ["train", "dev", "test"]:
        csv_path = metadata_root / f"IEMOCAP_{split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing metadata CSV: {csv_path}")
        with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            missing = REQUIRED_COLUMNS.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Missing columns in {csv_path}: {sorted(missing)}")
            rows: List[Dict[str, str]] = []
            for source_row in reader:
                row = {key: str(value or "").strip() for key, value in source_row.items()}
                emotion = row["Emotion"].lower()
                if emotion not in IEMOCAP_TO_ANJS:
                    raise ValueError(f"Unsupported emotion {emotion!r} in {csv_path}")
                sample_id = row["Speaker"]
                resolved_video = resolve_iemocap_video_path(row["Video_Path"], iemocap_root)
                row.update(
                    {
                        "Split": split,
                        "Sample_ID": sample_id,
                        "Source_Emotion": emotion,
                        "ANJS_Label": ANJS_LABELS[IEMOCAP_TO_ANJS[emotion]],
                        "Resolved_Video_Path": str(resolved_video),
                        "Frame_Dir": str(frame_root / split / sample_id),
                    }
                )
                rows.append(row)
        output[split] = rows
    return output


def inspect_video(path: Path) -> Tuple[float, int, float]:
    cv2 = import_cv2()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
    cap.release()
    return fps, frame_count, duration


def expected_frame_paths(frame_dir: Path, num_frames: int) -> List[Path]:
    return [frame_dir / f"{index:04d}.jpg" for index in range(num_frames)]


def frames_complete(frame_dir: Path, num_frames: int) -> bool:
    return all(path.is_file() and path.stat().st_size > 0 for path in expected_frame_paths(frame_dir, num_frames))


def extract_segment_frames(
    video_path: Path,
    frame_dir: Path,
    start_time: float,
    end_time: float,
    num_frames: int,
    margin_seconds: float,
    jpeg_quality: int,
) -> Dict[str, float]:
    cv2 = import_cv2()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
    clip_start = max(float(start_time) - margin_seconds, 0.0)
    clip_end = max(float(end_time) + margin_seconds, clip_start)
    if duration > 0:
        clip_end = min(clip_end, max(duration - (1.0 / fps if fps > 0 else 0.0), clip_start))
    timestamps = np.linspace(clip_start, clip_end, num=num_frames, dtype=np.float64)

    frame_dir.mkdir(parents=True, exist_ok=True)
    last_image = None
    for index, timestamp in enumerate(timestamps):
        cap.set(cv2.CAP_PROP_POS_MSEC, float(timestamp) * 1000.0)
        ok, image = cap.read()
        if (not ok or image is None) and fps > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(frame_count - 1, int(round(timestamp * fps)))))
            ok, image = cap.read()
        if not ok or image is None:
            if last_image is None:
                cap.release()
                raise RuntimeError(f"Could not decode frame at {timestamp:.3f}s from {video_path}")
            image = last_image.copy()
        last_image = image
        destination = frame_dir / f"{index:04d}.jpg"
        temporary = frame_dir / f".{index:04d}.tmp.jpg"
        if not cv2.imwrite(str(temporary), image, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]):
            cap.release()
            raise RuntimeError(f"Could not write frame: {temporary}")
        temporary.replace(destination)
    cap.release()
    return {"fps": fps, "frame_count": frame_count, "duration": duration, "clip_start": clip_start, "clip_end": clip_end}


def write_manifest(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def process(args: argparse.Namespace) -> Dict:
    iemocap_root = Path(args.iemocap_root).expanduser().resolve()
    metadata_root = Path(args.metadata_root).expanduser().resolve()
    frame_root = Path(args.frame_root).expanduser().resolve()
    manifest_root = Path(args.manifest_root).expanduser().resolve()
    if not iemocap_root.is_dir():
        raise FileNotFoundError(f"IEMOCAP root does not exist: {iemocap_root}")
    for session in range(1, 6):
        session_root = iemocap_root / f"Session{session}"
        if not session_root.is_dir():
            raise FileNotFoundError(f"Missing IEMOCAP session directory: {session_root}")

    rows_by_split = read_rows(metadata_root, iemocap_root, frame_root)
    failures: List[Dict[str, str]] = []
    split_summaries: Dict[str, Dict] = {}
    processed_total = 0

    for split, all_rows in rows_by_split.items():
        rows = all_rows[: args.max_rows_per_split] if args.max_rows_per_split > 0 else all_rows
        label_counts = Counter(row["ANJS_Label"] for row in rows)
        source_counts = Counter(row["Source_Emotion"] for row in rows)
        dialogue_count = len({row["Dialogue_ID"] for row in rows})
        missing_video = 0
        complete = 0
        extracted = 0
        probe_stats: List[Dict[str, float]] = []
        manifest_rows: List[Dict[str, str]] = []
        for row_index, row in enumerate(rows, start=1):
            video_path = Path(row["Resolved_Video_Path"])
            frame_dir = Path(row["Frame_Dir"])
            status = "existing" if frames_complete(frame_dir, args.num_frames) else "pending"
            if not video_path.is_file():
                missing_video += 1
                status = "missing_video"
                failures.append({"split": split, "sample_id": row["Sample_ID"], "error": f"missing video: {video_path}"})
            elif status == "existing":
                complete += 1
            elif not args.check_only:
                try:
                    stats = extract_segment_frames(
                        video_path,
                        frame_dir,
                        float(row["Start_Time"]),
                        float(row["End_Time"]),
                        args.num_frames,
                        args.margin_seconds,
                        args.jpeg_quality,
                    )
                    if not frames_complete(frame_dir, args.num_frames):
                        raise RuntimeError("frame set is incomplete after extraction")
                    status = "extracted"
                    extracted += 1
                    complete += 1
                    if len(probe_stats) < args.probe_videos:
                        probe_stats.append(stats)
                except Exception as exc:  # keep a complete failure manifest
                    status = "error"
                    failures.append({"split": split, "sample_id": row["Sample_ID"], "error": str(exc)})
            elif len(probe_stats) < args.probe_videos:
                try:
                    fps, frame_count, duration = inspect_video(video_path)
                    probe_stats.append({"fps": fps, "frame_count": frame_count, "duration": duration})
                except Exception as exc:
                    status = "probe_error"
                    failures.append({"split": split, "sample_id": row["Sample_ID"], "error": str(exc)})

            manifest_row = dict(row)
            manifest_row["Frame_Status"] = status
            manifest_rows.append(manifest_row)
            processed_total += 1
            if row_index == 1 or row_index % 250 == 0 or row_index == len(rows):
                print(
                    f"[IEMOCAP-FRAMES] split={split} {row_index}/{len(rows)} "
                    f"complete={complete} extracted={extracted} missing_video={missing_video} failures={len(failures)}"
                )

        write_manifest(manifest_root / f"iemocap_{split}_anjs4_frames.csv", manifest_rows)
        split_summaries[split] = {
            "rows": len(rows),
            "dialogues": dialogue_count,
            "source_emotion_counts": dict(source_counts),
            "anjs4_label_counts": {label: int(label_counts.get(label, 0)) for label in ANJS_LABELS},
            "complete_frames": complete,
            "newly_extracted": extracted,
            "missing_video": missing_video,
            "probe_stats": probe_stats,
        }

    expected = sum(len(rows[: args.max_rows_per_split] if args.max_rows_per_split > 0 else rows) for rows in rows_by_split.values())
    complete = sum(summary["complete_frames"] for summary in split_summaries.values())
    report = {
        "iemocap_root": str(iemocap_root),
        "metadata_root": str(metadata_root),
        "frame_root": str(frame_root),
        "manifest_root": str(manifest_root),
        "check_only": bool(args.check_only),
        "num_frames": int(args.num_frames),
        "margin_seconds": float(args.margin_seconds),
        "expected_utterances": expected,
        "complete_utterances": complete,
        "all_frames_complete": complete == expected,
        "splits": split_summaries,
        "failures": failures[:100],
        "failure_count": len(failures),
    }
    manifest_root.mkdir(parents=True, exist_ok=True)
    report_path = manifest_root / "iemocap_anjs4_frame_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[IEMOCAP-FRAMES] report={report_path}")
    if failures and args.strict:
        raise RuntimeError(f"IEMOCAP preparation had {len(failures)} failures; see {report_path}")
    if not args.check_only and complete != expected:
        raise RuntimeError(f"Only {complete}/{expected} utterances have complete frame sets; see {report_path}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate IEMOCAP metadata and extract 8 uniform frames per utterance segment.")
    parser.add_argument("--iemocap_root", default="/raid_zoe/home/lr/maokeyu/IEMOCAP_full_release")
    parser.add_argument("--metadata_root", default=str(REPO_ROOT / "new" / "MAGTKD" / "datasets" / "IEMOCAP"))
    parser.add_argument("--frame_root", default="/raid_zoe/home/lr/maokeyu/sign/iemocap_bobsl_ejsl/frames_8_margin1")
    parser.add_argument("--manifest_root", default="/raid_zoe/home/lr/maokeyu/sign/iemocap_bobsl_ejsl/manifests")
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--margin_seconds", type=float, default=1.0)
    parser.add_argument("--jpeg_quality", type=int, default=95)
    parser.add_argument("--probe_videos", type=int, default=10)
    parser.add_argument("--max_rows_per_split", type=int, default=0)
    parser.add_argument("--check_only", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    process(parse_args())
