from __future__ import annotations

import argparse
import html
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from manifest_utils import read_csv_rows, row_key, safe_sample_id, write_csv_rows


TIME_RE = re.compile(
    r"(?P<start>(?:\d{2}:)?\d{2}:\d{2}[\.,]\d{3})\s+-->\s+"
    r"(?P<end>(?:\d{2}:)?\d{2}:\d{2}[\.,]\d{3})"
)
TAG_RE = re.compile(r"<[^>]+>")
INLINE_TIME_RE = re.compile(r"<\d{2}:\d{2}:\d{2}[\.,]\d{3}>")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a J-Shuwa training manifest from CC rows by downloading YouTube videos and Japanese subtitles."
    )
    parser.add_argument("--metadata_csv", type=str, required=True)
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--subtitle_dir", type=str, required=True)
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--source", type=str, default="cc", choices=["cc", "all"])
    parser.add_argument("--yt_dlp_bin", type=str, default="yt-dlp")
    parser.add_argument("--subtitle_langs", type=str, default="ja,ja-JP,jp")
    parser.add_argument(
        "--video_format",
        type=str,
        default="bestvideo[ext=mp4]/best[ext=mp4]/best",
    )
    parser.add_argument("--download_videos", action="store_true")
    parser.add_argument("--download_subtitles", action="store_true")
    parser.add_argument("--skip_missing", action="store_true")
    parser.add_argument("--max_yids", type=int, default=0)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--min_duration", type=float, default=0.1)
    parser.add_argument("--max_duration", type=float, default=60.0)
    parser.add_argument("--min_text_chars", type=int, default=1)
    return parser.parse_args()


def run_cmd(cmd: List[str]) -> bool:
    print("[cmd]", " ".join(cmd))
    result = subprocess.run(cmd)
    return result.returncode == 0


def youtube_url(yid: str) -> str:
    return f"https://www.youtube.com/watch?v={yid}"


def find_video(video_dir: Path, yid: str) -> Optional[Path]:
    for ext in (".mp4", ".mkv", ".webm", ".avi", ".mov", ".m4v"):
        path = video_dir / f"{yid}{ext}"
        if path.exists():
            return path
    matches = [p for p in video_dir.glob(f"{yid}.*") if p.is_file()]
    return matches[0] if matches else None


def ensure_video(args, yid: str) -> Optional[Path]:
    video_dir = Path(args.video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)
    existing = find_video(video_dir, yid)
    if existing is not None:
        return existing
    if not args.download_videos:
        return None

    ok = run_cmd(
        [
            args.yt_dlp_bin,
            "--no-playlist",
            "--continue",
            "-f",
            args.video_format,
            "-o",
            str(video_dir / "%(id)s.%(ext)s"),
            youtube_url(yid),
        ]
    )
    if not ok:
        return None
    return find_video(video_dir, yid)


def find_vtt_files(subtitle_dir: Path, yid: str) -> List[Path]:
    files = sorted(subtitle_dir.glob(f"{yid}*.vtt"))
    return [p for p in files if p.is_file()]


def ensure_subtitles(args, yid: str) -> List[Path]:
    subtitle_dir = Path(args.subtitle_dir)
    subtitle_dir.mkdir(parents=True, exist_ok=True)
    existing = find_vtt_files(subtitle_dir, yid)
    if existing:
        return existing
    if not args.download_subtitles:
        return []

    ok = run_cmd(
        [
            args.yt_dlp_bin,
            "--no-playlist",
            "--skip-download",
            "--write-subs",
            "--write-auto-subs",
            "--sub-langs",
            args.subtitle_langs,
            "--sub-format",
            "vtt",
            "-o",
            str(subtitle_dir / "%(id)s.%(ext)s"),
            youtube_url(yid),
        ]
    )
    if not ok:
        return []
    return find_vtt_files(subtitle_dir, yid)


def time_to_seconds(text: str) -> float:
    text = text.replace(",", ".")
    parts = text.split(":")
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h, m, s = "0", parts[0], parts[1]
    else:
        raise ValueError(f"Invalid VTT timestamp: {text}")
    return int(h) * 3600 + int(m) * 60 + float(s)


def clean_vtt_line(line: str) -> str:
    line = INLINE_TIME_RE.sub("", line)
    line = TAG_RE.sub("", line)
    line = html.unescape(line)
    return " ".join(line.split()).strip()


def parse_vtt(path: Path) -> List[Tuple[float, float, str]]:
    cues: List[Tuple[float, float, str]] = []
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        match = TIME_RE.search(line)
        if not match:
            i += 1
            continue
        start = time_to_seconds(match.group("start"))
        end = time_to_seconds(match.group("end"))
        i += 1
        text_lines = []
        while i < len(lines) and lines[i].strip():
            cleaned = clean_vtt_line(lines[i])
            if cleaned and not cleaned.isdigit():
                text_lines.append(cleaned)
            i += 1
        text = " ".join(text_lines).strip()
        if text:
            cues.append((start, end, text))
    return cues


def select_segment_text(cues: List[Tuple[float, float, str]], start: float, end: float) -> str:
    chunks = []
    for cue_start, cue_end, text in cues:
        overlap = max(0.0, min(end, cue_end) - max(start, cue_start))
        mid = (cue_start + cue_end) / 2.0
        if overlap > 0.05 or start <= mid <= end:
            chunks.append(text)

    deduped = []
    for text in chunks:
        if text and (not deduped or deduped[-1] != text):
            deduped.append(text)
    return " ".join(deduped).strip()


def main():
    args = parse_args()
    metadata = read_csv_rows(Path(args.metadata_csv))
    if not metadata:
        raise RuntimeError(f"Empty metadata csv: {args.metadata_csv}")

    if args.source == "cc":
        metadata = [row for row in metadata if str(row.get("source", "")).strip() == "cc"]
    metadata = sorted(metadata, key=lambda row: (str(row["yid"]), float(row["start"]), float(row["end"])))

    yids = []
    seen = set()
    for row in metadata:
        yid = str(row["yid"])
        if yid not in seen:
            yids.append(yid)
            seen.add(yid)
        if args.max_yids > 0 and len(yids) >= args.max_yids:
            break
    selected_yids = set(yids)
    metadata = [row for row in metadata if str(row["yid"]) in selected_yids]

    video_by_yid: Dict[str, Path] = {}
    cues_by_yid: Dict[str, List[Tuple[float, float, str]]] = {}
    stats = {
        "metadata_rows": len(metadata),
        "kept": 0,
        "bad_duration": 0,
        "missing_video": 0,
        "missing_subtitle": 0,
        "missing_text": 0,
    }

    for yid in yids:
        video = ensure_video(args, yid)
        if video is not None:
            video_by_yid[yid] = video
        subtitles = ensure_subtitles(args, yid)
        all_cues = []
        for sub_path in subtitles:
            all_cues.extend(parse_vtt(sub_path))
        if all_cues:
            cues_by_yid[yid] = sorted(all_cues, key=lambda x: (x[0], x[1], x[2]))

    rows = []
    for row in metadata:
        yid = str(row["yid"])
        start = float(row["start"])
        end = float(row["end"])
        duration = end - start
        if duration < args.min_duration or duration > args.max_duration:
            stats["bad_duration"] += 1
            continue
        video = video_by_yid.get(yid)
        if video is None:
            stats["missing_video"] += 1
            if args.skip_missing:
                continue
            raise RuntimeError(f"Missing local/downloaded video for yid={yid}")
        cues = cues_by_yid.get(yid)
        if not cues:
            stats["missing_subtitle"] += 1
            if args.skip_missing:
                continue
            raise RuntimeError(f"Missing Japanese subtitle VTT for yid={yid}")
        text = select_segment_text(cues, start, end)
        if len(text) < args.min_text_chars:
            stats["missing_text"] += 1
            if args.skip_missing:
                continue
            raise RuntimeError(f"No subtitle text aligned to {row_key(yid, start, end)}")

        sample_id = safe_sample_id(str(row.get("vid") or row_key(yid, start, end).replace("|", "_")))
        rows.append(
            {
                "sample_id": sample_id,
                "source_id": str(row.get("vid", "")),
                "yid": yid,
                "start": f"{start:.3f}",
                "end": f"{end:.3f}",
                "source": str(row.get("source", "")),
                "video_path": str(video),
                "text": text,
                "split": "train",
            }
        )
        stats["kept"] += 1
        if args.max_rows > 0 and len(rows) >= args.max_rows:
            break

    if not rows:
        raise RuntimeError(f"No trainable CC rows were built. stats={stats}")

    write_csv_rows(
        Path(args.out_csv),
        rows,
        ["sample_id", "source_id", "yid", "start", "end", "source", "video_path", "text", "split"],
    )
    print(f"[J-Shuwa CC] wrote {len(rows)} rows to {args.out_csv}")
    print(f"[J-Shuwa CC] stats={stats}")


if __name__ == "__main__":
    main()
