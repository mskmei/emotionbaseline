from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List

from manifest_utils import find_video_files, first_nonempty, read_csv_rows, read_jsonl, row_key, safe_sample_id, write_csv_rows


REQUIRED_METADATA_COLUMNS = {"vid", "yid", "start", "end", "source"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Join J-Shuwa HF metadata with local videos and externally prepared Japanese subtitle text. "
            "The HF release itself has no video/text payload."
        )
    )
    parser.add_argument("--metadata_csv", type=str, default="", help="Optional local export of J-Shuwa metadata.")
    parser.add_argument("--hf_dataset", type=str, default="mouwjone/J-Shuwa")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--hf_token_env", type=str, default="HF_TOKEN")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing local full YouTube videos named by yid.")
    parser.add_argument(
        "--subtitle_text",
        type=str,
        required=True,
        help="JSONL/CSV with yid,start,end and text/subtitle_text, or key plus subtitle_text.",
    )
    parser.add_argument("--source", type=str, default="all", choices=["all", "cc", "hardsub"])
    parser.add_argument("--min_duration", type=float, default=0.1)
    parser.add_argument("--max_duration", type=float, default=60.0)
    parser.add_argument("--min_text_chars", type=int, default=1)
    parser.add_argument("--out_csv", type=str, required=True)
    return parser.parse_args()


def load_metadata(args) -> List[Dict[str, object]]:
    if args.metadata_csv:
        rows = read_csv_rows(Path(args.metadata_csv))
    else:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError("Install datasets or pass --metadata_csv.") from exc

        token = os.getenv(args.hf_token_env) if args.hf_token_env else None
        kwargs = {"split": args.split}
        if token:
            kwargs["token"] = token
        data = load_dataset(args.hf_dataset, **kwargs)
        rows = [dict(x) for x in data]

    if not rows:
        raise RuntimeError("No J-Shuwa metadata rows loaded")
    missing = REQUIRED_METADATA_COLUMNS - set(rows[0].keys())
    if missing:
        raise RuntimeError(f"J-Shuwa metadata is missing required columns: {sorted(missing)}")
    return rows


def load_text_map(path: Path) -> Dict[str, str]:
    if path.suffix.lower() == ".jsonl":
        rows = read_jsonl(path)
    else:
        rows = read_csv_rows(path)

    out: Dict[str, str] = {}
    for row in rows:
        text = first_nonempty(row, ["subtitle_text", "text", "translation", "utterance", "caption"])
        if not text:
            continue
        key = first_nonempty(row, ["key"])
        if not key:
            yid = first_nonempty(row, ["yid", "youtube_id"])
            start = first_nonempty(row, ["start", "start_time"])
            end = first_nonempty(row, ["end", "end_time"])
            if yid is None or start is None or end is None:
                continue
            key = row_key(yid, start, end)
        out[key] = text.strip()
    if not out:
        raise RuntimeError(f"No usable subtitle text rows found in {path}")
    return out


def main():
    args = parse_args()
    metadata = load_metadata(args)
    text_map = load_text_map(Path(args.subtitle_text))
    video_files = find_video_files(Path(args.video_dir))
    if not video_files:
        raise RuntimeError(f"No local video files found under {args.video_dir}")

    rows = []
    stats = {"metadata": len(metadata), "kept": 0, "missing_video": 0, "missing_text": 0, "bad_duration": 0}
    for row in metadata:
        source = str(row["source"])
        if args.source != "all" and source != args.source:
            continue
        start = float(row["start"])
        end = float(row["end"])
        duration = end - start
        if duration < args.min_duration or duration > args.max_duration:
            stats["bad_duration"] += 1
            continue

        yid = str(row["yid"])
        video_path = video_files.get(yid)
        if video_path is None:
            stats["missing_video"] += 1
            continue

        key = row_key(yid, start, end)
        text = text_map.get(key)
        if text is None or len(text) < args.min_text_chars:
            stats["missing_text"] += 1
            continue

        sample_id = safe_sample_id(str(row.get("vid") or key.replace("|", "_")))
        rows.append(
            {
                "sample_id": sample_id,
                "source_id": str(row.get("vid", "")),
                "yid": yid,
                "start": f"{start:.3f}",
                "end": f"{end:.3f}",
                "source": source,
                "video_path": str(video_path),
                "text": text,
                "split": "train",
            }
        )
        stats["kept"] += 1

    if not rows:
        raise RuntimeError(f"No trainable rows after joining metadata/video/text. stats={stats}")

    write_csv_rows(
        Path(args.out_csv),
        rows,
        ["sample_id", "source_id", "yid", "start", "end", "source", "video_path", "text", "split"],
    )
    print(f"[J-Shuwa] wrote {len(rows)} rows to {args.out_csv}")
    print(f"[J-Shuwa] stats={stats}")


if __name__ == "__main__":
    main()
