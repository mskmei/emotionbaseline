from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

from keypoints import extract_holistic_keypoints_from_video
from manifest_utils import first_nonempty, read_csv_rows, safe_sample_id, write_csv_rows


def parse_optional_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return float(text)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract MediaPipe Holistic keypoints for a video/text manifest.")
    parser.add_argument("--manifest_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--out_manifest_csv", type=str, required=True)
    parser.add_argument("--sample_id_col", type=str, default="sample_id")
    parser.add_argument("--video_col", type=str, default="video_path")
    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--sample_fps", type=float, default=10.0)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--model_complexity", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip_errors", action="store_true")
    return parser.parse_args()


def build_output_row(row: Dict[str, str], sample_id: str, keypoints_path: Path) -> Dict[str, object]:
    out = dict(row)
    out["sample_id"] = sample_id
    out["keypoints_path"] = str(keypoints_path)
    return out


def main():
    args = parse_args()
    rows = read_csv_rows(Path(args.manifest_csv))
    if not rows:
        raise RuntimeError(f"Empty manifest: {args.manifest_csv}")
    if args.video_col not in rows[0]:
        raise RuntimeError(f"Manifest missing video column {args.video_col!r}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_rows: List[Dict[str, object]] = []
    failures = []

    for i, row in enumerate(tqdm(rows, desc="mediapipe")):
        sample_raw = first_nonempty(row, [args.sample_id_col, "vid", "source_id"])
        if sample_raw is None:
            sample_raw = f"row_{i:08d}"
        sample_id = safe_sample_id(sample_raw)
        video_path = Path(str(row[args.video_col]))
        if not video_path.exists():
            msg = f"Missing video for {sample_id}: {video_path}"
            if args.skip_errors:
                failures.append(msg)
                continue
            raise FileNotFoundError(msg)

        keypoints_path = output_dir / f"{sample_id}.npz"
        if not (args.resume and keypoints_path.exists()):
            try:
                keypoints, timestamps = extract_holistic_keypoints_from_video(
                    video_path,
                    start=parse_optional_float(row.get("start")),
                    end=parse_optional_float(row.get("end")),
                    sample_fps=args.sample_fps,
                    max_frames=args.max_frames,
                    model_complexity=args.model_complexity,
                )
                np.savez_compressed(
                    keypoints_path,
                    keypoints=keypoints.astype(np.float32),
                    timestamps=timestamps.astype(np.float32),
                    video_path=str(video_path),
                    start=str(row.get("start", "")),
                    end=str(row.get("end", "")),
                    sample_id=sample_id,
                )
            except Exception as exc:
                msg = f"{sample_id}: {exc}"
                if args.skip_errors:
                    failures.append(msg)
                    continue
                raise

        out_rows.append(build_output_row(row, sample_id, keypoints_path))

    if not out_rows:
        raise RuntimeError(f"No keypoints extracted. failures={failures[:5]}")

    fieldnames = list(out_rows[0].keys())
    if "keypoints_path" not in fieldnames:
        fieldnames.append("keypoints_path")
    write_csv_rows(Path(args.out_manifest_csv), out_rows, fieldnames)
    print(f"[Keypoints] wrote manifest: {args.out_manifest_csv}")
    print(f"[Keypoints] rows={len(out_rows)} failures={len(failures)}")
    if failures:
        fail_path = Path(args.out_manifest_csv).with_suffix(".failures.txt")
        fail_path.write_text("\n".join(failures), encoding="utf-8")
        print(f"[Keypoints] failure details: {fail_path}")


if __name__ == "__main__":
    main()
