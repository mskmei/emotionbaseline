from __future__ import annotations

import argparse
import base64
import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
from tqdm import tqdm

from build_jshuwa_cc_manifest import ensure_video
from manifest_utils import append_jsonl, read_csv_rows, row_key, safe_sample_id, write_csv_rows


PROMPT = """
Analyze the target image and extract the main hard-coded Japanese subtitle text visible in it.
The surrounding images are only temporal context for deciding the target subtitle.
Ignore furigana, small annotations, background text, logos, watermarks, UI text, and signboards unless they are clearly part of the main subtitle.
If multiple subtitle fragments belong to one sentence, combine them in natural Japanese reading order.
Output only compact JSON in this exact format:
{"subtitle_text":"...","language":"ja"}
If no Japanese subtitle is visible in the target image, output:
{"subtitle_text":null,"language":null}
""".strip()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract J-Shuwa hardsub subtitle text with an OpenAI-compatible VLM endpoint."
    )
    parser.add_argument("--metadata_csv", type=str, required=True)
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--out_jsonl", type=str, required=True)
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--source", type=str, default="hardsub", choices=["hardsub"])

    parser.add_argument("--api_base", type=str, default=os.getenv("VLLM_API_BASE", "http://localhost:8000/v1"))
    parser.add_argument("--api_key", type=str, default=os.getenv("VLLM_API_KEY", os.getenv("OPENAI_API_KEY", "EMPTY")))
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-32B-Thinking")
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--request_retries", type=int, default=3)
    parser.add_argument("--retry_sleep", type=float, default=5.0)

    parser.add_argument("--context_segments", type=int, default=1)
    parser.add_argument("--max_image_side", type=int, default=1280)
    parser.add_argument("--jpeg_quality", type=int, default=90)
    parser.add_argument("--save_frames_dir", type=str, default="")
    parser.add_argument("--save_frames_every", type=int, default=0)

    parser.add_argument("--yt_dlp_bin", type=str, default="yt-dlp")
    parser.add_argument(
        "--video_format",
        type=str,
        default="18/best[height<=360][ext=mp4]/best[height<=480][ext=mp4]/best",
    )
    parser.add_argument("--extractor_args", type=str, default="youtube:player_client=android_vr")
    parser.add_argument(
        "--extractor_args_candidates",
        type=str,
        default="youtube:player_client=android_vr;youtube:player_client=android;youtube:player_client=ios;youtube:player_client=web;youtube:player_client=mweb",
    )
    parser.add_argument("--cookies", type=str, default="")
    parser.add_argument("--cookies_from_browser", type=str, default="")
    parser.add_argument("--sleep_interval", type=float, default=2.0)
    parser.add_argument("--max_sleep_interval", type=float, default=8.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--fragment_retries", type=int, default=5)
    parser.add_argument("--socket_timeout", type=float, default=30.0)
    parser.add_argument("--use_ytdlp_config", action="store_true")
    parser.add_argument("--verbose_ytdlp", action="store_true")
    parser.add_argument("--download_videos", action="store_true")
    parser.add_argument("--skip_missing", action="store_true")
    parser.add_argument("--skip_errors", action="store_true")

    parser.add_argument("--min_duration", type=float, default=0.1)
    parser.add_argument("--max_duration", type=float, default=60.0)
    parser.add_argument("--min_text_chars", type=int, default=1)
    parser.add_argument("--max_yids", type=int, default=0)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--build_csv_only", action="store_true")
    return parser.parse_args()


def parse_float(value: object) -> float:
    try:
        return float(str(value).strip())
    except Exception:
        return float("nan")


def clean_response_text(text: str) -> str:
    text = str(text or "").strip()
    end = text.rfind("</think>")
    if end >= 0:
        text = text[end + len("</think>") :].strip()
    text = text.replace("<think>", "").replace("</think>", "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    return text


def parse_model_json(text: str) -> Dict[str, object]:
    cleaned = clean_response_text(text)
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise ValueError(f"model response does not contain JSON: {cleaned[:200]}")
        obj = json.loads(match.group(0))
    if not isinstance(obj, dict):
        raise ValueError("model JSON response is not an object")
    return obj


def resize_frame(frame, max_side: int):
    if max_side <= 0:
        return frame
    height, width = frame.shape[:2]
    side = max(height, width)
    if side <= max_side:
        return frame
    scale = float(max_side) / float(side)
    return cv2.resize(frame, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)


def read_frame_at(video_path: Path, timestamp: float, max_side: int, jpeg_quality: int) -> Tuple[bytes, float, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            fps = 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_idx = max(0, int(round(float(timestamp) * fps)))
        if frame_count > 0:
            frame_idx = min(frame_idx, max(frame_count - 1, 0))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            raise RuntimeError(f"failed to read frame {frame_idx} at {timestamp:.3f}s from {video_path}")
        frame_bgr = resize_frame(frame_bgr, max_side=max_side)
        ok, encoded = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        if not ok:
            raise RuntimeError("failed to encode frame as JPEG")
        return encoded.tobytes(), frame_idx / fps, frame_idx
    finally:
        cap.release()


def data_url(jpeg_bytes: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes).decode("ascii")


def save_debug_frame(path: Path, jpeg_bytes: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(jpeg_bytes)


def api_endpoint(api_base: str) -> str:
    return api_base.rstrip("/") + "/chat/completions"


def request_vlm(args, target_url: str, context_urls: List[str]) -> str:
    content: List[Dict[str, object]] = []
    if context_urls:
        content.append({"type": "text", "text": "Surrounding frames for temporal context:"})
        for url in context_urls:
            content.append({"type": "image_url", "image_url": {"url": url}})
    content.append({"type": "text", "text": PROMPT})
    content.append({"type": "image_url", "image_url": {"url": target_url}})

    payload: Dict[str, object] = {
        "model": args.model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": int(args.max_tokens),
        "temperature": float(args.temperature),
    }
    raw = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key or 'EMPTY'}",
    }
    request = urllib.request.Request(api_endpoint(args.api_base), data=raw, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=float(args.timeout)) as response:
        data = json.loads(response.read().decode("utf-8"))
    return str(data["choices"][0]["message"]["content"])


def request_vlm_with_retries(args, target_url: str, context_urls: List[str]) -> str:
    last_error: Optional[BaseException] = None
    for attempt in range(max(int(args.request_retries), 1)):
        try:
            return request_vlm(args, target_url=target_url, context_urls=context_urls)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt + 1 < max(int(args.request_retries), 1):
                time.sleep(float(args.retry_sleep) * (attempt + 1))
    raise RuntimeError(f"VLM request failed after {args.request_retries} attempts: {last_error}") from last_error


def successful_keys(jsonl_path: Path) -> set:
    done = set()
    if not jsonl_path.exists():
        return done
    with jsonl_path.open("r", encoding="utf-8") as f:
        for raw in f:
            try:
                item = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if item.get("error") is None and item.get("subtitle_text"):
                done.add(str(item.get("key", "")))
    return done


def iter_jsonl(path: Path) -> Iterable[Dict[str, object]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError:
                continue
            yield item


def write_manifest_from_jsonl(jsonl_path: Path, out_csv: Path, min_text_chars: int) -> int:
    by_key: Dict[str, Dict[str, object]] = {}
    for item in iter_jsonl(jsonl_path):
        if item.get("error") is not None:
            continue
        text = str(item.get("subtitle_text") or "").strip()
        if len(text) < int(min_text_chars):
            continue
        key = str(item.get("key") or row_key(str(item["yid"]), item["start"], item["end"]))
        by_key[key] = item

    rows = []
    for item in by_key.values():
        rows.append(
            {
                "sample_id": safe_sample_id(str(item.get("sample_id") or item.get("source_id") or item["key"])),
                "source_id": str(item.get("source_id", "")),
                "yid": str(item.get("yid", "")),
                "start": f"{float(item.get('start')):.3f}",
                "end": f"{float(item.get('end')):.3f}",
                "source": "hardsub",
                "video_path": str(item.get("video_path", "")),
                "text": str(item.get("subtitle_text") or "").strip(),
                "split": "train",
            }
        )
    rows.sort(key=lambda row: (row["yid"], float(row["start"]), float(row["end"])))
    write_csv_rows(
        out_csv,
        rows,
        ["sample_id", "source_id", "yid", "start", "end", "source", "video_path", "text", "split"],
    )
    return len(rows)


def select_rows(args) -> List[Dict[str, str]]:
    rows = read_csv_rows(Path(args.metadata_csv))
    rows = [row for row in rows if str(row.get("source", "")).strip() == args.source]
    filtered = []
    for row in rows:
        start = parse_float(row.get("start"))
        end = parse_float(row.get("end"))
        duration = end - start
        if not (duration >= args.min_duration and duration <= args.max_duration):
            continue
        filtered.append(row)
    filtered.sort(key=lambda row: (str(row["yid"]), float(row["start"]), float(row["end"])))

    if args.max_yids > 0:
        selected_yids = []
        seen = set()
        for row in filtered:
            yid = str(row["yid"])
            if yid not in seen:
                selected_yids.append(yid)
                seen.add(yid)
            if len(selected_yids) >= args.max_yids:
                break
        allowed = set(selected_yids)
        filtered = [row for row in filtered if str(row["yid"]) in allowed]

    if args.max_rows > 0:
        filtered = filtered[: int(args.max_rows)]
    return filtered


def grouped_indices(rows: List[Dict[str, str]]) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for idx, row in enumerate(rows):
        out.setdefault(str(row["yid"]), []).append(idx)
    return out


def context_indices(indices: List[int], local_pos: int, n_context: int) -> List[int]:
    out = []
    for offset in range(int(n_context), 0, -1):
        pos = local_pos - offset
        if pos >= 0:
            out.append(indices[pos])
    for offset in range(1, int(n_context) + 1):
        pos = local_pos + offset
        if pos < len(indices):
            out.append(indices[pos])
    return out


def midpoint(row: Dict[str, str]) -> float:
    return (float(row["start"]) + float(row["end"])) / 2.0


def main():
    args = parse_args()
    out_jsonl = Path(args.out_jsonl)
    out_csv = Path(args.out_csv)

    if args.build_csv_only:
        n = write_manifest_from_jsonl(out_jsonl, out_csv, min_text_chars=args.min_text_chars)
        print(f"[J-Shuwa-hardsub] wrote manifest rows={n} out={out_csv}")
        return

    rows = select_rows(args)
    if not rows:
        raise RuntimeError("No hardsub metadata rows selected.")

    done = successful_keys(out_jsonl) if args.resume else set()
    groups = grouped_indices(rows)
    yid_pos = {
        idx: (indices, local_pos)
        for indices in groups.values()
        for local_pos, idx in enumerate(indices)
    }

    stats = {"selected": len(rows), "skipped_done": 0, "missing_video": 0, "ok": 0, "failed": 0}
    for idx, row in enumerate(tqdm(rows, desc="hardsub-vlm")):
        yid = str(row["yid"])
        start = float(row["start"])
        end = float(row["end"])
        key = row_key(yid, start, end)
        if key in done:
            stats["skipped_done"] += 1
            continue

        video = ensure_video(args, yid)
        if video is None:
            stats["missing_video"] += 1
            if not args.skip_missing:
                raise RuntimeError(f"Missing local/downloaded video for yid={yid}")
            continue

        sample_id = safe_sample_id(str(row.get("vid") or key.replace("|", "_")))
        item: Dict[str, object] = {
            "key": key,
            "sample_id": sample_id,
            "source_id": str(row.get("vid", "")),
            "yid": yid,
            "start": start,
            "end": end,
            "source": "hardsub",
            "video_path": str(video),
            "target_time": midpoint(row),
            "context_segments": [],
            "subtitle_text": None,
            "language": None,
            "raw_response": None,
            "error": None,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        try:
            target_jpeg, actual_time, frame_idx = read_frame_at(
                video,
                item["target_time"],
                max_side=args.max_image_side,
                jpeg_quality=args.jpeg_quality,
            )
            item["target_actual_time"] = actual_time
            item["target_frame_idx"] = frame_idx

            indices, local_pos = yid_pos[idx]
            context_urls: List[str] = []
            context_meta = []
            for cidx in context_indices(indices, local_pos, args.context_segments):
                ctx = rows[cidx]
                ctx_jpeg, ctx_time, ctx_frame_idx = read_frame_at(
                    video,
                    midpoint(ctx),
                    max_side=args.max_image_side,
                    jpeg_quality=args.jpeg_quality,
                )
                context_urls.append(data_url(ctx_jpeg))
                context_meta.append(
                    {
                        "key": row_key(str(ctx["yid"]), ctx["start"], ctx["end"]),
                        "start": float(ctx["start"]),
                        "end": float(ctx["end"]),
                        "actual_time": ctx_time,
                        "frame_idx": ctx_frame_idx,
                    }
                )
            item["context_segments"] = context_meta

            if args.save_frames_dir and args.save_frames_every > 0 and stats["ok"] % args.save_frames_every == 0:
                save_debug_frame(Path(args.save_frames_dir) / f"{sample_id}_target.jpg", target_jpeg)

            raw_response = request_vlm_with_retries(args, target_url=data_url(target_jpeg), context_urls=context_urls)
            parsed = parse_model_json(raw_response)
            text = parsed.get("subtitle_text")
            item["subtitle_text"] = str(text).strip() if text is not None else None
            item["language"] = parsed.get("language")
            item["raw_response"] = raw_response
            stats["ok"] += 1
        except Exception as exc:
            item["error"] = repr(exc)
            stats["failed"] += 1
            if not args.skip_errors:
                append_jsonl(out_jsonl, item)
                raise
        append_jsonl(out_jsonl, item)

    n = write_manifest_from_jsonl(out_jsonl, out_csv, min_text_chars=args.min_text_chars)
    print(f"[J-Shuwa-hardsub] stats={stats}")
    print(f"[J-Shuwa-hardsub] jsonl={out_jsonl}")
    print(f"[J-Shuwa-hardsub] manifest rows={n} csv={out_csv}")


if __name__ == "__main__":
    main()
