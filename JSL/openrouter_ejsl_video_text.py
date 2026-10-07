#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from manifest_utils import (
    append_jsonl,
    parse_ejsl_sample_id,
    read_dialogue_structure,
    read_ejsl_names,
    sanitize_generated_text,
)


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODELS = ["google/gemini-3.5-flash", "google/gemini-3-pro-preview"]
PLAIN_TRANSLATION_PROMPT = """Translate the Japanese Sign Language in this video into natural Japanese.

Output only the linguistic content expressed in the signing.

Do not describe facial expressions, emotions, gestures, appearance, or the scene.
Do not infer content from context or emotion.
Do not add information that is not explicitly conveyed by the signing.

Output only the Japanese translation."""
LABELS = ["A", "N", "J", "S"]
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")
VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a small non-oracle eJSL video-to-text sample with OpenRouter "
            "multimodal Gemini models and record exact generation costs."
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
    parser.add_argument(
        "--out_root",
        type=str,
        default="/raid_zoe/home/lr/maokeyu/sign/jsl_nonoracle/openrouter_ejsl_100",
    )
    parser.add_argument("--model", action="append", default=[], help="OpenRouter model slug. Repeatable.")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--selection", type=str, default="balanced", choices=["balanced", "first"])
    parser.add_argument("--input_mode", type=str, default="frames", choices=["frames", "video"])
    parser.add_argument("--num_frames", type=int, default=32, help="Maximum frames sampled across the full utterance.")
    parser.add_argument(
        "--sample_fps",
        type=float,
        default=0.0,
        help="When positive, sample frames across the full utterance at this FPS, then uniformly cap to --num_frames.",
    )
    parser.add_argument(
        "--frame_dir_fps",
        type=float,
        default=4.0,
        help="Assumed FPS for pre-extracted eJSL frame directories when --sample_fps is used.",
    )
    parser.add_argument("--image_max_side", type=int, default=640)
    parser.add_argument("--jpeg_quality", type=int, default=78)
    parser.add_argument("--max_video_mb", type=float, default=18.0)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=96)
    parser.add_argument("--reasoning_effort", type=str, default="minimal", help="Set empty string to omit.")
    parser.add_argument("--response_mode", choices=["json", "plain_translation"], default="json")
    parser.add_argument("--api_key_env", type=str, default="OPENROUTER_API_KEY")
    parser.add_argument(
        "--structure_txt_root",
        type=str,
        default="/raid_elmo/home/lr/wangyi/PTR/STUDIES-Japanese/Short_dialogue",
        help="Oracle txt root used only for speaker/dialogue/emotion structure, never sent to OpenRouter.",
    )
    parser.add_argument(
        "--output_txt_root",
        type=str,
        default="",
        help="Optional root for replacement eJSL txt trees. With multiple models, each model gets a subdirectory.",
    )
    parser.add_argument("--write_txt_tree", action="store_true")
    parser.add_argument("--sleep_sec", type=float, default=0.2)
    parser.add_argument("--generation_poll_retries", type=int, default=3)
    parser.add_argument("--generation_poll_sleep", type=float, default=2.0)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--fail_fast", action="store_true")
    parser.add_argument("--no_resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", value).strip("_")


def choose_sample_ids(sample_ids: List[str], limit: int, selection: str) -> List[str]:
    if limit <= 0 or limit >= len(sample_ids):
        return sample_ids
    if selection == "first":
        return sample_ids[:limit]

    indexed = list(enumerate(sample_ids))
    by_label: Dict[str, List[Tuple[int, str]]] = {label: [] for label in LABELS}
    leftovers: List[Tuple[int, str]] = []
    for item in indexed:
        _idx, sample_id = item
        label = sample_id[-1] if sample_id else ""
        if label in by_label:
            by_label[label].append(item)
        else:
            leftovers.append(item)

    base = limit // len(LABELS)
    remainder = limit % len(LABELS)
    selected: List[Tuple[int, str]] = []
    for i, label in enumerate(LABELS):
        quota = base + (1 if i < remainder else 0)
        selected.extend(by_label[label][:quota])

    selected_keys = {sample_id for _idx, sample_id in selected}
    if len(selected) < limit:
        remaining = [item for item in indexed if item[1] not in selected_keys]
        selected.extend(remaining[: limit - len(selected)])

    return [sample_id for _idx, sample_id in sorted(selected[:limit], key=lambda x: x[0])]


def list_frame_paths(frame_dir: Path) -> List[Path]:
    paths: List[Path] = []
    for ext in IMAGE_EXTS:
        paths.extend(frame_dir.glob(f"*{ext}"))
        paths.extend(frame_dir.glob(f"*{ext.upper()}"))
    return sorted(set(paths))


def sample_indices(length: int, count: int) -> np.ndarray:
    if length <= 0:
        return np.asarray([], dtype=np.int64)
    return np.linspace(0, length - 1, num=int(count), dtype=np.int64)


def uniform_cap(items: List[object], max_items: int) -> List[object]:
    if max_items <= 0 or len(items) <= max_items:
        return items
    return [items[int(i)] for i in sample_indices(len(items), max_items)]


def sample_frame_paths(frame_dir: Path, num_frames: int, sample_fps: float = 0.0, frame_dir_fps: float = 4.0) -> List[Path]:
    paths = list_frame_paths(frame_dir)
    if not paths:
        return []
    if sample_fps and sample_fps > 0:
        assumed_fps = frame_dir_fps if frame_dir_fps and frame_dir_fps > 0 else sample_fps
        step = max(int(round(float(assumed_fps) / float(sample_fps))), 1)
        selected = paths[::step]
        return uniform_cap(selected, num_frames)
    if num_frames <= 0 or len(paths) <= num_frames:
        return paths
    return [paths[int(i)] for i in sample_indices(len(paths), num_frames)]


def find_video(video_root: Path, sample_id: str) -> Optional[Path]:
    for ext in VIDEO_EXTS:
        path = video_root / f"{sample_id}{ext}"
        if path.exists():
            return path
    matches = [p for p in video_root.glob(f"{sample_id}.*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    return matches[0] if matches else None


def import_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for reading/resizing eJSL frames.") from exc
    return cv2


def encode_image_array(image_bgr, max_side: int, jpeg_quality: int) -> str:
    cv2 = import_cv2()
    height, width = image_bgr.shape[:2]
    longest = max(height, width)
    if max_side > 0 and longest > max_side:
        scale = float(max_side) / float(longest)
        image_bgr = cv2.resize(
            image_bgr,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    ok, encoded = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        raise RuntimeError("Failed to JPEG-encode frame")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def encode_image_path(path: Path, max_side: int, jpeg_quality: int) -> str:
    cv2 = import_cv2()
    image_bgr = cv2.imread(str(path))
    if image_bgr is None:
        raise RuntimeError(f"Failed to read frame image: {path}")
    return encode_image_array(image_bgr, max_side=max_side, jpeg_quality=jpeg_quality)


def encode_video_sampled_frames(
    video_path: Path,
    num_frames: int,
    max_side: int,
    jpeg_quality: int,
    sample_fps: float = 0.0,
) -> List[str]:
    cv2 = import_cv2()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= 0:
        cap.release()
        raise RuntimeError(f"Video has no readable frames: {video_path}")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if source_fps <= 0:
        source_fps = 30.0
    if sample_fps and sample_fps > 0:
        step = max(int(round(source_fps / float(sample_fps))), 1)
        indices = list(range(0, frame_count, step))
        indices = [int(i) for i in uniform_cap(indices, num_frames)]
    else:
        count = int(num_frames) if int(num_frames) > 0 else frame_count
        indices = [int(i) for i in sample_indices(frame_count, count)]

    data_urls: List[str] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame_bgr = cap.read()
        if ok and frame_bgr is not None:
            data_urls.append(encode_image_array(frame_bgr, max_side=max_side, jpeg_quality=jpeg_quality))
    cap.release()
    return data_urls[:num_frames] if num_frames > 0 else data_urls


def encode_video_data_url(video_path: Path, max_mb: float) -> str:
    size_mb = video_path.stat().st_size / (1024.0 * 1024.0)
    if max_mb > 0 and size_mb > max_mb:
        raise RuntimeError(f"Video is {size_mb:.2f} MB, larger than --max_video_mb={max_mb}")
    mime = "video/mp4" if video_path.suffix.lower() == ".mp4" else "video/webm"
    data = base64.b64encode(video_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def sample_media_payload(args, sample_id: str) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    frame_root = Path(args.frame_root)
    video_root = Path(args.video_root)
    frame_dir = frame_root / sample_id
    video_path = find_video(video_root, sample_id)

    if args.input_mode == "video":
        if video_path is None:
            raise RuntimeError(f"No eJSL video found for {sample_id}")
        data_url = encode_video_data_url(video_path, args.max_video_mb)
        return (
            [{"type": "video_url", "video_url": {"url": data_url}}],
            {"media_kind": "video", "video_path": str(video_path), "num_frames": None},
        )

    frame_paths = (
        sample_frame_paths(
            frame_dir,
            args.num_frames,
            sample_fps=args.sample_fps,
            frame_dir_fps=args.frame_dir_fps,
        )
        if frame_dir.exists()
        else []
    )
    if frame_paths:
        data_urls = [encode_image_path(path, args.image_max_side, args.jpeg_quality) for path in frame_paths]
        return (
            [{"type": "image_url", "image_url": {"url": url}} for url in data_urls],
            {
                "media_kind": "frame_dir",
                "frame_dir": str(frame_dir),
                "frames_used": [path.name for path in frame_paths],
                "num_frames": len(frame_paths),
            },
        )

    if video_path is None:
        raise RuntimeError(f"No eJSL frame dir or video found for {sample_id}")
    data_urls = encode_video_sampled_frames(
        video_path,
        args.num_frames,
        args.image_max_side,
        args.jpeg_quality,
        sample_fps=args.sample_fps,
    )
    return (
        [{"type": "image_url", "image_url": {"url": url}} for url in data_urls],
        {"media_kind": "video_sampled_frames", "video_path": str(video_path), "num_frames": len(data_urls)},
    )


def prompt_text(args, media_info: Dict[str, object]) -> str:
    if args.response_mode == "plain_translation":
        return PLAIN_TRANSLATION_PROMPT

    frame_count = media_info.get("num_frames")
    frame_text = f"{frame_count} frame images sampled across the full utterance" if frame_count else "one local video"
    return (
        "You are helping build non-oracle text for a Japanese Sign Language emotion dataset.\n"
        f"The input contains {frame_text} from one eJSL utterance clip, in temporal order.\n"
        "Task: infer the signed utterance and write exactly one concise natural Japanese sentence for this clip.\n"
        "Do not write a dialogue summary, a label, or an explanation.\n"
        "No filename, sample ID, dataset label, speaker name, or oracle transcript is available to you; use only the visual input.\n"
        "If the signs are not readable, give the most likely short Japanese description and lower confidence.\n"
        "Return only one compact JSON object with these keys and no extra whitespace:\n"
        "{"
        "\"text_ja\": string, "
        "\"confidence\": number between 0 and 1, "
        "\"uncertain\": boolean"
        "}."
    )


def request_json(method: str, url: str, api_key: Optional[str], payload: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    data = None
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    referer = os.environ.get("OPENROUTER_HTTP_REFERER", "")
    title = os.environ.get("OPENROUTER_APP_TITLE", "emotionbaseline-ejsl-nonoracle")
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-OpenRouter-Title"] = title
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            text = resp.read().decode("utf-8")
            return json.loads(text) if text.strip() else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter HTTP {exc.code}: {body}") from exc


def fetch_models(api_key: Optional[str], models: List[str], out_path: Path) -> Dict[str, object]:
    try:
        data = request_json("GET", f"{OPENROUTER_BASE_URL}/models", api_key)
    except Exception as exc:
        data = {"error": str(exc)}
    selected = []
    all_models = data.get("data") if isinstance(data, dict) else None
    if isinstance(all_models, list):
        by_id = {str(item.get("id")): item for item in all_models if isinstance(item, dict)}
        selected = [by_id.get(model, {"id": model, "missing_from_models_endpoint": True}) for model in models]
    out = {"requested_models": models, "selected_models": selected, "raw_error": data.get("error") if isinstance(data, dict) else None}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out


def chat_completion(
    args,
    api_key: str,
    model: str,
    sample_id: str,
    media_payload: List[Dict[str, object]],
    media_info: Dict[str, object],
) -> Dict[str, object]:
    content: List[Dict[str, object]] = [{"type": "text", "text": prompt_text(args, media_info)}]
    content.extend(media_payload)
    if args.response_mode == "plain_translation":
        system_content = (
            "You translate visual Japanese Sign Language clips into Japanese. "
            "Return only the Japanese translation text, with no explanation."
        )
    else:
        system_content = (
            "You translate visual Japanese Sign Language clips into concise Japanese. "
            "Answer with valid JSON only."
        )
    payload: Dict[str, object] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system_content,
            },
            {"role": "user", "content": content},
        ],
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "max_tokens": int(args.max_tokens),
        "usage": {"include": True},
    }
    if args.reasoning_effort:
        payload["reasoning"] = {"effort": args.reasoning_effort}
    return request_json("POST", f"{OPENROUTER_BASE_URL}/chat/completions", api_key, payload)


def fetch_generation(api_key: str, generation_id: str, retries: int, sleep_sec: float) -> Dict[str, object]:
    if not generation_id:
        return {}
    query = urllib.parse.urlencode({"id": generation_id})
    last_error = ""
    for attempt in range(max(1, retries)):
        if attempt:
            time.sleep(max(0.0, sleep_sec))
        try:
            return request_json("GET", f"{OPENROUTER_BASE_URL}/generation?{query}", api_key)
        except Exception as exc:
            last_error = str(exc)
    return {"error": last_error}


def response_text(response: Dict[str, object]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return ""
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                chunks.append(str(item.get("text", "")))
        return "\n".join(chunks)
    return str(content)


def extract_json_object(text: str) -> Dict[str, object]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(cleaned[start : end + 1])
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


def clean_plain_translation(text: str) -> str:
    cleaned = str(text or "").strip()
    end = cleaned.rfind("</think>")
    if end >= 0:
        cleaned = cleaned[end + len("</think>") :].strip()
    cleaned = cleaned.replace("<think>", "").replace("</think>", "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:\w+)?\s*", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    for prefix in ("日本語訳:", "翻訳:", "訳:", "Output:", "Translation:"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if len(lines) == 1:
        cleaned = lines[0]
    return sanitize_generated_text(cleaned)


def parse_response_text(args, text: str) -> Dict[str, object]:
    if args.response_mode == "plain_translation":
        return {"text_ja": clean_plain_translation(text)}
    return extract_json_object(text)


def value_from_nested(data: Dict[str, object], *keys: str) -> Optional[float]:
    for key in keys:
        current: object = data
        ok = True
        for part in key.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                ok = False
                break
        if not ok or current in (None, ""):
            continue
        try:
            value = float(current)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return None


def generation_data(generation_info: Dict[str, object]) -> Dict[str, object]:
    data = generation_info.get("data")
    return data if isinstance(data, dict) else generation_info


def row_cost(row: Dict[str, object]) -> Optional[float]:
    gen_info = row.get("generation_info")
    response = row.get("response")
    if isinstance(gen_info, dict):
        data = generation_data(gen_info)
        cost = value_from_nested(data, "total_cost", "totalCost", "usage")
        if cost is not None:
            return cost
    if isinstance(response, dict):
        return value_from_nested(response, "usage.cost", "usage.total_cost", "usage.totalCost")
    return None


def load_done(jsonl_path: Path) -> Dict[str, Dict[str, object]]:
    done: Dict[str, Dict[str, object]] = {}
    if not jsonl_path.exists():
        return done
    with jsonl_path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError:
                continue
            sample_id = str(item.get("sample_id", ""))
            if sample_id and item.get("status") == "ok":
                done[sample_id] = item
    return done


def write_review_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "sample_id",
        "gold_label",
        "status",
        "text_ja",
        "text_en",
        "confidence",
        "uncertain",
        "visual_notes",
        "cost_usd",
        "generation_id",
        "media_kind",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            parsed = row.get("parsed") if isinstance(row.get("parsed"), dict) else {}
            media = row.get("media") if isinstance(row.get("media"), dict) else {}
            writer.writerow(
                {
                    "model": row.get("model", ""),
                    "sample_id": row.get("sample_id", ""),
                    "gold_label": row.get("gold_label", ""),
                    "status": row.get("status", ""),
                    "text_ja": parsed.get("text_ja", ""),
                    "text_en": parsed.get("text_en", ""),
                    "confidence": parsed.get("confidence", ""),
                    "uncertain": parsed.get("uncertain", ""),
                    "visual_notes": parsed.get("visual_notes", ""),
                    "cost_usd": row_cost(row),
                    "generation_id": row.get("generation_id", ""),
                    "media_kind": media.get("media_kind", ""),
                    "error": row.get("error", ""),
                }
            )


def write_predictions_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    latest: Dict[str, str] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        sample_id = str(row.get("sample_id", "")).strip()
        text = text_ja_from_row(row)
        if sample_id and text:
            latest[sample_id] = text
    with path.open("w", encoding="utf-8") as f:
        for sample_id in sorted(latest):
            f.write(json.dumps({"sample_id": sample_id, "text": latest[sample_id]}, ensure_ascii=False) + "\n")


def summarize_rows(rows: Iterable[Dict[str, object]]) -> Dict[str, object]:
    rows = list(rows)
    costs = [row_cost(row) for row in rows]
    costs = [float(x) for x in costs if x is not None]
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    return {
        "n": len(rows),
        "ok": len(ok_rows),
        "failed": len(rows) - len(ok_rows),
        "total_cost_usd": sum(costs),
        "mean_cost_usd": (sum(costs) / len(costs)) if costs else None,
        "cost_rows": len(costs),
    }


def text_ja_from_row(row: Dict[str, object]) -> str:
    parsed = row.get("parsed")
    if isinstance(parsed, dict):
        text = sanitize_generated_text(str(parsed.get("text_ja", "")))
        if text:
            return text
    return ""


def write_model_txt_tree(
    sample_ids: List[str],
    rows: Iterable[Dict[str, object]],
    structure_txt_root: Path,
    output_txt_root: Path,
) -> None:
    translations: Dict[str, str] = {}
    errors: List[str] = []
    for row in rows:
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id:
            continue
        if row.get("status") != "ok":
            errors.append(f"{sample_id}: status={row.get('status')} error={row.get('error', '')}")
            continue
        text = text_ja_from_row(row)
        if not text:
            errors.append(f"{sample_id}: missing parsed text_ja")
            continue
        translations[sample_id] = text

    missing = [sample_id for sample_id in sample_ids if sample_id not in translations]
    if missing or errors:
        preview = "; ".join((missing[:5] + errors[:5]))
        raise RuntimeError(
            f"Cannot write replacement txt tree: missing={len(missing)} errors={len(errors)} first={preview}"
        )

    grouped: Dict[Tuple[str, int], Dict[int, str]] = defaultdict(dict)
    for sample_id in sample_ids:
        sd_id, dialogue_idx, utterance_idx, _label = parse_ejsl_sample_id(sample_id)
        grouped[(sd_id, dialogue_idx)][utterance_idx] = translations[sample_id]

    for (sd_id, dialogue_idx), items in grouped.items():
        structure_file = structure_txt_root / sd_id / "txt" / f"{sd_id}-Dialogue-{dialogue_idx:02d}.txt"
        if not structure_file.exists():
            raise FileNotFoundError(f"Missing structure txt file: {structure_file}")
        structure = read_dialogue_structure(structure_file)
        max_utt = max(items)
        if max_utt > len(structure):
            raise RuntimeError(f"{structure_file} has {len(structure)} turns, but sample needs turn {max_utt}")
        missing_turns = [idx for idx in range(1, max_utt + 1) if idx not in items]
        if missing_turns:
            raise RuntimeError(
                f"Cannot write {sd_id}-Dialogue-{dialogue_idx:02d}: missing turns {missing_turns[:10]}"
            )

        out_file = output_txt_root / sd_id / "txt" / f"{sd_id}-Dialogue-{dialogue_idx:02d}.txt"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for idx in range(1, max_utt + 1):
            speaker = structure[idx - 1]["speaker"]
            emotion = structure[idx - 1]["emotion"]
            lines.append(f"{speaker}|{emotion}|{items[idx]}")
        out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    models = args.model or DEFAULT_MODELS
    out_root = Path(args.out_root)
    responses_dir = out_root / "responses"
    reviews_dir = out_root / "reviews"
    out_root.mkdir(parents=True, exist_ok=True)
    responses_dir.mkdir(parents=True, exist_ok=True)
    reviews_dir.mkdir(parents=True, exist_ok=True)

    sample_ids_all = read_ejsl_names(Path(args.dial_list))
    sample_ids = choose_sample_ids(sample_ids_all, args.limit, args.selection)
    if not sample_ids:
        raise RuntimeError(f"No eJSL samples selected from {args.dial_list}")

    selected_path = out_root / "selected_samples.csv"
    with selected_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "gold_label", "sd_id", "dialogue_idx", "utterance_idx"])
        writer.writeheader()
        for sample_id in sample_ids:
            try:
                sd_id, dialogue_idx, utterance_idx, label = parse_ejsl_sample_id(sample_id)
            except Exception:
                sd_id, dialogue_idx, utterance_idx, label = "", "", "", sample_id[-1:]
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "gold_label": label,
                    "sd_id": sd_id,
                    "dialogue_idx": dialogue_idx,
                    "utterance_idx": utterance_idx,
                }
            )

    api_key = os.environ.get(args.api_key_env, "")
    if not api_key and not args.dry_run:
        raise RuntimeError(f"Set {args.api_key_env} before calling OpenRouter, or pass --dry_run.")

    fetch_models(api_key or None, models, out_root / "openrouter_models_selected.json")

    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "models": models,
                    "samples": len(sample_ids),
                    "selected_samples_csv": str(selected_path),
                    "out_root": str(out_root),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    all_summary: Dict[str, object] = {
        "out_root": str(out_root),
        "dial_list": args.dial_list,
        "frame_root": args.frame_root,
        "video_root": args.video_root,
        "input_mode": args.input_mode,
        "num_frames": args.num_frames,
        "sample_fps": args.sample_fps,
        "frame_dir_fps": args.frame_dir_fps,
        "response_mode": args.response_mode,
        "limit": len(sample_ids),
        "selection": args.selection,
        "models": {},
    }

    for model in models:
        model_name = safe_name(model)
        jsonl_path = responses_dir / f"{model_name}.jsonl"
        done = load_done(jsonl_path) if args.resume else {}
        model_rows: List[Dict[str, object]] = list(done.values())

        print(f"[OpenRouter] model={model} samples={len(sample_ids)} resume_ok={len(done)}")
        for index, sample_id in enumerate(sample_ids, start=1):
            if sample_id in done:
                continue
            try:
                media_payload, media_info = sample_media_payload(args, sample_id)
                response = chat_completion(args, api_key, model, sample_id, media_payload, media_info)
                text = response_text(response)
                parsed = parse_response_text(args, text)
                generation_id = str(response.get("id", ""))
                generation_info = fetch_generation(
                    api_key,
                    generation_id,
                    retries=args.generation_poll_retries,
                    sleep_sec=args.generation_poll_sleep,
                )
                try:
                    _sd_id, _dialogue_idx, _utterance_idx, gold_label = parse_ejsl_sample_id(sample_id)
                except Exception:
                    gold_label = sample_id[-1:]
                row: Dict[str, object] = {
                    "status": "ok",
                    "model": model,
                    "sample_id": sample_id,
                    "gold_label": gold_label,
                    "media": media_info,
                    "raw_text": text,
                    "parsed": parsed,
                    "generation_id": generation_id,
                    "response_usage": response.get("usage", {}),
                    "response": response,
                    "generation_info": generation_info,
                }
            except Exception as exc:
                row = {
                    "status": "error",
                    "model": model,
                    "sample_id": sample_id,
                    "gold_label": sample_id[-1:],
                    "error": str(exc),
                }
                if args.fail_fast:
                    append_jsonl(jsonl_path, row)
                    raise
            append_jsonl(jsonl_path, row)
            model_rows.append(row)

            parsed = row.get("parsed") if isinstance(row.get("parsed"), dict) else {}
            cost = row_cost(row)
            print(
                f"[OpenRouter] {model} {index}/{len(sample_ids)} {sample_id} "
                f"status={row.get('status')} cost={cost if cost is not None else 'NA'} "
                f"text_ja={parsed.get('text_ja', '') if parsed else ''}"
            )
            if args.sleep_sec > 0:
                time.sleep(args.sleep_sec)

        review_path = reviews_dir / f"review_{model_name}.csv"
        write_review_csv(review_path, model_rows)
        predictions_path = responses_dir / f"predictions_{model_name}.jsonl"
        write_predictions_jsonl(predictions_path, model_rows)

        txt_tree_root = ""
        if args.write_txt_tree:
            if not args.output_txt_root:
                raise RuntimeError("--write_txt_tree requires --output_txt_root")
            txt_tree_path = Path(args.output_txt_root)
            if len(models) > 1:
                txt_tree_path = txt_tree_path / model_name
            write_model_txt_tree(
                sample_ids,
                model_rows,
                structure_txt_root=Path(args.structure_txt_root),
                output_txt_root=txt_tree_path,
            )
            txt_tree_root = str(txt_tree_path)

        all_summary["models"][model] = {
            **summarize_rows(model_rows),
            "responses_jsonl": str(jsonl_path),
            "predictions_jsonl": str(predictions_path),
            "review_csv": str(review_path),
            "txt_tree_root": txt_tree_root,
        }

    summary_path = out_root / "cost_summary.json"
    summary_path.write_text(json.dumps(all_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(all_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
