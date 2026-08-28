# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from jsl_translation_model import JSLQwenPrefixTranslator
from keypoints import (
    extract_holistic_keypoints_from_frame_dir,
    extract_holistic_keypoints_from_video,
    sample_keypoints_sequence,
)
from manifest_utils import (
    append_jsonl,
    parse_ejsl_sample_id,
    read_dialogue_structure,
    read_ejsl_names,
    read_jsonl,
    sanitize_generated_text,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate non-oracle eJSL text files with a fine-tuned JSL translator.")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--dial_list", type=str, required=True)
    parser.add_argument("--output_txt_root", type=str, required=True)
    parser.add_argument("--structure_txt_root", type=str, required=True, help="Oracle txt root used only for speaker/dialogue structure.")
    parser.add_argument("--video_root", type=str, default="")
    parser.add_argument("--frame_root", type=str, default="")
    parser.add_argument("--keypoint_cache_dir", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_visual_tokens", type=int, default=64)
    parser.add_argument("--sample_fps", type=float, default=10.0)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--model_complexity", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--torch_dtype", type=str, default="auto")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--predictions_jsonl", type=str, default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def load_done(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    done: Dict[str, str] = {}
    for row in read_jsonl(path):
        sample_id = str(row.get("sample_id", "")).strip()
        text = str(row.get("text", "")).strip()
        if sample_id and text:
            done[sample_id] = text
    return done


def resolve_media(sample_id: str, video_root: Path | None, frame_root: Path | None) -> Tuple[str, Path]:
    if video_root is not None:
        path = video_root / f"{sample_id}.mp4"
        if path.exists():
            return "video", path
    if frame_root is not None:
        path = frame_root / sample_id
        if path.exists():
            return "frames", path
    raise FileNotFoundError(f"No video/frame media found for eJSL sample {sample_id}")


def extract_or_load_keypoints(args, sample_id: str, media_kind: str, media_path: Path) -> np.ndarray:
    cache_dir = Path(args.keypoint_cache_dir) if args.keypoint_cache_dir else None
    cache_path = cache_dir / f"{sample_id}.npz" if cache_dir is not None else None
    if cache_path is not None and args.resume and cache_path.exists():
        return np.asarray(np.load(cache_path)["keypoints"], dtype=np.float32)

    if media_kind == "video":
        keypoints, timestamps = extract_holistic_keypoints_from_video(
            media_path,
            sample_fps=args.sample_fps,
            max_frames=args.max_frames,
            model_complexity=args.model_complexity,
        )
    else:
        keypoints, timestamps = extract_holistic_keypoints_from_frame_dir(
            media_path,
            sample_fps=0.0,
            max_frames=args.max_frames,
            model_complexity=args.model_complexity,
        )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, keypoints=keypoints, timestamps=timestamps, sample_id=sample_id, media_path=str(media_path))
    return keypoints


def write_ejsl_txt_tree(
    sample_ids: List[str],
    translations: Dict[str, str],
    structure_txt_root: Path,
    output_txt_root: Path,
) -> None:
    grouped: Dict[Tuple[str, int], Dict[int, Tuple[str, str, str]]] = defaultdict(dict)
    needed_by_dialogue: Dict[Tuple[str, int], List[int]] = defaultdict(list)

    for sample_id in sample_ids:
        sd_id, dialogue_idx, utterance_idx, label = parse_ejsl_sample_id(sample_id)
        key = (sd_id, dialogue_idx)
        needed_by_dialogue[key].append(utterance_idx)
        text = sanitize_generated_text(translations[sample_id])
        if not text:
            raise RuntimeError(f"Generated empty text for {sample_id}")
        grouped[key][utterance_idx] = (sample_id, label, text)

    for (sd_id, dialogue_idx), items in grouped.items():
        structure_file = structure_txt_root / sd_id / "txt" / f"{sd_id}-Dialogue-{dialogue_idx:02d}.txt"
        if not structure_file.exists():
            raise FileNotFoundError(f"Missing structure txt file: {structure_file}")
        structure = read_dialogue_structure(structure_file)
        max_utt = max(needed_by_dialogue[(sd_id, dialogue_idx)])
        if max_utt > len(structure):
            raise RuntimeError(f"{structure_file} has {len(structure)} turns, but sample needs turn {max_utt}")

        missing = [idx for idx in range(1, max_utt + 1) if idx not in items]
        if missing:
            raise RuntimeError(
                f"Missing generated eJSL turns for {sd_id}-Dialogue-{dialogue_idx:02d}: {missing}. "
                "The non-oracle txt tree must contain all prefix turns used by TELME."
            )

        out_file = output_txt_root / sd_id / "txt" / f"{sd_id}-Dialogue-{dialogue_idx:02d}.txt"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for idx in range(1, max_utt + 1):
            _sample_id, label, text = items[idx]
            speaker = structure[idx - 1]["speaker"]
            emotion = structure[idx - 1]["emotion"] or label
            lines.append(f"{speaker}|{emotion}|{text}")
        out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    if not args.video_root and not args.frame_root:
        raise RuntimeError("Provide at least one of --video_root or --frame_root")

    sample_ids = read_ejsl_names(Path(args.dial_list))
    if args.limit > 0:
        sample_ids = sample_ids[: args.limit]
    if not sample_ids:
        raise RuntimeError(f"No sample ids loaded from {args.dial_list}")

    predictions_path = Path(args.predictions_jsonl) if args.predictions_jsonl else Path(args.output_txt_root) / "non_oracle_predictions.jsonl"
    translations = load_done(predictions_path) if args.resume else {}

    video_root = Path(args.video_root) if args.video_root else None
    frame_root = Path(args.frame_root) if args.frame_root else None

    model = JSLQwenPrefixTranslator.from_pretrained(
        args.model_dir,
        torch_dtype=args.torch_dtype,
        load_in_4bit=args.load_in_4bit,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.load_in_4bit:
        model.to(device)
    else:
        model.projector.to(device)
    model.eval()

    pending = [sample_id for sample_id in sample_ids if sample_id not in translations]
    batch_ids: List[str] = []
    batch_keypoints: List[torch.Tensor] = []

    for sample_id in tqdm(pending, desc="eJSL translate"):
        media_kind, media_path = resolve_media(sample_id, video_root, frame_root)
        keypoints = extract_or_load_keypoints(args, sample_id, media_kind, media_path)
        sampled = sample_keypoints_sequence(keypoints, args.num_visual_tokens)
        batch_ids.append(sample_id)
        batch_keypoints.append(torch.from_numpy(sampled).float())

        if len(batch_ids) >= args.batch_size:
            batch = torch.stack(batch_keypoints, dim=0)
            texts = model.generate_texts(
                batch,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            for sid, text in zip(batch_ids, texts):
                clean = sanitize_generated_text(text)
                translations[sid] = clean
                append_jsonl(predictions_path, {"sample_id": sid, "text": clean})
            batch_ids, batch_keypoints = [], []

    if batch_ids:
        batch = torch.stack(batch_keypoints, dim=0)
        texts = model.generate_texts(
            batch,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        for sid, text in zip(batch_ids, texts):
            clean = sanitize_generated_text(text)
            translations[sid] = clean
            append_jsonl(predictions_path, {"sample_id": sid, "text": clean})

    missing = [sid for sid in sample_ids if sid not in translations]
    if missing:
        raise RuntimeError(f"Missing translations for {len(missing)} samples; first={missing[:5]}")

    write_ejsl_txt_tree(
        sample_ids,
        translations,
        structure_txt_root=Path(args.structure_txt_root),
        output_txt_root=Path(args.output_txt_root),
    )
    print(f"[eJSL] wrote non-oracle txt root: {args.output_txt_root}")
    print(f"[eJSL] predictions jsonl: {predictions_path}")


if __name__ == "__main__":
    main()
