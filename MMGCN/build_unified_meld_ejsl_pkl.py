#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer, TimesformerModel


ANJS_LABELS = ["A", "N", "J", "S"]
MELD_TO_ANJS = {
    "anger": 0,
    "neutral": 1,
    "joy": 2,
    "sadness": 3,
}
EJSL_TO_ANJS = {"A": 0, "N": 1, "J": 2, "S": 3}


@dataclass
class UtteranceItem:
    text: str
    speaker: str
    label: int
    video_path: Optional[Path] = None
    frame_dir: Optional[Path] = None
    sample_id: str = ""


class FeatureCache:
    def __init__(self, root: Optional[Path]):
        self.root = root
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, kind: str, key: str) -> Optional[Path]:
        if self.root is None:
            return None
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.root / kind / f"{digest}.npy"

    def get(self, kind: str, key: str) -> Optional[np.ndarray]:
        path = self._path(kind, key)
        if path is None or not path.exists():
            return None
        return np.asarray(np.load(path), dtype=np.float32)

    def put(self, kind: str, key: str, value: np.ndarray) -> None:
        path = self._path(kind, key)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.asarray(value, dtype=np.float32))


class TextEncoder:
    def __init__(
        self,
        model_name: str,
        pooling: str,
        max_length: int,
        batch_size: int,
        device: torch.device,
        cache: FeatureCache,
    ):
        self.model_name = model_name
        self.pooling = pooling
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device
        self.cache = cache
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.dim = int(self.model.config.hidden_size)

    def _key(self, text: str) -> str:
        return json.dumps(
            {
                "model": self.model_name,
                "pooling": self.pooling,
                "max_length": self.max_length,
                "text": text,
            },
            sort_keys=True,
            ensure_ascii=False,
        )

    def _pool(self, last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return last_hidden[:, 0, :]
        if self.pooling == "last":
            lengths = attention_mask.sum(dim=1).clamp_min(1) - 1
            batch_idx = torch.arange(last_hidden.size(0), device=last_hidden.device)
            return last_hidden[batch_idx, lengths, :]
        if self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
            return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        raise ValueError(f"Unsupported pooling: {self.pooling}")

    def encode(self, texts: Sequence[str]) -> List[np.ndarray]:
        outputs: List[Optional[np.ndarray]] = [None] * len(texts)
        pending_indices: List[int] = []
        pending_texts: List[str] = []
        pending_keys: List[str] = []

        for idx, text in enumerate(texts):
            key = self._key(text)
            cached = self.cache.get("text", key)
            if cached is not None:
                outputs[idx] = cached
            else:
                pending_indices.append(idx)
                pending_texts.append(text)
                pending_keys.append(key)

        for start in range(0, len(pending_texts), self.batch_size):
            batch_texts = pending_texts[start : start + self.batch_size]
            batch_keys = pending_keys[start : start + self.batch_size]
            batch_indices = pending_indices[start : start + self.batch_size]
            encoded = self.tokenizer(
                list(batch_texts),
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.no_grad():
                hidden = self.model(**encoded).last_hidden_state
                pooled = self._pool(hidden, encoded["attention_mask"]).detach().cpu().numpy().astype(np.float32)
            for item_idx, key, vec in zip(batch_indices, batch_keys, pooled):
                outputs[item_idx] = vec
                self.cache.put("text", key, vec)

        return [np.asarray(x, dtype=np.float32) for x in outputs]


class VideoEncoder:
    def __init__(
        self,
        model_name: str,
        processor_name: str,
        device: torch.device,
        cache: FeatureCache,
        max_seconds: float,
        fp16: bool,
    ):
        self.model_name = model_name
        self.processor_name = processor_name
        self.device = device
        self.cache = cache
        self.max_seconds = max_seconds
        self.fp16 = fp16
        self.processor = AutoImageProcessor.from_pretrained(processor_name)
        self.model = TimesformerModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.dim = int(self.model.config.hidden_size)

    def _key(self, path: Optional[Path], frame_dir: Optional[Path]) -> str:
        if path is not None and path.exists():
            stat = path.stat()
            source = {"path": str(path.resolve()), "mtime": stat.st_mtime_ns, "size": stat.st_size}
        elif frame_dir is not None and frame_dir.exists():
            frame_paths = sorted(list(frame_dir.glob("*.jpg")) + list(frame_dir.glob("*.png")))
            source = {
                "frame_dir": str(frame_dir.resolve()),
                "count": len(frame_paths),
                "first": frame_paths[0].name if frame_paths else "",
                "last": frame_paths[-1].name if frame_paths else "",
            }
        else:
            source = {"missing": True, "path": str(path or ""), "frame_dir": str(frame_dir or "")}
        return json.dumps(
            {
                "model": self.model_name,
                "processor": self.processor_name,
                "max_seconds": self.max_seconds,
                "source": source,
            },
            sort_keys=True,
        )

    @staticmethod
    def _sample_frame_paths(frame_dir: Path) -> List[Path]:
        frame_paths = sorted(frame_dir.glob("*.jpg"))
        if not frame_paths:
            frame_paths = sorted(frame_dir.glob("*.png"))
        if not frame_paths:
            return []
        if len(frame_paths) >= 8:
            indices = np.linspace(0, len(frame_paths) - 1, num=8, dtype=int)
            return [frame_paths[int(i)] for i in indices]
        return frame_paths + [frame_paths[-1]] * (8 - len(frame_paths))

    @staticmethod
    def _read_frames_from_dir(frame_dir: Path) -> List[np.ndarray]:
        frames: List[np.ndarray] = []
        for path in VideoEncoder._sample_frame_paths(frame_dir):
            image = cv2.imread(str(path))
            if image is not None:
                frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return frames

    def _read_frames_from_mp4(self, path: Path) -> List[np.ndarray]:
        cap = cv2.VideoCapture(str(path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if frame_count <= 0:
            cap.release()
            return []
        if self.max_seconds > 0 and fps > 0 and frame_count / fps > self.max_seconds:
            cap.release()
            return []

        indices = np.linspace(0, frame_count - 1, num=8, dtype=int)
        frames: List[np.ndarray] = []
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, image = cap.read()
            if ok and image is not None:
                frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        cap.release()
        if frames and len(frames) < 8:
            frames.extend([frames[-1].copy() for _ in range(8 - len(frames))])
        return frames[:8]

    def encode_one(self, video_path: Optional[Path], frame_dir: Optional[Path]) -> Tuple[np.ndarray, str]:
        key = self._key(video_path, frame_dir)
        cached = self.cache.get("video", key)
        if cached is not None:
            return cached, "cache"

        frames: List[np.ndarray] = []
        status = "missing"
        if video_path is not None and video_path.exists():
            frames = self._read_frames_from_mp4(video_path)
            status = "mp4" if frames else "mp4_empty_or_long"
        if not frames and frame_dir is not None and frame_dir.exists():
            frames = self._read_frames_from_dir(frame_dir)
            status = "frames" if frames else "frames_empty"
        if not frames:
            vec = np.zeros(self.dim, dtype=np.float32)
            self.cache.put("video", key, vec)
            return vec, status

        inputs = self.processor(frames, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.device.type == "cuda" and self.fp16):
                hidden = self.model(pixel_values).last_hidden_state[:, 0, :]
        vec = hidden[0].detach().cpu().numpy().astype(np.float32)
        self.cache.put("video", key, vec)
        return vec, status


def resolve_path(raw: str, must_exist: bool = False) -> Path:
    path = Path(raw).expanduser()
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, Path(__file__).resolve().parent.parent / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    if must_exist:
        raise FileNotFoundError(f"Path not found: {raw}. Checked: {', '.join(str(x) for x in candidates)}")
    return candidates[0].resolve()


def make_speaker_mask(speakers: Sequence[str], n_speakers: int) -> np.ndarray:
    speaker_to_idx: Dict[str, int] = {}
    rows: List[np.ndarray] = []
    for speaker in speakers:
        if speaker not in speaker_to_idx:
            speaker_to_idx[speaker] = len(speaker_to_idx)
        idx = speaker_to_idx[speaker] % n_speakers
        row = np.zeros(n_speakers, dtype=np.float32)
        row[idx] = 1.0
        rows.append(row)
    return np.stack(rows, axis=0) if rows else np.zeros((0, n_speakers), dtype=np.float32)


def write_feature_pkl(
    dialogues: Dict[str, List[UtteranceItem]],
    train_vid: List[str],
    test_vid: List[str],
    out_pkl: Path,
    text_encoder: TextEncoder,
    video_encoder: VideoEncoder,
    audio_dim: int,
    n_speakers: int,
    meta: Dict,
) -> Dict:
    all_items: List[UtteranceItem] = []
    for vid in train_vid + test_vid:
        all_items.extend(dialogues[vid])

    print(f"[features] text encode items={len(all_items)} model={text_encoder.model_name}")
    text_vectors = text_encoder.encode([item.text for item in all_items])

    video_vectors: List[np.ndarray] = []
    video_status: Dict[str, int] = {}
    for idx, item in enumerate(all_items, start=1):
        vec, status = video_encoder.encode_one(item.video_path, item.frame_dir)
        video_vectors.append(vec)
        video_status[status] = video_status.get(status, 0) + 1
        if idx == 1 or idx % 250 == 0 or idx == len(all_items):
            print(f"[features] video encode items={idx}/{len(all_items)} status={video_status}")

    cursor = 0
    video_ids: Dict[str, str] = {}
    video_speakers: Dict[str, np.ndarray] = {}
    video_labels: Dict[str, List[int]] = {}
    video_text: Dict[str, np.ndarray] = {}
    video_audio: Dict[str, np.ndarray] = {}
    video_visual: Dict[str, np.ndarray] = {}
    video_sentence: Dict[str, List[str]] = {}

    for vid in train_vid + test_vid:
        items = dialogues[vid]
        n_items = len(items)
        text_seq = np.stack(text_vectors[cursor : cursor + n_items], axis=0).astype(np.float32)
        visual_seq = np.stack(video_vectors[cursor : cursor + n_items], axis=0).astype(np.float32)
        cursor += n_items

        video_ids[vid] = vid
        video_speakers[vid] = make_speaker_mask([item.speaker for item in items], n_speakers)
        video_labels[vid] = [int(item.label) for item in items]
        video_text[vid] = text_seq
        video_audio[vid] = np.zeros((n_items, audio_dim), dtype=np.float32)
        video_visual[vid] = visual_seq
        video_sentence[vid] = [item.text for item in items]

    payload = (
        video_ids,
        video_speakers,
        video_labels,
        video_text,
        video_audio,
        video_visual,
        video_sentence,
        train_vid,
        test_vid,
        {
            **meta,
            "label_names": ANJS_LABELS,
            "n_classes": len(ANJS_LABELS),
            "text_dim": text_encoder.dim,
            "audio_dim": audio_dim,
            "visual_dim": video_encoder.dim,
            "n_speakers": n_speakers,
            "video_status": video_status,
        },
    )
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with out_pkl.open("wb") as f:
        pickle.dump(payload, f)

    valid_labels = [item.label for item in all_items if item.label >= 0]
    counts = np.bincount(np.asarray(valid_labels, dtype=np.int64), minlength=len(ANJS_LABELS)).tolist()
    summary = {
        "out_pkl": str(out_pkl),
        "dialogues": len(train_vid) + len(test_vid),
        "train_dialogues": len(train_vid),
        "test_dialogues": len(test_vid),
        "utterances": len(all_items),
        "valid_labeled_utterances": len(valid_labels),
        "label_counts": dict(zip(ANJS_LABELS, counts)),
        "video_status": video_status,
    }
    out_pkl.with_suffix(out_pkl.suffix + ".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[features] wrote {out_pkl}")
    print(f"[features] label_counts={summary['label_counts']}")
    return summary


def resolve_meld_video_path(raw_video_path: str, meld_root: Path) -> Optional[Path]:
    raw = Path(raw_video_path.strip())
    candidates: List[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(Path.cwd() / raw)
        candidates.append(meld_root.parent.parent / raw)
        if len(raw.parts) >= 2:
            candidates.append(meld_root / raw.parts[-2] / raw.parts[-1])
        candidates.append(meld_root / raw.name)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0] if candidates else None


def read_meld_split(csv_path: Path, split: str, meld_root: Path, keep_non_anjs_context: bool) -> Dict[str, List[UtteranceItem]]:
    grouped: Dict[str, List[Tuple[int, UtteranceItem]]] = {}
    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            emotion = str(row["Emotion"]).strip().lower()
            label = MELD_TO_ANJS.get(emotion, -100)
            if label < 0 and not keep_non_anjs_context:
                continue
            dialogue_id = int(row["Dialogue_ID"])
            utterance_id = int(row["Utterance_ID"])
            vid = f"meld_{split}_dia{dialogue_id}"
            item = UtteranceItem(
                text=str(row["Utterance"]).strip(),
                speaker=str(row["Speaker"]).strip(),
                label=label,
                video_path=resolve_meld_video_path(str(row["Video_Path"]), meld_root),
                sample_id=f"{vid}_utt{utterance_id}",
            )
            grouped.setdefault(vid, []).append((utterance_id, item))

    return {vid: [item for _idx, item in sorted(items, key=lambda x: x[0])] for vid, items in grouped.items() if items}


def build_meld_dialogues(args) -> Tuple[Dict[str, List[UtteranceItem]], List[str], List[str]]:
    meld_root = resolve_path(args.meld_root, must_exist=True)
    split_files = {
        "train": meld_root / "train_meld_emo.csv",
        "dev": meld_root / "dev_meld_emo.csv",
        "test": meld_root / "test_meld_emo.csv",
    }
    for split, path in split_files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing MELD {split} csv: {path}")

    dialogues: Dict[str, List[UtteranceItem]] = {}
    train_vid: List[str] = []
    test_vid: List[str] = []
    for split in ["train", "dev", "test"]:
        split_dialogues = read_meld_split(split_files[split], split, meld_root, args.keep_non_anjs_context)
        keys = sorted(split_dialogues, key=lambda x: (x.split("_dia")[0], int(x.rsplit("dia", 1)[1])))
        if args.limit_meld_dialogues > 0:
            keys = keys[: args.limit_meld_dialogues]
            split_dialogues = {k: split_dialogues[k] for k in keys}
        dialogues.update(split_dialogues)
        if split in {"train", "dev"}:
            train_vid.extend(keys)
        else:
            test_vid.extend(keys)
        print(f"[MELD] split={split} dialogues={len(keys)}")

    return dialogues, train_vid, test_vid


def read_ejsl_names(path: Path) -> List[str]:
    lines = [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not lines:
        return []
    first = lines[0].lower()
    if "stem" in first or "clip" in first or "filename" in first:
        out: List[str] = []
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                value = row.get("clip_name") or row.get("stem") or row.get("filename")
                if value:
                    out.append(Path(str(value).strip()).stem)
        return out
    return [Path(x).stem for x in lines]


def parse_ejsl_sample_id(sample_id: str) -> Optional[Tuple[str, int, int, str]]:
    match = re.match(r"^(SD\d{2})-(\d{2})-(\d{2})([AJNS])$", sample_id)
    if match is None:
        return None
    sd_id, dialogue_idx, utterance_idx, label = match.groups()
    return sd_id, int(dialogue_idx), int(utterance_idx), label


def parse_ejsl_txt(txt_file: Path) -> List[Tuple[str, str, str]]:
    turns: List[Tuple[str, str, str]] = []
    for raw in txt_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split("|", 2)
        if len(parts) < 3:
            continue
        turns.append((parts[0].strip(), parts[1].strip(), parts[2].strip()))
    return turns


def find_ejsl_media(
    sd_id: str,
    dialogue_idx: int,
    utterance_idx: int,
    label: Optional[str],
    frame_root: Path,
    mp4_root: Path,
) -> Tuple[Optional[Path], Optional[Path], str]:
    stems: List[str] = []
    if label:
        stems.append(f"{sd_id}-{dialogue_idx:02d}-{utterance_idx:02d}{label}")
    else:
        prefix = f"{sd_id}-{dialogue_idx:02d}-{utterance_idx:02d}"
        stems.extend(sorted(path.name for path in frame_root.glob(prefix + "[AJNS]")))
        stems.extend(sorted(path.stem for path in mp4_root.glob(prefix + "[AJNS].mp4")))

    for stem in stems:
        mp4_path = mp4_root / f"{stem}.mp4"
        frame_dir = frame_root / stem
        if mp4_path.exists() or frame_dir.exists():
            return (mp4_path if mp4_path.exists() else None), (frame_dir if frame_dir.exists() else None), stem

    fallback = stems[0] if stems else f"{sd_id}-{dialogue_idx:02d}-{utterance_idx:02d}"
    return None, None, fallback


def build_ejsl_dialogues(args) -> Tuple[Dict[str, List[UtteranceItem]], List[str], List[str]]:
    txt_root = resolve_path(args.ejsl_txt_root, must_exist=True)
    dial_list = resolve_path(args.ejsl_dial_list, must_exist=True)
    frame_root = resolve_path(args.ejsl_frame_root, must_exist=True)
    mp4_root = resolve_path(args.ejsl_mp4_root, must_exist=False)

    grouped_required: Dict[Tuple[str, int], Dict[int, Tuple[str, str]]] = {}
    for sample_id in read_ejsl_names(dial_list):
        parsed = parse_ejsl_sample_id(sample_id)
        if parsed is None:
            continue
        sd_id, dialogue_idx, utterance_idx, label = parsed
        grouped_required.setdefault((sd_id, dialogue_idx), {})[utterance_idx] = (label, sample_id)

    keys = sorted(grouped_required)
    if args.limit_ejsl_dialogues > 0:
        keys = keys[: args.limit_ejsl_dialogues]

    dialogues: Dict[str, List[UtteranceItem]] = {}
    test_vid: List[str] = []
    dropped = 0
    for sd_id, dialogue_idx in keys:
        required = grouped_required[(sd_id, dialogue_idx)]
        max_utt = max(required)
        txt_file = txt_root / sd_id / "txt" / f"{sd_id}-Dialogue-{dialogue_idx:02d}.txt"
        if not txt_file.exists():
            dropped += len(required)
            continue
        turns = parse_ejsl_txt(txt_file)
        if len(turns) < max_utt:
            dropped += len(required)
            continue

        vid = f"ejsl_{sd_id}_dia{dialogue_idx:02d}"
        items: List[UtteranceItem] = []
        for utterance_idx in range(1, max_utt + 1):
            speaker, _emotion, text = turns[utterance_idx - 1]
            label_char = required.get(utterance_idx, (None, ""))[0]
            sample_id = required.get(utterance_idx, ("", ""))[1]
            label = EJSL_TO_ANJS[label_char] if label_char in EJSL_TO_ANJS else -100
            mp4_path, frame_dir, resolved_sample_id = find_ejsl_media(
                sd_id,
                dialogue_idx,
                utterance_idx,
                label_char,
                frame_root,
                mp4_root,
            )
            items.append(
                UtteranceItem(
                    text=text,
                    speaker=speaker,
                    label=label,
                    video_path=mp4_path,
                    frame_dir=frame_dir,
                    sample_id=sample_id or resolved_sample_id,
                )
            )
        dialogues[vid] = items
        test_vid.append(vid)

    print(f"[eJSL] dialogues={len(test_vid)} dropped_required_samples={dropped}")
    return dialogues, [], test_vid


def parse_args():
    parser = argparse.ArgumentParser(description="Build same-origin MMGCN feature pkl files for MELD and eJSL.")
    parser.add_argument("--meld_root", type=str, default="./dataset/MELD.Raw")
    parser.add_argument("--ejsl_txt_root", type=str, default="")
    parser.add_argument("--ejsl_dial_list", type=str, default="/home/lr/wangyi/Sign/RO-MAN/eJSL_dial_dataset/ejsldial_filenames.csv")
    parser.add_argument("--ejsl_frame_root", type=str, default="/raid_zoe/home/lr/wangyi/sign/eJSL_dial/frame")
    parser.add_argument("--ejsl_mp4_root", type=str, default="/raid_zoe/home/lr/wangyi/sign/eJSL_dial/video")
    parser.add_argument("--out_meld_pkl", type=str, default="./MMGCN/unified_features/meld_anjs4_unified.pkl")
    parser.add_argument("--out_ejsl_pkl", type=str, default="./MMGCN/unified_features/ejsl_anjs4_unified.pkl")
    parser.add_argument("--cache_dir", type=str, default="./MMGCN/unified_features/cache")
    parser.add_argument("--text_model", type=str, default="roberta-large")
    parser.add_argument("--text_pooling", choices=["mean", "cls", "last"], default="mean")
    parser.add_argument("--text_max_length", type=int, default=96)
    parser.add_argument("--text_batch_size", type=int, default=32)
    parser.add_argument("--video_model", type=str, default="facebook/timesformer-base-finetuned-k400")
    parser.add_argument("--video_processor", type=str, default="MCG-NJU/videomae-base")
    parser.add_argument("--video_max_seconds", type=float, default=30.0)
    parser.add_argument("--audio_dim", type=int, default=768)
    parser.add_argument("--n_speakers", type=int, default=9)
    parser.add_argument("--keep_non_anjs_context", action="store_true", default=True)
    parser.add_argument("--drop_non_anjs_context", dest="keep_non_anjs_context", action="store_false")
    parser.add_argument("--skip_meld", action="store_true")
    parser.add_argument("--skip_ejsl", action="store_true")
    parser.add_argument("--limit_meld_dialogues", type=int, default=0)
    parser.add_argument("--limit_ejsl_dialogues", type=int, default=0)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--no_cuda", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.skip_meld and args.skip_ejsl:
        raise RuntimeError("Both --skip_meld and --skip_ejsl were set")
    if not args.skip_ejsl and not args.ejsl_txt_root:
        raise RuntimeError("--ejsl_txt_root is required unless --skip_ejsl is set")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    cache = FeatureCache(resolve_path(args.cache_dir, must_exist=False) if args.cache_dir else None)
    print(f"[features] device={device}")
    print(f"[features] cache_dir={cache.root}")

    text_encoder = TextEncoder(
        args.text_model,
        args.text_pooling,
        args.text_max_length,
        args.text_batch_size,
        device,
        cache,
    )
    video_encoder = VideoEncoder(
        args.video_model,
        args.video_processor,
        device,
        cache,
        args.video_max_seconds,
        args.fp16,
    )

    summaries: Dict[str, Dict] = {}
    if not args.skip_meld:
        meld_dialogues, meld_train_vid, meld_test_vid = build_meld_dialogues(args)
        summaries["meld"] = write_feature_pkl(
            meld_dialogues,
            meld_train_vid,
            meld_test_vid,
            resolve_path(args.out_meld_pkl, must_exist=False),
            text_encoder,
            video_encoder,
            args.audio_dim,
            args.n_speakers,
            {
                "dataset": "MELD",
                "label_space": "ANJS4",
                "unknown_context_label": -100,
                "keep_non_anjs_context": args.keep_non_anjs_context,
                "text_model": args.text_model,
                "video_model": args.video_model,
                "video_processor": args.video_processor,
            },
        )

    if not args.skip_ejsl:
        ejsl_dialogues, ejsl_train_vid, ejsl_test_vid = build_ejsl_dialogues(args)
        summaries["ejsl"] = write_feature_pkl(
            ejsl_dialogues,
            ejsl_train_vid,
            ejsl_test_vid,
            resolve_path(args.out_ejsl_pkl, must_exist=False),
            text_encoder,
            video_encoder,
            args.audio_dim,
            args.n_speakers,
            {
                "dataset": "eJSL",
                "label_space": "ANJS4",
                "unknown_context_label": -100,
                "source_txt_root": args.ejsl_txt_root,
                "text_model": args.text_model,
                "video_model": args.video_model,
                "video_processor": args.video_processor,
            },
        )

    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
