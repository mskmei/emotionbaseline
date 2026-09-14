#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


ANJS_LABELS = ["A", "N", "J", "S"]
MELD_TO_ANJS = {
    "anger": 0,
    "neutral": 1,
    "joy": 2,
    "sadness": 3,
}
EJSL_TO_ANJS = {"A": 0, "N": 1, "J": 2, "S": 3}


@dataclass(frozen=True)
class UtteranceItem:
    text: str
    speaker: str
    label: int
    source: str
    split: str
    dialogue_id: str
    utterance_id: int
    video_path: Optional[Path] = None
    frame_dir: Optional[Path] = None
    frame_prefix: str = ""
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


def resolve_path(raw: str | Path, must_exist: bool = False, base: Optional[Path] = None) -> Path:
    path = Path(raw).expanduser()
    candidates = [path] if path.is_absolute() else []
    if not path.is_absolute():
        if base is not None:
            candidates.append(base / path)
        candidates.append(Path.cwd() / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    if must_exist:
        checked = ", ".join(str(x) for x in candidates) or str(path)
        raise FileNotFoundError(f"Path not found: {raw}. Checked: {checked}")
    return candidates[0].resolve() if candidates else path.resolve()


def normalize_text(value: str) -> str:
    return " ".join(str(value or "").replace("\t", " ").split())


def label_counts(dialogues: Dict[str, List[UtteranceItem]], keys: Iterable[str]) -> Dict[str, int]:
    labels: List[int] = []
    for key in keys:
        labels.extend(item.label for item in dialogues[key] if item.label >= 0)
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=len(ANJS_LABELS)).astype(int).tolist() if labels else [0] * len(ANJS_LABELS)
    return dict(zip(ANJS_LABELS, counts))


def make_speaker_onehot(speakers: Sequence[str], n_speakers: int) -> np.ndarray:
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


def make_speaker_ids(speakers: Sequence[str], max_speakers: int) -> np.ndarray:
    speaker_to_idx: Dict[str, int] = {}
    ids: List[int] = []
    for speaker in speakers:
        if speaker not in speaker_to_idx:
            speaker_to_idx[speaker] = len(speaker_to_idx)
        ids.append(int(speaker_to_idx[speaker] % max_speakers))
    return np.asarray(ids, dtype=np.int64)


def resolve_meld_video_path(raw_video_path: str, meld_root: Path) -> Optional[Path]:
    raw = Path(str(raw_video_path or "").strip())
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


def read_meld_split(
    csv_path: Path,
    split: str,
    meld_root: Path,
    keep_non_anjs_context: bool,
) -> Dict[str, List[UtteranceItem]]:
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
                text=normalize_text(str(row.get("Utterance", ""))),
                speaker=normalize_text(str(row.get("Speaker", ""))) or "speaker",
                label=label,
                source="MELD",
                split=split,
                dialogue_id=vid,
                utterance_id=utterance_id,
                video_path=resolve_meld_video_path(str(row.get("Video_Path", "")), meld_root),
                sample_id=f"{vid}_utt{utterance_id}",
            )
            grouped.setdefault(vid, []).append((utterance_id, item))
    return {vid: [item for _idx, item in sorted(items, key=lambda x: x[0])] for vid, items in grouped.items() if items}


def read_meld_dialogues(
    meld_root: str | Path,
    keep_non_anjs_context: bool = True,
    limit_meld_dialogues: int = 0,
) -> Tuple[Dict[str, List[UtteranceItem]], List[str], List[str], Dict]:
    root = resolve_path(meld_root, must_exist=True)
    split_files = {
        "train": root / "train_meld_emo.csv",
        "dev": root / "dev_meld_emo.csv",
        "test": root / "test_meld_emo.csv",
    }
    for split, path in split_files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing MELD {split} csv: {path}")

    dialogues: Dict[str, List[UtteranceItem]] = {}
    train_ids: List[str] = []
    test_ids: List[str] = []
    split_summary: Dict[str, Dict] = {}
    for split in ["train", "dev", "test"]:
        split_dialogues = read_meld_split(split_files[split], split, root, keep_non_anjs_context)
        keys = sorted(split_dialogues, key=lambda x: (x.split("_dia")[0], int(x.rsplit("dia", 1)[1])))
        if limit_meld_dialogues > 0:
            keys = keys[:limit_meld_dialogues]
            split_dialogues = {key: split_dialogues[key] for key in keys}
        dialogues.update(split_dialogues)
        if split in {"train", "dev"}:
            train_ids.extend(keys)
        else:
            test_ids.extend(keys)
        split_summary[split] = {
            "dialogues": len(keys),
            "utterances": sum(len(split_dialogues[key]) for key in keys),
            "label_counts": label_counts(split_dialogues, keys),
        }
        print(f"[MELD] split={split} dialogues={len(keys)} labels={split_summary[split]['label_counts']}")
    return dialogues, train_ids, test_ids, {"root": str(root), "splits": split_summary}


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
        turns.append((normalize_text(parts[0]), normalize_text(parts[1]), normalize_text(parts[2])))
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


def read_ejsl_dialogues(
    txt_root: str | Path,
    dial_list: str | Path,
    frame_root: str | Path,
    mp4_root: str | Path,
    limit_ejsl_dialogues: int = 0,
) -> Tuple[Dict[str, List[UtteranceItem]], List[str], List[str], Dict]:
    txt_root_path = resolve_path(txt_root, must_exist=True)
    dial_list_path = resolve_path(dial_list, must_exist=True)
    frame_root_path = resolve_path(frame_root, must_exist=True)
    mp4_root_path = resolve_path(mp4_root, must_exist=False)

    grouped_required: Dict[Tuple[str, int], Dict[int, Tuple[str, str]]] = {}
    for sample_id in read_ejsl_names(dial_list_path):
        parsed = parse_ejsl_sample_id(sample_id)
        if parsed is None:
            continue
        sd_id, dialogue_idx, utterance_idx, label = parsed
        grouped_required.setdefault((sd_id, dialogue_idx), {})[utterance_idx] = (label, sample_id)

    keys = sorted(grouped_required)
    if limit_ejsl_dialogues > 0:
        keys = keys[:limit_ejsl_dialogues]

    dialogues: Dict[str, List[UtteranceItem]] = {}
    test_ids: List[str] = []
    dropped = 0
    for sd_id, dialogue_idx in keys:
        required = grouped_required[(sd_id, dialogue_idx)]
        max_utt = max(required)
        txt_file = txt_root_path / sd_id / "txt" / f"{sd_id}-Dialogue-{dialogue_idx:02d}.txt"
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
            label_char, sample_id = required.get(utterance_idx, (None, ""))
            label = EJSL_TO_ANJS[label_char] if label_char in EJSL_TO_ANJS else -100
            mp4_path, frame_dir, resolved_sample_id = find_ejsl_media(
                sd_id,
                dialogue_idx,
                utterance_idx,
                label_char,
                frame_root_path,
                mp4_root_path,
            )
            items.append(
                UtteranceItem(
                    text=text,
                    speaker=speaker or "speaker",
                    label=label,
                    source="eJSL",
                    split="test",
                    dialogue_id=vid,
                    utterance_id=utterance_idx,
                    video_path=mp4_path,
                    frame_dir=frame_dir,
                    sample_id=sample_id or resolved_sample_id,
                )
            )
        dialogues[vid] = items
        test_ids.append(vid)

    summary = {
        "txt_root": str(txt_root_path),
        "dial_list": str(dial_list_path),
        "frame_root": str(frame_root_path),
        "mp4_root": str(mp4_root_path),
        "dialogues": len(test_ids),
        "utterances": sum(len(dialogues[key]) for key in test_ids),
        "dropped_required_samples": dropped,
        "label_counts": label_counts(dialogues, test_ids),
    }
    print(f"[eJSL] dialogues={len(test_ids)} dropped={dropped} labels={summary['label_counts']}")
    return dialogues, [], test_ids, summary


def resolve_under_root(raw: str, root: Path, default_name: str, must_exist: bool = True) -> Path:
    value = raw or default_name
    path = Path(value).expanduser()
    candidates = [path] if path.is_absolute() else [root / path, Path.cwd() / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    if must_exist:
        raise FileNotFoundError(f"Path not found: {value}. Checked: {', '.join(str(x) for x in candidates)}")
    return candidates[0].resolve()


def resolve_bobsl_video_path(bobsl_root: Path, video_subdir: str, stem: str, clip_name: str) -> Optional[Path]:
    candidates = [
        bobsl_root / video_subdir / stem / clip_name,
        bobsl_root / "video" / stem / clip_name,
        bobsl_root / stem / clip_name,
        bobsl_root / clip_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0]


def read_bobsl_split(
    csv_path: Path,
    split_name: str,
    bobsl_root: Path,
    video_subdir: str,
    frame_subdir: str,
    min_score: float,
    limit: int,
) -> Tuple[Dict[str, List[UtteranceItem]], List[str], Dict]:
    dialogues: Dict[str, List[UtteranceItem]] = {}
    keys: List[str] = []
    skipped_emotion = 0
    skipped_score = 0
    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            emotion = str(row.get("emotion", "")).strip().lower()
            label = MELD_TO_ANJS.get(emotion, -100)
            if label < 0:
                skipped_emotion += 1
                continue
            try:
                score = float(row.get("score", 0.0) or 0.0)
            except ValueError:
                score = 0.0
            if score < min_score:
                skipped_score += 1
                continue

            stem = str(row.get("stem", "")).strip()
            clip_name = str(row.get("clip_name", "")).strip()
            if not stem or not clip_name:
                skipped_emotion += 1
                continue
            clip_stem = Path(clip_name).stem
            base = f"bobsl_{split_name}_{stem}_{clip_stem}"
            vid = base
            index = 2
            while vid in dialogues:
                vid = f"{base}_{index}"
                index += 1
            item = UtteranceItem(
                text=normalize_text(str(row.get("text", ""))),
                speaker="bobsl",
                label=label,
                source="BOBSL",
                split=split_name,
                dialogue_id=vid,
                utterance_id=0,
                video_path=resolve_bobsl_video_path(bobsl_root, video_subdir, stem, clip_name),
                frame_dir=bobsl_root / frame_subdir / stem,
                frame_prefix=clip_stem,
                sample_id=f"{stem}/{clip_name}",
            )
            dialogues[vid] = [item]
            keys.append(vid)
            if limit > 0 and len(keys) >= limit:
                break

    summary = {
        "csv_path": str(csv_path),
        "split": split_name,
        "dialogues": len(keys),
        "utterances": len(keys),
        "label_counts": label_counts(dialogues, keys),
        "skipped_emotion": skipped_emotion,
        "skipped_score": skipped_score,
        "min_score": min_score,
    }
    print(f"[BOBSL] split={split_name} dialogues={len(keys)} labels={summary['label_counts']}")
    return dialogues, keys, summary


def read_bobsl_dialogues(
    bobsl_root: str | Path,
    train_csv: str = "",
    val_csv: str = "",
    test_csv: str = "",
    video_subdir: str = "clip256",
    frame_subdir: str = "frame",
    min_score: float = 0.0,
    limit_train: int = 0,
    limit_val: int = 0,
    limit_test: int = 0,
) -> Tuple[Dict[str, List[UtteranceItem]], List[str], List[str], List[str], Dict]:
    root = resolve_path(bobsl_root, must_exist=True)
    train_path = resolve_under_root(train_csv, root, "train_clips_balanced_updated.csv")
    val_path = resolve_under_root(val_csv, root, "val_clips_balanced_updated.csv")
    test_path = resolve_under_root(test_csv, root, "test_clips_balanced_updated.csv")

    train_dialogues, train_ids, train_summary = read_bobsl_split(
        train_path, "train", root, video_subdir, frame_subdir, min_score, limit_train
    )
    val_dialogues, val_ids, val_summary = read_bobsl_split(
        val_path, "val", root, video_subdir, frame_subdir, min_score, limit_val
    )
    test_dialogues, test_ids, test_summary = read_bobsl_split(
        test_path, "test", root, video_subdir, frame_subdir, min_score, limit_test
    )
    dialogues = {**train_dialogues, **val_dialogues, **test_dialogues}
    return dialogues, train_ids, val_ids, test_ids, {
        "root": str(root),
        "video_subdir": video_subdir,
        "frame_subdir": frame_subdir,
        "splits": {"train": train_summary, "val": val_summary, "test": test_summary},
    }


def choose_keys(keys: Sequence[str], max_count: int, seed: int) -> List[str]:
    chosen = list(keys)
    rng = random.Random(seed)
    rng.shuffle(chosen)
    if max_count > 0:
        chosen = chosen[:max_count]
    return chosen


def make_joint_dialogues(
    meld_dialogues: Dict[str, List[UtteranceItem]],
    meld_train_ids: Sequence[str],
    meld_test_ids: Sequence[str],
    bobsl_dialogues: Dict[str, List[UtteranceItem]],
    bobsl_train_ids: Sequence[str],
    max_bobsl_train_dialogues: int,
    bobsl_sample_seed: int,
) -> Tuple[Dict[str, List[UtteranceItem]], List[str], List[str], List[str]]:
    out: Dict[str, List[UtteranceItem]] = {}
    train_ids: List[str] = []
    test_ids: List[str] = []

    for key in meld_train_ids:
        out[key] = list(meld_dialogues[key])
        train_ids.append(key)
    for key in meld_test_ids:
        out[key] = list(meld_dialogues[key])
        test_ids.append(key)

    selected_bobsl = choose_keys(bobsl_train_ids, max_bobsl_train_dialogues, bobsl_sample_seed)
    for key in selected_bobsl:
        target_key = f"joint_{key}"
        suffix = 2
        while target_key in out:
            target_key = f"joint_{key}_{suffix}"
            suffix += 1
        out[target_key] = [replace(item, dialogue_id=target_key) for item in bobsl_dialogues[key]]
        train_ids.append(target_key)

    return out, train_ids, test_ids, selected_bobsl


def _import_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for video/frame loading.") from exc
    return cv2


def sample_indices(length: int, num_frames: int) -> np.ndarray:
    if length <= 0:
        return np.asarray([], dtype=np.int64)
    return np.linspace(0, length - 1, num=num_frames, dtype=np.int64)


def read_frames_from_video(path: Path, num_frames: int, max_seconds: float) -> Tuple[List[np.ndarray], str]:
    cv2 = _import_cv2()
    cap = cv2.VideoCapture(str(path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if frame_count <= 0:
        cap.release()
        return [], "mp4_empty"
    if max_seconds > 0 and fps > 0 and frame_count / fps > max_seconds:
        cap.release()
        return [], "mp4_too_long"

    frames: List[np.ndarray] = []
    for index in sample_indices(frame_count, num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, image = cap.read()
        if ok and image is not None:
            frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    cap.release()
    if frames and len(frames) < num_frames:
        frames.extend([frames[-1].copy() for _ in range(num_frames - len(frames))])
    return frames[:num_frames], "mp4" if frames else "mp4_empty"


def sample_frame_paths(frame_dir: Path, frame_prefix: str, num_frames: int) -> List[Path]:
    if frame_prefix:
        paths = sorted(frame_dir.glob(f"{frame_prefix}_*.jpg")) + sorted(frame_dir.glob(f"{frame_prefix}_*.png"))
    else:
        paths = sorted(frame_dir.glob("*.jpg")) + sorted(frame_dir.glob("*.png"))
    if not paths:
        return []
    indices = sample_indices(len(paths), num_frames)
    selected = [paths[int(index)] for index in indices]
    if selected and len(selected) < num_frames:
        selected.extend([selected[-1]] * (num_frames - len(selected)))
    return selected[:num_frames]


def read_frames_from_dir(frame_dir: Path, frame_prefix: str, num_frames: int) -> Tuple[List[np.ndarray], str]:
    cv2 = _import_cv2()
    frame_paths = sample_frame_paths(frame_dir, frame_prefix, num_frames)
    frames: List[np.ndarray] = []
    for path in frame_paths:
        image = cv2.imread(str(path))
        if image is not None:
            frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if frames and len(frames) < num_frames:
        frames.extend([frames[-1].copy() for _ in range(num_frames - len(frames))])
    return frames[:num_frames], "frames" if frames else "frames_empty"


def read_item_frames(item: UtteranceItem, num_frames: int, max_seconds: float) -> Tuple[List[np.ndarray], str]:
    if item.video_path is not None and item.video_path.exists():
        frames, status = read_frames_from_video(item.video_path, num_frames, max_seconds)
        if frames:
            return frames, status
    if item.frame_dir is not None and item.frame_dir.exists():
        frames, status = read_frames_from_dir(item.frame_dir, item.frame_prefix, num_frames)
        if frames:
            return frames, status
        return [], status
    return [], "missing"


def item_media_key(item: UtteranceItem, num_frames: int, max_seconds: float) -> Dict:
    if item.video_path is not None and item.video_path.exists():
        stat = item.video_path.stat()
        source = {"path": str(item.video_path.resolve()), "mtime": stat.st_mtime_ns, "size": stat.st_size}
    elif item.frame_dir is not None and item.frame_dir.exists():
        paths = sample_frame_paths(item.frame_dir, item.frame_prefix, num_frames)
        if paths:
            first, last = paths[0], paths[-1]
            source = {
                "frame_dir": str(item.frame_dir.resolve()),
                "frame_prefix": item.frame_prefix,
                "count": len(paths),
                "first": first.name,
                "last": last.name,
                "first_mtime": first.stat().st_mtime_ns,
                "last_mtime": last.stat().st_mtime_ns,
            }
        else:
            source = {"frame_dir": str(item.frame_dir.resolve()), "frame_prefix": item.frame_prefix, "empty": True}
    else:
        source = {"missing": True, "sample_id": item.sample_id}
    return {"source": source, "num_frames": num_frames, "max_seconds": max_seconds}


def resolve_hf_or_local(name: str, local_root: str | Path = "") -> str:
    if not local_root:
        return name
    root = Path(local_root).expanduser()
    candidates = [root / name, root / Path(name).name]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return name


class VideoEmbeddingEncoder:
    def __init__(
        self,
        video_model: str,
        video_processor: str,
        cache: FeatureCache,
        device: torch.device,
        local_model_root: str | Path = "",
        num_frames: int = 8,
        max_seconds: float = 30.0,
        fp16: bool = False,
    ):
        from transformers import AutoImageProcessor, TimesformerModel

        self.video_model = resolve_hf_or_local(video_model, local_model_root)
        self.video_processor = resolve_hf_or_local(video_processor, local_model_root)
        self.cache = cache
        self.device = device
        self.num_frames = int(num_frames)
        self.max_seconds = float(max_seconds)
        self.fp16 = bool(fp16)
        self.processor = AutoImageProcessor.from_pretrained(self.video_processor)
        self.model = TimesformerModel.from_pretrained(self.video_model).to(device)
        self.model.eval()
        self.dim = int(self.model.config.hidden_size)

    def _key(self, item: UtteranceItem) -> str:
        return json.dumps(
            {
                "encoder": "timesformer_cls_embedding",
                "video_model": self.video_model,
                "video_processor": self.video_processor,
                **item_media_key(item, self.num_frames, self.max_seconds),
            },
            sort_keys=True,
        )

    def encode_item(self, item: UtteranceItem) -> Tuple[np.ndarray, str]:
        key = self._key(item)
        cached = self.cache.get("video_embedding", key)
        if cached is not None:
            return cached, "cache"
        frames, status = read_item_frames(item, self.num_frames, self.max_seconds)
        if not frames:
            vec = np.zeros(self.dim, dtype=np.float32)
            self.cache.put("video_embedding", key, vec)
            return vec, status
        inputs = self.processor(frames, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.device.type == "cuda" and self.fp16):
                hidden = self.model(pixel_values).last_hidden_state[:, 0, :]
        vec = hidden[0].detach().cpu().numpy().astype(np.float32)
        self.cache.put("video_embedding", key, vec)
        return vec, status

