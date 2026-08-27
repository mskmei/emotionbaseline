import argparse
import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from sklearn import metrics
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoProcessor, RobertaTokenizer

from model import ASF, Student_Audio, Student_Video, Teacher_model


IEMOCAP_LABELS = ["ang", "exc", "fru", "hap", "neu", "sad"]
TARGET_LABELS = ["A", "N", "J", "S"]
TARGET_TO_ID = {name: idx for idx, name in enumerate(TARGET_LABELS)}
MODALITY_ALIASES = {
    "full": "full",
    "avl": "full",
    "text": "text",
    "t": "text",
    "l": "text",
    "video": "video",
    "visual": "video",
    "v": "video",
}
MISSING_AUDIO_ALIASES = {
    "silence": "silence",
    "zero_waveform": "silence",
    "old_zero": "silence",
    "zero": "silence",
    "zero_hidden": "zero_hidden",
    "noise": "pure_noise",
    "pure_noise": "pure_noise",
    "mean": "source_mean",
    "source_mean": "source_mean",
}


audio_processor = AutoProcessor.from_pretrained("facebook/data2vec-audio-base-960h")
video_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
speaker_list = ["<s1>", "<s2>", "<s3>", "<s4>", "<s5>", "<s6>", "<s7>", "<s8>", "<s9>"]
roberta_tokenizer.add_special_tokens({"additional_special_tokens": speaker_list})


@dataclass
class Config:
    mask_time_length: int = 3


@dataclass
class Sample:
    sample_id: str
    sd_id: str
    dialogue_idx: int
    utterance_idx: int
    label_name: str
    frame_dir: Path
    mp4_path: Optional[Path] = None


def encode_right_truncated(text: str, tokenizer: RobertaTokenizer, max_length: int = 511) -> List[int]:
    tokenized = tokenizer.tokenize(text)
    truncated = tokenized[-max_length:]
    ids = tokenizer.convert_tokens_to_ids(truncated)
    return ids + [tokenizer.mask_token_id]


def padding(ids_list: List[List[int]], tokenizer: RobertaTokenizer) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(ids) for ids in ids_list)
    pad_ids = []
    attention_masks = []
    for ids in ids_list:
        pad_len = max_len - len(ids)
        add_ids = [tokenizer.pad_token_id for _ in range(pad_len)]
        attention_mask = [1 for _ in range(len(ids))]
        add_attention = [0 for _ in range(len(add_ids))]
        pad_ids.append(add_ids + ids)
        attention_masks.append(add_attention + attention_mask)
    return torch.tensor(pad_ids), torch.tensor(attention_masks)


def padding_audio(batch: List[torch.Tensor]) -> torch.Tensor:
    max_len = max(len(x) for x in batch)
    padded = []
    for x in batch:
        pad_len = max_len - len(x)
        if pad_len > 0:
            x = torch.cat([torch.zeros(pad_len, dtype=x.dtype), x], dim=0)
        padded.append(x)
    return torch.stack(padded, dim=0)


def normalize_eval_modality(value: str) -> str:
    key = (value or "full").strip().lower()
    if key not in MODALITY_ALIASES:
        valid = ", ".join(sorted(MODALITY_ALIASES))
        raise ValueError(f"Unsupported eval_modality={value!r}; valid values: {valid}")
    return MODALITY_ALIASES[key]


def normalize_missing_audio_strategy(value: str) -> str:
    key = (value or "silence").strip().lower()
    if key not in MISSING_AUDIO_ALIASES:
        valid = ", ".join(sorted(MISSING_AUDIO_ALIASES))
        raise ValueError(f"Unsupported missing_audio_strategy={value!r}; valid values: {valid}")
    return MISSING_AUDIO_ALIASES[key]


def sample_eight_frames(frame_paths: List[Path]) -> List[np.ndarray]:
    if not frame_paths:
        raise ValueError("empty frame folder")
    if len(frame_paths) >= 8:
        idx = np.linspace(0, len(frame_paths) - 1, num=8, dtype=int)
        chosen = [frame_paths[i] for i in idx]
    else:
        chosen = frame_paths + [frame_paths[-1]] * (8 - len(frame_paths))

    frames_rgb = []
    for p in chosen:
        img = cv2.imread(str(p))
        if img is None:
            raise ValueError(f"failed to read frame: {p}")
        frames_rgb.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return frames_rgb


def get_video_from_frame_dir(frame_dir: Path) -> torch.Tensor:
    frame_paths = sorted(frame_dir.glob("*.jpg"))
    if not frame_paths:
        frame_paths = sorted(frame_dir.glob("*.png"))
    frames_rgb = sample_eight_frames(frame_paths)
    inputs = video_processor(frames_rgb, return_tensors="pt")
    return inputs["pixel_values"][0]


def get_video_from_mp4(video_path: Path) -> torch.Tensor:
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        raise ValueError(f"failed to read any frame from mp4: {video_path}")

    if len(frames) >= 8:
        idx = np.linspace(0, len(frames) - 1, num=8, dtype=int)
        chosen = [frames[i] for i in idx]
    else:
        chosen = frames + [frames[-1]] * (8 - len(frames))

    inputs = video_processor(chosen[:8], return_tensors="pt")
    return inputs["pixel_values"][0]


def get_audio_from_file_or_silence(wav_path: Optional[Path], silence_seconds: float = 1.0) -> torch.Tensor:
    if wav_path is not None and wav_path.exists():
        import librosa
        try:
            audio, _ = librosa.load(str(wav_path), sr=16000)
        except Exception:
            audio = np.zeros(int(16000 * silence_seconds), dtype=np.float32)
    else:
        audio = np.zeros(int(16000 * silence_seconds), dtype=np.float32)

    inputs = audio_processor(audio, sampling_rate=16000, return_tensors="pt")
    return inputs["input_values"][0]


def load_audio_from_file(wav_path: Path) -> Optional[torch.Tensor]:
    if not wav_path.exists() or not wav_path.is_file():
        return None

    import librosa

    try:
        audio, _ = librosa.load(str(wav_path), sr=16000)
    except Exception:
        return None
    inputs = audio_processor(audio, sampling_rate=16000, return_tensors="pt")
    return inputs["input_values"][0]


def resolve_csv_media_path(csv_path: Path, raw_path: str) -> Path:
    p = Path(str(raw_path).strip())
    if p.is_absolute():
        return p
    cwd_candidate = (Path.cwd() / p).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (csv_path.parent / p).resolve()


def load_audio_hidden_stats(stats_path: Path, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, int]:
    if not stats_path.exists():
        raise FileNotFoundError(f"audio stats file not found: {stats_path}")

    data = np.load(stats_path)
    if isinstance(data, np.lib.npyio.NpzFile):
        mean = data["mean"].astype(np.float32)
        std = data["std"].astype(np.float32) if "std" in data.files else np.ones_like(mean, dtype=np.float32)
        count = int(data["count"]) if "count" in data.files else -1
    else:
        mean = np.asarray(data, dtype=np.float32)
        std = np.ones_like(mean, dtype=np.float32)
        count = -1

    return torch.from_numpy(mean).to(device), torch.from_numpy(std).to(device), count


def collect_audio_paths_from_iemocap_csv(csv_path: Path, max_samples: int = 0) -> List[Path]:
    if not csv_path.exists():
        raise FileNotFoundError(f"audio stats source csv not found: {csv_path}")

    paths = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_path = row.get("Wav_Path", "")
            if not raw_path:
                continue
            p = resolve_csv_media_path(csv_path, raw_path)
            if p.exists() and p.is_file():
                paths.append(p)
            if max_samples > 0 and len(paths) >= max_samples:
                break
    return paths


def compute_audio_hidden_stats(
    audio_s,
    source_csv: Path,
    device: torch.device,
    batch_size: int,
    max_samples: int,
    save_path: Optional[Path] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    audio_paths = collect_audio_paths_from_iemocap_csv(source_csv, max_samples=max_samples)
    if not audio_paths:
        raise RuntimeError(
            "No usable source wav files found for audio mean/std. "
            f"Check --audio_stats_source_csv or provide --audio_stats_path: {source_csv}"
        )

    hidden_chunks = []
    batch = []
    with torch.no_grad():
        for p in audio_paths:
            audio_input = load_audio_from_file(p)
            if audio_input is None:
                continue
            batch.append(audio_input)
            if len(batch) >= batch_size:
                audios = padding_audio(batch).to(device)
                audio_hidden, _ = audio_s(audios)
                hidden_chunks.append(audio_hidden.detach().cpu())
                batch = []
        if batch:
            audios = padding_audio(batch).to(device)
            audio_hidden, _ = audio_s(audios)
            hidden_chunks.append(audio_hidden.detach().cpu())

    if not hidden_chunks:
        raise RuntimeError(f"No audio hidden vectors could be computed from: {source_csv}")

    hidden = torch.cat(hidden_chunks, dim=0)
    mean = hidden.mean(dim=0).to(device)
    std = hidden.std(dim=0, unbiased=False).clamp_min(1e-6).to(device)
    count = int(hidden.shape[0])

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            save_path,
            mean=mean.detach().cpu().numpy().astype(np.float32),
            std=std.detach().cpu().numpy().astype(np.float32),
            count=np.array(count, dtype=np.int64),
        )
        print(f"[AudioStats] saved mean/std/count={count}: {save_path}")

    return mean, std, count


def get_or_build_audio_hidden_stats(args, audio_s, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, int]:
    stats_path = Path(args.audio_stats_path) if args.audio_stats_path else None
    if stats_path is not None and stats_path.exists():
        mean, std, count = load_audio_hidden_stats(stats_path, device)
        print(f"[AudioStats] loaded count={count}: {stats_path}")
        return mean, std, count

    if not args.audio_stats_source_csv:
        raise RuntimeError("source_mean imputation requires --audio_stats_path or --audio_stats_source_csv")

    return compute_audio_hidden_stats(
        audio_s=audio_s,
        source_csv=Path(args.audio_stats_source_csv),
        device=device,
        batch_size=max(int(args.batch_size), 1),
        max_samples=int(args.audio_stats_max_samples),
        save_path=stats_path,
    )


def make_imputed_audio_hidden(
    reference_hidden: torch.Tensor,
    strategy: str,
    mean: Optional[torch.Tensor],
    std: Optional[torch.Tensor],
    noise_scale: float,
) -> torch.Tensor:
    if strategy == "zero_hidden":
        return torch.zeros_like(reference_hidden)
    if strategy == "pure_noise":
        return torch.randn_like(reference_hidden) * float(noise_scale)
    if strategy == "source_mean":
        if mean is None:
            raise RuntimeError("source_mean audio imputation requires source audio hidden mean stats")
        return mean.to(reference_hidden.device, dtype=reference_hidden.dtype).view(1, -1).expand_as(reference_hidden)
    raise ValueError(f"Unsupported hidden audio imputation strategy: {strategy}")


def parse_sample_id(name: str) -> Optional[Sample]:
    m = re.match(r"^(SD\d{2})-(\d{2})-(\d{2})([AJNS])$", name)
    if m is None:
        return None
    sd_id, dialogue, utterance, label = m.groups()
    return Sample(
        sample_id=name,
        sd_id=sd_id,
        dialogue_idx=int(dialogue),
        utterance_idx=int(utterance),
        label_name=label,
        frame_dir=Path(),
    )


def parse_dialogue_txt(txt_file: Path) -> List[Tuple[str, str, str]]:
    turns = []
    for raw in txt_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < 3:
            continue
        speaker = parts[0].strip()
        emotion = parts[1].strip()
        utterance = parts[2].strip()
        turns.append((speaker, emotion, utterance))
    return turns


def build_session_for_sample(
    sample: Sample,
    txt_root: Path,
    wav_map: Dict[str, Path],
) -> Optional[List[Tuple[int, str, Path, Optional[Path], Optional[Path], str]]]:
    txt_file = txt_root / sample.sd_id / "txt" / f"{sample.sd_id}-Dialogue-{sample.dialogue_idx:02d}.txt"
    if not txt_file.exists():
        return None

    turns = parse_dialogue_txt(txt_file)
    if sample.utterance_idx <= 0 or sample.utterance_idx > len(turns):
        return None

    speaker_to_idx: Dict[str, int] = {}
    session = []
    for i in range(sample.utterance_idx):
        speaker, _emotion_jp, utterance = turns[i]
        if speaker not in speaker_to_idx:
            speaker_to_idx[speaker] = len(speaker_to_idx)
        speaker_idx = speaker_to_idx[speaker]

        # TelME's batch format uses only the last turn's frame/audio and label.
        if i == sample.utterance_idx - 1:
            frame_dir = sample.frame_dir
            wav_path = wav_map.get(sample.sample_id)
            mp4_path = sample.mp4_path
            label_name = sample.label_name
        else:
            frame_dir = sample.frame_dir
            wav_path = None
            mp4_path = sample.mp4_path
            label_name = sample.label_name

        session.append((speaker_idx, utterance, frame_dir, wav_path, mp4_path, label_name))

    return session


def build_wav_map(wav_map_csv: Optional[Path]) -> Dict[str, Path]:
    if wav_map_csv is None:
        return {}
    if not wav_map_csv.exists():
        raise FileNotFoundError(f"wav_map_csv not found: {wav_map_csv}")

    mapping: Dict[str, Path] = {}
    for raw in wav_map_csv.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 2:
            continue
        mapping[parts[0]] = Path(parts[1])
    return mapping


class EJSLDataset(Dataset):
    def __init__(self, sessions: List[List[Tuple[int, str, Path, Optional[Path], Optional[Path], str]]]):
        self.sessions = sessions

    def __len__(self) -> int:
        return len(self.sessions)

    def __getitem__(self, idx: int):
        return self.sessions[idx]


def make_batchs(sessions, eval_modality: str = "full", missing_audio_strategy: str = "silence"):
    eval_modality = normalize_eval_modality(eval_modality)
    missing_audio_strategy = normalize_missing_audio_strategy(missing_audio_strategy)
    batch_input = []
    batch_audio = []
    batch_video = []
    batch_labels = []

    for session in sessions:
        input_string = ""
        now_speaker = None

        for speaker, utt, _frame_dir, _wav_path, _mp4_path, _label_name in session:
            input_string += f"<s{speaker + 1}> "
            input_string += utt + " "
            now_speaker = speaker

        _last_speaker, _last_utt, last_frame_dir, last_wav_path, last_mp4_path, last_label = session[-1]

        # Build text prompt consistent with original TelME.
        prompt = f"Now <s{now_speaker + 1}> feels"
        concat_string = input_string.strip() + " </s> " + prompt
        batch_input.append(encode_right_truncated(concat_string, roberta_tokenizer))

        if eval_modality == "full" and missing_audio_strategy == "silence":
            audio_source = last_wav_path if last_wav_path is not None else last_mp4_path
            audio_input = get_audio_from_file_or_silence(audio_source)
        else:
            audio_input = get_audio_from_file_or_silence(None)

        if eval_modality in {"full", "video"}:
            if last_mp4_path is not None and last_mp4_path.exists():
                video_input = get_video_from_mp4(last_mp4_path)
            else:
                video_input = get_video_from_frame_dir(last_frame_dir)
        else:
            video_input = torch.zeros((8, 3, 224, 224), dtype=torch.float32)
        batch_audio.append(audio_input)
        batch_video.append(video_input)

        batch_labels.append(TARGET_TO_ID[last_label])

    batch_input_tokens, batch_attention_masks = padding(batch_input, roberta_tokenizer)
    batch_audio = padding_audio(batch_audio)
    batch_video = torch.stack(batch_video)
    batch_labels = torch.tensor(batch_labels)

    return batch_input_tokens, batch_attention_masks, batch_audio, batch_video, batch_labels


def map_logits_6_to_probs_4(logits_6: torch.Tensor) -> torch.Tensor:
    probs_6 = torch.softmax(logits_6, dim=-1)
    p_ang = probs_6[:, 0]
    p_exc = probs_6[:, 1]
    p_fru = probs_6[:, 2]
    p_hap = probs_6[:, 3]
    p_neu = probs_6[:, 4]
    p_sad = probs_6[:, 5]

    p_a = p_ang + p_fru
    p_n = p_neu
    p_j = p_hap + p_exc
    p_s = p_sad
    return torch.stack([p_a, p_n, p_j, p_s], dim=-1)


def evaluation(
    model_t,
    audio_s,
    video_s,
    fusion,
    dataloader,
    device,
    eval_modality: str = "full",
    missing_audio_strategy: str = "silence",
    audio_mean: Optional[torch.Tensor] = None,
    audio_std: Optional[torch.Tensor] = None,
    audio_noise_scale: float = 1.0,
):
    eval_modality = normalize_eval_modality(eval_modality)
    missing_audio_strategy = normalize_missing_audio_strategy(missing_audio_strategy)
    gold_list = []
    pred_list = []

    with torch.no_grad():
        for data in dataloader:
            batch_input_tokens, attention_masks, audio_inputs, video_inputs, batch_labels = data
            batch_input_tokens = batch_input_tokens.to(device)
            attention_masks = attention_masks.to(device)
            audio_inputs = audio_inputs.to(device)
            video_inputs = video_inputs.to(device)
            batch_labels = batch_labels.to(device)

            if eval_modality == "text":
                text_hidden, _ = model_t(batch_input_tokens, attention_masks)
                audio_hidden = torch.zeros_like(text_hidden)
                video_hidden = torch.zeros_like(text_hidden)
            elif eval_modality == "video":
                video_hidden, _ = video_s(video_inputs)
                text_hidden = torch.zeros_like(video_hidden)
                audio_hidden = torch.zeros_like(video_hidden)
            else:
                text_hidden, _ = model_t(batch_input_tokens, attention_masks)
                video_hidden, _ = video_s(video_inputs)
                if missing_audio_strategy == "silence":
                    audio_hidden, _ = audio_s(audio_inputs)
                else:
                    audio_hidden = make_imputed_audio_hidden(
                        text_hidden,
                        missing_audio_strategy,
                        audio_mean,
                        audio_std,
                        audio_noise_scale,
                    )

            logits_6 = fusion(text_hidden, video_hidden, audio_hidden)

            probs_4 = map_logits_6_to_probs_4(logits_6)
            pred_4 = probs_4.argmax(dim=1)

            pred_list.extend(pred_4.cpu().numpy().tolist())
            gold_list.extend(batch_labels.cpu().numpy().tolist())

    return pred_list, gold_list


def save_prediction_details(
    samples: List[Sample],
    all_golds: List[int],
    all_preds: List[int],
    save_dir: Path,
    prefix: str,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    out_csv = save_dir / f"{prefix}_predictions.csv"
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sample_id",
                "sd_id",
                "dialogue_idx",
                "utterance_idx",
                "frame_dir",
                "gold_label",
                "pred_label",
                "correct",
            ]
        )
        for sample, g, p in zip(samples, all_golds, all_preds):
            writer.writerow(
                [
                    sample.sample_id,
                    sample.sd_id,
                    sample.dialogue_idx,
                    sample.utterance_idx,
                    str(sample.frame_dir),
                    TARGET_LABELS[int(g)],
                    TARGET_LABELS[int(p)],
                    int(int(g) == int(p)),
                ]
            )
    print(f"[Eval] prediction details saved: {out_csv}")


def save_reports(all_golds: List[int], all_preds: List[int], save_dir: Path, prefix: str) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    labels_idx = list(range(len(TARGET_LABELS)))
    cm = metrics.confusion_matrix(all_golds, all_preds, labels=labels_idx)
    report = metrics.classification_report(
        all_golds,
        all_preds,
        labels=labels_idx,
        target_names=TARGET_LABELS,
        digits=4,
        zero_division=0,
    )

    np.save(save_dir / f"{prefix}_confusion_matrix.npy", cm)
    with open(save_dir / f"{prefix}_confusion_matrix.txt", "w", encoding="utf-8") as f:
        f.write("labels: " + ",".join(TARGET_LABELS) + "\n")
        for row in cm:
            f.write(" ".join(str(int(x)) for x in row) + "\n")
    with open(save_dir / f"{prefix}_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print("[Eval] confusion matrix labels:", TARGET_LABELS)
    print(cm)
    print("[Eval] classification report:")
    print(report)


def load_state_dict_compat(path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    # Prefer safer loading mode on newer torch; fall back for older versions.
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)

    # Some checkpoints may be wrapped in {'state_dict': ...}.
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    if not isinstance(state, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {path}")

    # Compatibility with older HF checkpoints.
    state.pop("text_model.embeddings.position_ids", None)
    return state


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate TelME pretrained on EJSL frame-selected samples")
    parser.add_argument("--frame_root", type=str, required=True)
    parser.add_argument("--txt_root", type=str, required=True)
    parser.add_argument("--wav_map_csv", type=str, default="")
    parser.add_argument("--mp4_root", type=str, default="")
    parser.add_argument("--save_model_root", type=str, default="./IEMOCAP/save_model")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="./IEMOCAP/outputs_ejsl")
    parser.add_argument("--report_prefix", type=str, default="telme_ejsl")
    parser.add_argument(
        "--eval_modality",
        type=str,
        default="full",
        help="full/TELME fusion, text-only fusion, or video-only fusion. Non-selected fusion inputs are zeroed.",
    )
    parser.add_argument(
        "--missing_audio_strategy",
        type=str,
        default="silence",
        help="For full fusion on datasets without audio: silence/zero/old zero waveform, zero_hidden, pure_noise/noise, or source_mean/mean.",
    )
    parser.add_argument(
        "--audio_stats_path",
        type=str,
        default="",
        help="Optional .npz/.npy with source audio hidden mean/std for source_mean imputation.",
    )
    parser.add_argument(
        "--audio_stats_source_csv",
        type=str,
        default="./dataset/IEMOCAP_full_release/IEMOCAP_train.csv",
        help="IEMOCAP-style CSV used to compute audio hidden mean/std if --audio_stats_path is missing.",
    )
    parser.add_argument(
        "--audio_stats_max_samples",
        type=int,
        default=512,
        help="Maximum source wavs used to estimate audio hidden mean/std; 0 means all available wavs.",
    )
    parser.add_argument("--audio_noise_scale", type=float, default=1.0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--save_predictions", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.eval_modality = normalize_eval_modality(args.eval_modality)
    args.missing_audio_strategy = normalize_missing_audio_strategy(args.missing_audio_strategy)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    frame_root = Path(args.frame_root)
    txt_root = Path(args.txt_root)
    wav_map_csv = Path(args.wav_map_csv) if args.wav_map_csv else None
    mp4_root = Path(args.mp4_root) if args.mp4_root else None
    wav_map = build_wav_map(wav_map_csv)

    all_dirs = sorted([p for p in frame_root.iterdir() if p.is_dir()])
    samples: List[Sample] = []
    for d in all_dirs:
        parsed = parse_sample_id(d.name)
        if parsed is None:
            continue
        parsed.frame_dir = d
        if mp4_root is not None:
            candidate_mp4 = mp4_root / f"{parsed.sample_id}.mp4"
            if candidate_mp4.exists():
                parsed.mp4_path = candidate_mp4
        samples.append(parsed)

    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    sessions = []
    valid_samples: List[Sample] = []
    dropped = defaultdict(int)
    for s in samples:
        session = build_session_for_sample(s, txt_root, wav_map)
        if session is None:
            dropped["txt_or_turn_mismatch"] += 1
            continue
        has_frames = bool(list(s.frame_dir.glob("*.jpg")) or list(s.frame_dir.glob("*.png")))
        has_mp4 = s.mp4_path is not None and s.mp4_path.exists()
        if not has_frames and not has_mp4:
            dropped["empty_frame_dir"] += 1
            continue
        sessions.append(session)
        valid_samples.append(s)

    if not sessions:
        raise RuntimeError("No valid sessions. Check frame/txt root and naming format.")

    print(f"[Data] total_frame_dirs={len(all_dirs)} parsed_samples={len(samples)} valid_sessions={len(sessions)}")
    if dropped:
        print(f"[Data] dropped={dict(dropped)}")

    test_dataset = EJSLDataset(sessions)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=partial(
            make_batchs,
            eval_modality=args.eval_modality,
            missing_audio_strategy=args.missing_audio_strategy,
        ),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    text_model = "roberta-large"
    audio_model = "facebook/data2vec-audio-base-960h"
    video_model = "facebook/timesformer-base-finetuned-k400"
    cls_num = len(IEMOCAP_LABELS)

    need_text = args.eval_modality in {"full", "text"}
    need_audio = args.eval_modality == "full"
    need_video = args.eval_modality in {"full", "video"}

    model_t = None
    audio_s = None
    video_s = None
    fusion = None

    if need_text:
        model_t = Teacher_model(text_model, cls_num)
        teacher_sd = load_state_dict_compat(Path(args.save_model_root) / "teacher.bin", device)
        model_t.load_state_dict(teacher_sd, strict=False)
        model_t = model_t.to(device).eval()

    if need_audio:
        audio_s = Student_Audio(audio_model, cls_num, Config())
        audio_sd = load_state_dict_compat(Path(args.save_model_root) / "student_audio" / "total_student.bin", device)
        audio_s.load_state_dict(audio_sd, strict=True)
        audio_s = audio_s.to(device).eval()

    if need_video:
        video_s = Student_Video(video_model, cls_num)
        video_sd = load_state_dict_compat(Path(args.save_model_root) / "student_video" / "total_student.bin", device)
        video_s.load_state_dict(video_sd, strict=True)
        video_s = video_s.to(device).eval()

    hidden_size, beta_shift, dropout_prob, num_head = 768, 2e-1, 0.2, 4
    fusion = ASF(cls_num, hidden_size, beta_shift, dropout_prob, num_head)
    fusion_sd = load_state_dict_compat(Path(args.save_model_root) / "total_fusion.bin", device)
    fusion.load_state_dict(fusion_sd, strict=True)
    fusion = fusion.to(device).eval()

    audio_mean = None
    audio_std = None
    audio_stats_count = 0
    if args.eval_modality == "full" and args.missing_audio_strategy == "source_mean":
        audio_mean, audio_std, audio_stats_count = get_or_build_audio_hidden_stats(args, audio_s, device)

    print(
        f"[Eval] modality={args.eval_modality} "
        f"missing_audio_strategy={args.missing_audio_strategy} "
        f"audio_stats_count={audio_stats_count}"
    )
    with torch.no_grad():
        preds, golds = evaluation(
            model_t,
            audio_s,
            video_s,
            fusion,
            test_loader,
            device,
            args.eval_modality,
            args.missing_audio_strategy,
            audio_mean,
            audio_std,
            args.audio_noise_scale,
        )

    if len(valid_samples) != len(golds):
        raise RuntimeError(f"Prediction count mismatch: samples={len(valid_samples)} vs golds={len(golds)}")

    acc = np.mean(np.array(preds) == np.array(golds))
    print(f"[Eval] samples={len(golds)} acc={acc:.4f}")

    labels_idx = list(range(len(TARGET_LABELS)))
    support = metrics.confusion_matrix(golds, preds, labels=labels_idx).sum(axis=1).tolist()
    gold_counts = np.bincount(np.array(golds, dtype=np.int64), minlength=len(TARGET_LABELS)).tolist()
    pred_counts = np.bincount(np.array(preds, dtype=np.int64), minlength=len(TARGET_LABELS)).tolist()
    print(f"[Eval] support(A,N,J,S)={support} total={sum(support)}")
    print(f"[Eval] gold_counts(A,N,J,S)={gold_counts}")
    print(f"[Eval] pred_counts(A,N,J,S)={pred_counts}")

    save_reports(golds, preds, Path(args.save_dir), args.report_prefix)
    if args.save_predictions:
        save_prediction_details(
            valid_samples,
            golds,
            preds,
            Path(args.save_dir),
            args.report_prefix,
        )


if __name__ == "__main__":
    main()
