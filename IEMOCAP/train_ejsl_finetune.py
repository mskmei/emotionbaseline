import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn import metrics
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoProcessor, RobertaTokenizer

from model import ASF, Student_Audio, Student_Video, Teacher_model


IEMOCAP_LABELS = ["ang", "exc", "fru", "hap", "neu", "sad"]
TARGET_LABELS = ["A", "N", "J", "S"]
TARGET_TO_ID = {name: idx for idx, name in enumerate(TARGET_LABELS)}


audio_processor = AutoProcessor.from_pretrained("facebook/data2vec-audio-base-960h")
video_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
speaker_list = ["<s1>", "<s2>", "<s3>", "<s4>", "<s5>", "<s6>", "<s7>", "<s8>", "<s9>"]
roberta_tokenizer.add_special_tokens({"additional_special_tokens": speaker_list})


@dataclass
class Config:
    mask_time_length: int = 3


@dataclass
class TrainSample:
    sample_id: str
    text: str
    label_id: int
    media_path: Path


@dataclass
class DialSample:
    sample_id: str
    text: str
    label_id: int
    media_path: Path


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


def get_video_from_file(video_path: Path) -> torch.Tensor:
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        raise ValueError(f"failed to read any frame from video: {video_path}")

    if len(frames) >= 8:
        idx = np.linspace(0, len(frames) - 1, num=8, dtype=int)
        chosen = [frames[i] for i in idx]
    else:
        chosen = frames + [frames[-1]] * (8 - len(frames))

    inputs = video_processor(chosen[:8], return_tensors="pt")
    return inputs["pixel_values"][0]


def get_video_from_frame_dir(frame_dir: Path) -> torch.Tensor:
    frame_paths = sorted(frame_dir.glob("*.jpg"))
    if not frame_paths:
        frame_paths = sorted(frame_dir.glob("*.png"))
    if not frame_paths:
        raise ValueError(f"empty frame dir: {frame_dir}")

    if len(frame_paths) >= 8:
        idx = np.linspace(0, len(frame_paths) - 1, num=8, dtype=int)
        chosen = [frame_paths[i] for i in idx]
    else:
        chosen = frame_paths + [frame_paths[-1]] * (8 - len(frame_paths))

    frames = []
    for p in chosen:
        img = cv2.imread(str(p))
        if img is None:
            raise ValueError(f"failed to read frame: {p}")
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    inputs = video_processor(frames, return_tensors="pt")
    return inputs["pixel_values"][0]


def get_video_from_source(path: Path) -> torch.Tensor:
    if path.is_dir():
        return get_video_from_frame_dir(path)
    return get_video_from_file(path)


def get_audio_from_file_or_silence(media_path: Path, silence_seconds: float = 1.0) -> torch.Tensor:
    import librosa

    if media_path.exists() and media_path.is_file():
        try:
            audio, _ = librosa.load(str(media_path), sr=16000)
        except Exception:
            audio = np.zeros(int(16000 * silence_seconds), dtype=np.float32)
    else:
        audio = np.zeros(int(16000 * silence_seconds), dtype=np.float32)

    inputs = audio_processor(audio, sampling_rate=16000, return_tensors="pt")
    return inputs["input_values"][0]


def normalize_label(x: str) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    u = s.upper()
    if u in TARGET_TO_ID:
        return u

    t = s.lower()
    mapping = {
        "ang": "A",
        "anger": "A",
        "fru": "A",
        "怒り": "A",
        "怒り1": "A",
        "neu": "N",
        "neutral": "N",
        "平静": "N",
        "hap": "J",
        "happy": "J",
        "exc": "J",
        "joy": "J",
        "喜び": "J",
        "sad": "S",
        "sadness": "S",
        "悲しみ": "S",
    }
    return mapping.get(t)


def resolve_video_path(video_root: Path, clip_name: str, stem: str) -> Optional[Path]:
    cands = []
    if clip_name:
        cands.append(video_root / clip_name)
    if stem:
        cands.append(video_root / stem)
        cands.append(video_root / f"{stem}.mp4")
        cands.append(video_root / f"{stem}.avi")

    for p in cands:
        if p.exists() and p.is_file():
            return p

    if stem:
        for p in sorted(video_root.glob(f"{stem}.*")):
            if p.is_file():
                return p
    return None


def resolve_frame_dir(frame_root: Path, clip_name: str, stem: str) -> Optional[Path]:
    cands = []
    if clip_name:
        cands.append(frame_root / Path(clip_name).stem)
        cands.append(frame_root / clip_name)
    if stem:
        cands.append(frame_root / stem)

    for p in cands:
        if p.exists() and p.is_dir():
            return p
    return None


def parse_dial_sample_id(name: str) -> Optional[Tuple[str, int, int, str]]:
    m = re.match(r"^(SD\d{2})-(\d{2})-(\d{2})([AJNS])$", name)
    if m is None:
        return None
    sd_id, dialogue, utterance, label = m.groups()
    return sd_id, int(dialogue), int(utterance), label


def parse_dialogue_txt(txt_file: Path) -> List[Tuple[str, str, str]]:
    turns = []
    for raw in txt_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < 3:
            continue
        turns.append((parts[0].strip(), parts[1].strip(), parts[2].strip()))
    return turns


def build_dial_text(txt_root: Path, sample_id: str) -> Optional[str]:
    parsed = parse_dial_sample_id(sample_id)
    if parsed is None:
        return None
    sd_id, dialogue_idx, utterance_idx, _ = parsed
    txt_file = txt_root / sd_id / "txt" / f"{sd_id}-Dialogue-{dialogue_idx:02d}.txt"
    if not txt_file.exists():
        return None

    turns = parse_dialogue_txt(txt_file)
    if utterance_idx <= 0 or utterance_idx > len(turns):
        return None

    speaker_to_idx: Dict[str, int] = {}
    context = []
    for i in range(utterance_idx):
        speaker, _emo, utterance = turns[i]
        if speaker not in speaker_to_idx:
            speaker_to_idx[speaker] = len(speaker_to_idx)
        spk_idx = speaker_to_idx[speaker] + 1
        context.append(f"<s{spk_idx}> {utterance}")

    last_spk = speaker_to_idx[turns[utterance_idx - 1][0]] + 1
    return " ".join(context) + f" </s> Now <s{last_spk}> feels"


def build_train_samples(csv_path: Path, video_root: Optional[Path], frame_root: Optional[Path]) -> List[TrainSample]:
    df = pd.read_csv(csv_path)
    samples = []
    for _, row in df.iterrows():
        stem = str(row.get("stem", "")).strip()
        clip_name = str(row.get("clip_name", "")).strip()
        raw_label = row.get("emotion", None)
        norm_label = normalize_label(str(raw_label))
        if norm_label is None:
            continue

        sample_id = stem or Path(clip_name).stem
        if not sample_id:
            continue

        media_path = None
        if video_root is not None:
            media_path = resolve_video_path(video_root, clip_name, sample_id)
        if media_path is None and frame_root is not None:
            media_path = resolve_frame_dir(frame_root, clip_name, sample_id)
        if media_path is None:
            continue

        text = str(row.get("text", "")).strip()
        if not text:
            text = sample_id
        prompt_text = f"<s1> {text} </s> Now <s1> feels"

        samples.append(
            TrainSample(
                sample_id=sample_id,
                text=prompt_text,
                label_id=TARGET_TO_ID[norm_label],
                media_path=media_path,
            )
        )
    return samples


def read_dial_list(dial_csv_or_txt: Path) -> List[str]:
    lines = [x.strip() for x in dial_csv_or_txt.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not lines:
        return []

    # Accept one-column file list, with or without header.
    if len(lines) == 1:
        return [Path(lines[0]).stem]

    first = lines[0].lower()
    if "stem" in first or "clip" in first or "filename" in first:
        names = []
        with open(dial_csv_or_txt, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                val = row.get("clip_name") or row.get("stem") or row.get("filename")
                if val:
                    names.append(Path(str(val).strip()).stem)
        return names

    return [Path(x).stem for x in lines]


def build_dial_samples(
    dial_csv_or_txt: Path,
    dial_video_root: Optional[Path],
    dial_frame_root: Optional[Path],
    txt_root: Path,
) -> List[DialSample]:
    names = read_dial_list(dial_csv_or_txt)
    samples = []
    for n in names:
        parsed = parse_dial_sample_id(n)
        if parsed is None:
            continue
        _sd_id, _dialogue, _utt, label = parsed

        media_path = None
        if dial_video_root is not None:
            media_path = resolve_video_path(dial_video_root, f"{n}.mp4", n)
        if media_path is None and dial_frame_root is not None:
            media_path = resolve_frame_dir(dial_frame_root, f"{n}.mp4", n)
        if media_path is None:
            continue

        text = build_dial_text(txt_root, n)
        if text is None:
            continue

        samples.append(DialSample(sample_id=n, text=text, label_id=TARGET_TO_ID[label], media_path=media_path))
    return samples


class SingleTurnDataset(Dataset):
    def __init__(self, samples: List):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def collate_single_turn(batch):
    texts = []
    audios = []
    videos = []
    labels = []

    for s in batch:
        texts.append(encode_right_truncated(s.text, roberta_tokenizer))
        audios.append(get_audio_from_file_or_silence(s.media_path))
        videos.append(get_video_from_source(s.media_path))
        labels.append(s.label_id)

    tokens, masks = padding(texts, roberta_tokenizer)
    audios = padding_audio(audios)
    videos = torch.stack(videos)
    labels = torch.tensor(labels)
    return tokens, masks, audios, videos, labels


def map_logits_6_to_logits_4(logits_6: torch.Tensor) -> torch.Tensor:
    # A = ang + fru, N = neu, J = hap + exc, S = sad
    l_a = torch.logsumexp(torch.stack([logits_6[:, 0], logits_6[:, 2]], dim=1), dim=1)
    l_n = logits_6[:, 4]
    l_j = torch.logsumexp(torch.stack([logits_6[:, 3], logits_6[:, 1]], dim=1), dim=1)
    l_s = logits_6[:, 5]
    return torch.stack([l_a, l_n, l_j, l_s], dim=1)


def evaluate(model_t, audio_s, video_s, fusion, loader, device):
    all_preds, all_golds = [], []
    fusion.eval()
    with torch.no_grad():
        for tokens, masks, audios, videos, labels in loader:
            tokens = tokens.to(device)
            masks = masks.to(device)
            audios = audios.to(device)
            videos = videos.to(device)
            labels = labels.to(device)

            text_hidden, _ = model_t(tokens, masks)
            audio_hidden, _ = audio_s(audios)
            video_hidden, _ = video_s(videos)

            logits_6 = fusion(text_hidden, video_hidden, audio_hidden)
            logits_4 = map_logits_6_to_logits_4(logits_6)
            preds = logits_4.argmax(dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_golds.extend(labels.cpu().tolist())

    acc = float(np.mean(np.array(all_preds) == np.array(all_golds))) if all_golds else 0.0
    f1 = metrics.f1_score(all_golds, all_preds, average="weighted", zero_division=0) if all_golds else 0.0
    return acc, f1, all_golds, all_preds


def save_eval_report(golds: List[int], preds: List[int], save_dir: Path, prefix: str):
    save_dir.mkdir(parents=True, exist_ok=True)
    labels_idx = list(range(len(TARGET_LABELS)))
    cm = metrics.confusion_matrix(golds, preds, labels=labels_idx)
    report = metrics.classification_report(
        golds,
        preds,
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


def load_state_dict_compat(path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)

    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    if not isinstance(state, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {path}")

    state.pop("text_model.embeddings.position_ids", None)
    return state


def parse_args():
    p = argparse.ArgumentParser(description="Train TelME on BOBSL and evaluate every epoch on EJSL dial")
    p.add_argument("--train_csv", type=str, required=True)
    p.add_argument("--val_csv", type=str, required=True)
    p.add_argument("--train_video_root", type=str, default="")
    p.add_argument("--train_frame_root", type=str, default="")
    p.add_argument("--val_video_root", type=str, default="")
    p.add_argument("--val_frame_root", type=str, default="")

    p.add_argument("--dial_test_csv", type=str, required=True)
    p.add_argument("--dial_video_root", type=str, default="")
    p.add_argument("--dial_frame_root", type=str, default="")
    p.add_argument("--txt_root", type=str, required=True)

    p.add_argument("--save_model_root", type=str, default="./IEMOCAP/save_model")
    p.add_argument("--save_dir", type=str, default="./IEMOCAP/outputs_finetune")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--dial_eval_every", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--unfreeze_students", action="store_true", help="Unfreeze audio/video students during finetuning")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_video_root = Path(args.train_video_root) if args.train_video_root else None
    train_frame_root = Path(args.train_frame_root) if args.train_frame_root else None
    val_video_root = Path(args.val_video_root) if args.val_video_root else train_video_root
    val_frame_root = Path(args.val_frame_root) if args.val_frame_root else train_frame_root
    dial_video_root = Path(args.dial_video_root) if args.dial_video_root else None
    dial_frame_root = Path(args.dial_frame_root) if args.dial_frame_root else None

    if train_video_root is None and train_frame_root is None:
        raise RuntimeError("Provide at least one of --train_video_root or --train_frame_root")
    if dial_video_root is None and dial_frame_root is None:
        raise RuntimeError("Provide at least one of --dial_video_root or --dial_frame_root")

    train_samples = build_train_samples(Path(args.train_csv), train_video_root, train_frame_root)
    val_samples = build_train_samples(Path(args.val_csv), val_video_root, val_frame_root)
    dial_samples = build_dial_samples(Path(args.dial_test_csv), dial_video_root, dial_frame_root, Path(args.txt_root))

    if not train_samples:
        raise RuntimeError("No train samples parsed. Check train_csv and train_video_root.")
    if not val_samples:
        raise RuntimeError("No val samples parsed. Check val_csv and val_video_root.")
    if not dial_samples:
        raise RuntimeError("No dial samples parsed. Check dial_test_csv, dial_video_root, txt_root.")

    print(f"[Data] train={len(train_samples)} val={len(val_samples)} dial={len(dial_samples)}")

    train_loader = DataLoader(
        SingleTurnDataset(train_samples),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_single_turn,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        SingleTurnDataset(val_samples),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_single_turn,
        pin_memory=torch.cuda.is_available(),
    )
    dial_loader = DataLoader(
        SingleTurnDataset(dial_samples),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_single_turn,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    cls_num = len(IEMOCAP_LABELS)
    model_t = Teacher_model("roberta-large", cls_num)
    teacher_sd = load_state_dict_compat(Path(args.save_model_root) / "teacher.bin", device)
    model_t.load_state_dict(teacher_sd, strict=False)
    model_t = model_t.to(device).eval()

    audio_s = Student_Audio("facebook/data2vec-audio-base-960h", cls_num, Config())
    audio_sd = load_state_dict_compat(Path(args.save_model_root) / "student_audio" / "total_student.bin", device)
    audio_s.load_state_dict(audio_sd, strict=True)
    audio_s = audio_s.to(device).eval()

    video_s = Student_Video("facebook/timesformer-base-finetuned-k400", cls_num)
    video_sd = load_state_dict_compat(Path(args.save_model_root) / "student_video" / "total_student.bin", device)
    video_s.load_state_dict(video_sd, strict=True)
    video_s = video_s.to(device).eval()

    fusion = ASF(cls_num, hidden_size=768, beta_shift=2e-1, dropout_prob=0.2, num_head=4)
    fusion_sd = load_state_dict_compat(Path(args.save_model_root) / "total_fusion.bin", device)
    fusion.load_state_dict(fusion_sd, strict=True)
    fusion = fusion.to(device)

    for p in model_t.parameters():
        p.requires_grad = False
    for p in audio_s.parameters():
        p.requires_grad = args.unfreeze_students
    for p in video_s.parameters():
        p.requires_grad = args.unfreeze_students

    trainable_params = list(fusion.parameters())
    if args.unfreeze_students:
        trainable_params += [p for p in audio_s.parameters() if p.requires_grad]
        trainable_params += [p for p in video_s.parameters() if p.requires_grad]

    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    best_dial = -1.0
    best_val = -1.0

    for epoch in range(1, args.epochs + 1):
        fusion.train()
        if args.unfreeze_students:
            audio_s.train()
            video_s.train()

        losses = []
        for tokens, masks, audios, videos, labels in train_loader:
            tokens = tokens.to(device)
            masks = masks.to(device)
            audios = audios.to(device)
            videos = videos.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                text_hidden, _ = model_t(tokens, masks)

            audio_hidden, _ = audio_s(audios)
            video_hidden, _ = video_s(videos)
            logits_6 = fusion(text_hidden, video_hidden, audio_hidden)
            logits_4 = map_logits_6_to_logits_4(logits_6)
            loss = F.cross_entropy(logits_4, labels)

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        train_loss = float(np.mean(losses)) if losses else 0.0
        val_acc, val_f1, _, _ = evaluate(model_t, audio_s, video_s, fusion, val_loader, device)
        print(f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "state": {
                        "fusion": fusion.state_dict(),
                        "audio_s": audio_s.state_dict(),
                        "video_s": video_s.state_dict(),
                    },
                    "args": vars(args),
                },
                save_dir / "best_val.pt",
            )

        if epoch % args.dial_eval_every == 0:
            dial_acc, dial_f1, dial_golds, dial_preds = evaluate(model_t, audio_s, video_s, fusion, dial_loader, device)
            print(f"[Epoch {epoch:03d}] dial_acc={dial_acc:.4f} dial_f1={dial_f1:.4f}")
            save_eval_report(dial_golds, dial_preds, save_dir, f"dial_epoch{epoch:03d}")

            if dial_acc > best_dial:
                best_dial = dial_acc
                torch.save(
                    {
                        "epoch": epoch,
                        "dial_acc": dial_acc,
                        "state": {
                            "fusion": fusion.state_dict(),
                            "audio_s": audio_s.state_dict(),
                            "video_s": video_s.state_dict(),
                        },
                        "args": vars(args),
                    },
                    save_dir / "best_dial.pt",
                )

    print(f"[Done] best_val={best_val:.4f} best_dial={best_dial:.4f}")


if __name__ == "__main__":
    main()
