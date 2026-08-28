import argparse
import os
import random
import shutil
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from dataset import meld_dataset
from meld_kd import Feature_Loss, Logit_Loss
from model import ASF, Student_Video, Teacher_model
from preprocessing import preprocessing
from utils import encode_right_truncated, padding, roberta_tokenizer, video_processor


MELD_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
MODALITY_ALIASES = {
    "text": "text",
    "t": "text",
    "l": "text",
    "video": "video",
    "visual": "video",
    "v": "video",
    "tv": "tv",
    "vt": "tv",
    "vl": "tv",
    "text_video": "tv",
    "video_text": "tv",
}


@dataclass
class Config:
    mask_time_length: int = 3


def normalize_modality(value: str) -> str:
    key = (value or "").strip().lower()
    if key not in MODALITY_ALIASES:
        valid = ", ".join(sorted(MODALITY_ALIASES))
        raise ValueError(f"Unsupported modality={value!r}; valid values: {valid}")
    return MODALITY_ALIASES[key]


def parse_modalities(raw_values: Sequence[str]) -> List[str]:
    out: List[str] = []
    for value in raw_values:
        for part in str(value).replace(",", " ").split():
            mod = normalize_modality(part)
            if mod not in out:
                out.append(mod)
    return out


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def save_report(labels: Iterable[int], preds: Iterable[int], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = classification_report(
        list(labels),
        list(preds),
        labels=list(range(len(MELD_LABELS))),
        target_names=MELD_LABELS,
        digits=4,
        zero_division=0,
    )
    out_path.write_text(report, encoding="utf-8")


def get_video_or_zeros(video_path: str, max_seconds: float = 30.0) -> torch.Tensor:
    path = Path(video_path)
    video = cv2.VideoCapture(str(path))
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(video.get(cv2.CAP_PROP_FPS) or 0.0)
    if frame_count <= 0:
        video.release()
        return torch.zeros(8, 3, 224, 224)

    if fps > 0 and frame_count / fps > max_seconds:
        video.release()
        return torch.zeros(8, 3, 224, 224)

    wanted = set(np.linspace(0, frame_count - 1, num=8, dtype=int).tolist())
    frames = []
    index = 0
    while video.isOpened():
        ok, image = video.read()
        if not ok:
            break
        if index in wanted:
            frames.append(image)
        index += 1
    video.release()

    if not frames:
        return torch.zeros(8, 3, 224, 224)
    while len(frames) < 8:
        frames.append(frames[-1].copy())

    inputs = video_processor(frames[:8], return_tensors="pt")
    return inputs["pixel_values"][0]


def make_telme_batch(sessions, load_video: bool):
    batch_input = []
    batch_audio = []
    batch_video = []
    batch_labels = []

    for session in sessions:
        input_string = ""
        now_speaker = 0
        video_path = ""
        emotion = "neutral"

        for line in session:
            speaker, utt, video_path, emotion = line
            input_string += f"<s{speaker + 1}> {utt} "
            now_speaker = speaker

        prompt = f"Now <s{now_speaker + 1}> feels"
        concat_string = input_string.strip() + " </s> " + prompt
        batch_input.append(encode_right_truncated(concat_string, roberta_tokenizer))
        batch_labels.append(MELD_LABELS.index(emotion))

        batch_audio.append(torch.zeros(1, dtype=torch.float32))
        if load_video:
            batch_video.append(get_video_or_zeros(video_path))
        else:
            batch_video.append(torch.zeros(8, 3, 224, 224))

    tokens, masks = padding(batch_input, roberta_tokenizer)
    audio = torch.stack(batch_audio)
    video = torch.stack(batch_video)
    labels = torch.tensor(batch_labels, dtype=torch.long)
    return tokens, masks, audio, video, labels


def build_loaders(args, load_video: bool) -> Tuple[DataLoader, DataLoader, DataLoader]:
    data_root = Path(args.data_root)
    paths = {
        "train": data_root / "train_meld_emo.csv",
        "dev": data_root / "dev_meld_emo.csv",
        "test": data_root / "test_meld_emo.csv",
    }
    for split, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing MELD {split} csv: {path}")

    train_data = preprocessing(str(paths["train"]))
    dev_data = preprocessing(str(paths["dev"]))
    test_data = preprocessing(str(paths["test"]))

    if args.max_train_samples > 0:
        train_data = train_data[: args.max_train_samples]
    if args.max_eval_samples > 0:
        dev_data = dev_data[: args.max_eval_samples]
        test_data = test_data[: args.max_eval_samples]

    collate = partial(make_telme_batch, load_video=load_video)
    train_loader = DataLoader(
        meld_dataset(train_data),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
    )
    dev_loader = DataLoader(
        meld_dataset(dev_data),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        meld_dataset(test_data),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )
    return train_loader, dev_loader, test_loader


def ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return nn.CrossEntropyLoss()(logits, labels)


def evaluate_teacher(model: Teacher_model, loader: DataLoader, device: torch.device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for tokens, masks, _audio, _video, batch_labels in loader:
            tokens = tokens.to(device)
            masks = masks.to(device)
            batch_labels = batch_labels.to(device)
            _hidden, logits = model(tokens, masks)
            preds.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())
            labels.extend(batch_labels.detach().cpu().numpy().tolist())
    return preds, labels


def train_teacher(args, device: torch.device, shared_root: Path) -> Teacher_model:
    teacher_path = shared_root / "teacher.bin"
    model = Teacher_model("roberta-large", len(MELD_LABELS)).to(device)
    if args.reuse_shared and teacher_path.exists():
        print(f"[TELME][teacher] reuse {teacher_path}")
        model.load_state_dict(torch.load(teacher_path, map_location=device))
        return model.eval()

    train_loader, dev_loader, test_loader = build_loaders(args, load_video=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.teacher_lr)
    total_steps = max(len(train_loader.dataset) * max(args.teacher_epochs, 1), 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(len(train_loader.dataset), 1),
        num_training_steps=total_steps,
    )
    best_dev_f1 = -1.0

    for epoch in tqdm(range(args.teacher_epochs), desc="[TELME][teacher]"):
        model.train()
        for tokens, masks, _audio, _video, labels in train_loader:
            tokens = tokens.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            _hidden, logits = model(tokens, masks)
            loss = ce_loss(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

        dev_preds, dev_labels = evaluate_teacher(model, dev_loader, device)
        dev_f1 = precision_recall_fscore_support(dev_labels, dev_preds, average="weighted", zero_division=0)[2]
        print(f"[TELME][teacher] epoch={epoch + 1} dev_f1={dev_f1:.6f}")
        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            shared_root.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), teacher_path)
            test_preds, test_labels = evaluate_teacher(model, test_loader, device)
            test_f1 = precision_recall_fscore_support(test_labels, test_preds, average="weighted", zero_division=0)[2]
            print(f"[TELME][teacher] epoch={epoch + 1} test_f1={test_f1:.6f}")
            save_report(test_labels, test_preds, shared_root / "teacher_meld_test_report.txt")

    if not teacher_path.exists():
        shared_root.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), teacher_path)
    model.load_state_dict(torch.load(teacher_path, map_location=device))
    return model.eval()


def evaluate_video_student(model: Student_Video, loader: DataLoader, device: torch.device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for _tokens, _masks, _audio, video, batch_labels in loader:
            video = video.to(device)
            batch_labels = batch_labels.to(device)
            _hidden, logits = model(video)
            preds.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())
            labels.extend(batch_labels.detach().cpu().numpy().tolist())
    return preds, labels


def train_video_student(args, teacher: Teacher_model, device: torch.device, shared_root: Path) -> Student_Video:
    student_path = shared_root / "student_video" / "total_student.bin"
    model = Student_Video("facebook/timesformer-base-finetuned-k400", len(MELD_LABELS)).to(device)
    if args.reuse_shared and student_path.exists():
        print(f"[TELME][video-student] reuse {student_path}")
        model.load_state_dict(torch.load(student_path, map_location=device))
        return model.eval()

    for param in teacher.parameters():
        param.requires_grad = False
    teacher.eval()

    train_loader, dev_loader, test_loader = build_loaders(args, load_video=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.student_lr)
    total_steps = max(len(train_loader.dataset) * max(args.student_epochs, 1), 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(len(train_loader.dataset), 1),
        num_training_steps=total_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    logit_loss = Logit_Loss().to(device)
    feature_loss = Feature_Loss().to(device)
    best_dev_f1 = -1.0

    for epoch in tqdm(range(args.student_epochs), desc="[TELME][video-student]"):
        model.train()
        for tokens, masks, _audio, video, labels in train_loader:
            tokens = tokens.to(device)
            masks = masks.to(device)
            video = video.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                hidden_s, logits_s = model(video)
                with torch.no_grad():
                    hidden_t, logits_t = teacher(tokens, masks)
                loss = ce_loss(logits_s, labels) + logit_loss(logits_s, logits_t) + feature_loss(hidden_s, hidden_t)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

        dev_preds, dev_labels = evaluate_video_student(model, dev_loader, device)
        dev_f1 = precision_recall_fscore_support(dev_labels, dev_preds, average="weighted", zero_division=0)[2]
        print(f"[TELME][video-student] epoch={epoch + 1} dev_f1={dev_f1:.6f}")
        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            student_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), student_path)
            test_preds, test_labels = evaluate_video_student(model, test_loader, device)
            test_f1 = precision_recall_fscore_support(test_labels, test_preds, average="weighted", zero_division=0)[2]
            print(f"[TELME][video-student] epoch={epoch + 1} test_f1={test_f1:.6f}")
            save_report(test_labels, test_preds, shared_root / "video_student_meld_test_report.txt")

    if not student_path.exists():
        student_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), student_path)
    model.load_state_dict(torch.load(student_path, map_location=device))
    return model.eval()


def modality_hiddens(
    modality: str,
    teacher: Teacher_model,
    video_student: Student_Video,
    tokens: torch.Tensor,
    masks: torch.Tensor,
    video: torch.Tensor,
):
    text_hidden = None
    video_hidden = None

    if modality in {"text", "tv"}:
        text_hidden, _ = teacher(tokens, masks)
    if modality in {"video", "tv"}:
        video_hidden, _ = video_student(video)

    if text_hidden is None:
        text_hidden = torch.zeros_like(video_hidden)
    if video_hidden is None:
        video_hidden = torch.zeros_like(text_hidden)
    audio_hidden = torch.zeros_like(text_hidden)
    return text_hidden, audio_hidden, video_hidden


def stage_modality_checkpoints(save_root: Path, modality: str) -> None:
    target_root = save_root / modality
    copy_file(save_root / "shared" / "teacher.bin", target_root / "teacher.bin")
    if modality in {"video", "tv"}:
        copy_file(save_root / "shared" / "student_video" / "total_student.bin", target_root / "student_video" / "total_student.bin")


def evaluate_fusion(
    modality: str,
    teacher: Teacher_model,
    video_student: Student_Video,
    fusion: ASF,
    loader: DataLoader,
    device: torch.device,
):
    fusion.eval()
    preds, labels = [], []
    with torch.no_grad():
        for tokens, masks, _audio, video, batch_labels in loader:
            tokens = tokens.to(device)
            masks = masks.to(device)
            video = video.to(device)
            batch_labels = batch_labels.to(device)
            text_hidden, audio_hidden, video_hidden = modality_hiddens(modality, teacher, video_student, tokens, masks, video)
            logits = fusion(text_hidden, audio_hidden, video_hidden)
            preds.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())
            labels.extend(batch_labels.detach().cpu().numpy().tolist())
    return preds, labels


def train_fusion_modality(
    args,
    modality: str,
    teacher: Teacher_model,
    video_student: Student_Video,
    device: torch.device,
    save_root: Path,
) -> None:
    target_root = save_root / modality
    fusion_path = target_root / "total_fusion.bin"
    if args.reuse_fusion and fusion_path.exists():
        print(f"[TELME][fusion:{modality}] reuse {fusion_path}")
        stage_modality_checkpoints(save_root, modality)
        return

    for model in (teacher, video_student):
        for param in model.parameters():
            param.requires_grad = False
        model.eval()

    train_loader, dev_loader, test_loader = build_loaders(args, load_video=modality in {"video", "tv"})
    fusion = ASF(len(MELD_LABELS), hidden_size=768, beta_shift=1e-1, dropout_prob=0.2, num_head=3).to(device)
    optimizer = torch.optim.AdamW(fusion.parameters(), lr=args.fusion_lr)
    total_steps = max(len(train_loader.dataset) * max(args.fusion_epochs, 1), 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(len(train_loader.dataset), 1),
        num_training_steps=total_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best_dev_f1 = -1.0

    for epoch in tqdm(range(args.fusion_epochs), desc=f"[TELME][fusion:{modality}]"):
        fusion.train()
        for tokens, masks, _audio, video, labels in train_loader:
            tokens = tokens.to(device)
            masks = masks.to(device)
            video = video.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                text_hidden, audio_hidden, video_hidden = modality_hiddens(modality, teacher, video_student, tokens, masks, video)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = fusion(text_hidden, audio_hidden, video_hidden)
                loss = ce_loss(logits, labels)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(fusion.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

        dev_preds, dev_labels = evaluate_fusion(modality, teacher, video_student, fusion, dev_loader, device)
        dev_f1 = precision_recall_fscore_support(dev_labels, dev_preds, average="weighted", zero_division=0)[2]
        print(f"[TELME][fusion:{modality}] epoch={epoch + 1} dev_f1={dev_f1:.6f}")
        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            target_root.mkdir(parents=True, exist_ok=True)
            torch.save(fusion.state_dict(), fusion_path)
            test_preds, test_labels = evaluate_fusion(modality, teacher, video_student, fusion, test_loader, device)
            test_f1 = precision_recall_fscore_support(test_labels, test_preds, average="weighted", zero_division=0)[2]
            print(f"[TELME][fusion:{modality}] epoch={epoch + 1} test_f1={test_f1:.6f}")
            save_report(test_labels, test_preds, target_root / "meld_test_report.txt")

    if not fusion_path.exists():
        target_root.mkdir(parents=True, exist_ok=True)
        torch.save(fusion.state_dict(), fusion_path)

    stage_modality_checkpoints(save_root, modality)


def parse_args():
    parser = argparse.ArgumentParser(description="Train TELME on raw MELD for T/V/TV modality ablations")
    parser.add_argument("--data_root", type=str, default="./dataset/MELD.Raw")
    parser.add_argument("--save_root", type=str, default="./MELD/save_model_meld_modalities")
    parser.add_argument("--modalities", nargs="+", default=["text", "video", "tv"])
    parser.add_argument("--teacher_epochs", type=int, default=10)
    parser.add_argument("--student_epochs", type=int, default=10)
    parser.add_argument("--fusion_epochs", type=int, default=10)
    parser.add_argument("--teacher_lr", type=float, default=1e-6)
    parser.add_argument("--student_lr", type=float, default=1e-5)
    parser.add_argument("--fusion_lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_grad_norm", type=float, default=10.0)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_eval_samples", type=int, default=0)
    parser.add_argument("--reuse_shared", action="store_true")
    parser.add_argument("--reuse_fusion", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    modalities = parse_modalities(args.modalities)
    if not modalities:
        raise ValueError("No valid modalities requested")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_root = Path(args.save_root)
    shared_root = save_root / "shared"
    print(f"[TELME] device={device} data_root={args.data_root} save_root={save_root}")
    print(f"[TELME] modalities={modalities}")

    teacher = train_teacher(args, device, shared_root)
    video_student = train_video_student(args, teacher, device, shared_root)

    for modality in modalities:
        train_fusion_modality(args, modality, teacher, video_student, device, save_root)

    print("[TELME] done")


if __name__ == "__main__":
    main()
