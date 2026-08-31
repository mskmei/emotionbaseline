#!/usr/bin/env python3
from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader

import train_telme_meld_modalities as base


MELD4_CLASS_LABELS = ["A", "N", "J", "S"]
MELD4_EMOTION_TO_ID = {
    "anger": 0,
    "neutral": 1,
    "joy": 2,
    "sadness": 3,
}


def make_telme_meld4_batch(sessions, load_video: bool):
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

        if emotion not in MELD4_EMOTION_TO_ID:
            raise RuntimeError(f"Unexpected non-ANJS MELD target in MELD4 batch: {emotion}")

        prompt = f"Now <s{now_speaker + 1}> feels"
        concat_string = input_string.strip() + " </s> " + prompt
        batch_input.append(base.encode_right_truncated(concat_string, base.roberta_tokenizer))
        batch_labels.append(MELD4_EMOTION_TO_ID[emotion])

        batch_audio.append(torch.zeros(1, dtype=torch.float32))
        if load_video:
            batch_video.append(base.get_video_or_zeros(video_path))
        else:
            batch_video.append(torch.zeros(8, 3, 224, 224))

    tokens, masks = base.padding(batch_input, base.roberta_tokenizer)
    audio = torch.stack(batch_audio)
    video = torch.stack(batch_video)
    labels = torch.tensor(batch_labels, dtype=torch.long)
    return tokens, masks, audio, video, labels


def filter_meld4_samples(samples):
    return [session for session in samples if session and session[-1][3] in MELD4_EMOTION_TO_ID]


def build_meld4_loaders(args, load_video: bool) -> Tuple[DataLoader, DataLoader, DataLoader]:
    data_root = Path(args.data_root)
    paths = {
        "train": data_root / "train_meld_emo.csv",
        "dev": data_root / "dev_meld_emo.csv",
        "test": data_root / "test_meld_emo.csv",
    }
    for split, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing MELD {split} csv: {path}")

    train_data = filter_meld4_samples(base.preprocessing(str(paths["train"])))
    dev_data = filter_meld4_samples(base.preprocessing(str(paths["dev"])))
    test_data = filter_meld4_samples(base.preprocessing(str(paths["test"])))
    print(
        f"[TELME-MELD4] filtered samples train={len(train_data)} "
        f"dev={len(dev_data)} test={len(test_data)}"
    )

    if args.max_train_samples > 0:
        train_data = train_data[: args.max_train_samples]
    if args.max_eval_samples > 0:
        dev_data = dev_data[: args.max_eval_samples]
        test_data = test_data[: args.max_eval_samples]

    collate = partial(make_telme_meld4_batch, load_video=load_video)
    train_loader = DataLoader(
        base.meld_dataset(train_data),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
    )
    dev_loader = DataLoader(
        base.meld_dataset(dev_data),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        base.meld_dataset(test_data),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )
    return train_loader, dev_loader, test_loader


def main() -> None:
    base.MELD_LABELS = MELD4_CLASS_LABELS
    base.build_loaders = build_meld4_loaders

    args = base.parse_args()
    if args.save_root == "./MELD/save_model_meld_modalities":
        args.save_root = "./MELD/save_model_meld4_modalities"

    modalities = base.parse_modalities(args.modalities)
    if not modalities:
        raise ValueError("No valid modalities requested")

    base.seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_root = Path(args.save_root)
    shared_root = save_root / "shared"
    print(f"[TELME-MELD4] device={device} data_root={args.data_root} save_root={save_root}")
    print(f"[TELME-MELD4] label_order={MELD4_CLASS_LABELS} modalities={modalities}")

    teacher = base.train_teacher(args, device, shared_root)
    video_student = base.train_video_student(args, teacher, device, shared_root)

    for modality in modalities:
        base.train_fusion_modality(args, modality, teacher, video_student, device, save_root)

    print("[TELME-MELD4] done")


if __name__ == "__main__":
    main()
