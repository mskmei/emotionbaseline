#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(REPO_ROOT / "new"))

from train_eval_cmerc_visual_anjs import (  # noqa: E402
    CMERCVisualDataset,
    build_model,
    inspect_pkl,
    prepare_inputs,
    sequence_lengths,
)
from train_eval_domain_routed_cmerc import flatten_seq_first, load_checkpoint  # noqa: E402


def make_loader(path: Path, split: str, batch_size: int) -> DataLoader:
    dataset = CMERCVisualDataset(path, split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn)


@torch.no_grad()
def collect(source_model, joint_model, dataloader: DataLoader, device: torch.device) -> Dict[str, np.ndarray]:
    source_model.eval()
    joint_model.eval()
    source_probs: List[np.ndarray] = []
    joint_probs: List[np.ndarray] = []
    visuals: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []
    lengths_all: List[np.ndarray] = []
    ids: List[str] = []
    for data in dataloader:
        text1, text2, text3, text4, visuf, acouf, qmask, umask, labels = [
            value.to(device) for value in data[:-1]
        ]
        texts, visuf, acouf = prepare_inputs([text1, text2, text3, text4], visuf, acouf, "visual")
        lengths = sequence_lengths(umask)
        source_log_prob, *_ = source_model(texts, qmask, umask, lengths, acouf, visuf, 0)
        joint_log_prob, *_ = joint_model(texts, qmask, umask, lengths, acouf, visuf, 0)
        labels_flat = torch.cat([labels[index][:length] for index, length in enumerate(lengths)])
        visual_flat = flatten_seq_first(visuf, lengths)
        valid = labels_flat.ge(0)
        source_probs.append(source_log_prob[valid].cpu().numpy().astype(np.float32))
        joint_probs.append(joint_log_prob[valid].cpu().numpy().astype(np.float32))
        visuals.append(visual_flat[valid].cpu().numpy().astype(np.float32))
        labels_all.append(labels_flat[valid].cpu().numpy().astype(np.int64))
        valid_lengths: List[int] = []
        label_matrix = labels.cpu()
        for batch_index, vid in enumerate(data[-1]):
            for utterance_index in range(lengths[batch_index]):
                if int(label_matrix[batch_index, utterance_index]) >= 0:
                    ids.append(f"{vid}:utt{utterance_index + 1:02d}")
                    valid_lengths.append(lengths[batch_index])
        lengths_all.append(np.asarray(valid_lengths, dtype=np.float32))
    return {
        "source_log_prob": np.concatenate(source_probs),
        "joint_log_prob": np.concatenate(joint_probs),
        "visual": np.concatenate(visuals),
        "labels": np.concatenate(labels_all),
        "lengths": np.concatenate(lengths_all),
        "ids": np.asarray(ids, dtype=str),
    }


def save(path: Path, payload: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    print(f"[CMERC-pair-cache] {path} n={len(payload['labels'])}")


def load_model(checkpoint_path: Path, info: Dict, device: torch.device):
    checkpoint = load_checkpoint(checkpoint_path, device)
    model_args = SimpleNamespace(**checkpoint["args"])
    model = build_model(model_args, info, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_pkl", required=True)
    parser.add_argument("--external_pkl", required=True)
    parser.add_argument("--source_checkpoint", required=True)
    parser.add_argument("--joint_checkpoint", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    source_pkl = Path(args.source_pkl).expanduser().resolve()
    external_pkl = Path(args.external_pkl).expanduser().resolve()
    source_checkpoint = Path(args.source_checkpoint).expanduser().resolve()
    joint_checkpoint = Path(args.joint_checkpoint).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    for path in (source_pkl, external_pkl, source_checkpoint, joint_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info = inspect_pkl(source_pkl)
    source_model = load_model(source_checkpoint, info, device)
    joint_model = load_model(joint_checkpoint, info, device)
    for filename, path, split in (
        ("source_train.npz", source_pkl, "train"),
        ("source_test.npz", source_pkl, "test"),
        ("external_test.npz", external_pkl, "test"),
    ):
        save(out_dir / filename, collect(source_model, joint_model, make_loader(path, split, args.batch_size), device))


if __name__ == "__main__":
    main()
