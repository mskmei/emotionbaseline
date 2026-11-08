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
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))

from dataloader import MELDDataset  # noqa: E402
from train_eval_mmgcn_unified import (  # noqa: E402
    build_model,
    inspect_feature_pkl,
    load_torch_checkpoint,
    sequence_lengths,
    zero_unused_modalities,
)


def loader(path: Path, train: bool, batch_size: int) -> DataLoader:
    dataset = MELDDataset(str(path), train=train)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        num_workers=0,
    )


@torch.no_grad()
def collect(model, joint_model, dataloader: DataLoader, device: torch.device) -> Dict[str, np.ndarray]:
    model.eval()
    joint_model.eval()
    base_log_probs: List[np.ndarray] = []
    joint_log_probs: List[np.ndarray] = []
    visuals: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []
    lengths_all: List[np.ndarray] = []
    domains: List[np.ndarray] = []
    ids: List[str] = []
    for data in dataloader:
        textf, visuf, acouf, qmask, umask, labels = [value.to(device) for value in data[:-1]]
        textf, visuf, acouf = zero_unused_modalities(textf, visuf, acouf, "video")
        lengths = sequence_lengths(umask)
        log_prob, *_ = model(textf, qmask, umask, lengths, acouf, visuf)
        joint_log_prob, *_ = joint_model(textf, qmask, umask, lengths, acouf, visuf)
        labels_flat = torch.cat([labels[index][:length] for index, length in enumerate(lengths)])
        visual_flat = torch.cat([visuf[:length, index] for index, length in enumerate(lengths)], dim=0)
        valid = labels_flat.ge(0)
        base_log_probs.append(log_prob[valid].cpu().numpy().astype(np.float32))
        joint_log_probs.append(joint_log_prob[valid].cpu().numpy().astype(np.float32))
        visuals.append(visual_flat[valid].cpu().numpy().astype(np.float32))
        labels_all.append(labels_flat[valid].cpu().numpy().astype(np.int64))
        valid_lengths: List[int] = []
        valid_domains: List[bool] = []
        label_matrix = labels.cpu()
        for batch_index, vid in enumerate(data[-1]):
            is_bobsl = "bobsl" in str(vid).lower()
            for utterance_index in range(lengths[batch_index]):
                if int(label_matrix[batch_index, utterance_index]) >= 0:
                    ids.append(f"{vid}:utt{utterance_index + 1:02d}")
                    valid_lengths.append(lengths[batch_index])
                    valid_domains.append(is_bobsl)
        lengths_all.append(np.asarray(valid_lengths, dtype=np.float32))
        domains.append(np.asarray(valid_domains, dtype=bool))
    return {
        "base_log_prob": np.concatenate(base_log_probs),
        "source_log_prob": np.concatenate(base_log_probs),
        "joint_log_prob": np.concatenate(joint_log_probs),
        "visual": np.concatenate(visuals),
        "labels": np.concatenate(labels_all),
        "lengths": np.concatenate(lengths_all),
        "domain": np.concatenate(domains),
        "ids": np.asarray(ids, dtype=str),
    }


def save_cache(path: Path, payload: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    print(
        f"[MMGCN-cache] {path} n={len(payload['labels'])} "
        f"bobsl={int(payload['domain'].sum())} visual_dim={payload['visual'].shape[1]}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_pkl", required=True)
    parser.add_argument("--joint_pkl", required=True)
    parser.add_argument("--external_pkl", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--joint_checkpoint", default="")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    source_pkl = Path(args.source_pkl).expanduser().resolve()
    joint_pkl = Path(args.joint_pkl).expanduser().resolve()
    external_pkl = Path(args.external_pkl).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    for path in (source_pkl, joint_pkl, external_pkl, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = load_torch_checkpoint(checkpoint_path, device)
    base_args = SimpleNamespace(**checkpoint["args"])
    base_args.no_cuda = device.type != "cuda"
    info = inspect_feature_pkl(source_pkl)
    model, _, _ = build_model(base_args, info, "video", device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    joint_model = model
    if args.joint_checkpoint:
        joint_checkpoint = load_torch_checkpoint(Path(args.joint_checkpoint).expanduser().resolve(), device)
        joint_args = SimpleNamespace(**joint_checkpoint["args"])
        joint_args.no_cuda = device.type != "cuda"
        joint_model, _, _ = build_model(joint_args, info, "video", device)
        joint_model.load_state_dict(joint_checkpoint["model_state_dict"], strict=True)
    jobs = (
        ("source_train.npz", source_pkl, True),
        ("joint_train.npz", joint_pkl, True),
        ("source_test.npz", source_pkl, False),
        ("external_test.npz", external_pkl, False),
    )
    for filename, path, train in jobs:
        save_cache(out_dir / filename, collect(model, joint_model, loader(path, train, args.batch_size), device))


if __name__ == "__main__":
    main()
