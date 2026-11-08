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

from train_eval_domain_expert_cmerc import DomainExpertCMERC  # noqa: E402
from train_eval_domain_routed_cmerc import (  # noqa: E402
    CMERCVisualDataset,
    build_model,
    flatten_seq_first,
    inspect_pkl,
    load_checkpoint,
    namespace_from_checkpoint,
    sequence_lengths,
)


def make_loader(path: Path, split: str, batch_size: int) -> DataLoader:
    dataset = CMERCVisualDataset(path, split)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        num_workers=0,
    )


@torch.no_grad()
def collect(model: DomainExpertCMERC, dataloader: DataLoader, device: torch.device) -> Dict[str, np.ndarray]:
    model.eval()
    base_log_probs: List[np.ndarray] = []
    visuals: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []
    lengths_all: List[np.ndarray] = []
    domains: List[np.ndarray] = []
    ids: List[str] = []
    for data in dataloader:
        text1, text2, text3, text4, visuf, acouf, qmask, umask, labels = [
            value.to(device) for value in data[:-1]
        ]
        texts = [torch.zeros_like(text) for text in (text1, text2, text3, text4)]
        acouf = torch.zeros_like(acouf)
        lengths = sequence_lengths(umask)
        output = model(texts, qmask, umask, lengths, acouf, visuf, 0)
        labels_flat = torch.cat([labels[index][:length] for index, length in enumerate(lengths)])
        visual_flat = flatten_seq_first(visuf, lengths)
        valid = labels_flat.ge(0)
        base_log_probs.append(output["fused_log_prob"][valid].cpu().numpy().astype(np.float32))
        visuals.append(visual_flat[valid].cpu().numpy().astype(np.float32))
        labels_all.append(labels_flat[valid].cpu().numpy().astype(np.int64))
        label_matrix = labels.cpu()
        valid_lengths: List[int] = []
        valid_domains: List[bool] = []
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
        "visual": np.concatenate(visuals),
        "labels": np.concatenate(labels_all),
        "lengths": np.concatenate(lengths_all),
        "domain": np.concatenate(domains),
        "ids": np.asarray(ids, dtype=str),
    }


def save(path: Path, payload: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    print(f"[CMERC-cache] {path} n={len(payload['labels'])} bobsl={int(payload['domain'].sum())}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_pkl", required=True)
    parser.add_argument("--joint_pkl", required=True)
    parser.add_argument("--external_pkl", required=True)
    parser.add_argument("--source_checkpoint", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    source_pkl = Path(args.source_pkl).expanduser().resolve()
    joint_pkl = Path(args.joint_pkl).expanduser().resolve()
    external_pkl = Path(args.external_pkl).expanduser().resolve()
    source_checkpoint_path = Path(args.source_checkpoint).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    for path in (source_pkl, joint_pkl, external_pkl, source_checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_checkpoint = load_checkpoint(source_checkpoint_path, device)
    source_args = SimpleNamespace(**source_checkpoint["args"])
    context_path = Path(source_checkpoint["context_checkpoint"]).expanduser().resolve()
    context_checkpoint = load_checkpoint(context_path, device)
    info = inspect_pkl(source_pkl)
    context_args = namespace_from_checkpoint(context_checkpoint, source_args)
    context_model = build_model(context_args, info, device)
    context_model.load_state_dict(context_checkpoint["model_state_dict"], strict=True)
    model = DomainExpertCMERC(context_model, int(info["d_visual"]), source_args, has_bobsl=False).to(device)
    model.load_state_dict(source_checkpoint["model_state_dict"], strict=False)
    model.has_bobsl = False
    jobs = (
        ("source_train.npz", source_pkl, "train"),
        ("joint_train.npz", joint_pkl, "train"),
        ("source_test.npz", source_pkl, "test"),
        ("external_test.npz", external_pkl, "test"),
    )
    for filename, path, split in jobs:
        save(out_dir / filename, collect(model, make_loader(path, split, args.batch_size), device))


if __name__ == "__main__":
    main()
