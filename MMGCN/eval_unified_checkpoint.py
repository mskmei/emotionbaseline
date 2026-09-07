#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataloader import MELDDataset
from train_eval_mmgcn_unified import (
    build_model,
    extract_state_dict,
    inspect_feature_pkl,
    load_torch_checkpoint,
    parse_modalities,
    partially_load_checkpoint,
    resolve_path,
    run_epoch,
    save_report,
)


TRAIN_DEFAULTS = {
    "base_model": "LSTM",
    "graph_type": "DeepGCN",
    "graph_construct": "direct",
    "mm_fusion_mthd": "concat_subsequently",
    "modals": "auto",
    "deep_gcn_nlayers": 4,
    "windowp": 10,
    "windowf": 10,
    "attention": "general",
    "alpha": 0.2,
    "multiheads": 6,
    "nodal_attention": True,
    "active_listener": False,
    "use_gcn": False,
    "use_topic": False,
    "use_speaker": True,
    "use_modal": False,
    "av_using_lstm": False,
    "no_residue": False,
    "d_g": 150,
    "d_p": 150,
    "d_e": 100,
    "d_h": 100,
    "d_a": 100,
    "graph_hidden": 100,
    "max_seq_len": 300,
    "no_cuda": False,
}


def build_args_from_checkpoint(checkpoint, device: torch.device, graph_type_override: str) -> SimpleNamespace:
    saved_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    values = {**TRAIN_DEFAULTS, **saved_args}
    if isinstance(checkpoint, dict) and checkpoint.get("graph_type"):
        values["graph_type"] = checkpoint["graph_type"]
    if graph_type_override:
        values["graph_type"] = graph_type_override
    values["no_cuda"] = values.get("no_cuda", False) or device.type != "cuda"
    return SimpleNamespace(**values)


def evaluate_one(args, checkpoint_path: Path, checkpoint, modality: str, test_pkl: Path, out_dir: Path, device: torch.device) -> None:
    info = inspect_feature_pkl(test_pkl)
    train_args = build_args_from_checkpoint(checkpoint, device, args.graph_type)
    model, modals, graph_type = build_model(train_args, info, modality, device)
    partially_load_checkpoint(model, checkpoint_path, device)

    testset = MELDDataset(str(test_pkl), train=False)
    loader = DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=testset.collate_fn,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available() and not train_args.no_cuda,
    )
    result = run_epoch(
        model,
        nn.NLLLoss(),
        loader,
        device,
        modality,
        optimizer=None,
        max_grad_norm=0.0,
    )

    label_names = list(info["label_names"])
    extra = {
        "script": "MMGCN/eval_unified_checkpoint.py",
        "checkpoint": str(checkpoint_path),
        "test_pkl": str(test_pkl),
        "modality": modality,
        "modals": modals,
        "graph_type": graph_type,
        "label_names": label_names,
        "dims": {
            "text": int(info["d_text"]),
            "audio": int(info["d_audio"]),
            "visual": int(info["d_visual"]),
            "n_speakers": int(info["n_speakers"]),
            "n_classes": int(info["n_classes"]),
        },
    }
    prefix = args.prefix or f"{modality}_{Path(test_pkl).stem}"
    summary = save_report(result, out_dir, prefix, label_names, extra)
    if summary:
        print(
            f"[MMGCN-EVAL][{modality}] {prefix} "
            f"acc={summary['accuracy']:.4f} macro_f1={summary['macro_f1']:.4f} "
            f"weighted_f1={summary['weighted_f1']:.4f}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a saved unified MMGCN checkpoint on one or more pkl test splits.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_pkl", type=str, nargs="+", required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--modalities", nargs="+", default=[])
    parser.add_argument("--graph_type", type=str, default="", help="Optional override; default uses the checkpoint graph_type.")
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--no_cuda", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = resolve_path(args.checkpoint, must_exist=True)
    test_pkls = [resolve_path(x, must_exist=True) for x in args.test_pkl]
    out_dir = resolve_path(args.out_dir, must_exist=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    checkpoint = load_torch_checkpoint(checkpoint_path, device)
    state = extract_state_dict(checkpoint)
    print(f"[MMGCN-EVAL] checkpoint={checkpoint_path} tensors={len(state)}")
    print(f"[MMGCN-EVAL] test_pkls={json.dumps([str(x) for x in test_pkls], ensure_ascii=False)}")
    print(f"[MMGCN-EVAL] out_dir={out_dir} device={device}")

    modalities = parse_modalities(args.modalities or [checkpoint.get("modality", "video") if isinstance(checkpoint, dict) else "video"])
    for modality in modalities:
        for test_pkl in test_pkls:
            evaluate_one(args, checkpoint_path, checkpoint, modality, test_pkl, out_dir, device)


if __name__ == "__main__":
    main()
