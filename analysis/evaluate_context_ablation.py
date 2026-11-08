#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


LABELS = ("A", "N", "J", "S")
REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINERS = {
    "MMGCN": REPO_ROOT / "MMGCN" / "train_eval_mmgcn_unified.py",
    "MAGTKD": REPO_ROOT / "new" / "MAGTKD" / "MELD" / "train_eval_magtkd_video_anjs.py",
    "CMERC": REPO_ROOT / "new2" / "CMERC" / "train_eval_cmerc_visual_anjs.py",
}


class ContextAblationDataset(Dataset):
    def __init__(self, base: Dataset, mode: str, seed: int):
        self.base = base
        self.mode = mode
        self.seed = int(seed)
        self.index: List[Tuple[int, Optional[int]]] = []
        if mode == "isolate":
            for dialogue_index in range(len(base)):
                item = base[dialogue_index]
                sequence_length = int(item[-2].shape[0])
                self.index.extend((dialogue_index, utterance_index) for utterance_index in range(sequence_length))
        else:
            self.index = [(dialogue_index, None) for dialogue_index in range(len(base))]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int):
        dialogue_index, utterance_index = self.index[index]
        item = self.base[dialogue_index]
        tensors = list(item[:-1])
        item_id = str(item[-1])
        sequence_length = int(tensors[-1].shape[0])
        if self.mode == "isolate":
            assert utterance_index is not None
            tensors = [tensor[utterance_index : utterance_index + 1] for tensor in tensors]
            item_id = f"{item_id}__isolated_{utterance_index + 1:03d}"
        elif self.mode == "shuffle":
            rng = np.random.default_rng(self.seed + 1000003 * dialogue_index)
            permutation = torch.as_tensor(rng.permutation(sequence_length), dtype=torch.long)
            tensors = [tensor.index_select(0, permutation) for tensor in tensors]
            item_id = f"{item_id}__shuffled"
        elif self.mode != "normal":
            raise ValueError(f"Unsupported context mode: {self.mode}")
        return (*tensors, item_id)

    def collate_fn(self, data):
        return self.base.collate_fn(data)


def load_module(baseline: str):
    path = TRAINERS[baseline]
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(f"context_ablation_{baseline.lower()}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import trainer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_checkpoint(path: Path, device: torch.device) -> Dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def checkpoint_args(checkpoint: Dict, overrides: Dict[str, object]) -> SimpleNamespace:
    values = dict(checkpoint.get("args") or {})
    values.update(overrides)
    return SimpleNamespace(**values)


def make_loader(dataset: Dataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def build_datasets(module, baseline: str, source_pkl: Path, external_pkl: Path, mode: str, seed: int):
    if baseline == "MMGCN":
        source_base = module.MELDDataset(str(source_pkl), train=False)
        external_base = module.MELDDataset(str(external_pkl), train=False)
    elif baseline == "MAGTKD":
        source_base = module.MAGTKDFeatureDataset(source_pkl, "test")
        external_base = module.MAGTKDFeatureDataset(external_pkl, "test")
    elif baseline == "CMERC":
        source_base = module.CMERCVisualDataset(source_pkl, "test")
        external_base = module.CMERCVisualDataset(external_pkl, "test")
    else:
        raise ValueError(baseline)
    return ContextAblationDataset(source_base, mode, seed), ContextAblationDataset(external_base, mode, seed)


def evaluate_mmgcn(module, args, checkpoint, train_pkl, source_loader, external_loader, device):
    info = module.inspect_feature_pkl(train_pkl)
    modality = str(checkpoint.get("modality") or "video")
    model, _, _ = module.build_model(args, info, modality, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    labels = module.collect_train_labels(train_pkl)
    loss_function = module.build_loss(args, labels, int(info["n_classes"]), device)
    return (
        module.run_epoch(model, loss_function, source_loader, device, modality),
        module.run_epoch(model, loss_function, external_loader, device, modality),
    )


def evaluate_magtkd(module, args, checkpoint, train_pkl, source_loader, external_loader, device):
    info = module.inspect_pkl(train_pkl)
    args.clsNum = len(LABELS)
    output_head = module.resolve_output_head(args.modality, args.output_head)
    model = module.Transformer_Based_Model(args).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    labels = module.collect_train_labels(train_pkl)
    weights = module.make_class_weights(labels, int(info["n_classes"]), device) if args.class_weight else None
    loss_function = nn.CrossEntropyLoss(weight=weights)
    return (
        module.run_epoch(model, loss_function, source_loader, device, output_head, modality=args.modality),
        module.run_epoch(model, loss_function, external_loader, device, output_head, modality=args.modality),
    )


def evaluate_cmerc(module, args, checkpoint, train_pkl, source_loader, external_loader, device):
    info = module.inspect_pkl(train_pkl)
    model = module.build_model(args, info, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    labels = module.collect_train_labels(train_pkl)
    weights = (
        module.make_class_weights(labels, int(info["n_classes"]), device)
        if args.class_weight and args.loss == "ce"
        else None
    )
    loss_function = module.FocalLoss(gamma=args.focal_gamma) if args.loss == "focal" else nn.NLLLoss(weight=weights)
    epoch = int(checkpoint.get("best_epoch") or checkpoint.get("epoch") or 1)
    return (
        module.run_epoch(model, loss_function, source_loader, device, epoch, modality=args.modality),
        module.run_epoch(model, loss_function, external_loader, device, epoch, modality=args.modality),
    )


def finite(value: float) -> Optional[float]:
    return float(value) if math.isfinite(float(value)) else None


def result_summary(result: Dict, common: Dict[str, object], split: str) -> Dict[str, object]:
    gold = np.asarray(result["golds"], dtype=np.int64)
    pred = np.asarray(result["preds"], dtype=np.int64)
    output: Dict[str, object] = {
        **common,
        "split": split,
        "n": len(gold),
        "loss": finite(result["loss"]),
        "accuracy": finite(result["accuracy"]),
        "macro_f1": finite(result["macro_f1"]),
        "weighted_f1": finite(result["weighted_f1"]),
        "gold_counts": {label: int(np.sum(gold == index)) for index, label in enumerate(LABELS)},
        "pred_counts": {label: int(np.sum(pred == index)) for index, label in enumerate(LABELS)},
        "per_class": {},
    }
    for class_id, label in enumerate(LABELS):
        tp = int(np.sum((gold == class_id) & (pred == class_id)))
        fp = int(np.sum((gold != class_id) & (pred == class_id)))
        fn = int(np.sum((gold == class_id) & (pred != class_id)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        output["per_class"][label] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}
    return output


def write_predictions(path: Path, result: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    probs = np.asarray(result["probs"], dtype=np.float64)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item_id", "gold_id", "gold_label", "pred_id", "pred_label", *[f"prob_{x}" for x in LABELS]])
        for item_id, gold, pred, prob in zip(result["ids"], result["golds"], result["preds"], probs):
            writer.writerow([item_id, int(gold), LABELS[int(gold)], int(pred), LABELS[int(pred)], *[float(x) for x in prob]])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved visual checkpoint with dialogue context ablations.")
    parser.add_argument("--baseline", choices=sorted(TRAINERS), required=True)
    parser.add_argument("--corpus", choices=["MELD", "IEMOCAP"], required=True)
    parser.add_argument("--config", choices=["source_only", "joint_b5k"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mode", choices=["normal", "isolate", "shuffle"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train_pkl", required=True, help="Original training pickle used to construct the checkpoint.")
    parser.add_argument("--source_pkl", required=True, help="Source corpus pickle providing the test dialogues.")
    parser.add_argument("--external_pkl", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=0, help="Zero reuses the checkpoint batch size.")
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Context ablation uses the original GPU model implementations and requires CUDA.")
    checkpoint_path = Path(cli.checkpoint).expanduser().resolve()
    train_pkl = Path(cli.train_pkl).expanduser().resolve()
    source_pkl = Path(cli.source_pkl).expanduser().resolve()
    external_pkl = Path(cli.external_pkl).expanduser().resolve()
    for path in (checkpoint_path, train_pkl, source_pkl, external_pkl):
        if not path.is_file():
            raise FileNotFoundError(path)

    module = load_module(cli.baseline)
    checkpoint = load_checkpoint(checkpoint_path, device)
    overrides = {"seed": cli.seed, "no_cuda": False, "num_workers": cli.num_workers}
    args = checkpoint_args(checkpoint, overrides)
    batch_size = cli.batch_size or int(getattr(args, "batch_size", 8))
    source_dataset, external_dataset = build_datasets(module, cli.baseline, source_pkl, external_pkl, cli.mode, cli.seed)
    source_loader = make_loader(source_dataset, batch_size, cli.num_workers)
    external_loader = make_loader(external_dataset, batch_size, cli.num_workers)

    if cli.baseline == "MMGCN":
        source_result, external_result = evaluate_mmgcn(
            module, args, checkpoint, train_pkl, source_loader, external_loader, device
        )
    elif cli.baseline == "MAGTKD":
        source_result, external_result = evaluate_magtkd(
            module, args, checkpoint, train_pkl, source_loader, external_loader, device
        )
    else:
        source_result, external_result = evaluate_cmerc(
            module, args, checkpoint, train_pkl, source_loader, external_loader, device
        )

    out_dir = Path(cli.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "baseline": cli.baseline,
        "corpus": cli.corpus,
        "config": cli.config,
        "seed": cli.seed,
        "context_mode": cli.mode,
        "checkpoint": str(checkpoint_path),
        "train_pkl": str(train_pkl),
    }
    summaries = [
        result_summary(source_result, common, cli.corpus),
        result_summary(external_result, common, "eJSL"),
    ]
    (out_dir / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_predictions(out_dir / "source_predictions.csv", source_result)
    write_predictions(out_dir / "ejsl_predictions.csv", external_result)
    for summary in summaries:
        print(
            f"[context-ablation] {cli.baseline}/{cli.corpus}/{cli.config}/seed{cli.seed}/{cli.mode} "
            f"split={summary['split']} wf1={summary['weighted_f1']:.4f} macro={summary['macro_f1']:.4f}"
        )
    print(f"[context-ablation] out_dir={out_dir}")


if __name__ == "__main__":
    main()
