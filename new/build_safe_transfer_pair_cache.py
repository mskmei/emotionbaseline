#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_script(path: Path, name: str):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_checkpoint(path: Path, device: torch.device) -> Dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def flatten_visual(visual: torch.Tensor, lengths: List[int], labels_flat: torch.Tensor) -> torch.Tensor:
    values = torch.cat([visual[:length, index] for index, length in enumerate(lengths)], dim=0)
    return values[labels_flat.ge(0)]


def item_ids(vids, lengths: List[int], labels: torch.Tensor) -> List[str]:
    labels_cpu = labels.detach().cpu()
    output: List[str] = []
    for batch_index, vid in enumerate(vids):
        for utterance_index in range(lengths[batch_index]):
            if int(labels_cpu[batch_index, utterance_index]) >= 0:
                output.append(f"{vid}:utt{utterance_index + 1:02d}")
    return output


def save_cache(path: Path, chunks: Dict[str, List[np.ndarray]]) -> None:
    payload = {key: np.concatenate(values) if key != "ids" else np.asarray(sum(values, []), dtype=str) for key, values in chunks.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    print(f"[portable-pair-cache] {path} n={len(payload['labels'])} visual_dim={payload['visual'].shape[1]}")


def empty_chunks() -> Dict[str, List]:
    return {key: [] for key in ("source_log_prob", "joint_log_prob", "visual", "labels", "lengths", "ids")}


def build_magtkd(args, device: torch.device) -> None:
    module = load_script(REPO_ROOT / "new/MAGTKD/MELD/train_eval_magtkd_video_anjs.py", "portable_magtkd")
    source_checkpoint = load_checkpoint(args.source_checkpoint, device)
    joint_checkpoint = load_checkpoint(args.joint_checkpoint, device)

    def make_model(checkpoint):
        model_args = SimpleNamespace(**checkpoint["args"])
        model_args.clsNum = 4
        model = module.Transformer_Based_Model(model_args).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        return model, module.resolve_output_head(model_args.modality, model_args.output_head)

    source_model, source_head = make_model(source_checkpoint)
    joint_model, joint_head = make_model(joint_checkpoint)

    @torch.no_grad()
    def collect(path: Path, split: str):
        dataset = module.MAGTKDFeatureDataset(path, split)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate_fn)
        chunks = empty_chunks()
        for data in loader:
            text, audio, video, audio_kd, video_kd, qmask, umask, labels = [value.to(device) for value in data[:-1]]
            lengths = module.sequence_lengths(umask)
            inputs = module.prepare_inputs(text, audio, video, audio_kd, video_kd, "video")
            source_outputs = source_model(inputs[0], inputs[3], inputs[4], umask, qmask, lengths)
            joint_outputs = joint_model(inputs[0], inputs[3], inputs[4], umask, qmask, lengths)
            source_logits = module.choose_logits(source_head, *source_outputs[:3])
            joint_logits = module.choose_logits(joint_head, *joint_outputs[:3])
            valid = umask.bool() & labels.ge(0)
            labels_flat = torch.cat([labels[index][:length] for index, length in enumerate(lengths)])
            chunks["source_log_prob"].append(F.log_softmax(source_logits[valid], dim=-1).cpu().numpy().astype(np.float32))
            chunks["joint_log_prob"].append(F.log_softmax(joint_logits[valid], dim=-1).cpu().numpy().astype(np.float32))
            chunks["visual"].append(flatten_visual(video, lengths, labels_flat).cpu().numpy().astype(np.float32))
            chunks["labels"].append(labels[valid].cpu().numpy().astype(np.int64))
            chunks["lengths"].append(np.concatenate([np.full(int((labels[index][:length] >= 0).sum()), length, dtype=np.float32) for index, length in enumerate(lengths)]))
            chunks["ids"].append(item_ids(data[-1], lengths, labels))
        return chunks

    for filename, path, split in cache_jobs(args):
        save_cache(args.out_dir / filename, collect(path, split))


def build_ecerc(args, device: torch.device) -> None:
    module = load_script(REPO_ROOT / "new/ECERC/MELD/train_eval_ecerc_video_anjs.py", "portable_ecerc")
    info = module.inspect_pkl(args.source_pkl)
    source_checkpoint = load_checkpoint(args.source_checkpoint, device)
    joint_checkpoint = load_checkpoint(args.joint_checkpoint, device)

    def make_model(checkpoint):
        model_args = SimpleNamespace(**checkpoint["args"])
        model = module.build_model(model_args, info, device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        return model

    source_model = make_model(source_checkpoint)
    joint_model = make_model(joint_checkpoint)

    @torch.no_grad()
    def collect(path: Path, split: str):
        dataset = module.ECERCFeatureDataset(path, split)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate_fn)
        chunks = empty_chunks()
        for data in loader:
            emo_text, sem_text, visual, audio, qmask, umask, labels = [value.to(device) for value in data[:-1]]
            lengths = module.sequence_lengths(umask)
            values = module.prepare_inputs(emo_text, sem_text, visual, audio, "video")
            model_input = torch.cat([values[0], values[3], values[2]], dim=-1)
            source_log_prob = source_model(model_input, values[1], qmask, umask, lengths)
            joint_log_prob = joint_model(model_input, values[1], qmask, umask, lengths)
            labels_flat = torch.cat([labels[index][:length] for index, length in enumerate(lengths)])
            valid = labels_flat.ge(0)
            chunks["source_log_prob"].append(source_log_prob[valid].cpu().numpy().astype(np.float32))
            chunks["joint_log_prob"].append(joint_log_prob[valid].cpu().numpy().astype(np.float32))
            chunks["visual"].append(flatten_visual(visual, lengths, labels_flat).cpu().numpy().astype(np.float32))
            chunks["labels"].append(labels_flat[valid].cpu().numpy().astype(np.int64))
            chunks["lengths"].append(np.concatenate([np.full(int((labels[index][:length] >= 0).sum()), length, dtype=np.float32) for index, length in enumerate(lengths)]))
            chunks["ids"].append(item_ids(data[-1], lengths, labels))
        return chunks

    for filename, path, split in cache_jobs(args):
        save_cache(args.out_dir / filename, collect(path, split))


def build_conxgnn(args, device: torch.device) -> None:
    module = load_script(REPO_ROOT / "new2/ConxGNN/train_eval_conxgnn_visual_anjs.py", "portable_conxgnn")
    source_payload = module.load_payload(args.source_pkl)
    info = module.inspect_pkl(args.source_pkl)
    source_checkpoint = load_checkpoint(args.source_checkpoint, device)
    joint_checkpoint = load_checkpoint(args.joint_checkpoint, device)

    config_sets = {
        split: module.ConxVisualDataset(source_payload.get(split, []), args.batch_size, info["d_text"], info["d_audio"], info["d_visual"])
        for split in ("train", "dev", "test")
    }

    def make_model(checkpoint):
        model_args = SimpleNamespace(**checkpoint["args"])
        model_args.device = str(device)
        config = module.build_config(model_args, info, config_sets["train"], config_sets["dev"], config_sets["test"])
        model = module.src.MainModel(config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        return model

    source_model = make_model(source_checkpoint)
    joint_model = make_model(joint_checkpoint)

    @torch.no_grad()
    def collect(path: Path, split: str):
        payload = module.load_payload(path)
        split_key = "test" if split == "test" else "train"
        dataset = module.ConxVisualDataset(payload.get(split_key, []), args.batch_size, info["d_text"], info["d_audio"], info["d_visual"])
        chunks = empty_chunks()
        for batch_index in range(len(dataset)):
            raw = dataset[batch_index]
            data = module.apply_modality_inputs(module.move_batch(raw, device), "visual", False, 0.0, "utterance")
            labels = data["label_tensor"]
            valid = labels.ge(0)
            source_logits, _ = module.conx_forward_logits(source_model, data)
            joint_logits, _ = module.conx_forward_logits(joint_model, data)
            lengths = data["text_len_tensor"].cpu().tolist()
            visual = data["visual_tensor"]
            visual_flat = torch.cat([visual[index, :length] for index, length in enumerate(lengths)], dim=0)
            chunks["source_log_prob"].append(F.log_softmax(source_logits[valid], dim=-1).cpu().numpy().astype(np.float32))
            chunks["joint_log_prob"].append(F.log_softmax(joint_logits[valid], dim=-1).cpu().numpy().astype(np.float32))
            chunks["visual"].append(visual_flat[valid].cpu().numpy().astype(np.float32))
            chunks["labels"].append(labels[valid].cpu().numpy().astype(np.int64))
            chunks["lengths"].append(np.concatenate([np.full(length, length, dtype=np.float32) for length in lengths])[valid.cpu().numpy()])
            chunks["ids"].append([item for item, keep in zip(raw["item_ids"], valid.cpu().tolist()) if keep])
        return chunks

    for filename, path, split in cache_jobs(args):
        save_cache(args.out_dir / filename, collect(path, split))


def cache_jobs(args) -> List[Tuple[str, Path, str]]:
    return [
        ("source_train.npz", args.source_pkl, "train"),
        ("source_test.npz", args.source_pkl, "test"),
        ("external_test.npz", args.external_pkl, "test"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", choices=["MAGTKD", "ECERC", "ConxGNN"], required=True)
    parser.add_argument("--source_pkl", type=Path, required=True)
    parser.add_argument("--external_pkl", type=Path, required=True)
    parser.add_argument("--source_checkpoint", type=Path, required=True)
    parser.add_argument("--joint_checkpoint", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    for field in ("source_pkl", "external_pkl", "source_checkpoint", "joint_checkpoint"):
        path = getattr(args, field).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        setattr(args, field, path)
    args.out_dir = args.out_dir.expanduser().resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Portable cache extraction requires a GPU for these baselines")
    builders = {"MAGTKD": build_magtkd, "ECERC": build_ecerc, "ConxGNN": build_conxgnn}
    builders[args.baseline](args, device)


if __name__ == "__main__":
    main()
