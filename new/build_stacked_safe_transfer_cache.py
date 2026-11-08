#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from residual_transfer_module import SafeTransferGate


def build_model(checkpoint: dict, visual_dim: int, device: torch.device) -> SafeTransferGate:
    args = checkpoint["args"]
    model = SafeTransferGate(
        visual_dim=visual_dim,
        n_classes=4,
        hidden_dim=int(args.get("hidden_dim", 32)),
        dropout=float(args.get("dropout", 0.1)),
        bias=float(args.get("gate_bias", -2.0)),
        max_scale=float(args.get("max_scale", 1.0)),
        mode=str(args.get("mode", "reliability")),
        visual_gate_dim=int(args.get("visual_gate_dim", 16)),
        fusion_space=str(args.get("fusion_space", "probability")),
        delta_clip=float(args.get("delta_clip", 6.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def transform(
    input_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    device: torch.device,
    batch_size: int,
) -> dict:
    payload = np.load(input_path, allow_pickle=True)
    source = payload["source_log_prob"].astype(np.float32)
    joint = payload["joint_log_prob"].astype(np.float32)
    visual = payload["visual"].astype(np.float32)
    lengths = payload["lengths"].astype(np.float32)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(checkpoint, visual.shape[1], device)
    output_branch = "source_log_prob" if checkpoint.get("source_fallback", False) else "fused_log_prob"
    stacked = []
    with torch.no_grad():
        for start in range(0, len(source), batch_size):
            stop = start + batch_size
            output = model(
                torch.from_numpy(source[start:stop]).to(device),
                torch.from_numpy(joint[start:stop]).to(device),
                torch.from_numpy(visual[start:stop]).to(device),
                torch.from_numpy(lengths[start:stop]).to(device),
            )
            stacked.append(output[output_branch].cpu().numpy().astype(np.float32))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        source_log_prob=np.concatenate(stacked),
        joint_log_prob=joint,
        visual=visual,
        labels=payload["labels"],
        lengths=lengths,
        ids=payload["ids"],
    )
    return {
        "input": str(input_path),
        "output": str(output_path),
        "checkpoint": str(checkpoint_path),
        "output_branch": output_branch,
        "n": len(source),
        "visual_dim": visual.shape[1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    cache_dir = Path(args.cache_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    checkpoint = Path(args.checkpoint).expanduser()
    device = torch.device(args.device)
    summaries = []
    for filename in ("source_train.npz", "source_test.npz", "external_test.npz"):
        summaries.append(
            transform(
                cache_dir / filename,
                out_dir / filename,
                checkpoint,
                device,
                args.batch_size,
            )
        )
    (out_dir / "stack_cache_manifest.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[stack-cache] wrote {out_dir}")


if __name__ == "__main__":
    main()
