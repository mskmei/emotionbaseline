#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from jsl_translation_model import JSLQwenPrefixTranslator
from keypoints import sample_keypoints_sequence
from manifest_utils import read_csv_rows, sanitize_generated_text


class ManifestKeypointDataset(Dataset):
    def __init__(self, manifest_csv: Path, split: str = "", max_samples: int = 0):
        rows = read_csv_rows(manifest_csv)
        if split:
            rows = [row for row in rows if str(row.get("split", "")).strip() == split]
        if max_samples > 0:
            rows = rows[:max_samples]
        if not rows:
            raise RuntimeError(f"No rows selected from {manifest_csv}; split={split!r}")
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        keypoints_path = Path(row["keypoints_path"])
        data = np.load(keypoints_path)
        return {
            "sample_id": str(row.get("sample_id", keypoints_path.stem)),
            "keypoints": np.asarray(data["keypoints"], dtype=np.float32),
            "ref": str(row.get("text", "")),
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Generate predictions for a keypoint/text manifest with a JSL model.")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--manifest_csv", type=str, required=True)
    parser.add_argument("--out_jsonl", type=str, required=True)
    parser.add_argument("--split", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_visual_tokens", type=int, default=64)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--torch_dtype", type=str, default="auto")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--max_samples", type=int, default=0)
    return parser.parse_args()


def collate(batch: List[Dict[str, object]], num_visual_tokens: int) -> Dict[str, object]:
    return {
        "sample_id": [str(item["sample_id"]) for item in batch],
        "ref": [str(item.get("ref", "")) for item in batch],
        "keypoints": torch.stack(
            [
                torch.from_numpy(sample_keypoints_sequence(item["keypoints"], num_visual_tokens))
                for item in batch
            ],
            dim=0,
        ).float(),
    }


def main():
    args = parse_args()
    dtype_arg = "bf16" if args.bf16 else "fp16" if args.fp16 else args.torch_dtype
    model = JSLQwenPrefixTranslator.from_pretrained(
        args.model_dir,
        torch_dtype=dtype_arg,
        load_in_4bit=args.load_in_4bit,
        adapter_is_trainable=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.load_in_4bit:
        model.to(device)
    else:
        model.projector.to(device)

    dataset = ManifestKeypointDataset(Path(args.manifest_csv), split=args.split, max_samples=args.max_samples)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate(batch, args.num_visual_tokens),
    )

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for batch in tqdm(loader, desc=f"generate {args.split or 'all'}"):
            texts = model.generate_texts(
                batch["keypoints"],
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            for sample_id, text, ref in zip(batch["sample_id"], texts, batch["ref"]):
                row = {"sample_id": sample_id, "text": sanitize_generated_text(text), "ref": ref}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[JSL-generate-manifest] rows={len(dataset)} out={out_path}")


if __name__ == "__main__":
    main()
