# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from jsl_translation_model import JSLQwenPrefixTranslator
from keypoints import KEYPOINT_DIM, sample_keypoints_sequence
from manifest_utils import first_nonempty, read_csv_rows


class KeypointTextDataset(Dataset):
    def __init__(self, manifest_csv: Path, split: Optional[str] = None, max_samples: int = 0):
        rows = read_csv_rows(manifest_csv)
        if not rows:
            raise RuntimeError(f"Empty manifest: {manifest_csv}")
        required = {"keypoints_path", "text"}
        missing = required - set(rows[0].keys())
        if missing:
            raise RuntimeError(f"Manifest missing required columns: {sorted(missing)}")
        if split and "split" in rows[0]:
            rows = [r for r in rows if str(r.get("split", "")).strip() == split]
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
        if not keypoints_path.exists():
            raise FileNotFoundError(f"Missing keypoints: {keypoints_path}")
        data = np.load(keypoints_path)
        keypoints = np.asarray(data["keypoints"], dtype=np.float32)
        text = first_nonempty(row, ["text", "subtitle_text", "translation"])
        if text is None:
            raise RuntimeError(f"Missing text for row {idx}")
        return {
            "sample_id": str(row.get("sample_id", keypoints_path.stem)),
            "keypoints": keypoints,
            "text": text,
        }


class JSLDataCollator:
    def __init__(self, model: JSLQwenPrefixTranslator, num_visual_tokens: int, max_target_tokens: int):
        self.model = model
        self.tokenizer = model.tokenizer
        self.num_visual_tokens = int(num_visual_tokens)
        self.max_target_tokens = int(max_target_tokens)
        self.prompt_ids = model.prompt_ids(device=None)

    def __call__(self, batch: List[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        keypoints = [
            torch.from_numpy(sample_keypoints_sequence(item["keypoints"], self.num_visual_tokens))
            for item in batch
        ]
        prompt_input_ids = self.prompt_ids.unsqueeze(0).expand(len(batch), -1).clone()
        target_ids = []
        for item in batch:
            text = str(item["text"]).strip()
            if not text:
                raise RuntimeError(f"Empty target text for sample {item['sample_id']}")
            ids = self.tokenizer(
                text + self.tokenizer.eos_token,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_target_tokens,
            ).input_ids
            target_ids.append(torch.tensor(ids, dtype=torch.long))

        target_input_ids = torch.nn.utils.rnn.pad_sequence(
            target_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        return {
            "keypoints": torch.stack(keypoints, dim=0).float(),
            "prompt_input_ids": prompt_input_ids.long(),
            "target_input_ids": target_input_ids.long(),
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Train MediaPipe-keypoint MLP + Qwen3 LoRA for JSL-to-Japanese translation.")
    parser.add_argument("--manifest_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--valid_split", type=str, default="")
    parser.add_argument("--num_visual_tokens", type=int, default=64)
    parser.add_argument("--projector_hidden", type=int, default=2048)
    parser.add_argument("--max_target_tokens", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--save_epochs", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch_dtype", type=str, default="auto")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, default="all", help="all, qkvo, qv, or comma-separated module names.")
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_eval(model, loader, device, autocast_dtype) -> float:
    if loader is None:
        return float("nan")
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="valid", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=autocast_dtype is not None, dtype=autocast_dtype):
                out = model(**batch)
            losses.append(float(out.loss.detach().cpu()))
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def main():
    args = parse_args()
    seed_everything(args.seed)

    dtype_arg = "bf16" if args.bf16 else "fp16" if args.fp16 else args.torch_dtype
    model = JSLQwenPrefixTranslator(
        base_model=args.base_model,
        keypoint_dim=KEYPOINT_DIM,
        num_visual_tokens=args.num_visual_tokens,
        projector_hidden=args.projector_hidden,
        torch_dtype=dtype_arg,
        load_in_4bit=args.load_in_4bit,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.load_in_4bit:
        model.to(device)
    else:
        model.projector.to(device)

    train_dataset = KeypointTextDataset(Path(args.manifest_csv), split=args.train_split or None, max_samples=args.max_samples)
    valid_dataset = None
    if args.valid_split:
        valid_dataset = KeypointTextDataset(Path(args.manifest_csv), split=args.valid_split, max_samples=0)

    collator = JSLDataCollator(model, args.num_visual_tokens, args.max_target_tokens)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
    )
    valid_loader = None
    if valid_dataset is not None:
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collator)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    total_updates = math.ceil(len(train_loader) / max(args.gradient_accumulation_steps, 1)) * max(args.epochs, 1)
    warmup_steps = int(total_updates * args.warmup_ratio)
    from transformers import get_linear_schedule_with_warmup

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=max(total_updates, 1))

    autocast_dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else None
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and torch.cuda.is_available())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    print(f"[Train] device={device} rows={len(train_dataset)} trainable_params={sum(p.numel() for p in trainable)}")
    for epoch in range(args.epochs):
        model.train()
        running = []
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for step, batch in enumerate(pbar, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=autocast_dtype is not None, dtype=autocast_dtype):
                out = model(**batch)
                loss = out.loss / max(args.gradient_accumulation_steps, 1)
            scaler.scale(loss).backward()
            running.append(float(out.loss.detach().cpu()))

            if step % args.gradient_accumulation_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                if global_step % args.logging_steps == 0:
                    pbar.set_postfix(loss=f"{np.mean(running[-args.logging_steps:]):.4f}", update=global_step)

        valid_loss = maybe_eval(model, valid_loader, device, autocast_dtype)
        if valid_loader is not None:
            print(f"[Train] epoch={epoch + 1} valid_loss={valid_loss:.6f}")

        if args.save_epochs > 0 and (epoch + 1) % args.save_epochs == 0:
            ckpt_dir = output_dir / f"checkpoint-epoch{epoch + 1:03d}"
            model.save_pretrained(ckpt_dir)
            print(f"[Train] saved checkpoint: {ckpt_dir}")

    model.save_pretrained(output_dir)
    print(f"[Train] saved final model: {output_dir}")


if __name__ == "__main__":
    main()
