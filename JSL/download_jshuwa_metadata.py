from __future__ import annotations

import argparse
import csv
import os
from collections import Counter
from pathlib import Path


REQUIRED_COLUMNS = ["vid", "yid", "start", "end", "source"]


def parse_args():
    parser = argparse.ArgumentParser(description="Download the gated J-Shuwa metadata table from Hugging Face.")
    parser.add_argument("--dataset", type=str, default="mouwjone/J-Shuwa")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--hf_token_env", type=str, default="HF_TOKEN")
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets first: pip install datasets") from exc

    token = os.getenv(args.hf_token_env) if args.hf_token_env else None
    kwargs = {"split": args.split}
    if token:
        kwargs["token"] = token

    ds = load_dataset(args.dataset, **kwargs)
    rows = [dict(row) for row in ds]
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError(f"No rows loaded from {args.dataset}:{args.split}")

    missing = [col for col in REQUIRED_COLUMNS if col not in rows[0]]
    if missing:
        raise RuntimeError(f"Unexpected J-Shuwa columns. missing={missing} available={list(rows[0].keys())}")

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    source_counts = Counter(str(row.get("source", "")) for row in rows)
    print(f"[J-Shuwa] rows={len(rows)} out={out}")
    print(f"[J-Shuwa] columns={fieldnames}")
    print(f"[J-Shuwa] source_counts={dict(source_counts)}")


if __name__ == "__main__":
    main()
