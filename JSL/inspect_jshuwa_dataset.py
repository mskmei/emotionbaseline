from __future__ import annotations

import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect the gated mouwjone/J-Shuwa HF dataset structure.")
    parser.add_argument("--dataset", type=str, default="mouwjone/J-Shuwa")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--hf_token_env", type=str, default="HF_TOKEN")
    parser.add_argument("--num_rows", type=int, default=3)
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
    print(ds)
    print("features:", ds.features)
    for i in range(min(args.num_rows, len(ds))):
        print(f"row[{i}]:", ds[i])


if __name__ == "__main__":
    main()
