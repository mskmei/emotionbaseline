#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import importlib
import json


def parse_args():
    parser = argparse.ArgumentParser(description="Check whether the active environment can load the JSL Qwen tokenizer.")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-1.7B")
    return parser.parse_args()


def module_version(name: str) -> str:
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        return f"<not importable: {exc}>"
    return str(getattr(module, "__version__", "<unknown>"))


def main():
    args = parse_args()
    info = {
        "base_model": args.base_model,
        "python_packages": {
            "torch": module_version("torch"),
            "transformers": module_version("transformers"),
            "tokenizers": module_version("tokenizers"),
            "huggingface_hub": module_version("huggingface_hub"),
            "sentencepiece": module_version("sentencepiece"),
            "protobuf": module_version("google.protobuf"),
            "peft": module_version("peft"),
            "accelerate": module_version("accelerate"),
        },
        "tokenizer_ok": False,
    }

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
        info["tokenizer_ok"] = True
        info["tokenizer_class"] = tokenizer.__class__.__name__
        info["pad_token_id"] = tokenizer.pad_token_id
        info["eos_token_id"] = tokenizer.eos_token_id
        info["chat_template"] = bool(getattr(tokenizer, "chat_template", None))
    except Exception as exc:
        info["tokenizer_error"] = repr(exc)
        info["suggestion"] = (
            "Run: pip install -U 'transformers>=4.51.0' tokenizers accelerate sentencepiece protobuf "
            "or set BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct for a more widely supported fallback."
        )

    print(json.dumps(info, ensure_ascii=False, indent=2))
    if not info["tokenizer_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
