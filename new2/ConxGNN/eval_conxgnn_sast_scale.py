#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(REPO_ROOT / "new"))

import train_eval_conxgnn_sast_visual_anjs as sast  # noqa: E402
import train_eval_conxgnn_visual_anjs as base_script  # noqa: E402
from anjs_video_common import resolve_path  # noqa: E402


def parse_scales(value: str):
    scales = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not scales or scales[0] < 0:
        raise ValueError("--scale_factors must contain non-negative values")
    return scales


def load_model(adapter_checkpoint: Path, device: torch.device):
    checkpoint = torch.load(adapter_checkpoint, map_location="cpu")
    args = SimpleNamespace(**checkpoint["args"])
    train_pkl = resolve_path(args.train_pkl, must_exist=True)
    external_pkl = resolve_path(args.external_test_pkl, must_exist=True)
    payload = base_script.load_payload(train_pkl)
    external_payload = base_script.load_payload(external_pkl)
    info = base_script.inspect_pkl(train_pkl)
    source_samples, _ = sast.split_samples(payload["train"])
    source_train = sast.make_dataset(source_samples, args.source_batch_size, info)
    source_test = sast.make_dataset(payload["test"], args.eval_batch_size, info)
    external_test = sast.make_dataset(external_payload["test"], args.eval_batch_size, info)

    base_checkpoint_path = resolve_path(args.base_checkpoint, must_exist=True)
    base_checkpoint = torch.load(base_checkpoint_path, map_location="cpu")
    base_model = sast.load_base_model(args, info, source_train, source_test, base_checkpoint, device)
    state = checkpoint["model_state_dict"]
    basis = state["adapter.emotion_basis"]
    model = sast.SASTConxGNN(base_model, int(info["d_visual"]), basis, args).to(device)
    model.load_state_dict(state, strict=True)
    return model, args, info, source_test, external_test


def main():
    parser = argparse.ArgumentParser(description="Select a SAST residual gate using source data only.")
    parser.add_argument("--adapter_checkpoint", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--scale_factors", default="0,0.1,0.25,0.5,0.75,1.0,1.25")
    parser.add_argument("--selection_metric", choices=["weighted_f1", "macro_f1", "accuracy"], default="weighted_f1")
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    if device.type != "cuda":
        raise RuntimeError("ConxGNN scale evaluation requires CUDA")
    checkpoint_path = resolve_path(args.adapter_checkpoint, must_exist=True)
    out_dir = resolve_path(args.out_dir, must_exist=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, train_args, info, source_test, external_test = load_model(checkpoint_path, device)
    original_scale = float(model.adapter.residual_scale)
    base_source = sast.evaluate(model, source_test, device, adapted=False)
    base_external = sast.evaluate(model, external_test, device, adapted=False)
    best_source = copy.deepcopy(base_source)
    best_external = copy.deepcopy(base_external)
    best_factor = 0.0
    best_score = float(base_source[args.selection_metric])
    rows = []
    for factor in parse_scales(args.scale_factors):
        model.adapter.residual_scale = original_scale * factor
        source = sast.evaluate(model, source_test, device, adapted=factor > 0)
        external = sast.evaluate(model, external_test, device, adapted=factor > 0)
        score = float(source[args.selection_metric])
        rows.append(
            {
                "scale_factor": factor,
                "source": source["weighted_f1"],
                "external": external["weighted_f1"],
                "delta_source": source["weighted_f1"] - base_source["weighted_f1"],
                "delta_external": external["weighted_f1"] - base_external["weighted_f1"],
            }
        )
        if score > best_score:
            best_score = score
            best_factor = factor
            best_source = copy.deepcopy(source)
            best_external = copy.deepcopy(external)

    model.adapter.residual_scale = original_scale * best_factor
    common = {
        "script": "new2/ConxGNN/eval_conxgnn_sast_scale.py",
        "method": "SAST-ConxGNN-source-calibrated-gate",
        "seed": int(getattr(train_args, "seed", 0)),
        "train_pkl": str(train_args.train_pkl),
        "external_test_pkl": str(train_args.external_test_pkl),
        "base_checkpoint": str(train_args.base_checkpoint),
        "adapter_checkpoint": str(checkpoint_path),
        "best_epoch": int(torch.load(checkpoint_path, map_location="cpu").get("epoch", 0)),
        "selected_scale_factor": best_factor,
        "base_source_weighted_f1": base_source["weighted_f1"],
        "base_external_weighted_f1": base_external["weighted_f1"],
        "scale_results": rows,
    }
    base_script.save_report(base_source, out_dir, "source_test_base", {**common, "selection": "base"})
    base_script.save_report(base_external, out_dir, "external_test_base", {**common, "selection": "base"})
    base_script.save_report(best_source, out_dir, "source_test_best", {**common, "selection": "source"})
    base_script.save_report(best_external, out_dir, "external_test_best", {**common, "selection": "source"})
    summary = {
        **common,
        "base_source": base_source["weighted_f1"],
        "base_external": base_external["weighted_f1"],
        "best_source": best_source["weighted_f1"],
        "best_external": best_external["weighted_f1"],
        "delta_source": best_source["weighted_f1"] - base_source["weighted_f1"],
        "delta_external": best_external["weighted_f1"] - base_external["weighted_f1"],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "method": common["method"],
            "args": vars(train_args),
            "info": info,
            "selected_scale_factor": best_factor,
            "source_score": best_score,
        },
        out_dir / "model_best.pt",
    )
    print(
        f"[SAST-SCALE] factor={best_factor:g} source={best_source['weighted_f1']:.4f} "
        f"({summary['delta_source']:+.4f}) external={best_external['weighted_f1']:.4f} "
        f"({summary['delta_external']:+.4f}) out={out_dir}"
    )


if __name__ == "__main__":
    main()
