#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


LABELS = ("A", "N", "J", "S")
SEEDS = (35, 36, 37, 41, 42)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_inputs(values: Sequence[str]):
    output = {}
    for value in values:
        baseline, paths = value.split("=", 1)
        anchor_root, source_root, weights_csv = paths.split(",", 2)
        output[baseline] = (
            Path(anchor_root).expanduser(),
            Path(source_root).expanduser(),
            Path(weights_csv).expanduser(),
        )
    return output


def dialogue_id(item_id: str) -> str:
    return str(item_id).rsplit(":utt", 1)[0]


def validation_dialogues(ids: np.ndarray, ratio: float, split_seed: int):
    dialogues = sorted({dialogue_id(item_id) for item_id in ids})
    generator = random.Random(split_seed)
    generator.shuffle(dialogues)
    n_valid = max(1, min(len(dialogues) - 1, int(round(len(dialogues) * ratio))))
    return set(dialogues[:n_valid])


def load_experts(anchor_path: Path, source_path: Path):
    anchor = np.load(anchor_path, allow_pickle=True)
    source = np.load(source_path, allow_pickle=True)
    if not np.array_equal(anchor["ids"], source["ids"]):
        raise ValueError(f"ID mismatch: {anchor_path} vs {source_path}")
    probabilities = np.stack(
        [
            np.exp(anchor["source_log_prob"].astype(np.float64)),
            np.exp(source["joint_log_prob"].astype(np.float64)),
            np.exp(anchor["joint_log_prob"].astype(np.float64)),
        ],
        axis=-1,
    )
    return probabilities, anchor["labels"].astype(np.int64), anchor["ids"].astype(str)


def normalize_weights(raw: str) -> np.ndarray:
    weights = np.asarray(json.loads(raw), dtype=np.float64)
    weights = np.clip(weights, 0, None)
    return weights / np.clip(weights.sum(axis=1, keepdims=True), 1e-12, None)


def fuse(probabilities: np.ndarray, weights: np.ndarray) -> np.ndarray:
    fused = np.einsum("nce,ce->nc", probabilities, weights, optimize=True)
    fused = np.clip(fused, 1e-12, None)
    return fused / fused.sum(axis=1, keepdims=True)


def metrics(labels: np.ndarray, probabilities: np.ndarray) -> Dict[str, object]:
    predictions = probabilities.argmax(axis=1)
    per_class = f1_score(labels, predictions, labels=range(4), average=None, zero_division=0)
    return {
        "n": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, labels=range(4), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predictions, labels=range(4), average="weighted", zero_division=0)),
        "per_class_f1": {label: float(per_class[index]) for index, label in enumerate(LABELS)},
    }


def subset_metrics(labels, fused, anchor, mask):
    return metrics(labels[mask], fused[mask]), metrics(labels[mask], anchor[mask])


def aggregate(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    output = {
        "n_seeds": len(rows),
        "weighted_f1_mean": float(np.mean([row["weighted_f1"] for row in rows])),
        "weighted_f1_std": float(np.std([row["weighted_f1"] for row in rows])),
        "macro_f1_mean": float(np.mean([row["macro_f1"] for row in rows])),
        "accuracy_mean": float(np.mean([row["accuracy"] for row in rows])),
    }
    for label in LABELS:
        output[f"{label}_f1_mean"] = float(np.mean([row["per_class_f1"][label] for row in rows]))
    return output


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields: List[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="BASELINE=joint_stack_root,source_stack_root,heldout_weights_csv",
    )
    parser.add_argument("--main_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--external_valid_ratio", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=2026)
    args = parser.parse_args()

    inputs = parse_inputs(args.input)
    main_rows = read_csv(Path(args.main_csv).expanduser())
    old = {(row["baseline"], row["corpus"]): row for row in main_rows}
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    aggregate_rows = []
    all_seed_results = {}
    for baseline, (anchor_root, source_root, weights_path) in inputs.items():
        for weight_row in read_csv(weights_path):
            corpus = weight_row["corpus"]
            weights = normalize_weights(weight_row["weights_json"])
            split_rows = {name: [] for name in ("source", "external_full", "external_valid", "external_heldout")}
            anchor_rows = {name: [] for name in split_rows}
            for seed in SEEDS:
                seed_key = f"{baseline}/{corpus}/seed{seed}"
                source_prob, source_labels, _ = load_experts(
                    anchor_root / corpus / f"seed{seed}" / "source_test.npz",
                    source_root / corpus / f"seed{seed}" / "source_test.npz",
                )
                source_fused = fuse(source_prob, weights)
                split_rows["source"].append(metrics(source_labels, source_fused))
                anchor_rows["source"].append(metrics(source_labels, source_prob[:, :, 0]))

                external_prob, external_labels, external_ids = load_experts(
                    anchor_root / corpus / f"seed{seed}" / "external_test.npz",
                    source_root / corpus / f"seed{seed}" / "external_test.npz",
                )
                external_fused = fuse(external_prob, weights)
                valid_dialogues = validation_dialogues(
                    external_ids, args.external_valid_ratio, args.split_seed
                )
                valid_mask = np.asarray([dialogue_id(item_id) in valid_dialogues for item_id in external_ids])
                heldout_mask = ~valid_mask
                external_results = {
                    "external_full": (
                        metrics(external_labels, external_fused),
                        metrics(external_labels, external_prob[:, :, 0]),
                    ),
                    "external_valid": subset_metrics(
                        external_labels, external_fused, external_prob[:, :, 0], valid_mask
                    ),
                    "external_heldout": subset_metrics(
                        external_labels, external_fused, external_prob[:, :, 0], heldout_mask
                    ),
                }
                for split, (fused_result, anchor_result) in external_results.items():
                    split_rows[split].append(fused_result)
                    anchor_rows[split].append(anchor_result)
                all_seed_results[seed_key] = {
                    "weights": weights.tolist(),
                    "source": split_rows["source"][-1],
                    **{split: split_rows[split][-1] for split in external_results},
                }

            for split, rows in split_rows.items():
                summary = aggregate(rows)
                anchor_summary = aggregate(anchor_rows[split])
                reference = (
                    float(old[(baseline, corpus)]["source_fused_wf1"])
                    if split == "source"
                    else (
                        float(old[(baseline, corpus)]["external_fused_wf1"])
                        if split == "external_full"
                        else float(anchor_summary["weighted_f1_mean"])
                    )
                )
                aggregate_rows.append(
                    {
                        "baseline": baseline,
                        "corpus": corpus,
                        "split": split,
                        **summary,
                        "anchor_wf1_mean": anchor_summary["weighted_f1_mean"],
                        "reference_wf1": reference,
                        "delta": float(summary["weighted_f1_mean"]) - reference,
                        "weights_json": json.dumps(weights.tolist()),
                    }
                )

    aggregate_rows.sort(key=lambda row: (row["baseline"], row["corpus"], row["split"]))
    write_csv(out_dir / "aggregate_metrics.csv", aggregate_rows)
    (out_dir / "per_seed_metrics.json").write_text(
        json.dumps(all_seed_results, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Classwise Pareto Expert Mixer",
        "",
        "Weights are selected on 20% of eJSL dialogues; held-out metrics use the remaining 80%.",
        "",
        "| Baseline | Source | Split | wF1 | Delta | A | N | J | S |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        lines.append(
            f"| {row['baseline']} | {row['corpus']} | {row['split']} | "
            f"{float(row['weighted_f1_mean']):.4f} | {float(row['delta']):+.4f} | "
            f"{float(row['A_f1_mean']):.4f} | {float(row['N_f1_mean']):.4f} | "
            f"{float(row['J_f1_mean']):.4f} | {float(row['S_f1_mean']):.4f} |"
        )
    (out_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    heldout_rows = [row for row in aggregate_rows if row["split"] == "external_heldout"]
    source_rows = [row for row in aggregate_rows if row["split"] == "source"]
    source_pass = {(row["baseline"], row["corpus"]): float(row["delta"]) > 0 for row in source_rows}
    pass_count = sum(
        float(row["delta"]) > 0 and source_pass[(row["baseline"], row["corpus"])]
        for row in heldout_rows
    )
    print(f"[classwise-results] heldout_strict_pareto={pass_count}/{len(heldout_rows)} out={out_dir}")


if __name__ == "__main__":
    main()
