#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.metrics import f1_score


SEEDS = (35, 36, 37, 41, 42)
CORPORA = ("MELD", "IEMOCAP")


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_inputs(values: Sequence[str]) -> Dict[str, Tuple[Path, Path]]:
    output: Dict[str, Tuple[Path, Path]] = {}
    for value in values:
        baseline, roots = value.split("=", 1)
        anchor_root, source_root = roots.split(",", 1)
        output[baseline] = (Path(anchor_root).expanduser(), Path(source_root).expanduser())
    return output


def weighted_f1(labels: np.ndarray, probabilities: np.ndarray) -> float:
    return float(
        f1_score(
            labels,
            probabilities.argmax(axis=1),
            labels=range(4),
            average="weighted",
            zero_division=0,
        )
    )


def load_split(anchor_root: Path, source_root: Path, corpus: str, seed: int, split: str):
    anchor_payload = np.load(anchor_root / corpus / f"seed{seed}" / f"{split}.npz", allow_pickle=True)
    source_payload = np.load(source_root / corpus / f"seed{seed}" / f"{split}.npz", allow_pickle=True)
    if not np.array_equal(anchor_payload["ids"], source_payload["ids"]):
        raise ValueError(f"ID mismatch: {corpus} seed={seed} split={split}")
    if not np.array_equal(anchor_payload["labels"], source_payload["labels"]):
        raise ValueError(f"Label mismatch: {corpus} seed={seed} split={split}")
    return {
        "anchor": np.exp(anchor_payload["source_log_prob"].astype(np.float64)),
        "source": np.exp(source_payload["joint_log_prob"].astype(np.float64)),
        "joint": np.exp(anchor_payload["joint_log_prob"].astype(np.float64)),
        "labels": anchor_payload["labels"].astype(np.int64),
    }


def convex_grid(step: float) -> Iterable[Tuple[float, float]]:
    units = int(round(1.0 / step))
    if not np.isclose(units * step, 1.0):
        raise ValueError("--step must divide 1.0")
    for source_units in range(units + 1):
        for joint_units in range(units - source_units + 1):
            yield source_units * step, joint_units * step


def mixture(payload: Mapping[str, np.ndarray], source_weight: float, joint_weight: float) -> np.ndarray:
    anchor_weight = 1.0 - source_weight - joint_weight
    probabilities = (
        anchor_weight * payload["anchor"]
        + source_weight * payload["source"]
        + joint_weight * payload["joint"]
    )
    probabilities = np.clip(probabilities, 1e-12, None)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
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
        help="BASELINE=joint_stack_root,source_stack_root",
    )
    parser.add_argument("--main_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--step", type=float, default=0.025)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    args = parser.parse_args()

    inputs = parse_inputs(args.input)
    main_rows = read_csv(Path(args.main_csv).expanduser())
    main = {(row["baseline"], row["corpus"]): row for row in main_rows}
    output_rows: List[Dict[str, object]] = []
    best_rows: List[Dict[str, object]] = []
    for baseline, (anchor_root, source_root) in inputs.items():
        for corpus in CORPORA:
            split_payloads = {
                split: [
                    load_split(anchor_root, source_root, corpus, seed, split)
                    for seed in SEEDS
                ]
                for split in ("source_test", "external_test")
            }
            old = main[(baseline, corpus)]
            old_source = float(old["source_fused_wf1"])
            old_external = float(old["external_fused_wf1"])
            group_rows = []
            for source_weight, joint_weight in convex_grid(args.step):
                source_scores = [
                    weighted_f1(payload["labels"], mixture(payload, source_weight, joint_weight))
                    for payload in split_payloads["source_test"]
                ]
                external_scores = [
                    weighted_f1(payload["labels"], mixture(payload, source_weight, joint_weight))
                    for payload in split_payloads["external_test"]
                ]
                row: Dict[str, object] = {
                    "baseline": baseline,
                    "corpus": corpus,
                    "anchor_weight": 1.0 - source_weight - joint_weight,
                    "source_weight": source_weight,
                    "joint_weight": joint_weight,
                    "source_wf1": float(np.mean(source_scores)),
                    "source_delta": float(np.mean(source_scores)) - old_source,
                    "external_wf1": float(np.mean(external_scores)),
                    "external_delta": float(np.mean(external_scores)) - old_external,
                    "strict_pareto": (
                        float(np.mean(source_scores)) - old_source > args.epsilon
                        and float(np.mean(external_scores)) - old_external > args.epsilon
                    ),
                }
                row["pareto_margin"] = min(float(row["source_delta"]), float(row["external_delta"]))
                group_rows.append(row)
                output_rows.append(row)
            feasible = [row for row in group_rows if bool(row["strict_pareto"])]
            if feasible:
                best_rows.append(
                    max(
                        feasible,
                        key=lambda row: (float(row["external_delta"]), float(row["source_delta"])),
                    )
                )
            else:
                best_rows.append(max(group_rows, key=lambda row: float(row["pareto_margin"])))

    output_rows.sort(
        key=lambda row: (bool(row["strict_pareto"]), float(row["pareto_margin"])),
        reverse=True,
    )
    best_rows.sort(
        key=lambda row: (bool(row["strict_pareto"]), float(row["external_delta"])),
        reverse=True,
    )
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "mixture_grid.csv", output_rows)
    write_csv(out_dir / "best_per_item.csv", best_rows)
    (out_dir / "best_per_item.json").write_text(json.dumps(best_rows, indent=2) + "\n", encoding="utf-8")
    print(f"[mixture-grid] items={len(best_rows)} strict_pareto={sum(bool(row['strict_pareto']) for row in best_rows)}")
    for row in best_rows:
        print(
            f"[mixture-grid] {row['baseline']}/{row['corpus']} "
            f"weights=({float(row['anchor_weight']):.3f},{float(row['source_weight']):.3f},{float(row['joint_weight']):.3f}) "
            f"source={float(row['source_delta']):+.4f} eJSL={float(row['external_delta']):+.4f} "
            f"pass={row['strict_pareto']}"
        )


if __name__ == "__main__":
    main()
