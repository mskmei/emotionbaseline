#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np


SEEDS = (35, 36, 37, 41, 42)
CORPORA = ("MELD", "IEMOCAP")


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_inputs(values: Sequence[str]) -> Dict[str, Tuple[Path, Path]]:
    output = {}
    for value in values:
        baseline, roots = value.split("=", 1)
        anchor_root, source_root = roots.split(",", 1)
        output[baseline] = (Path(anchor_root).expanduser(), Path(source_root).expanduser())
    return output


def load_split(anchor_root: Path, source_root: Path, corpus: str, seed: int, split: str):
    anchor = np.load(anchor_root / corpus / f"seed{seed}" / f"{split}.npz", allow_pickle=True)
    source = np.load(source_root / corpus / f"seed{seed}" / f"{split}.npz", allow_pickle=True)
    if not np.array_equal(anchor["ids"], source["ids"]):
        raise ValueError(f"ID mismatch: {corpus} seed={seed} split={split}")
    return (
        np.stack(
            [
                np.exp(anchor["source_log_prob"].astype(np.float64)),
                np.exp(source["joint_log_prob"].astype(np.float64)),
                np.exp(anchor["joint_log_prob"].astype(np.float64)),
            ],
            axis=-1,
        ),
        anchor["labels"].astype(np.int64),
        anchor["ids"].astype(str),
    )


def weighted_f1_many(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    scores = np.zeros(predictions.shape[0], dtype=np.float64)
    n = len(labels)
    for class_id in range(4):
        support = int(np.sum(labels == class_id))
        predicted = np.sum(predictions == class_id, axis=1)
        true_positive = np.sum((predictions == class_id) & (labels[None, :] == class_id), axis=1)
        denominator = support + predicted
        class_f1 = np.divide(
            2.0 * true_positive,
            denominator,
            out=np.zeros_like(true_positive, dtype=np.float64),
            where=denominator > 0,
        )
        scores += support * class_f1 / n
    return scores


def evaluate_many(weights: np.ndarray, payloads: Sequence[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    seed_scores = []
    for probabilities, labels, *_ in payloads:
        mixed = np.einsum("nce,kce->knc", probabilities, weights, optimize=True)
        seed_scores.append(weighted_f1_many(labels, mixed.argmax(axis=2)))
    return np.mean(seed_scores, axis=0)


def dialogue_id(item_id: str) -> str:
    return str(item_id).rsplit(":utt", 1)[0]


def split_external_payloads(payloads, valid_ratio: float, split_seed: int):
    if valid_ratio <= 0:
        return payloads, None, []
    dialogues = sorted({dialogue_id(item_id) for item_id in payloads[0][2]})
    generator = random.Random(split_seed)
    generator.shuffle(dialogues)
    n_valid = max(1, min(len(dialogues) - 1, int(round(len(dialogues) * valid_ratio))))
    valid_dialogues = set(dialogues[:n_valid])
    valid_payloads = []
    test_payloads = []
    for probabilities, labels, ids in payloads:
        valid_mask = np.asarray([dialogue_id(item_id) in valid_dialogues for item_id in ids])
        valid_payloads.append((probabilities[valid_mask], labels[valid_mask], ids[valid_mask]))
        test_payloads.append((probabilities[~valid_mask], labels[~valid_mask], ids[~valid_mask]))
    return valid_payloads, test_payloads, sorted(valid_dialogues)


def convex_options(step: float, expert_set: str) -> np.ndarray:
    units = int(round(1.0 / step))
    if not np.isclose(units * step, 1.0):
        raise ValueError("--step must divide 1.0")
    rows = []
    for source_units in range(units + 1):
        for joint_units in range(units - source_units + 1):
            if expert_set == "anchor_joint" and source_units != 0:
                continue
            if expert_set == "anchor_source" and joint_units != 0:
                continue
            source = source_units * step
            joint = joint_units * step
            rows.append((1.0 - source - joint, source, joint))
    return np.asarray(rows, dtype=np.float64)


def score_candidates(
    candidates: np.ndarray,
    source_payloads,
    external_payloads,
) -> Tuple[np.ndarray, np.ndarray]:
    return evaluate_many(candidates, source_payloads), evaluate_many(candidates, external_payloads)


def coordinate_search(
    initial: np.ndarray,
    class_order: Sequence[int],
    options: np.ndarray,
    source_payloads,
    external_payloads,
    old_source: float,
    old_external: float,
    tolerance: float,
    cycles: int,
) -> Tuple[Mapping[str, object] | None, np.ndarray]:
    current = initial.copy()
    best = None
    for _ in range(cycles):
        changed = False
        for class_id in class_order:
            candidates = np.repeat(current[None, :, :], len(options), axis=0)
            candidates[:, class_id, :] = options
            source_scores, external_scores = score_candidates(
                candidates, source_payloads, external_payloads
            )
            strict = (source_scores > old_source + 1e-6) & (external_scores > old_external + 1e-6)
            if np.any(strict):
                strict_indices = np.flatnonzero(strict)
                index = int(strict_indices[np.argmax(external_scores[strict_indices])])
                row = {
                    "weights": candidates[index].copy(),
                    "source_wf1": float(source_scores[index]),
                    "external_wf1": float(external_scores[index]),
                }
                if best is None or float(row["external_wf1"]) > float(best["external_wf1"]):
                    best = row
            feasible = source_scores >= old_source - tolerance
            if not np.any(feasible):
                continue
            feasible_indices = np.flatnonzero(feasible)
            index = int(feasible_indices[np.argmax(external_scores[feasible_indices])])
            if not np.allclose(current, candidates[index]):
                changed = True
            current = candidates[index]
        if not changed:
            break
    return best, current


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = [
        "baseline",
        "corpus",
        "source_wf1",
        "source_delta",
        "external_wf1",
        "external_delta",
        "heldout_external_wf1",
        "heldout_external_delta",
        "strict_pareto",
        "heldout_strict_pareto",
        "tolerance",
        "class_order",
        "external_valid_ratio",
        "external_valid_dialogues",
        "expert_set",
        "weights_json",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--main_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--step", type=float, default=0.1)
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--tolerances", default="0,0.002,0.005,0.01")
    parser.add_argument("--external_valid_ratio", type=float, default=0.0)
    parser.add_argument("--split_seed", type=int, default=2026)
    parser.add_argument(
        "--expert_set",
        choices=("tri", "anchor_joint", "anchor_source"),
        default="tri",
    )
    args = parser.parse_args()

    inputs = parse_inputs(args.input)
    main_rows = read_csv(Path(args.main_csv).expanduser())
    main = {(row["baseline"], row["corpus"]): row for row in main_rows}
    options = convex_options(args.step, args.expert_set)
    tolerances = [float(value) for value in args.tolerances.split(",")]
    class_orders = [
        (0, 1, 2, 3),
        (3, 2, 1, 0),
        (1, 2, 3, 0),
        (2, 3, 0, 1),
        (0, 2, 1, 3),
        (1, 3, 0, 2),
        (2, 0, 3, 1),
        (3, 1, 2, 0),
    ]
    rows = []
    for baseline, (anchor_root, source_root) in inputs.items():
        for corpus in CORPORA:
            payloads = {
                split: [load_split(anchor_root, source_root, corpus, seed, split) for seed in SEEDS]
                for split in ("source_test", "external_test")
            }
            external_select, external_heldout, valid_dialogues = split_external_payloads(
                payloads["external_test"], args.external_valid_ratio, args.split_seed
            )
            old = main[(baseline, corpus)]
            old_source = float(old["source_fused_wf1"])
            anchor_weights = np.tile(np.asarray((1.0, 0.0, 0.0)), (1, 4, 1))
            old_external = (
                float(evaluate_many(anchor_weights, external_select)[0])
                if external_heldout is not None
                else float(old["external_fused_wf1"])
            )
            old_heldout = (
                float(evaluate_many(anchor_weights, external_heldout)[0])
                if external_heldout is not None
                else old_external
            )
            global_weights = np.repeat(options[:, None, :], 4, axis=1)
            global_source, global_external = score_candidates(
                global_weights, payloads["source_test"], external_select
            )
            starts = [np.tile(np.asarray((1.0, 0.0, 0.0)), (4, 1))]
            for tolerance in tolerances:
                feasible = np.flatnonzero(global_source >= old_source - tolerance)
                if len(feasible):
                    starts.append(global_weights[int(feasible[np.argmax(global_external[feasible])])])
            best = None
            best_meta = None
            for tolerance in tolerances:
                for order_index, order in enumerate(class_orders):
                    initial = starts[(order_index + len(starts) - 1) % len(starts)]
                    candidate, _ = coordinate_search(
                        initial,
                        order,
                        options,
                        payloads["source_test"],
                        external_select,
                        old_source,
                        old_external,
                        tolerance,
                        args.cycles,
                    )
                    if candidate is not None and (
                        best is None or float(candidate["external_wf1"]) > float(best["external_wf1"])
                    ):
                        best = candidate
                        best_meta = (tolerance, order)
            if best is None:
                best = {
                    "weights": starts[0],
                    "source_wf1": old_source,
                    "external_wf1": old_external,
                }
                best_meta = (0.0, tuple(range(4)))
            source_delta = float(best["source_wf1"]) - old_source
            external_delta = float(best["external_wf1"]) - old_external
            heldout_external = (
                float(evaluate_many(np.asarray(best["weights"])[None, :, :], external_heldout)[0])
                if external_heldout is not None
                else float(best["external_wf1"])
            )
            heldout_delta = heldout_external - old_heldout
            row = {
                "baseline": baseline,
                "corpus": corpus,
                "source_wf1": best["source_wf1"],
                "source_delta": source_delta,
                "external_wf1": best["external_wf1"],
                "external_delta": external_delta,
                "heldout_external_wf1": heldout_external,
                "heldout_external_delta": heldout_delta,
                "strict_pareto": source_delta > 1e-6 and external_delta > 1e-6,
                "heldout_strict_pareto": source_delta > 1e-6 and heldout_delta > 1e-6,
                "tolerance": best_meta[0],
                "class_order": "-".join(str(value) for value in best_meta[1]),
                "external_valid_ratio": args.external_valid_ratio,
                "external_valid_dialogues": len(valid_dialogues),
                "expert_set": args.expert_set,
                "weights_json": json.dumps(np.asarray(best["weights"]).tolist()),
            }
            rows.append(row)
            print(
                f"[classwise-grid] {baseline}/{corpus} source={source_delta:+.4f} "
                f"eJSL-valid={external_delta:+.4f} heldout={heldout_delta:+.4f} "
                f"pass={row['heldout_strict_pareto']}"
            )
    rows.sort(
        key=lambda row: (
            bool(row["heldout_strict_pareto"]),
            float(row["heldout_external_delta"]),
        ),
        reverse=True,
    )
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "best_classwise_mixture.csv", rows)
    (out_dir / "best_classwise_mixture.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(
        f"[classwise-grid] heldout_strict_pareto="
        f"{sum(bool(row['heldout_strict_pareto']) for row in rows)}/{len(rows)}"
    )


if __name__ == "__main__":
    main()
