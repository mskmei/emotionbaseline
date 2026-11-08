#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score


LABELS = ("A", "N", "J", "S")


def read_predictions(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids = [row["item_id"] for row in rows]
    gold = np.asarray([int(row["gold_id"]) for row in rows], dtype=np.int64)
    probability = np.asarray([[float(row[f"prob_{label}"]) for label in LABELS] for row in rows])
    return ids, gold, probability


def aligned_pair(base_path: Path, adapted_path: Path):
    base_ids, gold, base_probability = read_predictions(base_path)
    adapted_ids, adapted_gold, adapted_probability = read_predictions(adapted_path)
    if base_ids != adapted_ids or not np.array_equal(gold, adapted_gold):
        raise ValueError(f"Prediction order mismatch: {base_path} vs {adapted_path}")
    return gold, base_probability, adapted_probability


def weighted_f1(gold, probability):
    return float(f1_score(gold, probability.argmax(axis=1), labels=range(4), average="weighted", zero_division=0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--alpha_step", type=float, default=0.05)
    parser.add_argument("--class_alpha_step", type=float, default=0.25)
    args = parser.parse_args()
    root = Path(args.root).expanduser()
    output = []
    for run_dir in sorted(path.parent for path in root.glob("*/*_seed*/source_test_best_predictions.csv")):
        source_gold, source_base, source_adapted = aligned_pair(
            run_dir / "source_test_base_predictions.csv", run_dir / "source_test_best_predictions.csv"
        )
        external_gold, external_base, external_adapted = aligned_pair(
            run_dir / "external_test_base_predictions.csv", run_dir / "external_test_best_predictions.csv"
        )
        best = None
        for alpha in np.arange(0.0, 1.0 + args.alpha_step / 2, args.alpha_step):
            source_probability = (1.0 - alpha) * source_base + alpha * source_adapted
            source_score = weighted_f1(source_gold, source_probability)
            candidate = (source_score, -alpha, float(alpha))
            if best is None or candidate > best:
                best = candidate
        alpha = best[2]
        source_score = best[0]
        external_score = weighted_f1(external_gold, (1.0 - alpha) * external_base + alpha * external_adapted)
        base_source_score = weighted_f1(source_gold, source_base)
        base_external_score = weighted_f1(external_gold, external_base)
        profile, seed = run_dir.name.rsplit("_seed", 1)
        output.append({
            "corpus": run_dir.parent.name.upper(), "profile": profile, "seed": int(seed), "alpha": alpha,
            "base_source": base_source_score, "source": source_score, "delta_source": source_score - base_source_score,
            "base_external": base_external_score, "external": external_score,
            "delta_external": external_score - base_external_score,
        })
        class_grid = np.arange(0.0, 1.0 + args.class_alpha_step / 2, args.class_alpha_step)
        best_classwise = None
        for alpha_values in itertools.product(class_grid, repeat=len(LABELS)):
            class_alpha = np.asarray(alpha_values, dtype=float)[None, :]
            mixed = (1.0 - class_alpha) * source_base + class_alpha * source_adapted
            mixed = mixed / mixed.sum(axis=1, keepdims=True)
            score = weighted_f1(source_gold, mixed)
            candidate = (score, -float(class_alpha.mean()), tuple(float(x) for x in alpha_values))
            if best_classwise is None or candidate > best_classwise:
                best_classwise = candidate
        alpha_values = np.asarray(best_classwise[2], dtype=float)[None, :]
        external_mixed = (1.0 - alpha_values) * external_base + alpha_values * external_adapted
        external_mixed = external_mixed / external_mixed.sum(axis=1, keepdims=True)
        class_external_score = weighted_f1(external_gold, external_mixed)
        output[-1].update({
            "class_alpha": "/".join(f"{value:.2f}" for value in best_classwise[2]),
            "class_source": best_classwise[0],
            "class_delta_source": best_classwise[0] - base_source_score,
            "class_external": class_external_score,
            "class_delta_external": class_external_score - base_external_score,
        })
    if not output:
        raise RuntimeError(f"No completed runs under {root}")
    fields = list(output[0])
    with (root / "source_selected_probability_mix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)
    for row in output:
        print(
            f"[MIX] {row['corpus']}/{row['profile']} seed={row['seed']} alpha={row['alpha']:.2f} "
            f"delta_source={row['delta_source']:+.4f} delta_eJSL={row['delta_external']:+.4f} "
            f"class_alpha={row['class_alpha']} class_delta={row['class_delta_source']:+.4f}/"
            f"{row['class_delta_external']:+.4f}"
        )


if __name__ == "__main__":
    main()
