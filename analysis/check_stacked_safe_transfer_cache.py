#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    root = Path(args.root).expanduser()
    with Path(args.thresholds).expanduser().open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    failures = []
    for row in rows:
        path = root / row["baseline"] / row["corpus"] / f"seed{row['seed']}" / "source_test.npz"
        payload = np.load(path, allow_pickle=True)
        pred = payload["source_log_prob"].argmax(axis=1)
        score = f1_score(payload["labels"], pred, labels=range(4), average="weighted", zero_division=0)
        error = abs(score - float(row["source_floor"]))
        if error > args.tolerance:
            failures.append((path, score, float(row["source_floor"]), error))
    if failures:
        for path, score, expected, error in failures[:10]:
            print(f"[stack-check] mismatch path={path} actual={score:.8f} expected={expected:.8f} error={error:.3g}")
        raise RuntimeError(f"Stacked cache mismatches: {len(failures)}/{len(rows)}")
    print(f"[stack-check] all {len(rows)} source anchors match within {args.tolerance:g}")


if __name__ == "__main__":
    main()
