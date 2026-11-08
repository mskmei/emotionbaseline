#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence


def read_csv(path: str | Path) -> List[Dict[str, str]]:
    with Path(path).expanduser().open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_inputs(values: Sequence[str]) -> Dict[str, Path]:
    output: Dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected BASELINE=CSV, got {value!r}")
        baseline, filename = value.split("=", 1)
        output[baseline] = Path(filename).expanduser()
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected_main_csv", required=True)
    parser.add_argument("--per_seed", action="append", required=True, help="BASELINE=per_seed_metrics.csv")
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    selected = read_csv(args.selected_main_csv)
    per_seed_inputs = parse_inputs(args.per_seed)
    output: List[Dict[str, object]] = []
    for main in selected:
        baseline = main["baseline"]
        rows = read_csv(per_seed_inputs[baseline])
        for seed in (35, 36, 37, 41, 42):
            matches = [
                row
                for row in rows
                if row["corpus"] == main["corpus"]
                and row["profile"] == main["profile"]
                and int(row["seed"]) == seed
            ]
            source = next(row for row in matches if row["split"] == "source")
            external = next(row for row in matches if row["split"] == "external")
            output.append(
                {
                    "baseline": baseline,
                    "corpus": main["corpus"],
                    "seed": seed,
                    "main_profile": main["profile"],
                    "source_floor": source["fused_weighted_f1"],
                    "external_floor": external["fused_weighted_f1"],
                }
            )

    out_path = Path(args.out_csv).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(f"[pareto-thresholds] wrote {len(output)} rows to {out_path}")


if __name__ == "__main__":
    main()
