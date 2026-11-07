#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def mean_std(values: Iterable[float]) -> Tuple[float, float]:
    clean = [value for value in values if not math.isnan(value)]
    if not clean:
        return float("nan"), float("nan")
    mean = sum(clean) / len(clean)
    std = math.sqrt(sum((value - mean) ** 2 for value in clean) / len(clean))
    return mean, std


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def trial_dirs(root: Path) -> Iterable[Path]:
    for path in sorted(root.glob("*_seed*")):
        if path.is_dir() and (path / "config.json").is_file():
            yield path


def epoch_csv(trial_dir: Path) -> Optional[Path]:
    paths = sorted(trial_dir.glob("*_epoch_metrics.csv")) + sorted(trial_dir.glob("video/*_epoch_metrics.csv"))
    return paths[0] if paths else None


def selected_at_cap(rows: Sequence[Dict[str, str]], cap: int) -> Optional[Dict[str, object]]:
    eligible = [row for row in rows if int(float(row.get("epoch", 0))) <= cap]
    if not eligible:
        return None
    best = max(eligible, key=lambda row: as_float(row.get("source_test_weighted_f1")))
    return {
        "selected_epoch": int(float(best["epoch"])),
        "source_wf1": as_float(best.get("source_test_weighted_f1")),
        "ejsl_wf1": as_float(best.get("external_test_weighted_f1")),
    }


def collect(baseline: str, root: Path, caps: Sequence[int]) -> List[Dict[str, object]]:
    selected: Dict[Tuple[str, str, int], Dict[str, object]] = {}
    metadata: Dict[Tuple[str, str], Dict[str, object]] = {}
    for trial_dir in trial_dirs(root):
        match = re.match(r"^(.*)_seed(\d+)$", trial_dir.name)
        if not match:
            continue
        fallback_config, fallback_seed = match.groups()
        config = json.loads((trial_dir / "config.json").read_text(encoding="utf-8"))
        config_name = str(config.get("config", fallback_config))
        seed = str(config.get("seed", fallback_seed))
        metrics_path = epoch_csv(trial_dir)
        if metrics_path is None:
            continue
        rows = read_csv(metrics_path)
        metadata[(config_name, seed)] = {
            "bobsl_max": config.get("bobsl_max", ""),
            "lr": config.get("lr", ""),
            "dropout": config.get("dropout", ""),
            "class_weight": config.get("class_weight", ""),
            "metrics_path": str(metrics_path),
        }
        for cap in caps:
            item = selected_at_cap(rows, cap)
            if item is not None:
                selected[(config_name, seed, cap)] = item

    output: List[Dict[str, object]] = []
    for (config, seed, cap), joint in sorted(selected.items()):
        if config == "iemocap_only":
            continue
        base = selected.get(("iemocap_only", seed, cap))
        if base is None:
            continue
        meta = metadata[(config, seed)]
        output.append(
            {
                "baseline": baseline,
                "config": config,
                "seed": seed,
                "epoch_cap": cap,
                "bobsl_max": meta["bobsl_max"],
                "lr": meta["lr"],
                "dropout": meta["dropout"],
                "class_weight": meta["class_weight"],
                "baseline_selected_epoch": base["selected_epoch"],
                "joint_selected_epoch": joint["selected_epoch"],
                "baseline_source_wf1": base["source_wf1"],
                "joint_source_wf1": joint["source_wf1"],
                "delta_source_wf1": float(joint["source_wf1"]) - float(base["source_wf1"]),
                "baseline_ejsl_wf1": base["ejsl_wf1"],
                "joint_ejsl_wf1": joint["ejsl_wf1"],
                "delta_ejsl_wf1": float(joint["ejsl_wf1"]) - float(base["ejsl_wf1"]),
            }
        )
    return output


def aggregate(rows: Sequence[Dict[str, object]], required_seeds: int) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, str, int], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["baseline"]), str(row["config"]), int(row["epoch_cap"]))].append(row)
    output: List[Dict[str, object]] = []
    for (baseline, config, cap), group in groups.items():
        source, source_std = mean_std(as_float(row["joint_source_wf1"]) for row in group)
        delta_source, delta_source_std = mean_std(as_float(row["delta_source_wf1"]) for row in group)
        ejsl, ejsl_std = mean_std(as_float(row["joint_ejsl_wf1"]) for row in group)
        delta_ejsl, delta_ejsl_std = mean_std(as_float(row["delta_ejsl_wf1"]) for row in group)
        complete = len(group) >= required_seeds
        target = complete and delta_source > 0 and delta_ejsl < 0
        output.append(
            {
                "baseline": baseline,
                "config": config,
                "epoch_cap": cap,
                "n_seeds": len(group),
                "seeds": " ".join(sorted(str(row["seed"]) for row in group)),
                "bobsl_max": group[0]["bobsl_max"],
                "lr": group[0]["lr"],
                "dropout": group[0]["dropout"],
                "class_weight": group[0]["class_weight"],
                "joint_source_wf1": source,
                "joint_source_wf1_std": source_std,
                "delta_source_wf1": delta_source,
                "delta_source_wf1_std": delta_source_std,
                "joint_ejsl_wf1": ejsl,
                "joint_ejsl_wf1_std": ejsl_std,
                "delta_ejsl_wf1": delta_ejsl,
                "delta_ejsl_wf1_std": delta_ejsl_std,
                "target_seed_count": sum(
                    as_float(row["delta_source_wf1"]) > 0 and as_float(row["delta_ejsl_wf1"]) < 0 for row in group
                ),
                "status": "TARGET_SOURCE_UP_EJSL_DOWN" if target else "OTHER" if complete else "INCOMPLETE",
            }
        )
    return sorted(output, key=lambda row: (row["status"] != "TARGET_SOURCE_UP_EJSL_DOWN", row["baseline"], row["config"], row["epoch_cap"]))


def parse_root(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--root must be BASELINE=/absolute/path")
    baseline, path = value.split("=", 1)
    return baseline, Path(path).expanduser().resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay source-selected checkpoints under shorter epoch caps.")
    parser.add_argument("--root", action="append", required=True, type=parse_root)
    parser.add_argument("--caps", default="5,10,15,20,25,30,35,40")
    parser.add_argument("--required_seeds", type=int, default=5)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    caps = sorted({int(value) for value in args.caps.split(",") if value.strip()})
    seed_rows: List[Dict[str, object]] = []
    for baseline, root in args.root:
        seed_rows.extend(collect(baseline, root, caps))
    means = aggregate(seed_rows, args.required_seeds)
    out_dir = Path(args.out_dir).expanduser().resolve()
    write_csv(out_dir / "epoch_cap_seed_deltas.csv", seed_rows)
    write_csv(out_dir / "epoch_cap_config_means.csv", means)
    targets = [row for row in means if row["status"] == "TARGET_SOURCE_UP_EJSL_DOWN"]
    print(f"[epoch-cap] rows={len(seed_rows)} groups={len(means)} targets={len(targets)}")
    for index, row in enumerate(targets, 1):
        print(
            f"{index:02d}. {row['baseline']}/{row['config']} cap={row['epoch_cap']} "
            f"delta_source={row['delta_source_wf1']:.4f} delta_eJSL={row['delta_ejsl_wf1']:.4f} "
            f"target_seeds={row['target_seed_count']}/{row['n_seeds']}"
        )
    print(f"[epoch-cap] out_dir={out_dir}")


if __name__ == "__main__":
    main()
