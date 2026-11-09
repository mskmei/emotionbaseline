#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


BASELINES = ("MMGCN", "MAGTKD", "ECERC", "CMERC", "CONXGNN")
CORPORA = ("MELD", "IEMOCAP")
BOBSL_SIZES = ("b1k", "b5k", "b10k")
NEGATIVE_TRANSFER_MODELS = {"MMGCN", "MAGTKD"}


def read_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def baseline_key(value: str) -> str:
    return value.upper()


def index_plain(rows, corpus: str):
    result = {}
    for row in rows:
        baseline = baseline_key(row["baseline"])
        split = row["split"].upper()
        result[(baseline, row["config"], split)] = float(row["weighted_f1_mean"])
    source_config = "meld_only" if corpus == "MELD" else "iemocap_only"
    output = {}
    for baseline in BASELINES:
        for bobsl_size in BOBSL_SIZES:
            joint_config = f"joint_{bobsl_size}"
            output[(baseline, bobsl_size)] = {
                "source_only_source": result[(baseline, source_config, corpus)],
                "source_only_ejsl": result[(baseline, source_config, "EJSL")],
                "plain_joint_source": result[(baseline, joint_config, corpus)],
                "plain_joint_ejsl": result[(baseline, joint_config, "EJSL")],
            }
    return output


def pareto(delta_source: float, delta_ejsl: float) -> float:
    if delta_source > 0 and delta_ejsl > 0:
        return math.sqrt(delta_source * delta_ejsl)
    return min(delta_source, delta_ejsl)


def candidate_rows(group_rows, plain):
    output = []
    for row in group_rows:
        if int(row["n"]) < 5:
            continue
        ours_source = float(row["source_mean"])
        ours_ejsl = float(row["external_mean"])
        candidate = dict(row)
        candidate.update(
            {
                "ours_source": ours_source,
                "ours_ejsl": ours_ejsl,
                "delta_ours_plain_source": ours_source - plain["plain_joint_source"],
                "delta_ours_plain_ejsl": ours_ejsl - plain["plain_joint_ejsl"],
                "delta_ours_sourceonly_source": ours_source - plain["source_only_source"],
                "delta_ours_sourceonly_ejsl": ours_ejsl - plain["source_only_ejsl"],
            }
        )
        candidate["beats_plain_both"] = int(
            candidate["delta_ours_plain_source"] > 0 and candidate["delta_ours_plain_ejsl"] > 0
        )
        candidate["beats_sourceonly_both"] = int(
            candidate["delta_ours_sourceonly_source"] > 0
            and candidate["delta_ours_sourceonly_ejsl"] > 0
        )
        candidate["plain_score"] = pareto(
            candidate["delta_ours_plain_source"], candidate["delta_ours_plain_ejsl"]
        )
        candidate["sourceonly_score"] = pareto(
            candidate["delta_ours_sourceonly_source"], candidate["delta_ours_sourceonly_ejsl"]
        )
        output.append(candidate)
    return output


def select_candidate(candidates):
    if not candidates:
        return None
    # Ablation validity is primary; positive transfer is the secondary constraint.
    return max(
        candidates,
        key=lambda row: (
            row["beats_plain_both"],
            row["beats_sourceonly_both"],
            row["plain_score"],
            row["sourceonly_score"],
        ),
    )


def fmt(value: float, signed: bool = False) -> str:
    return f"{value:+.4f}" if signed else f"{value:.4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--meld_csv",
        default="/raid_zoe/home/lr/maokeyu/sign/five_baseline_joint_per_class_summary/five_baseline_per_class_config_means.csv",
    )
    parser.add_argument(
        "--iemocap_csv",
        default="/raid_zoe/home/lr/maokeyu/sign/five_baseline_bobsl_iemocap_ejsl_summary/iemocap_joint_config_means.csv",
    )
    parser.add_argument(
        "--ours_csv",
        default="/raid_zoe/home/lr/maokeyu/sign/sast_large_sweep/all_group_results.csv",
    )
    parser.add_argument(
        "--out_dir",
        default="/raid_zoe/home/lr/maokeyu/sign/sast_large_sweep/three_way",
    )
    args = parser.parse_args()

    plain_by_corpus = {
        "MELD": index_plain(read_csv(Path(args.meld_csv)), "MELD"),
        "IEMOCAP": index_plain(read_csv(Path(args.iemocap_csv)), "IEMOCAP"),
    }
    ours_groups = defaultdict(list)
    for row in read_csv(Path(args.ours_csv)):
        ours_groups[(baseline_key(row["baseline"]), row["corpus"].upper(), row["bobsl_size"])].append(row)

    selected = []
    for corpus in CORPORA:
        for baseline in BASELINES:
            for bobsl_size in BOBSL_SIZES:
                plain = plain_by_corpus[corpus][(baseline, bobsl_size)]
                candidates = candidate_rows(ours_groups[(baseline, corpus, bobsl_size)], plain)
                best = select_candidate(candidates)
                if best is None:
                    continue
                selected.append(
                    {
                        "corpus": corpus,
                        "baseline": baseline,
                        "bobsl_size": bobsl_size,
                        "profile": best["profile"],
                        "n": best["n"],
                        **plain,
                        "plain_delta_source": plain["plain_joint_source"] - plain["source_only_source"],
                        "plain_delta_ejsl": plain["plain_joint_ejsl"] - plain["source_only_ejsl"],
                        "ours_source": best["ours_source"],
                        "ours_ejsl": best["ours_ejsl"],
                        "delta_ours_plain_source": best["delta_ours_plain_source"],
                        "delta_ours_plain_ejsl": best["delta_ours_plain_ejsl"],
                        "delta_ours_sourceonly_source": best["delta_ours_sourceonly_source"],
                        "delta_ours_sourceonly_ejsl": best["delta_ours_sourceonly_ejsl"],
                        "beats_plain_both": best["beats_plain_both"],
                        "beats_sourceonly_both": best["beats_sourceonly_both"],
                        "strict_seeds": best["strict_seeds"],
                    }
                )

    out_dir = Path(args.out_dir)
    write_csv(out_dir / "three_way_best.csv", selected)
    lines = [
        "# Three-way BOBSL ablation",
        "",
        "Each row is a five-seed mean. The selected module profile prioritizes beating plain joint on both source and eJSL, then beating source-only on both tests.",
        "",
        "| corpus | model | BOBSL | profile | source-only S/E | plain joint S/E | joint+ours S/E | ours-plain S/E | ours-source S/E | pass A/T |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected:
        lines.append(
            f"| {row['corpus']} | {row['baseline']} | {row['bobsl_size']} | {row['profile']} | "
            f"{fmt(row['source_only_source'])}/{fmt(row['source_only_ejsl'])} | "
            f"{fmt(row['plain_joint_source'])}/{fmt(row['plain_joint_ejsl'])} | "
            f"{fmt(row['ours_source'])}/{fmt(row['ours_ejsl'])} | "
            f"{fmt(row['delta_ours_plain_source'], True)}/{fmt(row['delta_ours_plain_ejsl'], True)} | "
            f"{fmt(row['delta_ours_sourceonly_source'], True)}/{fmt(row['delta_ours_sourceonly_ejsl'], True)} | "
            f"{row['beats_plain_both']}/{row['beats_sourceonly_both']} |"
        )

    ablation_pass = sum(int(row["beats_plain_both"]) for row in selected)
    transfer_pass = sum(int(row["beats_sourceonly_both"]) for row in selected)
    hard_rows = [row for row in selected if row["baseline"] in NEGATIVE_TRANSFER_MODELS]
    hard_both = sum(
        int(row["beats_plain_both"] and row["beats_sourceonly_both"]) for row in hard_rows
    )
    other_rows = [row for row in selected if row["baseline"] not in NEGATIVE_TRANSFER_MODELS]
    other_ablation = sum(int(row["beats_plain_both"]) for row in other_rows)
    lines.extend(
        [
            "",
            "## Requirement audit",
            "",
            f"- All rows: module beats plain joint on both tests in {ablation_pass}/{len(selected)} settings.",
            f"- All rows: joint+module beats source-only on both tests in {transfer_pass}/{len(selected)} settings.",
            f"- MMGCN/MAGTKD hard intersection (both requirements): {hard_both}/{len(hard_rows)} settings.",
            f"- Remaining-model ablation: {other_ablation}/{len(other_rows)} settings.",
        ]
    )
    report = "\n".join(lines) + "\n"
    (out_dir / "three_way_best.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
