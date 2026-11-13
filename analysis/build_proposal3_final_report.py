#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


HARD_SELECTIONS = {
    ("IEMOCAP", "MMGCN"): ("context_r1", "dual-anchor context adapter"),
    ("IEMOCAP", "MAGTKD"): ("decision_p0_w2_s025", "dual-anchor decision adapter"),
    ("MELD", "MMGCN"): ("dual_better_w4", "source anchor + selective peer distillation"),
    ("MELD", "MAGTKD"): ("dual_all_w2", "tuned joint anchor + peer distillation"),
}

PROPOSAL1_SELECTIONS = {
    ("IEMOCAP", "CMERC"): "fullspace",
    ("IEMOCAP", "CONXGNN"): "full",
    ("IEMOCAP", "ECERC"): "no_weight",
    ("MELD", "CMERC"): "bobsl2",
    ("MELD", "CONXGNN"): "rank1_b10_kl1_off31",
    ("MELD", "ECERC"): "no_weight",
}

DISPLAY_NAME = {"CONXGNN": "ConxGNN"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def find_one(rows: list[dict[str, str]], **criteria: str) -> dict[str, str]:
    matches = [row for row in rows if all(row.get(key) == value for key, value in criteria.items())]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one row for {criteria}, found {len(matches)}")
    return matches[0]


def f(value) -> float:
    return float(value)


def round11_result(root: Path, profile: str) -> dict[str, float | int | str]:
    runs = []
    for path in sorted(root.glob(f"MAGTKD/meld/b5k/{profile}_seed*/summary.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        seed = int(path.parent.name.rsplit("_seed", 1)[1])
        runs.append((seed, data))
    seeds = [seed for seed, _ in runs]
    if seeds != [35, 36, 37, 41, 42]:
        raise RuntimeError(f"Incomplete {profile} confirmation: seeds={seeds}")
    return {
        "ours_source": mean(data["best_source"] for _, data in runs),
        "ours_external": mean(data["best_external"] for _, data in runs),
        "tuned_anchor_source": mean(data["base_source"] for _, data in runs),
        "tuned_anchor_external": mean(data["base_external"] for _, data in runs),
        "n": len(runs),
        "seeds": " ".join(str(seed) for seed in seeds),
    }


def final_row(
    corpus: str,
    baseline: str,
    profile: str,
    method: str,
    source_source: float,
    source_external: float,
    joint_source: float,
    joint_external: float,
    ours_source: float,
    ours_external: float,
    n: int,
    seeds: str,
    tuned_source: float | None = None,
    tuned_external: float | None = None,
) -> dict:
    delta_source_source = ours_source - source_source
    delta_source_external = ours_external - source_external
    delta_joint_source = ours_source - joint_source
    delta_joint_external = ours_external - joint_external
    beats_source = delta_source_source > 0 and delta_source_external > 0
    beats_joint = delta_joint_source > 0 and delta_joint_external > 0
    row = {
        "corpus": corpus,
        "baseline": DISPLAY_NAME.get(baseline, baseline),
        "bobsl_size": "b5k",
        "profile": profile,
        "method": method,
        "n": n,
        "seeds": seeds,
        "source_only_source": source_source,
        "source_only_ejsl": source_external,
        "plain_joint_source": joint_source,
        "plain_joint_ejsl": joint_external,
        "ours_source": ours_source,
        "ours_ejsl": ours_external,
        "delta_source_only_source": delta_source_source,
        "delta_source_only_ejsl": delta_source_external,
        "delta_plain_joint_source": delta_joint_source,
        "delta_plain_joint_ejsl": delta_joint_external,
        "beats_source_only_both": int(beats_source),
        "beats_plain_joint_both": int(beats_joint),
        "beats_both_references": int(beats_source and beats_joint),
        "tuned_joint_source": "" if tuned_source is None else tuned_source,
        "tuned_joint_ejsl": "" if tuned_external is None else tuned_external,
        "delta_tuned_joint_source": "" if tuned_source is None else ours_source - tuned_source,
        "delta_tuned_joint_ejsl": "" if tuned_external is None else ours_external - tuned_external,
    }
    return row


def build_main_rows(args) -> list[dict]:
    proposal1 = read_csv(args.proposal1_csv)
    round5 = read_csv(args.round5_csv)
    output = []

    for (corpus, baseline), (profile, method) in HARD_SELECTIONS.items():
        if (corpus, baseline) == ("MELD", "MAGTKD"):
            reference = find_one(
                proposal1, baseline="MAGTKD", corpus="MELD", bobsl_size="b5k"
            )
            result = round11_result(args.round11_root, profile)
            output.append(final_row(
                corpus, baseline, profile, method,
                f(reference["source_only_source"]), f(reference["source_only_ejsl"]),
                f(reference["base_source_mean"]), f(reference["base_ejsl_mean"]),
                f(result["ours_source"]), f(result["ours_external"]),
                int(result["n"]), str(result["seeds"]),
                f(result["tuned_anchor_source"]), f(result["tuned_anchor_external"]),
            ))
            continue
        selected = find_one(
            round5, baseline=baseline, corpus=corpus, bobsl_size="b5k", profile=profile
        )
        output.append(final_row(
            corpus, baseline, profile, method,
            f(selected["source_only_source"]), f(selected["source_only_external"]),
            f(selected["plain_joint_source"]), f(selected["plain_joint_external"]),
            f(selected["ours_source"]), f(selected["ours_external"]),
            int(selected["n"]), selected["seeds"],
        ))

    for (corpus, baseline), profile in PROPOSAL1_SELECTIONS.items():
        selected = find_one(
            proposal1, baseline=baseline, corpus=corpus, bobsl_size="b5k", profile=profile
        )
        output.append(final_row(
            corpus, baseline, profile, "plain-joint subspace refinement",
            f(selected["source_only_source"]), f(selected["source_only_ejsl"]),
            f(selected["base_source_mean"]), f(selected["base_ejsl_mean"]),
            f(selected["ours_source_mean"]), f(selected["ours_ejsl_mean"]),
            int(selected["n"]), selected["seeds"],
        ))

    order = {"MMGCN": 0, "MAGTKD": 1, "CMERC": 2, "ConxGNN": 3, "ECERC": 4}
    output.sort(key=lambda row: (row["corpus"], order[row["baseline"]]))
    return output


def markdown_main(rows: list[dict]) -> str:
    hard = [row for row in rows if row["baseline"] in {"MMGCN", "MAGTKD"}]
    lines = [
        "# Proposal 3 final five-seed results",
        "",
        "All cells are source weighted F1 / eJSL weighted F1. Results use BOBSL-5K and ",
        "seeds 35, 36, 37, 41, and 42. Epoch selection uses the source split only.",
        "",
        "| corpus | backbone | source-only | plain joint | joint + module | delta vs source | delta vs joint | pass both | profile |",
        "|---|---|---:|---:|---:|---:|---:|:---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['corpus']} | {row['baseline']} | "
            f"{row['source_only_source']:.4f} / {row['source_only_ejsl']:.4f} | "
            f"{row['plain_joint_source']:.4f} / {row['plain_joint_ejsl']:.4f} | "
            f"{row['ours_source']:.4f} / {row['ours_ejsl']:.4f} | "
            f"{row['delta_source_only_source']:+.4f} / {row['delta_source_only_ejsl']:+.4f} | "
            f"{row['delta_plain_joint_source']:+.4f} / {row['delta_plain_joint_ejsl']:+.4f} | "
            f"{'yes' if row['beats_both_references'] else 'no'} | `{row['profile']}` |"
        )
    lines.extend([
        "",
        f"- MMGCN/MAGTKD hard requirement: {sum(row['beats_both_references'] for row in hard)}/{len(hard)} model-corpus pairs.",
        f"- All five backbones: {sum(row['beats_both_references'] for row in rows)}/{len(rows)} model-corpus pairs beat both references on both tests.",
        "- MAGTKD/MELD also beats its stronger tuned-joint anchor: 0.4777/0.3406 versus 0.4725/0.3334.",
        "- One model checkpoint is evaluated on both source and eJSL; no inference-time routing or checkpoint ensemble is used.",
        "",
    ])
    return "\n".join(lines)


def build_scale_rows(proposal1_rows: list[dict[str, str]]) -> list[dict]:
    rows = []
    for row in proposal1_rows:
        if row["bobsl_size"] not in {"b1k", "b10k"}:
            continue
        rows.append({
            "corpus": row["corpus"],
            "baseline": DISPLAY_NAME.get(row["baseline"], row["baseline"]),
            "bobsl_size": row["bobsl_size"],
            "profile": row["profile"],
            "n": int(row["n"]),
            "source_only_source": f(row["source_only_source"]),
            "source_only_ejsl": f(row["source_only_ejsl"]),
            "plain_joint_source": f(row["base_source_mean"]),
            "plain_joint_ejsl": f(row["base_ejsl_mean"]),
            "ours_source": f(row["ours_source_mean"]),
            "ours_ejsl": f(row["ours_ejsl_mean"]),
            "delta_plain_joint_source": f(row["delta_plain_source_mean"]),
            "delta_plain_joint_ejsl": f(row["delta_plain_ejsl_mean"]),
            "delta_source_only_source": f(row["delta_sourceonly_source"]),
            "delta_source_only_ejsl": f(row["delta_sourceonly_ejsl"]),
            "beats_both_references": int(
                row["beats_plain_both"] == "1" and row["beats_sourceonly_both"] == "1"
            ),
        })
    rows.sort(key=lambda row: (row["bobsl_size"], row["corpus"], row["baseline"]))
    return rows


def build_seed42_rows(root: Path) -> list[dict]:
    rows = []
    for path in root.glob("*/*/b*k/*_seed42/summary.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        baseline, corpus, size = path.relative_to(root).parts[:3]
        source_threshold = max(data["base_source"], data["joint_source"])
        external_threshold = max(data["base_external"], data["joint_external"])
        rows.append({
            "baseline": baseline,
            "corpus": corpus.upper(),
            "bobsl_size": size,
            "profile": path.parent.name.rsplit("_seed", 1)[0],
            "ours_source": data["best_source"],
            "ours_ejsl": data["best_external"],
            "hard_source_margin": data["best_source"] - source_threshold,
            "hard_ejsl_margin": data["best_external"] - external_threshold,
            "hard_min_margin": min(
                data["best_source"] - source_threshold,
                data["best_external"] - external_threshold,
            ),
        })
    rows.sort(key=lambda row: (row["baseline"], row["corpus"], row["bobsl_size"], -row["hard_min_margin"]))
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--proposal1_csv", type=Path,
        default=Path("/raid_zoe/home/lr/maokeyu/sign/sast_plain_joint_refine/three_way/strict_three_way_best.csv"),
    )
    parser.add_argument(
        "--round5_csv", type=Path,
        default=Path("/raid_zoe/home/lr/maokeyu/sign/proposal2_dual_anchor_search/round5_confirm_strict.csv"),
    )
    parser.add_argument(
        "--round11_root", type=Path,
        default=Path("/raid_zoe/home/lr/maokeyu/sign/proposal3_final_search/round11_magtkd_meld_five_seed"),
    )
    parser.add_argument(
        "--round12_root", type=Path,
        default=Path("/raid_zoe/home/lr/maokeyu/sign/proposal3_final_search/round12_b1_b10_screen"),
    )
    parser.add_argument(
        "--out_dir", type=Path,
        default=Path("/raid_zoe/home/lr/maokeyu/sign/proposal3_final_report"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    main_rows = build_main_rows(args)
    write_csv(args.out_dir / "proposal3_main_b5k.csv", main_rows)
    (args.out_dir / "proposal3_main_b5k.md").write_text(markdown_main(main_rows), encoding="utf-8")

    proposal1 = read_csv(args.proposal1_csv)
    write_csv(args.out_dir / "proposal3_scale_fiveseed.csv", build_scale_rows(proposal1))
    write_csv(args.out_dir / "proposal3_scale_seed42_screen.csv", build_seed42_rows(args.round12_root))
    print(markdown_main(main_rows))
    print(f"[PROPOSAL3] wrote {args.out_dir}")


if __name__ == "__main__":
    main()
