#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


LABELS = ("A", "N", "J", "S")
LABEL_IDS = {label: index for index, label in enumerate(LABELS)}


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields: List[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values: Iterable[float]) -> Tuple[float, float]:
    values = list(values)
    if not values:
        return float("nan"), float("nan")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(variance)


def percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def exact_sign_flip_pvalue(deltas: Sequence[float]) -> float:
    observed = abs(sum(deltas) / len(deltas))
    null_values = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(deltas)):
        null_values.append(abs(sum(sign * value for sign, value in zip(signs, deltas)) / len(deltas)))
    return sum(value >= observed - 1e-15 for value in null_values) / len(null_values)


def bootstrap_ci(deltas: Sequence[float], samples: int, seed: int) -> Tuple[float, float]:
    generator = random.Random(seed)
    means = []
    for _ in range(samples):
        draw = [generator.choice(deltas) for _ in deltas]
        means.append(sum(draw) / len(draw))
    return percentile(means, 0.025), percentile(means, 0.975)


def find_one(rows: Sequence[Mapping[str, str]], **criteria: str) -> Mapping[str, str]:
    matches = [row for row in rows if all(row.get(key) == value for key, value in criteria.items())]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one row for {criteria}, found {len(matches)}")
    return matches[0]


def build_baseline_rows(meld_csv: Path, iemocap_csv: Path) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    specs = (
        ("MELD", read_csv(meld_csv), "meld_only"),
        ("IEMOCAP", read_csv(iemocap_csv), "iemocap_only"),
    )
    for corpus, rows, source_config in specs:
        baselines = sorted({row["baseline"] for row in rows})
        for baseline in baselines:
            selected = {}
            for condition, config in (("source_only", source_config), ("joint_b5k", "joint_b5k")):
                selected[(condition, "source")] = find_one(
                    rows, baseline=baseline, config=config, split=corpus
                )
                selected[(condition, "ejsl")] = find_one(
                    rows, baseline=baseline, config=config, split="eJSL"
                )
            source_only_source = float(selected[("source_only", "source")]["weighted_f1_mean"])
            source_only_ejsl = float(selected[("source_only", "ejsl")]["weighted_f1_mean"])
            joint_source = float(selected[("joint_b5k", "source")]["weighted_f1_mean"])
            joint_ejsl = float(selected[("joint_b5k", "ejsl")]["weighted_f1_mean"])
            output.append(
                {
                    "corpus": corpus,
                    "model": baseline,
                    "source_only_n": int(selected[("source_only", "source")]["n_seeds"]),
                    "source_only_source_wf1": source_only_source,
                    "source_only_source_std": float(
                        selected[("source_only", "source")]["weighted_f1_std"]
                    ),
                    "source_only_ejsl_wf1": source_only_ejsl,
                    "source_only_ejsl_std": float(selected[("source_only", "ejsl")]["weighted_f1_std"]),
                    "joint_n": int(selected[("joint_b5k", "source")]["n_seeds"]),
                    "joint_source_wf1": joint_source,
                    "joint_source_std": float(selected[("joint_b5k", "source")]["weighted_f1_std"]),
                    "joint_ejsl_wf1": joint_ejsl,
                    "joint_ejsl_std": float(selected[("joint_b5k", "ejsl")]["weighted_f1_std"]),
                    "delta_source_wf1": joint_source - source_only_source,
                    "delta_ejsl_wf1": joint_ejsl - source_only_ejsl,
                }
            )
    return output


def build_proposed_rows(all_runs_csv: Path) -> List[Dict[str, object]]:
    rows = read_csv(all_runs_csv)
    output: List[Dict[str, object]] = []
    for corpus in ("MELD", "IEMOCAP"):
        source_rows = [
            row
            for row in rows
            if row["corpus"] == corpus and row["condition"] == "source_only" and row["profile"] == "transfer"
        ]
        joint_rows = [
            row
            for row in rows
            if row["corpus"] == corpus and row["condition"] == "joint_b5k" and row["profile"] == "residual-mid"
        ]
        if len(source_rows) != 5 or len(joint_rows) != 5:
            raise RuntimeError(f"Expected five source/joint proposed runs for {corpus}")
        metrics: Dict[str, Tuple[float, float]] = {}
        for prefix, group in (("source_only", source_rows), ("joint", joint_rows)):
            for split, field in (("source", "source_fused_wf1"), ("ejsl", "ejsl_fused_wf1")):
                metrics[f"{prefix}_{split}"] = mean_std(float(row[field]) for row in group)
        output.append(
            {
                "corpus": corpus,
                "model": "Ours-ResidualExpert",
                "source_only_n": len(source_rows),
                "source_only_source_wf1": metrics["source_only_source"][0],
                "source_only_source_std": metrics["source_only_source"][1],
                "source_only_ejsl_wf1": metrics["source_only_ejsl"][0],
                "source_only_ejsl_std": metrics["source_only_ejsl"][1],
                "joint_n": len(joint_rows),
                "joint_source_wf1": metrics["joint_source"][0],
                "joint_source_std": metrics["joint_source"][1],
                "joint_ejsl_wf1": metrics["joint_ejsl"][0],
                "joint_ejsl_std": metrics["joint_ejsl"][1],
                "delta_source_wf1": metrics["joint_source"][0] - metrics["source_only_source"][0],
                "delta_ejsl_wf1": metrics["joint_ejsl"][0] - metrics["source_only_ejsl"][0],
            }
        )
    return output


def classification_metrics(path: Path) -> Dict[str, float]:
    rows = read_csv(path)
    gold = [int(row["gold_id"]) for row in rows]
    pred = [int(row["fused_pred"]) for row in rows]
    metrics: Dict[str, float] = {"n": float(len(rows))}
    supports = []
    f1_scores = []
    for label, label_id in LABEL_IDS.items():
        true_positive = sum(g == label_id and p == label_id for g, p in zip(gold, pred))
        false_positive = sum(g != label_id and p == label_id for g, p in zip(gold, pred))
        false_negative = sum(g == label_id and p != label_id for g, p in zip(gold, pred))
        support = sum(g == label_id for g in gold)
        denominator = 2 * true_positive + false_positive + false_negative
        f1 = 2 * true_positive / denominator if denominator else 0.0
        metrics[f"{label}_f1"] = f1
        metrics[f"{label}_support"] = float(support)
        f1_scores.append(f1)
        supports.append(support)
    metrics["accuracy"] = sum(g == p for g, p in zip(gold, pred)) / len(gold)
    metrics["macro_f1"] = sum(f1_scores) / len(f1_scores)
    metrics["weighted_f1"] = sum(score * support for score, support in zip(f1_scores, supports)) / sum(supports)
    return metrics


def proposed_per_class(root: Path) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    seed_rows: List[Dict[str, object]] = []
    profiles = {"source_only": "transfer", "joint_b5k": "residual-mid"}
    for corpus in ("MELD", "IEMOCAP"):
        for condition, profile in profiles.items():
            for split, filename in (
                (corpus, "source_test_best_predictions.csv"),
                ("eJSL", "external_test_best_predictions.csv"),
            ):
                paths = sorted((root / corpus / condition).glob(f"{profile}_seed*/{filename}"))
                if len(paths) != 5:
                    raise RuntimeError(f"Expected five prediction files, found {len(paths)}: {paths}")
                for path in paths:
                    seed = int(path.parent.name.rsplit("_seed", 1)[1])
                    seed_rows.append(
                        {
                            "corpus": corpus,
                            "condition": condition,
                            "split": split,
                            "seed": seed,
                            **classification_metrics(path),
                        }
                    )
    mean_rows: List[Dict[str, object]] = []
    groups: Dict[Tuple[str, str, str], List[Mapping[str, object]]] = defaultdict(list)
    for row in seed_rows:
        groups[(str(row["corpus"]), str(row["condition"]), str(row["split"]))].append(row)
    metric_names = ("accuracy", "macro_f1", "weighted_f1") + tuple(f"{label}_f1" for label in LABELS)
    for (corpus, condition, split), group in sorted(groups.items()):
        result: Dict[str, object] = {
            "corpus": corpus,
            "condition": condition,
            "split": split,
            "n_seeds": len(group),
        }
        for metric in metric_names:
            mean, std = mean_std(float(row[metric]) for row in group)
            result[f"{metric}_mean"] = mean
            result[f"{metric}_std"] = std
        mean_rows.append(result)
    return seed_rows, mean_rows


def paired_statistics(paired_csv: Path, bootstrap_samples: int) -> List[Dict[str, object]]:
    rows = read_csv(paired_csv)
    output: List[Dict[str, object]] = []
    for corpus in ("MELD", "IEMOCAP"):
        group = [row for row in rows if row["corpus"] == corpus and row["profile"] == "residual-mid"]
        for split, field in (("source", "delta_source_wf1"), ("eJSL", "delta_ejsl_wf1")):
            deltas = [float(row[field]) for row in sorted(group, key=lambda row: int(row["seed"]))]
            mean, std = mean_std(deltas)
            ci_low, ci_high = bootstrap_ci(deltas, bootstrap_samples, seed=1701 + len(output))
            output.append(
                {
                    "corpus": corpus,
                    "split": split,
                    "n_seeds": len(deltas),
                    "mean_delta_wf1": mean,
                    "std_delta_wf1": std,
                    "positive_seeds": sum(value > 0 for value in deltas),
                    "negative_seeds": sum(value < 0 for value in deltas),
                    "bootstrap_95ci_low": ci_low,
                    "bootstrap_95ci_high": ci_high,
                    "exact_two_sided_sign_flip_p": exact_sign_flip_pvalue(deltas),
                    "seed_deltas": " ".join(f"{value:+.6f}" for value in deltas),
                }
            )
    return output


def collect_anchor_rows(root: Path) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    for source_path in sorted(root.glob("*/joint_b5k/anchor-*_seed*/source_test_best_summary.json")):
        external_path = source_path.with_name("external_test_best_summary.json")
        source = json.loads(source_path.read_text(encoding="utf-8"))
        external = json.loads(external_path.read_text(encoding="utf-8"))
        profile, seed = source_path.parent.name.rsplit("_seed", 1)
        output.append(
            {
                "corpus": source_path.parts[-4],
                "profile": profile,
                "seed": int(seed),
                "source_wf1": source["fused"]["weighted_f1"],
                "ejsl_wf1": external["fused"]["weighted_f1"],
                "source_gate_bobsl": source["mean_gate"]["bobsl_local"],
                "ejsl_gate_bobsl": external["mean_gate"]["bobsl_local"],
            }
        )
    return output


def fmt(value: object, signed: bool = False) -> str:
    return f"{float(value):+0.4f}" if signed else f"{float(value):0.4f}"


def build_report(
    main_rows: Sequence[Mapping[str, object]],
    per_class_rows: Sequence[Mapping[str, object]],
    stats_rows: Sequence[Mapping[str, object]],
    anchor_rows: Sequence[Mapping[str, object]],
    tv_rows: Sequence[Mapping[str, str]],
    prediction_rows: Sequence[Mapping[str, str]],
    geometry_rows: Sequence[Mapping[str, str]],
    prototype_rows: Sequence[Mapping[str, str]],
    context_rows: Sequence[Mapping[str, str]],
) -> str:
    lines = [
        "# BOBSL joint-transfer study",
        "",
        "## Protocol",
        "",
        "All main comparisons use the aligned `joint_b5k` condition and five seeds (35, 36, 37, 41, 42). "
        "Models are trained without eJSL labels; eJSL is an external transfer test. MELD and IEMOCAP keep "
        "their established model-specific optimization settings, while the BOBSL sample budget is aligned.",
        "",
        "## Main visual results",
        "",
        "| source | model | source-only source/eJSL | joint-b5k source/eJSL | delta source/eJSL | n |",
        "|---|---|---|---|---|---:|",
    ]
    for row in sorted(main_rows, key=lambda item: (str(item["corpus"]), str(item["model"]))):
        lines.append(
            f"| {row['corpus']} | {row['model']} | {fmt(row['source_only_source_wf1'])}/{fmt(row['source_only_ejsl_wf1'])} | "
            f"{fmt(row['joint_source_wf1'])}/{fmt(row['joint_ejsl_wf1'])} | "
            f"{fmt(row['delta_source_wf1'], True)}/{fmt(row['delta_ejsl_wf1'], True)} | {row['joint_n']} |"
        )
    lines.extend(["", "Among the compared `joint_b5k` methods:"])
    for corpus in ("MELD", "IEMOCAP"):
        group = [row for row in main_rows if row["corpus"] == corpus]
        best = max(group, key=lambda row: float(row["joint_source_wf1"]))
        proposed = find_one(group, model="Ours-ResidualExpert")
        rank = 1 + sum(float(row["joint_source_wf1"]) > float(proposed["joint_source_wf1"]) for row in group)
        lines.append(
            f"- {corpus}: our source wF1 is {fmt(proposed['joint_source_wf1'])} (rank {rank}/{len(group)}); "
            f"the best compared value is {fmt(best['joint_source_wf1'])} from {best['model']}."
        )

    lines.extend(
        [
            "",
            "## Proposed model",
            "",
            "The model freezes the source-trained CMERC context expert and source-local visual expert, then "
            "learns a separate BOBSL visual expert as a bounded residual. A reliability gate controls only the "
            "residual contribution, preserving the high-performing source decision function instead of forcing "
            "raw domain alignment. Profile selection and checkpoint selection use source performance only.",
            "",
            "### Per-class results",
            "",
            "| source | condition | test | wF1 | A | N | J | S |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in per_class_rows:
        lines.append(
            f"| {row['corpus']} | {row['condition']} | {row['split']} | {fmt(row['weighted_f1_mean'])} | "
            + " | ".join(fmt(row[f"{label}_f1_mean"]) for label in LABELS)
            + " |"
        )

    lines.extend(
        [
            "",
            "### Paired uncertainty",
            "",
            "| source | test | mean delta +/- SD | positive seeds | bootstrap 95% CI | exact sign-flip p |",
            "|---|---|---:|---:|---|---:|",
        ]
    )
    for row in stats_rows:
        lines.append(
            f"| {row['corpus']} | {row['split']} | {fmt(row['mean_delta_wf1'], True)} +/- {fmt(row['std_delta_wf1'])} | "
            f"{row['positive_seeds']}/{row['n_seeds']} | [{fmt(row['bootstrap_95ci_low'], True)}, "
            f"{fmt(row['bootstrap_95ci_high'], True)}] | {float(row['exact_two_sided_sign_flip_p']):.4f} |"
        )
    lines.append(
        "With five pairs, the exact two-sided test has coarse resolution (minimum attainable p=0.0625), so the "
        "result should be described through effect sizes and seed consistency rather than significance claims."
    )

    domain_probe = [
        float(row["class_balanced_domain_probe_accuracy"])
        for row in geometry_rows
        if row["label"] == "ALL" and row["class_balanced_domain_probe_accuracy"]
    ]
    bobsl_transfer = {
        row["test_domain"]: float(row["weighted_f1"])
        for row in prototype_rows
        if row["train_domain"] == "BOBSL_train"
    }
    mmgcn_shift = find_one(prediction_rows, corpus="MELD", baseline="MMGCN", split="eJSL")
    magtkd_shift = find_one(prediction_rows, corpus="IEMOCAP", baseline="MAGTKD", split="eJSL")
    lines.extend(
        [
            "",
            "## Mechanism evidence",
            "",
            "1. **Domain-separable features.** Class-balanced linear domain probes reach "
            f"{min(domain_probe):.4f}-{max(domain_probe):.4f} accuracy. A BOBSL nearest-prototype classifier "
            f"transfers at only {bobsl_transfer['MELD_test']:.4f} wF1 to MELD, "
            f"{bobsl_transfer['IEMOCAP_test']:.4f} to IEMOCAP, and {bobsl_transfer['eJSL_test']:.4f} to eJSL. "
            "The same extractor therefore does not make emotion geometry domain invariant.",
            "2. **Decision-prior shift.** BOBSL joint training changes predictions substantially. For example, "
            f"MMGCN on MELD->eJSL changes wF1 by {float(mmgcn_shift['delta_weighted_f1_mean']):+.4f}; "
            f"MAGTKD on IEMOCAP->eJSL changes it by {float(magtkd_shift['delta_weighted_f1_mean']):+.4f}. "
            "This opposite response under the same BOBSL budget motivates model-dependent transfer groups.",
            "3. **Context is causal, not uniformly helpful.** Isolating utterances separates local transfer from "
            "conversation amplification. The largest absolute context interactions include:",
        ]
    )
    ranked_context = sorted(
        context_rows,
        key=lambda row: abs(float(row["context_interaction_weighted_f1_mean"])),
        reverse=True,
    )[:6]
    for row in ranked_context:
        lines.append(
            f"   - {row['baseline']} {row['corpus']}->{row['split']}: normal joint gain "
            f"{float(row['joint_gain_normal_weighted_f1_mean']):+.4f}, isolated gain "
            f"{float(row['joint_gain_isolate_weighted_f1_mean']):+.4f}, interaction "
            f"{float(row['context_interaction_weighted_f1_mean']):+.4f}."
        )
    lines.extend(
        [
            "",
            "Together these experiments support two mechanisms: local class-prior/decision-boundary transfer, "
            "and architecture-dependent context amplification. This explains why the five baselines separate "
            "into negative-transfer and positive/neutral-transfer behavior without requiring one universal group "
            "assignment across both source corpora.",
            "",
            "## Non-oracle text extension",
            "",
            "The text is produced by the selected non-oracle JSL translator, translated with the existing cached "
            "pipeline, and never replaced by oracle eJSL text.",
            "",
            "| text dropout | condition | MELD wF1 | eJSL wF1 | delta MELD/eJSL |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in tv_rows:
        lines.append(
            f"| {row['profile']} | {row['condition']} | {float(row['source_wf1']):.4f} | "
            f"{float(row['ejsl_wf1']):.4f} | {float(row['delta_source_wf1']):+.4f}/"
            f"{float(row['delta_ejsl_wf1']):+.4f} |"
        )

    lines.extend(
        [
            "",
            "## Negative ablation: forced alignment",
            "",
            "Adding source classification and source-center alignment directly to the BOBSL expert did not improve "
            "the source criterion used for method selection:",
            "",
            "| source | profile | source wF1 | eJSL wF1 | BOBSL gate source/eJSL |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in anchor_rows:
        lines.append(
            f"| {row['corpus']} | {row['profile']} | {fmt(row['source_wf1'])} | {fmt(row['ejsl_wf1'])} | "
            f"{float(row['source_gate_bobsl']):.3f}/{float(row['ejsl_gate_bobsl']):.3f} |"
        )
    lines.extend(
        [
            "",
            "This negative ablation favors preserving a strong source expert and learning bounded residual transfer "
            "over imposing feature-center alignment between visibly separable domains.",
            "",
            "## Claims and limitations",
            "",
            "- The proposed method is the best source-domain `joint_b5k` result among the five reproduced baselines "
            "under this protocol; this is not a claim against untested methods or tuning budgets.",
            "- The visual main study has five paired seeds. The non-oracle TV extension reuses the available seed-42 "
            "CMERC checkpoints and is therefore supporting evidence, not a five-seed main result.",
            "- The inherited training scripts select checkpoints on the split named `source`; the paper must state "
            "the exact validation/test role of that split and avoid presenting selection-set scores as untouched test results.",
            "- eJSL labels are used only for final reporting and mechanism analysis, never for model/profile selection.",
            "- The domain-probe result demonstrates feature separability, not by itself a causal explanation; the "
            "prediction-shift and context interventions provide the complementary behavioral and causal evidence.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the paper-level BOBSL transfer report.")
    parser.add_argument(
        "--meld_csv",
        default="/raid_zoe/home/lr/maokeyu/sign/five_baseline_joint_per_class_summary/five_baseline_per_class_config_means.csv",
    )
    parser.add_argument(
        "--iemocap_csv",
        default="/raid_zoe/home/lr/maokeyu/sign/five_baseline_bobsl_iemocap_ejsl_summary/iemocap_joint_config_means.csv",
    )
    parser.add_argument("--proposed_root", default="/raid_zoe/home/lr/maokeyu/sign/domain_expert_cmerc/final")
    parser.add_argument("--anchor_root", default="/raid_zoe/home/lr/maokeyu/sign/domain_expert_cmerc/anchor_screen")
    parser.add_argument(
        "--tv_csv", default="/raid_zoe/home/lr/maokeyu/sign/domain_expert_cmerc/tv_nonoracle/runs/summary.csv"
    )
    parser.add_argument(
        "--prediction_csv",
        default="/raid_zoe/home/lr/maokeyu/sign/bobsl_b5k_mechanism_analysis/prediction_shift/mean_overall.csv",
    )
    parser.add_argument(
        "--geometry_csv",
        default="/raid_zoe/home/lr/maokeyu/sign/bobsl_b5k_mechanism_analysis/visual_domain_geometry/pairwise_geometry.csv",
    )
    parser.add_argument(
        "--prototype_csv",
        default="/raid_zoe/home/lr/maokeyu/sign/bobsl_b5k_mechanism_analysis/visual_domain_geometry/prototype_transfer.csv",
    )
    parser.add_argument(
        "--context_csv",
        default="/raid_zoe/home/lr/maokeyu/sign/bobsl_b5k_mechanism_analysis/context_ablation/mean_interactions.csv",
    )
    parser.add_argument(
        "--out_dir", default="/raid_zoe/home/lr/maokeyu/sign/domain_expert_cmerc/paper_summary"
    )
    parser.add_argument("--bootstrap_samples", type=int, default=20000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    proposed_root = Path(args.proposed_root).expanduser().resolve()
    input_paths = {
        "meld_baselines": Path(args.meld_csv).expanduser().resolve(),
        "iemocap_baselines": Path(args.iemocap_csv).expanduser().resolve(),
        "proposed_runs": proposed_root / "all_runs.csv",
        "proposed_pairs": proposed_root / "paired_deltas.csv",
        "anchor_root": Path(args.anchor_root).expanduser().resolve(),
        "nonoracle_tv": Path(args.tv_csv).expanduser().resolve(),
        "prediction_shift": Path(args.prediction_csv).expanduser().resolve(),
        "domain_geometry": Path(args.geometry_csv).expanduser().resolve(),
        "prototype_transfer": Path(args.prototype_csv).expanduser().resolve(),
        "context_interaction": Path(args.context_csv).expanduser().resolve(),
    }
    for name, path in input_paths.items():
        if name != "anchor_root" and not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")
        if name == "anchor_root" and not path.is_dir():
            raise FileNotFoundError(f"Missing {name}: {path}")

    main_rows = build_baseline_rows(input_paths["meld_baselines"], input_paths["iemocap_baselines"])
    main_rows.extend(build_proposed_rows(input_paths["proposed_runs"]))
    seed_class_rows, mean_class_rows = proposed_per_class(proposed_root)
    stats_rows = paired_statistics(input_paths["proposed_pairs"], args.bootstrap_samples)
    anchor_rows = collect_anchor_rows(input_paths["anchor_root"])
    tv_rows = read_csv(input_paths["nonoracle_tv"])
    prediction_rows = read_csv(input_paths["prediction_shift"])
    geometry_rows = read_csv(input_paths["domain_geometry"])
    prototype_rows = read_csv(input_paths["prototype_transfer"])
    context_rows = read_csv(input_paths["context_interaction"])

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "main_visual_b5k.csv", main_rows)
    write_csv(out_dir / "proposed_per_class_by_seed.csv", seed_class_rows)
    write_csv(out_dir / "proposed_per_class_means.csv", mean_class_rows)
    write_csv(out_dir / "proposed_paired_statistics.csv", stats_rows)
    write_csv(out_dir / "forced_alignment_ablation.csv", anchor_rows)
    report = build_report(
        main_rows,
        mean_class_rows,
        stats_rows,
        anchor_rows,
        tv_rows,
        prediction_rows,
        geometry_rows,
        prototype_rows,
        context_rows,
    )
    (out_dir / "paper_report.md").write_text(report, encoding="utf-8")
    artifacts = {
        "inputs": {name: str(path) for name, path in input_paths.items()},
        "outputs": {
            "main_visual_b5k": str(out_dir / "main_visual_b5k.csv"),
            "proposed_per_class_by_seed": str(out_dir / "proposed_per_class_by_seed.csv"),
            "proposed_per_class_means": str(out_dir / "proposed_per_class_means.csv"),
            "proposed_paired_statistics": str(out_dir / "proposed_paired_statistics.csv"),
            "forced_alignment_ablation": str(out_dir / "forced_alignment_ablation.csv"),
            "paper_report": str(out_dir / "paper_report.md"),
        },
        "main_config": {
            "bobsl_samples": 5000,
            "seeds": [35, 36, 37, 41, 42],
            "label_order": list(LABELS),
            "bootstrap_samples": args.bootstrap_samples,
            "proposed_source_profile": "transfer",
            "proposed_joint_profile": "residual-mid",
        },
    }
    (out_dir / "artifacts.json").write_text(json.dumps(artifacts, indent=2) + "\n", encoding="utf-8")
    print(report, end="")
    print(f"[paper-report] out_dir={out_dir}")


if __name__ == "__main__":
    main()
