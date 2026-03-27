import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

LABELS = ["A", "N", "J", "S"]
L2I = {k: i for i, k in enumerate(LABELS)}

JP2ANJS = {
    "怒り": "A",
    "平静": "N",
    "喜び": "J",
    "悲しみ": "S",
}


def parse_sample_id(name: str) -> Optional[Tuple[str, int, int, str]]:
    m = re.match(r"^(SD\d{2})-(\d{2})-(\d{2})([AJNS])$", name)
    if m is None:
        return None
    sd_id, d_idx, u_idx, lab = m.groups()
    return sd_id, int(d_idx), int(u_idx), lab


def parse_dialogue_txt(txt_file: Path) -> List[Tuple[str, str, str, str]]:
    turns: List[Tuple[str, str, str, str]] = []
    for raw in txt_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < 4:
            continue
        speaker = parts[0].strip()
        emotion_jp = parts[1].strip()
        utterance = parts[2].strip()
        kana = parts[3].strip()
        turns.append((speaker, emotion_jp, utterance, kana))
    return turns


def collect_frame_sample_ids(frame_root: Path) -> List[str]:
    out: List[str] = []
    for p in sorted(frame_root.iterdir()):
        if not p.is_dir():
            continue
        if parse_sample_id(p.name) is not None:
            out.append(p.name)
    return out


def build_transitions(frame_root: Path, txt_root: Path) -> Tuple[np.ndarray, List[dict]]:
    sample_ids = collect_frame_sample_ids(frame_root)
    matrix = np.zeros((4, 4), dtype=np.int64)
    rows: List[dict] = []

    missing_txt = 0
    first_turn = 0
    unknown_prev = Counter()

    for sid in sample_ids:
        parsed = parse_sample_id(sid)
        if parsed is None:
            continue
        sd_id, d_idx, u_idx, cur_lab = parsed
        txt_file = txt_root / sd_id / "txt" / f"{sd_id}-Dialogue-{d_idx:02d}.txt"
        if not txt_file.exists():
            missing_txt += 1
            continue

        turns = parse_dialogue_txt(txt_file)
        if u_idx <= 1:
            first_turn += 1
            continue
        if u_idx > len(turns):
            continue

        prev_emotion_jp = turns[u_idx - 2][1]
        prev_lab = JP2ANJS.get(prev_emotion_jp, "")
        if prev_lab == "":
            unknown_prev[prev_emotion_jp] += 1
            continue

        i, j = L2I[prev_lab], L2I[cur_lab]
        matrix[i, j] += 1

        rows.append(
            {
                "sample_id": sid,
                "prev_label": prev_lab,
                "curr_label": cur_lab,
                "prev_emotion_jp": prev_emotion_jp,
                "prev_text": turns[u_idx - 2][2],
                "curr_text": turns[u_idx - 1][2],
            }
        )

    print(f"[Data] frame_samples={len(sample_ids)}")
    print(f"[Data] valid_transitions={int(matrix.sum())}")
    print(f"[Data] missing_txt={missing_txt} first_turn_skipped={first_turn}")
    if unknown_prev:
        print(f"[Data] unknown_prev_labels={dict(unknown_prev)}")

    return matrix, rows


def save_transition_table(matrix: np.ndarray, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    total = float(matrix.sum()) if matrix.sum() > 0 else 1.0

    with open(out_dir / "transition_counts.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["prev\\curr"] + LABELS)
        for i, p in enumerate(LABELS):
            w.writerow([p] + [int(x) for x in matrix[i].tolist()])

    with open(out_dir / "transition_probs_global.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["prev", "curr", "count", "prob_global"])
        for i, p in enumerate(LABELS):
            for j, c in enumerate(LABELS):
                cnt = int(matrix[i, j])
                w.writerow([p, c, cnt, cnt / total])


def draw_heatmap(matrix: np.ndarray, out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception as exc:
        print(f"[Warn] skip heatmap (matplotlib/seaborn unavailable): {exc}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    total = int(matrix.sum())
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_probs = np.divide(matrix, np.maximum(row_sums, 1), where=np.ones_like(matrix, dtype=bool))

    sns.set_theme(style="ticks", context="talk")
    plt.rcParams["font.family"] = "DejaVu Sans"

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.8), dpi=260, gridspec_kw={"wspace": 0.25})
    fig.patch.set_alpha(0)

    ax0, ax1 = axes[0], axes[1]
    for ax in axes:
        ax.set_facecolor("none")

    # Panel A: absolute transition counts
    sns.heatmap(
        matrix,
        ax=ax0,
        cmap="crest",
        linewidths=1.1,
        linecolor="#FFFFFF",
        annot=True,
        fmt="d",
        cbar_kws={"label": "Count"},
        square=True,
        xticklabels=LABELS,
        yticklabels=LABELS,
        # annot_kws={"size": 12},
    )
    ax0.set_title("A. Transition Count", fontsize=12.5, weight="bold", pad=10)
    ax0.set_xlabel("Current Emotion", fontsize=16, fontweight="bold")
    ax0.set_ylabel("Previous Emotion", fontsize=16, fontweight="bold")
    ax0.set_title("")
    ax0.tick_params(labelsize=14)
    for lab in ax0.get_xticklabels() + ax0.get_yticklabels():
        lab.set_fontweight("bold")

    # Panel B: row-normalized transition probabilities
    annot_prob = np.empty_like(matrix, dtype=object)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            annot_prob[i, j] = f"{row_probs[i, j]:.1%}"
    sns.heatmap(
        row_probs,
        ax=ax1,
        cmap="YlOrBr",
        linewidths=1.1,
        linecolor="#FFFFFF",
        annot=annot_prob,
        fmt="",
        cbar_kws={"label": "P(curr | prev)", "format": "%.0f%%"},
        square=True,
        xticklabels=LABELS,
        yticklabels=LABELS,
        vmin=0.0,
        vmax=max(0.4, float(row_probs.max())),
        # annot_kws={"size": 12},
    )
    ax1.set_title("B. Conditional Probability", fontsize=12.5, weight="bold", pad=10)
    ax1.set_xlabel("Current Emotion", fontsize=16, fontweight="bold")
    ax1.set_ylabel("Previous Emotion", fontsize=16, fontweight="bold")
    ax1.set_title("")
    ax1.tick_params(labelsize=14)
    for lab in ax1.get_xticklabels() + ax1.get_yticklabels():
        lab.set_fontweight("bold")

    # No title per user request.
    fig.text(0.5, 0.015, "A=Anger, N=Neutral, J=Joy, S=Sadness", ha="center", fontsize=15, fontweight="bold", color="#3D3D3D")

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(out_dir / "transition_heatmap.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.savefig(out_dir / "transition_heatmap.pdf", bbox_inches="tight", facecolor=fig.get_facecolor(), transparent=True)
    plt.close(fig)


def draw_sankey(matrix: np.ndarray, out_dir: Path) -> None:
    try:
        import plotly.graph_objects as go
    except Exception as exc:
        print(f"[Warn] skip sankey (plotly unavailable): {exc}")
        return

    def _hex_to_rgba(hex_color: str, alpha: float) -> str:
        c = hex_color.strip().lstrip("#")
        r = int(c[0:2], 16)
        g = int(c[2:4], 16)
        b = int(c[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    out_dir.mkdir(parents=True, exist_ok=True)
    total = int(matrix.sum())
    left_nodes = [f"Prev {x}" for x in LABELS]
    right_nodes = [f"Curr {x}" for x in LABELS]
    labels = left_nodes + right_nodes
    base_colors = ["#2A9D8F", "#457B9D", "#E9C46A", "#E76F51"]

    sources: List[int] = []
    targets: List[int] = []
    values: List[int] = []
    link_colors: List[str] = []
    customdata: List[Tuple[str, str, float]] = []
    for i in range(4):
        for j in range(4):
            c = int(matrix[i, j])
            if c <= 0:
                continue
            sources.append(i)
            targets.append(4 + j)
            values.append(c)
            link_colors.append(_hex_to_rgba(base_colors[i], 0.42))
            customdata.append((LABELS[i], LABELS[j], c / max(total, 1)))

    x_pos = [0.03, 0.03, 0.03, 0.03, 0.97, 0.97, 0.97, 0.97]
    y_pos = [0.08, 0.33, 0.58, 0.83, 0.08, 0.33, 0.58, 0.83]

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="fixed",
                valueformat="d",
                node=dict(
                    pad=16,
                    thickness=20,
                    line=dict(color="rgba(38,38,38,0.35)", width=0.8),
                    label=labels,
                    color=base_colors + base_colors,
                    x=x_pos,
                    y=y_pos,
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color=link_colors,
                    customdata=customdata,
                    hovertemplate=(
                        "<b>%{customdata[0]} -> %{customdata[1]}</b><br>"
                        "Count: %{value:d}<br>"
                        "Global share: %{customdata[2]:.2%}<extra></extra>"
                    ),
                ),
            )
        ]
    )
    fig.update_layout(
        title=None,
        font=dict(size=18, family="Arial", color="#1F2430"),
        width=1080,
        height=680,
        margin=dict(l=24, r=24, t=24, b=24),
        paper_bgcolor="#F7F8FA",
        plot_bgcolor="#F7F8FA",
    )

    html_path = out_dir / "transition_sankey.html"
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    try:
        fig.write_image(str(out_dir / "transition_sankey.png"), scale=2)
    except Exception:
        pass
    print(f"[Save] sankey html: {html_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Build eJSL transition stats and polished plots from frame-selected samples")
    p.add_argument("--frame_root", type=str, required=True)
    p.add_argument("--txt_root", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="./analysis/outputs_transition")
    p.add_argument("--expected_samples", type=int, default=1920)
    p.add_argument("--strict", action="store_true", help="Fail when frame sample count != expected_samples")
    args = p.parse_args()

    frame_root = Path(args.frame_root)
    txt_root = Path(args.txt_root)
    out_dir = Path(args.out_dir)

    sample_count = len(collect_frame_sample_ids(frame_root))
    print(f"[Check] frame sample count={sample_count}")
    if args.expected_samples > 0 and sample_count != args.expected_samples:
        msg = f"frame sample count mismatch: expected={args.expected_samples}, got={sample_count}"
        if args.strict:
            raise RuntimeError(msg)
        print(f"[Warn] {msg}")

    matrix, rows = build_transitions(frame_root, txt_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_transition_table(matrix, out_dir)
    with open(out_dir / "transition_rows.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    draw_heatmap(matrix, out_dir)
    draw_sankey(matrix, out_dir)

    print("[Done] transition analysis complete")
    print(f"[Done] outputs: {out_dir}")


if __name__ == "__main__":
    main()
