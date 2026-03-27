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
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_probs = np.divide(matrix, np.maximum(row_sums, 1), where=np.ones_like(matrix, dtype=bool))

    annot = np.empty_like(matrix, dtype=object)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            annot[i, j] = f"{int(matrix[i, j])}\\n{row_probs[i, j]:.2%}"

    plt.figure(figsize=(8.8, 7.2), dpi=220)
    sns.set_theme(style="whitegrid")
    ax = sns.heatmap(
        row_probs,
        cmap="YlGnBu",
        linewidths=0.8,
        linecolor="#FFFFFF",
        annot=annot,
        fmt="",
        cbar_kws={"label": "P(curr | prev)"},
        square=True,
        xticklabels=LABELS,
        yticklabels=LABELS,
        vmin=0.0,
        vmax=max(0.4, float(row_probs.max())),
    )
    ax.set_title("eJSL Emotion Transition Matrix", pad=14, fontsize=14, weight="bold")
    ax.set_xlabel("Current Utterance Emotion", fontsize=12)
    ax.set_ylabel("Previous Utterance Emotion", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_dir / "transition_heatmap.png", bbox_inches="tight")
    plt.close()


def draw_sankey(matrix: np.ndarray, out_dir: Path) -> None:
    try:
        import plotly.graph_objects as go
    except Exception as exc:
        print(f"[Warn] skip sankey (plotly unavailable): {exc}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    left_nodes = [f"Prev {x}" for x in LABELS]
    right_nodes = [f"Curr {x}" for x in LABELS]
    labels = left_nodes + right_nodes

    sources: List[int] = []
    targets: List[int] = []
    values: List[int] = []
    for i in range(4):
        for j in range(4):
            c = int(matrix[i, j])
            if c <= 0:
                continue
            sources.append(i)
            targets.append(4 + j)
            values.append(c)

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                valueformat="d",
                node=dict(
                    pad=16,
                    thickness=18,
                    line=dict(color="rgba(30,30,30,0.35)", width=0.6),
                    label=labels,
                    color=["#2A9D8F", "#457B9D", "#E9C46A", "#E76F51"] * 2,
                ),
                link=dict(source=sources, target=targets, value=values),
            )
        ]
    )
    fig.update_layout(
        title_text="eJSL Emotion Flow (Prev -> Curr)",
        font=dict(size=13),
        width=980,
        height=620,
        margin=dict(l=16, r=16, t=56, b=16),
    )

    html_path = out_dir / "transition_sankey.html"
    fig.write_html(str(html_path), include_plotlyjs="cdn")
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
