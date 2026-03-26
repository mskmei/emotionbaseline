import argparse
import csv
import hashlib
import pickle
import re
from pathlib import Path

import cv2
import numpy as np


def read_names(path: Path):
    lines = [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not lines:
        return []

    first = lines[0].lower()
    if "stem" in first or "clip" in first or "filename" in first:
        out = []
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                v = row.get("clip_name") or row.get("stem") or row.get("filename")
                if v:
                    out.append(Path(str(v).strip()).stem)
        return out

    return [Path(x).stem for x in lines]


def parse_sample_id(stem: str):
    m = re.match(r"^(SD\d{2})-(\d{2})-(\d{2})([AJNS])$", stem)
    if m is None:
        return None
    sd_id, dial_idx, utt_idx, label = m.groups()
    return sd_id, int(dial_idx), int(utt_idx), label


def map_label(dataset: str, lb: str):
    if dataset.upper() == "IEMOCAP":
        # [ang, exc, fru, hap, neu, sad]
        table = {"A": 0, "J": 3, "N": 4, "S": 5}
        return table[lb]
    # MELD: [neutral, surprise, fear, sadness, joy, disgust, anger]
    table = {"A": 6, "J": 4, "N": 0, "S": 3}
    return table[lb]


def parse_dialogue_txt(txt_file: Path):
    turns = []
    for raw in txt_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < 3:
            continue
        speaker = parts[0].strip()
        text = parts[2].strip()
        turns.append((speaker, text))
    return turns


def hash_text_vec(text: str, dim: int):
    # Deterministic hashed bag-of-char ngram style vector.
    vec = np.zeros(dim, dtype=np.float32)
    if not text:
        return vec
    text = text.strip()
    tokens = list(text)
    for i in range(len(tokens)):
        ngram = "".join(tokens[max(0, i - 1): i + 2])
        h = hashlib.md5(ngram.encode("utf-8")).hexdigest()
        idx = int(h[:8], 16) % dim
        sign = 1.0 if (int(h[8:10], 16) % 2 == 0) else -1.0
        vec[idx] += sign
    n = np.linalg.norm(vec)
    if n > 0:
        vec = vec / n
    return vec


def frame_feature_342(frame_path: Path):
    img = cv2.imread(str(frame_path))
    if img is None:
        return np.zeros(342, dtype=np.float32)
    # 114 bins per channel -> 342 dims total.
    feats = []
    for c in range(3):
        hist = cv2.calcHist([img], [c], None, [114], [0, 256]).flatten()
        s = float(hist.sum())
        if s > 0:
            hist = hist / s
        feats.append(hist.astype(np.float32))
    return np.concatenate(feats, axis=0)


def build_visual_seq(frame_dir: Path, length: int):
    frame_paths = sorted(frame_dir.glob("*.jpg"))
    if not frame_paths:
        frame_paths = sorted(frame_dir.glob("*.png"))
    if not frame_paths:
        return np.zeros((length, 342), dtype=np.float32)

    if len(frame_paths) >= length:
        idx = np.linspace(0, len(frame_paths) - 1, num=length, dtype=int)
        chosen = [frame_paths[i] for i in idx]
    else:
        chosen = frame_paths + [frame_paths[-1]] * (length - len(frame_paths))
    return np.stack([frame_feature_342(p) for p in chosen], axis=0)


def build_single_visual(frame_dir: Path):
    seq = build_visual_seq(frame_dir, 1)
    return seq[0]


def main():
    parser = argparse.ArgumentParser(description="Build MMGCN external DIAL test pkl directly from txt+frame")
    parser.add_argument("--dial_list", type=str, required=True, help="CSV/TXT with DIAL sample names")
    parser.add_argument("--txt_root", type=str, required=True, help="Short_dialogue root")
    parser.add_argument("--frame_root", type=str, required=True, help="eJSL_dial/frame root")
    parser.add_argument("--dataset", type=str, default="IEMOCAP", choices=["IEMOCAP", "MELD"])
    parser.add_argument("--out_pkl", type=str, required=True)
    args = parser.parse_args()

    if args.dataset.upper() == "IEMOCAP":
        d_text, d_audio = 100, 1582
    else:
        d_text, d_audio = 600, 300

    names = read_names(Path(args.dial_list))
    txt_root = Path(args.txt_root)
    frame_root = Path(args.frame_root)

    videoIDs = {}
    videoSpeakers = {}
    videoLabels = {}
    videoText = {}
    videoAudio = {}
    videoVisual = {}
    videoSentence = {}
    testVid = []

    kept, dropped = 0, 0
    for stem in names:
        parsed = parse_sample_id(stem)
        if parsed is None:
            dropped += 1
            continue
        sd_id, dial_idx, utt_idx, lb = parsed
        y = map_label(args.dataset, lb)

        txt_file = txt_root / sd_id / "txt" / f"{sd_id}-Dialogue-{dial_idx:02d}.txt"
        frame_dir = frame_root / stem
        if not txt_file.exists() or not frame_dir.exists():
            dropped += 1
            continue

        turns = parse_dialogue_txt(txt_file)
        if utt_idx <= 0 or utt_idx > len(turns):
            dropped += 1
            continue

        # Keep exactly one labeled timestep per selected sample to align with
        # clip-level evaluation (N should equal number of selected clips).
        turns_ctx = turns[:utt_idx]
        if len(turns_ctx) <= 0:
            dropped += 1
            continue

        # Build one-step sequence using target utterance with context concatenated.
        target_speaker, _target_text = turns_ctx[-1]
        merged_text = ' '.join([t[1] for t in turns_ctx])
        text_seq = hash_text_vec(merged_text, d_text).reshape(1, -1)
        visual_seq = build_single_visual(frame_dir).reshape(1, -1)
        audio_seq = np.zeros((1, d_audio), dtype=np.float32)

        # Speakers for IEMOCAP loader: expects ['M','F',...]. Use deterministic per-dialog mapping.
        spk_raw = [t[0] for t in turns_ctx]
        uniq = []
        for s in spk_raw:
            if s not in uniq:
                uniq.append(s)
        spk_map = {s: ('M' if i % 2 == 0 else 'F') for i, s in enumerate(uniq)}
        spk_seq = [spk_map[target_speaker]]

        videoIDs[stem] = stem
        videoSpeakers[stem] = spk_seq
        videoLabels[stem] = [y]
        videoText[stem] = text_seq
        videoAudio[stem] = audio_seq
        videoVisual[stem] = visual_seq
        videoSentence[stem] = [merged_text]
        testVid.append(stem)
        kept += 1

    # Keep tuple format compatible with IEMOCAP loader implementation.
    payload = (
        videoIDs,
        videoSpeakers,
        videoLabels,
        videoText,
        videoAudio,
        videoVisual,
        videoSentence,
        [],
        testVid,
    )

    out = Path(args.out_pkl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(payload, f)

    print(f"[Build] dataset={args.dataset} total={len(names)} kept={kept} dropped={dropped}")
    print(f"[Build] out={out}")


if __name__ == "__main__":
    main()
