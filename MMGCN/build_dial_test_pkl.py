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


def extract_audio_feature_from_mp4(mp4_path: Path, dim: int):
    """Directly extract a fixed-size audio feature from mp4 (no intermediate npy)."""
    try:
        import librosa
    except Exception:
        return np.zeros(dim, dtype=np.float32), "librosa_missing"

    if not mp4_path.exists():
        return np.zeros(dim, dtype=np.float32), "mp4_missing"

    try:
        y, _ = librosa.load(str(mp4_path), sr=16000, mono=True)
    except Exception:
        return np.zeros(dim, dtype=np.float32), "decode_fail"

    if y.size == 0:
        return np.zeros(dim, dtype=np.float32), "empty_audio"

    # Keep bounded runtime.
    y = y[: 16000 * 20]
    spec = np.abs(np.fft.rfft(y, n=4096)).astype(np.float32)
    feat = np.log1p(spec)
    if feat.shape[0] >= dim:
        feat = feat[:dim]
    else:
        feat = np.pad(feat, (0, dim - feat.shape[0]), mode="constant")

    n = np.linalg.norm(feat)
    if n > 0:
        feat = feat / n
    return feat.astype(np.float32), "ok"


def _flatten_feature_dict(feat_dict):
    rows = []
    for _k, v in feat_dict.items():
        arr = np.asarray(v, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim == 2 and arr.shape[0] > 0:
            rows.append(arr)
    if not rows:
        return None
    return np.concatenate(rows, axis=0)


def load_source_stats(source_pkl: Path):
    obj = pickle.load(open(source_pkl, 'rb'), encoding='latin1')
    if not (isinstance(obj, tuple) or isinstance(obj, list)):
        raise RuntimeError(f"Unsupported source pkl format: {source_pkl}")

    text_dict = obj[3]
    audio_dict = obj[4]
    visual_dict = obj[5]

    t_all = _flatten_feature_dict(text_dict)
    a_all = _flatten_feature_dict(audio_dict)
    v_all = _flatten_feature_dict(visual_dict)
    if t_all is None or a_all is None or v_all is None:
        raise RuntimeError(f"Failed to build source stats from: {source_pkl}")

    eps = 1e-6
    return {
        'text_mean': t_all.mean(axis=0),
        'text_std': np.maximum(t_all.std(axis=0), eps),
        'audio_mean': a_all.mean(axis=0),
        'audio_std': np.maximum(a_all.std(axis=0), eps),
        'visual_mean': v_all.mean(axis=0),
        'visual_std': np.maximum(v_all.std(axis=0), eps),
    }


def match_stats(x, x_mean, x_std, src_mean, src_std):
    z = (x - x_mean) / np.maximum(x_std, 1e-6)
    return z * src_std + src_mean


def main():
    parser = argparse.ArgumentParser(description="Build MMGCN external DIAL test pkl directly from txt+frame")
    parser.add_argument("--dial_list", type=str, required=True, help="CSV/TXT with DIAL sample names")
    parser.add_argument("--txt_root", type=str, required=True, help="Short_dialogue root")
    parser.add_argument("--frame_root", type=str, required=True, help="eJSL_dial/frame root")
    parser.add_argument("--mp4_root", type=str, required=True, help="eJSL_dial/video root")
    parser.add_argument("--dataset", type=str, default="IEMOCAP", choices=["IEMOCAP", "MELD"])
    parser.add_argument("--source_pkl", type=str, default="", help="Source feature pkl for mean/std alignment")
    parser.add_argument("--disable_source_align", action="store_true", help="Disable source-domain stats alignment")
    parser.add_argument("--out_pkl", type=str, required=True)
    args = parser.parse_args()

    if args.dataset.upper() == "IEMOCAP":
        d_text, d_audio = 100, 1582
    else:
        d_text, d_audio = 600, 300

    names = read_names(Path(args.dial_list))
    txt_root = Path(args.txt_root)
    frame_root = Path(args.frame_root)
    mp4_root = Path(args.mp4_root)

    if args.source_pkl:
        source_pkl = Path(args.source_pkl)
    else:
        source_pkl = Path('./IEMOCAP_features/IEMOCAP_features.pkl') if args.dataset.upper() == 'IEMOCAP' else Path('./MELD_features/MELD_features_raw1.pkl')
    source_stats = None
    if not args.disable_source_align:
        source_stats = load_source_stats(source_pkl)

    videoIDs = {}
    videoSpeakers = {}
    videoLabels = {}
    videoText = {}
    videoAudio = {}
    videoVisual = {}
    videoSentence = {}
    testVid = []

    kept, dropped = 0, 0
    audio_stats = {
        "ok": 0,
        "mp4_missing": 0,
        "decode_fail": 0,
        "empty_audio": 0,
        "librosa_missing": 0,
    }
    raw_text_bank, raw_audio_bank, raw_visual_bank = [], [], []
    for stem in names:
        parsed = parse_sample_id(stem)
        if parsed is None:
            dropped += 1
            continue
        sd_id, dial_idx, utt_idx, lb = parsed
        y = map_label(args.dataset, lb)

        txt_file = txt_root / sd_id / "txt" / f"{sd_id}-Dialogue-{dial_idx:02d}.txt"
        frame_dir = frame_root / stem
        mp4_file = mp4_root / f"{stem}.mp4"
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
        audio_vec, audio_status = extract_audio_feature_from_mp4(mp4_file, d_audio)
        if audio_status not in audio_stats:
            audio_stats[audio_status] = 0
        audio_stats[audio_status] += 1
        audio_seq = audio_vec.reshape(1, -1)

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

        raw_text_bank.append(text_seq[0])
        raw_audio_bank.append(audio_seq[0])
        raw_visual_bank.append(visual_seq[0])

    # Optional source-domain mean/std alignment.
    if source_stats is not None and kept > 0:
        t_bank = np.stack(raw_text_bank, axis=0)
        a_bank = np.stack(raw_audio_bank, axis=0)
        v_bank = np.stack(raw_visual_bank, axis=0)
        t_mean, t_std = t_bank.mean(axis=0), np.maximum(t_bank.std(axis=0), 1e-6)
        a_mean, a_std = a_bank.mean(axis=0), np.maximum(a_bank.std(axis=0), 1e-6)
        v_mean, v_std = v_bank.mean(axis=0), np.maximum(v_bank.std(axis=0), 1e-6)

        for stem in testVid:
            tx = videoText[stem][0]
            ax = videoAudio[stem][0]
            vx = videoVisual[stem][0]

            tx2 = match_stats(tx, t_mean, t_std, source_stats['text_mean'], source_stats['text_std']).astype(np.float32)
            ax2 = match_stats(ax, a_mean, a_std, source_stats['audio_mean'], source_stats['audio_std']).astype(np.float32)
            vx2 = match_stats(vx, v_mean, v_std, source_stats['visual_mean'], source_stats['visual_std']).astype(np.float32)

            videoText[stem] = tx2.reshape(1, -1)
            videoAudio[stem] = ax2.reshape(1, -1)
            videoVisual[stem] = vx2.reshape(1, -1)

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
    print(
        "[Build] audio_status "
        + " ".join([f"{k}={v}" for k, v in audio_stats.items()])
    )
    if source_stats is not None:
        print(f"[Build] source_align=on source_pkl={source_pkl}")
    else:
        print("[Build] source_align=off")
    print(f"[Build] out={out}")


if __name__ == "__main__":
    main()
