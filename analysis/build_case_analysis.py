import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

LABELS = ["A", "N", "J", "S"]
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
    turns = []
    for raw in txt_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < 4:
            continue
        turns.append((parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip()))
    return turns


def collect_manifest(frame_root: Path, txt_root: Path) -> Dict[str, dict]:
    manifest: Dict[str, dict] = {}
    for p in sorted(frame_root.iterdir()):
        if not p.is_dir():
            continue
        sid = p.name
        parsed = parse_sample_id(sid)
        if parsed is None:
            continue
        sd_id, d_idx, u_idx, lab = parsed
        txt_file = txt_root / sd_id / "txt" / f"{sd_id}-Dialogue-{d_idx:02d}.txt"
        if not txt_file.exists():
            continue

        turns = parse_dialogue_txt(txt_file)
        if u_idx <= 0 or u_idx > len(turns):
            continue

        cur_turn = turns[u_idx - 1]
        prev_turn = turns[u_idx - 2] if u_idx > 1 else ("", "", "", "")

        manifest[sid] = {
            "sample_id": sid,
            "sd_id": sd_id,
            "dialogue_idx": d_idx,
            "utterance_idx": u_idx,
            "gold_label": lab,
            "frame_dir": str(p),
            "txt_file": str(txt_file),
            "prev_speaker": prev_turn[0],
            "prev_emotion_jp": prev_turn[1],
            "prev_emotion_anjs": JP2ANJS.get(prev_turn[1], ""),
            "prev_text": prev_turn[2],
            "curr_speaker": cur_turn[0],
            "curr_emotion_jp": cur_turn[1],
            "curr_text": cur_turn[2],
        }
    return manifest


def normalize_label(s: str) -> str:
    x = str(s).strip()
    return x if x in LABELS else ""


def load_pred_csv(path: Path, model_name: str) -> Dict[str, dict]:
    if not path.exists():
        raise FileNotFoundError(f"prediction csv not found: {path}")

    out: Dict[str, dict] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = str(row.get("sample_id", "")).strip()
            if not sid:
                continue

            g = normalize_label(row.get("gold_label", ""))
            p = normalize_label(row.get("pred_label", ""))
            if not g:
                g = normalize_label(row.get("gold_label_raw", ""))
            if not p:
                p = normalize_label(row.get("pred_label_raw", ""))

            if not g or not p:
                continue

            out[sid] = {
                "model": model_name,
                "gold_label": g,
                "pred_label": p,
                "correct": int(g == p),
            }
    print(f"[Load] {model_name}: {len(out)} samples from {path}")
    return out


def write_cases(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "sample_id",
                "gold_label",
                "telme_pred",
                "emotrans_pred",
                "eanwh_pred",
                "frame_dir",
                "prev_text",
                "curr_text",
            ])
        return

    cols = [
        "sample_id",
        "gold_label",
        "telme_pred",
        "emotrans_pred",
        "eanwh_pred",
        "telme_correct",
        "emotrans_correct",
        "eanwh_correct",
        "sd_id",
        "dialogue_idx",
        "utterance_idx",
        "frame_dir",
        "txt_file",
        "prev_speaker",
        "prev_emotion_jp",
        "prev_emotion_anjs",
        "prev_text",
        "curr_speaker",
        "curr_emotion_jp",
        "curr_text",
    ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in cols})


def main() -> None:
    p = argparse.ArgumentParser(description="Build EJSL case analysis from three baseline prediction CSV files")
    p.add_argument("--frame_root", type=str, required=True)
    p.add_argument("--txt_root", type=str, required=True)
    p.add_argument("--telme_pred_csv", type=str, required=True)
    p.add_argument("--emotrans_pred_csv", type=str, required=True)
    p.add_argument("--eanwh_pred_csv", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="./analysis/outputs_cases")
    p.add_argument("--expected_samples", type=int, default=1920)
    args = p.parse_args()

    frame_root = Path(args.frame_root)
    txt_root = Path(args.txt_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = collect_manifest(frame_root, txt_root)
    print(f"[Data] manifest samples={len(manifest)}")
    if args.expected_samples > 0 and len(manifest) != args.expected_samples:
        print(f"[Warn] expected_samples={args.expected_samples}, got={len(manifest)}")

    telme = load_pred_csv(Path(args.telme_pred_csv), "telme")
    emotrans = load_pred_csv(Path(args.emotrans_pred_csv), "emotrans")
    eanwh = load_pred_csv(Path(args.eanwh_pred_csv), "eanwh")

    common_ids = sorted(set(manifest.keys()) & set(telme.keys()) & set(emotrans.keys()) & set(eanwh.keys()))
    print(f"[Data] common samples across 3 models={len(common_ids)}")

    case_telme_vs_emotrans: List[dict] = []
    case_telme_vs_eanwh: List[dict] = []
    merged_all: List[dict] = []

    for sid in common_ids:
        base = dict(manifest[sid])
        base["telme_pred"] = telme[sid]["pred_label"]
        base["emotrans_pred"] = emotrans[sid]["pred_label"]
        base["eanwh_pred"] = eanwh[sid]["pred_label"]
        base["telme_correct"] = telme[sid]["correct"]
        base["emotrans_correct"] = emotrans[sid]["correct"]
        base["eanwh_correct"] = eanwh[sid]["correct"]
        merged_all.append(base)

        if base["telme_correct"] == 1 and base["emotrans_correct"] == 0:
            case_telme_vs_emotrans.append(base)
        if base["telme_correct"] == 1 and base["eanwh_correct"] == 0:
            case_telme_vs_eanwh.append(base)

    write_cases(merged_all, out_dir / "all_common_predictions.csv")
    write_cases(case_telme_vs_emotrans, out_dir / "cases_telme_correct_emotrans_wrong.csv")
    write_cases(case_telme_vs_eanwh, out_dir / "cases_telme_correct_eanwh_wrong.csv")

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "manifest_samples": len(manifest),
                "common_samples": len(common_ids),
                "case_telme_correct_emotrans_wrong": len(case_telme_vs_emotrans),
                "case_telme_correct_eanwh_wrong": len(case_telme_vs_eanwh),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("[Done] case analysis generated")
    print(f"[Done] out_dir={out_dir}")


if __name__ == "__main__":
    main()
