import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def normalize_label(x: str) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    u = s.upper()
    if u in {"A", "N", "J", "S"}:
        return u

    m = {
        "anger": "A",
        "ang": "A",
        "fru": "A",
        "怒り": "A",
        "怒り1": "A",
        "neutral": "N",
        "neu": "N",
        "平静": "N",
        "joy": "J",
        "happy": "J",
        "hap": "J",
        "exc": "J",
        "喜び": "J",
        "sadness": "S",
        "sad": "S",
        "悲しみ": "S",
    }
    return m.get(s.lower())


def read_csv_rows(csv_path: Path) -> List[dict]:
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_dial_sample_id(name: str) -> Optional[Tuple[str, int, int, str]]:
    m = re.match(r"^(SD\d{2})-(\d{2})-(\d{2})([AJNS])$", name)
    if m is None:
        return None
    sd_id, dialogue_idx, utterance_idx, label = m.groups()
    return sd_id, int(dialogue_idx), int(utterance_idx), label


def parse_dialogue_txt(txt_file: Path) -> List[Tuple[str, str, str]]:
    turns = []
    for raw in txt_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < 3:
            continue
        speaker, emotion, text = parts[0].strip(), parts[1].strip(), parts[2].strip()
        turns.append((speaker, emotion, text))
    return turns


def make_bobsl_conversations(rows: List[dict]) -> List[List[dict]]:
    convs = []
    for r in rows:
        text = str(r.get("text", "")).strip()
        label = normalize_label(str(r.get("emotion", "")))
        if not text or label is None:
            continue
        convs.append([
            {
                "text": text,
                "speaker": "S1",
                "emotion": label,
            }
        ])
    return convs


def read_dial_names(path: Path) -> List[str]:
    lines = [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not lines:
        return []

    # Header case.
    first = lines[0].lower()
    if "stem" in first or "clip" in first or "filename" in first:
        out = []
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                val = row.get("clip_name") or row.get("stem") or row.get("filename")
                if val:
                    out.append(Path(str(val).strip()).stem)
        return out

    # One-file-per-line case.
    return [Path(x).stem for x in lines]


def make_dial_conversations(dial_list_path: Path, txt_root: Path) -> List[List[dict]]:
    names = read_dial_names(dial_list_path)
    convs = []
    for name in names:
        parsed = parse_dial_sample_id(name)
        if parsed is None:
            continue
        sd_id, dialogue_idx, utterance_idx, forced_label = parsed

        txt_file = txt_root / sd_id / "txt" / f"{sd_id}-Dialogue-{dialogue_idx:02d}.txt"
        if not txt_file.exists():
            continue
        turns = parse_dialogue_txt(txt_file)
        if utterance_idx <= 0 or utterance_idx > len(turns):
            continue

        conv = []
        for i in range(utterance_idx):
            speaker, emo_jp, text = turns[i]
            lab = normalize_label(emo_jp)
            if lab is None:
                lab = forced_label if i == utterance_idx - 1 else "N"
            if i == utterance_idx - 1:
                lab = forced_label
            conv.append({"text": text, "speaker": speaker, "emotion": lab})
        convs.append(conv)
    return convs


def dump_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def dump_label_change(path: Path):
    # EmoTrans expects a label_change file mapping original label to prompt label token.
    mapping = [["A", "Ang"], ["N", "Neu"], ["J", "Joy"], ["S", "Sad"]]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)


def main():
    p = argparse.ArgumentParser(description="Build EmoTrans dataset jsons from BOBSL train/val and DIAL test list")
    p.add_argument("--train_csv", required=True, type=str)
    p.add_argument("--val_csv", required=True, type=str)
    p.add_argument("--dial_test_csv", required=True, type=str)
    p.add_argument("--txt_root", required=True, type=str)
    p.add_argument("--out_dir", default="./datasets/bobsl_dial", type=str)
    args = p.parse_args()

    train_rows = read_csv_rows(Path(args.train_csv))
    val_rows = read_csv_rows(Path(args.val_csv))

    train_convs = make_bobsl_conversations(train_rows)
    val_convs = make_bobsl_conversations(val_rows)
    test_convs = make_dial_conversations(Path(args.dial_test_csv), Path(args.txt_root))

    out_dir = Path(args.out_dir)
    dump_json(out_dir / "train.json", train_convs)
    dump_json(out_dir / "valid.json", val_convs)
    dump_json(out_dir / "test.json", test_convs)
    dump_label_change(out_dir / "label_change")

    print(f"[Build] train={len(train_convs)} valid={len(val_convs)} test={len(test_convs)}")
    print(f"[Build] output_dir={out_dir}")


if __name__ == "__main__":
    main()
