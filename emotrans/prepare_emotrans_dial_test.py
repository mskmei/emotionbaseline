import argparse
import csv
import json
import shutil
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_dial_sample_id(name: str) -> Optional[Tuple[str, int, int, str]]:
    m = re.match(r"^(SD\d{2})-(\d{2})-(\d{2})([AJNS])$", name)
    if m is None:
        return None
    sd_id, dialogue_idx, utterance_idx, label = m.groups()
    return sd_id, int(dialogue_idx), int(utterance_idx), label


def dial_to_meld_label(label: str) -> str:
    # Keep test labels compatible with MELD label_change.
    mapping = {
        "A": "anger",
        "N": "neutral",
        "J": "joy",
        "S": "sadness",
    }
    return mapping.get(label, "neutral")


def read_dial_names(path: Path) -> List[str]:
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


def scan_dial_names_from_frame_root(frame_root: Path) -> List[str]:
    if not frame_root.exists():
        return []
    names = []
    for p in sorted(frame_root.iterdir()):
        # Typical layout: frame/SD01-01-01A/*.jpg
        if p.is_dir() and parse_dial_sample_id(p.name) is not None:
            names.append(p.name)
            continue
        # Fallback for possible file-style entries.
        stem = p.stem
        if parse_dial_sample_id(stem) is not None:
            names.append(stem)
    return names


def collect_dial_names(dial_list: Optional[Path], frame_root: Optional[Path]) -> List[str]:
    if dial_list is not None and dial_list.exists():
        names = read_dial_names(dial_list)
        print(f"[Prepare] using dial list: {dial_list}  n={len(names)}")
        return names

    if dial_list is not None:
        print(f"[Warn] dial list not found: {dial_list}")

    if frame_root is not None:
        names = scan_dial_names_from_frame_root(frame_root)
        print(f"[Prepare] using frame scan: {frame_root}  n={len(names)}")
        return names

    return []


def parse_dialogue_txt(txt_file: Path):
    turns = []
    for raw in txt_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < 3:
            continue
        speaker, text = parts[0].strip(), parts[2].strip()
        turns.append((speaker, text))
    return turns


class JaEnTranslator:
    def __init__(self, model_name: str, batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self._cache: Dict[str, str] = {}
        self.failed = 0

        try:
            from transformers import pipeline
            import torch

            # Use GPU when available; otherwise run on CPU.
            device = 0 if torch.cuda.is_available() else -1
            self._pipe = pipeline(
                "translation",
                model=model_name,
                tokenizer=model_name,
                device=device,
            )
        except Exception as exc:
            self._pipe = None
            print(f"[Warn] translation disabled: cannot init model '{model_name}': {exc}")

    def translate(self, text: str) -> str:
        if not text:
            return text
        if text in self._cache:
            return self._cache[text]
        if self._pipe is None:
            return text

        try:
            out = self._pipe(text, max_length=256)
            if out and isinstance(out, list):
                translated = str(out[0].get("translation_text", "")).strip()
                if translated:
                    self._cache[text] = translated
                    return translated
        except Exception:
            self.failed += 1
        return text


def build_dial_test_json(dial_names: List[str], txt_root: Path, translator: Optional[JaEnTranslator] = None):
    names = dial_names
    convs = []
    for name in names:
        parsed = parse_dial_sample_id(name)
        if parsed is None:
            continue
        sd_id, dialogue_idx, utterance_idx, label = parsed

        txt_file = txt_root / sd_id / "txt" / f"{sd_id}-Dialogue-{dialogue_idx:02d}.txt"
        if not txt_file.exists():
            continue
        turns = parse_dialogue_txt(txt_file)
        if utterance_idx <= 0 or utterance_idx > len(turns):
            continue

        conv = []
        for i in range(utterance_idx):
            speaker, text = turns[i]
            if translator is not None:
                text = translator.translate(text)
            # Keep earlier turns as neutral to avoid injecting unknown labels.
            emo = dial_to_meld_label(label) if i == utterance_idx - 1 else "neutral"
            conv.append({"text": text, "speaker": speaker, "emotion": emo})
        convs.append(conv)

    return convs


def main():
    p = argparse.ArgumentParser(description="Prepare EmoTrans dataset alias: keep base train/valid, replace test with DIAL")
    p.add_argument("--base_dataset", type=str, default="meld", help="Existing dataset under ./datasets, e.g. meld")
    p.add_argument("--new_dataset", type=str, default="meld_dial", help="New dataset folder name under ./datasets")
    p.add_argument("--dial_test_csv", type=str, default="", help="CSV/TXT list of clip names; optional when --frame_root is provided")
    p.add_argument("--frame_root", type=str, default="", help="Frame root like .../eJSL_dial/frame; used when dial list is missing")
    p.add_argument("--txt_root", type=str, required=True)
    p.add_argument("--datasets_root", type=str, default="./datasets")
    p.add_argument("--translate_to_en", action="store_true", help="Translate Japanese utterances to English before writing test.json")
    p.add_argument("--translation_model", type=str, default="Helsinki-NLP/opus-mt-ja-en")
    args = p.parse_args()

    datasets_root = Path(args.datasets_root)
    base_dir = datasets_root / args.base_dataset
    new_dir = datasets_root / args.new_dataset

    if not base_dir.exists():
        raise RuntimeError(f"base dataset not found: {base_dir}")

    new_dir.mkdir(parents=True, exist_ok=True)

    # Follow original training style: keep train/valid from base dataset.
    shutil.copy2(base_dir / "train.json", new_dir / "train.json")
    shutil.copy2(base_dir / "valid.json", new_dir / "valid.json")

    # label_change should be inherited from base dataset for model prompt tokens.
    shutil.copy2(base_dir / "label_change", new_dir / "label_change")

    translator = None
    if args.translate_to_en:
        translator = JaEnTranslator(args.translation_model)
        if translator._pipe is None:
            print("[Warn] translation requested but model is unavailable, fallback to original Japanese text")

    dial_list = Path(args.dial_test_csv) if args.dial_test_csv else None
    frame_root = Path(args.frame_root) if args.frame_root else None
    dial_names = collect_dial_names(dial_list, frame_root)
    if not dial_names:
        raise RuntimeError(
            "No DIAL samples found. Provide --dial_test_csv (existing file) or --frame_root (existing dir)."
        )

    test_convs = build_dial_test_json(dial_names, Path(args.txt_root), translator=translator)
    with open(new_dir / "test.json", "w", encoding="utf-8") as f:
        json.dump(test_convs, f, ensure_ascii=False, indent=2)

    print(f"[Prepare] base={base_dir}")
    print(f"[Prepare] new ={new_dir}")
    print(f"[Prepare] train copied: {new_dir / 'train.json'}")
    print(f"[Prepare] valid copied: {new_dir / 'valid.json'}")
    print(f"[Prepare] test(dial): {new_dir / 'test.json'}  n={len(test_convs)}")
    if translator is not None:
        print(f"[Prepare] translation_model={args.translation_model}")
        print(f"[Prepare] translation_cache={len(translator._cache)} failed={translator.failed}")


if __name__ == "__main__":
    main()
