from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_csv_rows(path: Path, rows: Iterable[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def append_jsonl(path: Path, item: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def row_key(yid: str, start: object, end: object) -> str:
    return f"{str(yid)}|{float(start):.3f}|{float(end):.3f}"


def safe_sample_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    safe = safe.strip("._")
    if not safe:
        raise RuntimeError(f"Cannot make a safe sample id from {value!r}")
    return safe


def find_video_files(video_dir: Path) -> Dict[str, Path]:
    exts = {".mp4", ".mkv", ".webm", ".avi", ".mov", ".flv", ".wmv", ".m4v"}
    out: Dict[str, Path] = {}
    for path in video_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in exts:
            out[path.stem] = path
    return out


def first_nonempty(row: Dict[str, object], names: Iterable[str]) -> Optional[str]:
    for name in names:
        value = row.get(name)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def sanitize_generated_text(text: str) -> str:
    text = " ".join(str(text).replace("|", " ").replace("\r", " ").replace("\n", " ").split())
    return text.strip()


def read_ejsl_names(path: Path) -> List[str]:
    lines = [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not lines:
        return []
    first = lines[0].lower()
    if "stem" in first or "clip" in first or "filename" in first:
        names: List[str] = []
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                value = first_nonempty(row, ["clip_name", "stem", "filename", "sample_id"])
                if value:
                    names.append(Path(value).stem)
        return names
    return [Path(x).stem for x in lines]


def parse_ejsl_sample_id(sample_id: str):
    match = re.match(r"^(SD\d{2})-(\d{2})-(\d{2})([AJNS])$", sample_id)
    if match is None:
        raise RuntimeError(f"Unexpected eJSL sample id: {sample_id}")
    sd_id, dialogue_idx, utterance_idx, label = match.groups()
    return sd_id, int(dialogue_idx), int(utterance_idx), label


def read_dialogue_structure(txt_file: Path) -> List[Dict[str, str]]:
    rows = []
    for line_no, raw in enumerate(txt_file.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < 3:
            raise RuntimeError(f"Expected speaker|emotion|text at {txt_file}:{line_no}")
        rows.append({"speaker": parts[0].strip(), "emotion": parts[1].strip() or "NA"})
    return rows
