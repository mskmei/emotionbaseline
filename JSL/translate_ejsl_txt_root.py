#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


SYSTEM_INSTRUCTIONS = (
    "Translate the Japanese dialogue utterance into natural English for emotion recognition. "
    "Preserve names, fillers, interjections, sarcasm cues, negation, and emotional intensity when possible. "
    "Return only the English translation, with no quotes, labels, explanations, or extra lines."
)


def read_ejsl_names(path: Path) -> List[str]:
    lines = [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not lines:
        return []

    first = lines[0].lower()
    if "stem" in first or "clip" in first or "filename" in first:
        out: List[str] = []
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                value = row.get("clip_name") or row.get("stem") or row.get("filename")
                if value:
                    out.append(Path(str(value).strip()).stem)
        return out

    return [Path(x).stem for x in lines]


def parse_sample_id(sample_id: str) -> Optional[Tuple[str, int, int, str]]:
    match = re.match(r"^(SD\d{2})-(\d{2})-(\d{2})([AJNS])$", sample_id)
    if match is None:
        return None
    sd_id, dialogue_idx, utterance_idx, label = match.groups()
    return sd_id, int(dialogue_idx), int(utterance_idx), label


def parse_dialogue_filename(path: Path) -> Optional[Tuple[str, int]]:
    match = re.match(r"^(SD\d{2})-Dialogue-(\d{2})\.txt$", path.name)
    if match is None:
        return None
    sd_id, dialogue_idx = match.groups()
    return sd_id, int(dialogue_idx)


def collect_dialogue_limits(dial_list: Optional[Path]) -> Dict[Tuple[str, int], int]:
    if dial_list is None:
        return {}
    limits: Dict[Tuple[str, int], int] = {}
    for sample_id in read_ejsl_names(dial_list):
        parsed = parse_sample_id(sample_id)
        if parsed is None:
            continue
        sd_id, dialogue_idx, utterance_idx, _label = parsed
        key = (sd_id, dialogue_idx)
        limits[key] = max(limits.get(key, 0), utterance_idx)
    return limits


def iter_dialogue_files(input_root: Path, limits: Dict[Tuple[str, int], int]) -> Iterable[Tuple[Path, Optional[int]]]:
    if limits:
        for (sd_id, dialogue_idx), max_utterance in sorted(limits.items()):
            path = input_root / sd_id / "txt" / f"{sd_id}-Dialogue-{dialogue_idx:02d}.txt"
            yield path, max_utterance
        return

    for path in sorted(input_root.glob("SD*/txt/*-Dialogue-*.txt")):
        yield path, None


def parse_line(raw: str) -> Optional[Tuple[str, str, str]]:
    parts = raw.rstrip("\n").split("|", 2)
    if len(parts) < 3:
        return None
    return parts[0].strip(), parts[1].strip(), parts[2].strip()


def clean_translation(text: str) -> str:
    value = html.unescape(text or "")
    value = value.replace("|", "/")
    value = " ".join(value.split())
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1].strip()
    return value


def cache_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_cache(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    cache: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            source = str(row.get("source", ""))
            translation = str(row.get("translation", ""))
            if source and translation:
                cache[source] = translation
    return cache


def append_cache(path: Path, source: str, translation: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "source_hash": cache_key(source),
        "source": source,
        "translation": translation,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_openai_output_text(payload: Dict) -> str:
    direct = payload.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct

    texts: List[str] = []
    for item in payload.get("output", []) or []:
        for content in item.get("content", []) or []:
            text = content.get("text")
            if isinstance(text, str):
                texts.append(text)
    return "\n".join(texts)


def translate_openai(text: str, args) -> str:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for --backend openai")

    body = {
        "model": args.openai_model,
        "instructions": args.instructions,
        "input": text,
    }
    if args.temperature >= 0:
        body["temperature"] = args.temperature

    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    last_error: Optional[BaseException] = None
    for attempt in range(args.retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            translated = clean_translation(extract_openai_output_text(payload))
            if translated:
                return translated
            raise RuntimeError(f"OpenAI response did not contain output text: {payload}")
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"OpenAI HTTP {exc.code}: {body_text}")
        except Exception as exc:  # noqa: BLE001
            last_error = exc

        if attempt < args.retries:
            time.sleep(args.retry_sleep * (2**attempt))

    raise RuntimeError(f"OpenAI translation failed after retries: {last_error}")


def translate_google_cloud(text: str, args) -> str:
    try:
        from google.cloud import translate_v2 as translate  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "google-cloud-translate is required for --backend google_cloud. "
            "Install it on the server and set GOOGLE_APPLICATION_CREDENTIALS."
        ) from exc

    client = translate.Client()
    result = client.translate(
        text,
        source_language=args.source_language,
        target_language=args.target_language,
        format_="text",
    )
    return clean_translation(str(result.get("translatedText", "")))


def translate_command(text: str, args) -> str:
    if not args.command:
        raise RuntimeError("--command is required for --backend command")
    proc = subprocess.run(
        args.command,
        input=text,
        text=True,
        shell=True,
        capture_output=True,
        timeout=args.command_timeout,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"translation command failed with code {proc.returncode}: {proc.stderr.strip()}")
    return clean_translation(proc.stdout)


def translate_text(text: str, args) -> str:
    if args.backend == "openai":
        return translate_openai(text, args)
    if args.backend == "google_cloud":
        return translate_google_cloud(text, args)
    if args.backend == "command":
        return translate_command(text, args)
    if args.backend == "identity":
        return clean_translation(text)
    raise ValueError(f"Unsupported backend: {args.backend}")


def translate_dialogue_file(
    input_file: Path,
    output_file: Path,
    max_utterance: Optional[int],
    cache: Dict[str, str],
    cache_path: Path,
    args,
) -> Tuple[int, int, int]:
    if not input_file.exists():
        raise FileNotFoundError(f"Missing dialogue txt file: {input_file}")

    translated_lines: List[str] = []
    translated_count = 0
    cache_hits = 0
    copied_count = 0

    for idx, raw in enumerate(input_file.read_text(encoding="utf-8").splitlines(), start=1):
        if max_utterance is not None and idx > max_utterance:
            break
        parsed = parse_line(raw)
        if parsed is None:
            translated_lines.append(raw)
            copied_count += 1
            continue

        speaker, emotion, utterance = parsed
        if not utterance:
            translation = ""
        elif utterance in cache:
            translation = cache[utterance]
            cache_hits += 1
        else:
            translation = translate_text(utterance, args)
            if not translation:
                raise RuntimeError(f"Empty translation for line {idx} in {input_file}: {utterance!r}")
            cache[utterance] = translation
            append_cache(cache_path, utterance, translation)
            translated_count += 1
            if args.sleep > 0:
                time.sleep(args.sleep)

        translated_lines.append(f"{speaker}|{emotion}|{translation}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(translated_lines) + "\n", encoding="utf-8")
    return translated_count, cache_hits, copied_count


def parse_args():
    parser = argparse.ArgumentParser(description="Translate an eJSL dialogue txt tree to English.")
    parser.add_argument("--input_txt_root", type=str, required=True)
    parser.add_argument("--output_txt_root", type=str, required=True)
    parser.add_argument("--dial_list", type=str, default="", help="Optional eJSL filename CSV/TXT; only needed prefixes are translated.")
    parser.add_argument("--cache_jsonl", type=str, default="", help="Translation cache. Defaults to output_txt_root/translation_cache.jsonl.")
    parser.add_argument("--backend", choices=["openai", "google_cloud", "command", "identity"], default="openai")
    parser.add_argument("--openai_model", type=str, default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument("--instructions", type=str, default=SYSTEM_INSTRUCTIONS)
    parser.add_argument("--source_language", type=str, default="ja")
    parser.add_argument("--target_language", type=str, default="en")
    parser.add_argument("--command", type=str, default="", help="Shell command for --backend command. Input text is passed on stdin.")
    parser.add_argument("--command_timeout", type=float, default=60.0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry_sleep", type=float, default=2.0)
    parser.add_argument("--temperature", type=float, default=-1.0, help="Negative means do not send temperature.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep after each uncached request.")
    parser.add_argument("--limit_files", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_txt_root).expanduser()
    output_root = Path(args.output_txt_root).expanduser()
    dial_list = Path(args.dial_list).expanduser() if args.dial_list else None
    cache_path = Path(args.cache_jsonl).expanduser() if args.cache_jsonl else output_root / "translation_cache.jsonl"

    input_resolved = input_root.resolve(strict=True)
    output_resolved = output_root.resolve(strict=False)
    if output_resolved == input_resolved or input_resolved in output_resolved.parents:
        raise RuntimeError(
            "Refusing to write translated txt inside the source txt root. "
            f"input_txt_root={input_resolved} output_txt_root={output_resolved}"
        )

    limits = collect_dialogue_limits(dial_list)
    cache = load_cache(cache_path)
    files = list(iter_dialogue_files(input_root, limits))
    if args.limit_files > 0:
        files = files[: args.limit_files]
    if not files:
        raise RuntimeError(f"No eJSL dialogue txt files found under {input_root}")

    total_new = 0
    total_hits = 0
    total_copied = 0
    for file_idx, (input_file, max_utterance) in enumerate(files, start=1):
        parsed = parse_dialogue_filename(input_file)
        if parsed is None:
            raise RuntimeError(f"Unexpected dialogue filename: {input_file}")
        sd_id, dialogue_idx = parsed
        output_file = output_root / sd_id / "txt" / f"{sd_id}-Dialogue-{dialogue_idx:02d}.txt"
        new_count, hit_count, copied_count = translate_dialogue_file(
            input_file,
            output_file,
            max_utterance,
            cache,
            cache_path,
            args,
        )
        total_new += new_count
        total_hits += hit_count
        total_copied += copied_count
        if file_idx == 1 or file_idx % 25 == 0 or file_idx == len(files):
            print(
                f"[translate] files={file_idx}/{len(files)} new={total_new} "
                f"cache_hits={total_hits} copied={total_copied}"
            )

    marker = output_root / ".translation_complete"
    marker.write_text(
        json.dumps(
            {
                "input_txt_root": str(input_root),
                "output_txt_root": str(output_root),
                "backend": args.backend,
                "files": len(files),
                "new_translations": total_new,
                "cache_hits": total_hits,
                "cache_jsonl": str(cache_path),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[translate] output_txt_root={output_root}")
    print(f"[translate] cache_jsonl={cache_path}")


if __name__ == "__main__":
    main()
