from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from manifest_utils import read_csv_rows, write_csv_rows


FIELDS = ["sample_id", "source_id", "yid", "start", "end", "source", "video_path", "text", "split"]


def parse_args():
    parser = argparse.ArgumentParser(description="Combine J-Shuwa manifest CSV files.")
    parser.add_argument("--in_csv", action="append", required=True, help="Input manifest CSV. Repeatable.")
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--dedupe", choices=["sample_id", "source_id", "segment", "none"], default="sample_id")
    return parser.parse_args()


def dedupe_key(row: Dict[str, str], mode: str) -> str:
    if mode == "none":
        return ""
    if mode == "sample_id":
        return str(row.get("sample_id", ""))
    if mode == "source_id":
        return str(row.get("source_id") or row.get("sample_id", ""))
    return f"{row.get('yid','')}|{float(row.get('start')):.3f}|{float(row.get('end')):.3f}"


def main():
    args = parse_args()
    rows: List[Dict[str, str]] = []
    seen = set()
    for path_text in args.in_csv:
        path = Path(path_text)
        if not path.exists():
            raise FileNotFoundError(path)
        for row in read_csv_rows(path):
            out = {field: row.get(field, "") for field in FIELDS}
            if not out["sample_id"] or not out["text"]:
                continue
            if args.dedupe != "none":
                key = dedupe_key(out, args.dedupe)
                if key in seen:
                    continue
                seen.add(key)
            rows.append(out)
    rows.sort(key=lambda row: (row.get("source", ""), row.get("yid", ""), float(row.get("start") or 0.0), float(row.get("end") or 0.0)))
    write_csv_rows(Path(args.out_csv), rows, FIELDS)
    print(f"[J-Shuwa-combine] rows={len(rows)} out={args.out_csv}")


if __name__ == "__main__":
    main()
