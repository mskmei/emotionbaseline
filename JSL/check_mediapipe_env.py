#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import json

from keypoints import load_mediapipe_holistic


def main():
    try:
        import mediapipe as mp
    except Exception as exc:
        print(json.dumps({"import_ok": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        raise

    info = {
        "import_ok": True,
        "version": getattr(mp, "__version__", "<unknown>"),
        "file": getattr(mp, "__file__", "<unknown>"),
        "has_top_level_solutions": hasattr(mp, "solutions"),
    }
    holistic_cls = load_mediapipe_holistic()
    info["holistic_class"] = f"{holistic_cls.__module__}.{holistic_cls.__name__}"
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
