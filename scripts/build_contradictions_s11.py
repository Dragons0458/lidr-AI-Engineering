#!/usr/bin/env python3
"""Ingest the Session 11 contradiction corpus for synthesis demos."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_task_corpus import ingest_corpus  # noqa: E402

CORPUS_PATH = ROOT / "data" / "task_corpus_contradictions.json"


def main() -> None:
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    ingest_corpus(corpus)


if __name__ == "__main__":
    main()
