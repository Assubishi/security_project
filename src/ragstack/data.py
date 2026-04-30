from __future__ import annotations

import json
from pathlib import Path

from .types import Passage, QAExample


def _read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_passages(path: str | Path) -> list[Passage]:
    rows = _read_jsonl(path)
    return [Passage(**row) for row in rows]


def load_examples(path: str | Path) -> list[QAExample]:
    rows = _read_jsonl(path)
    return [QAExample(**row) for row in rows]
