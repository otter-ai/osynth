"""Small file I/O helpers shared by the pipeline steps."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def makedirs(path: str | Path, exist_ok: bool = True) -> None:
    os.makedirs(str(path), exist_ok=exist_ok)


def write_text(path: str | Path, text: str, encoding: str = "utf-8") -> None:
    p = Path(path)
    makedirs(p.parent)
    p.write_text(text, encoding=encoding)


def write_json(path: str | Path, data: Any) -> None:
    write_text(path, json.dumps(data, indent=2, default=str, ensure_ascii=False) + "\n")


def read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)
