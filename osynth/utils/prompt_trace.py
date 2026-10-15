"""Utilities for persisting raw LLM prompt inputs and outputs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import fileio


class PromptTraceRecorder:
    """Persist raw prompt inputs and outputs for later inspection."""

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        fileio.makedirs(self.base_dir)
        self._counter = 0

    def record(
        self,
        *,
        label: str,
        system_prompt: str | None,
        user_prompt: str,
        response_text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._counter += 1
        trace_dir = self.base_dir / f"{self._counter:04d}_{label}"
        fileio.makedirs(trace_dir)
        if system_prompt is not None:
            fileio.write_text(trace_dir / "system_prompt.txt", system_prompt)
        fileio.write_text(trace_dir / "user_prompt.txt", user_prompt)
        fileio.write_text(trace_dir / "response.txt", response_text)
        if metadata is not None:
            fileio.write_text(
                trace_dir / "metadata.json",
                json.dumps(metadata, indent=2, ensure_ascii=False, default=str),
            )
