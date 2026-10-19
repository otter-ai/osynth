"""Helpers for parsing JSON returned by LLMs."""
from __future__ import annotations

import json
import re
from typing import Any

_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL)


def extract_json(text: str, *, repair_string_newlines: bool = False) -> Any:
    """Parse JSON, accepting optional fenced code blocks around the payload.

    On strict-parse failure the helper falls back to scanning the payload for
    the first parseable JSON value (``[...]`` or ``{...}``) so a stray prose
    preamble or suffix from the model doesn't lose the call. The original
    strict-parse exception is re-raised when no JSON value is recoverable.
    """
    payload = strip_json_fence(text)
    try:
        return json.loads(payload)
    except json.JSONDecodeError as strict_exc:
        try:
            return _scan_first_json_value(payload)
        except json.JSONDecodeError:
            pass
        if repair_string_newlines:
            return json.loads(_escape_string_control_chars(payload))
        raise strict_exc


def strip_json_fence(text: str) -> str:
    """Remove a surrounding Markdown JSON fence when present."""
    fenced = _FENCED_JSON_RE.search(text)
    if fenced:
        return fenced.group(1).strip()
    return text.strip()


def _scan_first_json_value(text: str) -> Any:
    """Return the first parseable ``[...]`` / ``{...}`` value found in ``text``.

    Walks through every ``[`` and ``{`` character and uses
    ``json.JSONDecoder.raw_decode`` to attempt a parse from that index. The
    first one that succeeds wins — this tolerates a prose preamble before the
    JSON value (e.g. ``"Here are the classifications: [...]"``) and trailing
    prose after it. Raises ``json.JSONDecodeError`` if no candidate parses.
    """
    decoder = json.JSONDecoder()
    last_exc: json.JSONDecodeError | None = None
    for i, ch in enumerate(text):
        if ch not in "[{":
            continue
        try:
            obj, _ = decoder.raw_decode(text, i)
            return obj
        except json.JSONDecodeError as exc:
            last_exc = exc
            continue
    if last_exc is not None:
        raise last_exc
    raise json.JSONDecodeError("No JSON value found", text, 0)


def _escape_string_control_chars(text: str) -> str:
    """Escape raw newlines/tabs inside JSON string values."""
    result: list[str] = []
    in_string = False
    escape = False
    for ch in text:
        if escape:
            result.append(ch)
            escape = False
            continue
        if ch == "\\" and in_string:
            result.append(ch)
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            result.append(ch)
            continue
        if in_string:
            if ch == "\n":
                result.append("\\n")
                continue
            if ch == "\r":
                result.append("\\r")
                continue
            if ch == "\t":
                result.append("\\t")
                continue
        result.append(ch)
    return "".join(result)
