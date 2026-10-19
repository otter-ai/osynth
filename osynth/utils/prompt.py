"""Shared prompt template loading and simple placeholder rendering.

Prompt files use two substitution layers:

1. **Language variables** — ``{lang:key}`` markers are replaced with values
   from the language config (see ``osynth.utils.language``). Processed
   first.
2. **Context variables** — ``{{PLACEHOLDER}}`` markers are replaced by
   callers through ``render_prompt()``. Processed after.

This module is intentionally small: prompts are plain text, and placeholder
rendering remains deterministic string substitution rather than a template
engine with control flow.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

from .language import get_lang_config

# Matches {lang:key} or {lang:key|join:", "} for list values
_LANG_RE = re.compile(r"\{lang:(\w+)(?:\|join:\"([^\"]*)\")?\}")


def read_prompt(path: Path, language: str = "en") -> str:
    """Read a prompt template and substitute language variables.

    Args:
        path: Path to the prompt template file.
        language: Language code (en, de, es, fr).

    Returns:
        Prompt text with ``{lang:*}`` markers resolved.
    """
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    return render_lang(text, language)


def render_prompt(template: str, variables: Mapping[str, object]) -> str:
    """Replace ``{{PLACEHOLDER}}`` markers in *template*.

    Variable keys may be provided either as ``"NAME"`` or ``"{{NAME}}"``.
    Values are converted with ``str()`` to preserve the previous call-site
    behavior of chained ``.replace()`` calls.
    """
    rendered = template
    for key, value in variables.items():
        key_text = str(key)
        placeholder = (
            key_text
            if key_text.startswith("{{") and key_text.endswith("}}")
            else "{{" + key_text + "}}"
        )
        rendered = rendered.replace(placeholder, str(value))
    return rendered


def render_lang(text: str, language: str = "en") -> str:
    """Substitute ``{lang:key}`` markers in *text* using language config."""
    config = get_lang_config(language)

    def _replace(match: re.Match) -> str:
        key = match.group(1)
        join_sep = match.group(2)
        value = config.get(key, match.group(0))  # leave unknown keys as-is
        if isinstance(value, list):
            if join_sep is not None:
                return join_sep.join(value)
            return ", ".join(value)
        return str(value)

    return _LANG_RE.sub(_replace, text)
