"""Text normalization shared by transcript-producing steps."""
from __future__ import annotations

# Some writers emit typographic quotes. Fold them to ASCII so
# TTS input and reference transcripts tokenize like every other run
# ("That’s" vs "That's" would otherwise count as a WER substitution).
# CJK corner brackets and guillemets are left alone.
_TYPOGRAPHIC_QUOTES = str.maketrans({
    "‘": "'", "’": "'", "‛": "'", "′": "'",
    "“": '"', "”": '"', "‟": '"', "″": '"',
})

TURN_TEXT_FIELDS = ("text", "canonical_text", "semantic_verbatim_text", "tts_text")


def ascii_quotes(text: str) -> str:
    return text.translate(_TYPOGRAPHIC_QUOTES)


def ascii_quotes_in_turns(turns: list[dict]) -> list[dict]:
    """Normalize the text fields of formatted turns in place; returns ``turns``."""
    for turn in turns:
        for field in TURN_TEXT_FIELDS:
            value = turn.get(field)
            if isinstance(value, str):
                turn[field] = ascii_quotes(value)
    return turns
