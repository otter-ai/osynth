"""Derive an AMI-style tokenized verbatim view from formatted turns.

The AMI corpus annotates its manual orthographic transcripts as a list of
words with span-level disfluency tags (filled pauses, repetitions, false
starts / repairs, prolongations). We mirror that shape for our synthetic
turns by tokenizing ``tts_text`` and mapping each character-offset TTS edit
(plus the semantic-layer edits that carry over to the spoken form) to
word-index ranges.

This module is pure text manipulation; no LLM calls, no audio. It is the
text-side floor of the AMI view. Acoustic events (vocal sounds, unclear
regions) are added post-finalize from acoustic alignment.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from ..models.tts import (
    AMIDisfluencySpan,
    AMITurn,
    AMIWord,
    TranscriptEdit,
    UnclearSpan,
)

logger = logging.getLogger(__name__)


_WORD_RE = re.compile(r"\S+")


def tokenize_for_ami(text: str) -> list[tuple[str, int, int]]:
    """Split ``text`` into whitespace-separated tokens with their char spans.

    Returns a list of ``(token, char_start, char_end)`` triples. Tokens
    preserve their attached punctuation (e.g. ``"Q3,"`` stays one token);
    the AMI view marks token-level punctuation via ``AMIWord.is_punctuation``
    only for tokens that are *entirely* punctuation.
    """
    return [(m.group(0), m.start(), m.end()) for m in _WORD_RE.finditer(text)]


def _char_span_to_word_range(
    tokens: list[tuple[str, int, int]],
    char_start: int,
    char_end: int,
) -> tuple[int, int]:
    """Map a [char_start, char_end) span to a [word_start, word_end) range.

    Words whose char span overlaps the input span are included. Returns
    ``(0, 0)`` when no token overlaps.
    """
    if char_end <= char_start or not tokens:
        return (0, 0)
    word_start: int | None = None
    word_end = 0
    for idx, (_, t_start, t_end) in enumerate(tokens):
        if t_end <= char_start:
            continue
        if t_start >= char_end:
            break
        if word_start is None:
            word_start = idx
        word_end = idx + 1
    if word_start is None:
        return (0, 0)
    return (word_start, word_end)


_PUNCT_RE = re.compile(r"^[^\w]+$", re.UNICODE)


def derive_ami_turn(turn: dict[str, Any]) -> AMITurn:
    """Build a single ``AMITurn`` from a ``FormattedTurn`` dict."""
    tts_text = str(turn.get("tts_text") or turn.get("text") or "")
    speaker_id = str(turn.get("speaker_id") or "")

    tokens = tokenize_for_ami(tts_text)
    word_texts = [tok for tok, _, _ in tokens]

    disfluencies: list[AMIDisfluencySpan] = []
    for raw in turn.get("tts_edit_layer", []) or []:
        edit = raw if isinstance(raw, TranscriptEdit) else TranscriptEdit.model_validate(raw)
        word_start, word_end = _char_span_to_word_range(
            tokens, edit.span_start, edit.span_end,
        )
        if word_end <= word_start:
            continue
        disfluencies.append(
            AMIDisfluencySpan(
                type=edit.type,
                word_start=word_start,
                word_end=word_end,
            ),
        )

    truncation_word_indices: set[int] = set()
    semantic_verbatim_text = str(turn.get("semantic_verbatim_text") or "")
    for raw in turn.get("semantic_edit_layer", []) or []:
        edit = raw if isinstance(raw, TranscriptEdit) else TranscriptEdit.model_validate(raw)
        if edit.type != "word_truncation":
            continue
        fragment = edit.surface.strip()
        if not fragment.endswith("-"):
            continue
        tts_fragment = fragment[:-1] + "--"
        char_idx = tts_text.find(tts_fragment)
        if char_idx < 0:
            continue
        word_start, word_end = _char_span_to_word_range(
            tokens, char_idx, char_idx + len(tts_fragment),
        )
        if word_end <= word_start:
            continue
        truncation_word_indices.update(range(word_start, word_end))
        disfluencies.append(
            AMIDisfluencySpan(
                type=edit.type,
                word_start=word_start,
                word_end=word_end,
            ),
        )

    # Collapse the double-trailing-hyphen TTS convention (`aparata--`) down to
    # the single-trailing-hyphen AMI/read surface (`aparata-`) for semantic
    # `word_truncation` spans. The double hyphen is a Qwen3-TTS rendering hint,
    # not part of the orthography.
    words: list[AMIWord] = []
    for idx, tok in enumerate(word_texts):
        text = tok
        if idx in truncation_word_indices and text.endswith("--"):
            text = text[:-1]
        words.append(AMIWord(text=text, is_punctuation=bool(_PUNCT_RE.match(text))))

    # Semantic-layer edits that survived into the TTS surface contribute repair
    # structure to the AMI view. Their offsets refer to semantic_verbatim_text,
    # not tts_text, so we re-map by searching the surface within tts_text.
    for raw in turn.get("semantic_edit_layer", []) or []:
        edit = raw if isinstance(raw, TranscriptEdit) else TranscriptEdit.model_validate(raw)
        if edit.type == "word_truncation":
            continue
        surface = edit.surface
        if not surface:
            continue
        # Try to locate the same surface in tts_text. If the TTS pass inserted
        # disfluencies around it, the surface usually survives as a substring.
        char_idx = tts_text.find(surface)
        if char_idx < 0:
            # Fall back to mapping by relative position in semantic_verbatim_text.
            if not semantic_verbatim_text:
                continue
            relative = edit.span_start / max(1, len(semantic_verbatim_text))
            char_idx = max(0, min(len(tts_text) - 1, int(relative * len(tts_text))))
            char_end = min(len(tts_text), char_idx + len(surface))
        else:
            char_end = char_idx + len(surface)
        word_start, word_end = _char_span_to_word_range(tokens, char_idx, char_end)
        if word_end <= word_start:
            continue
        disfluencies.append(
            AMIDisfluencySpan(
                type=edit.type,
                word_start=word_start,
                word_end=word_end,
            ),
        )

    unclear_raw = turn.get("unclear_spans", []) or []
    unclear_spans = [
        u if isinstance(u, UnclearSpan) else UnclearSpan.model_validate(u)
        for u in unclear_raw
    ]

    return AMITurn(
        speaker_id=speaker_id,
        words=words,
        disfluencies=disfluencies,
        unclear_spans=unclear_spans,
        is_backchannel=bool(turn.get("is_backchannel")),
    )


def derive_ami_view(turns: list[dict[str, Any]]) -> list[AMITurn]:
    """Build an AMI-style view for a meeting's formatted turns."""
    return [derive_ami_turn(turn) for turn in turns]
