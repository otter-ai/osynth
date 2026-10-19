"""Deterministic sampler for unclear word ranges in TTS-form turns.

Selects a small fraction of words across a meeting's turns to mark as
acoustically unclear. The sampler runs at step 7 (text-side); the audio
attenuation that realizes the unclear regions runs at step 10.

Determinism: the sampler is seeded by ``(run_seed, meeting_id)`` so the
same run/meeting always produces the same unclear spans.
"""
from __future__ import annotations

import hashlib
import logging
import random
from typing import Any

from ..models.tts import TranscriptEdit, UnclearSpan
from .ami_view import _char_span_to_word_range, tokenize_for_ami

logger = logging.getLogger(__name__)


def _meeting_seed(run_seed: int, meeting_id: str) -> int:
    digest = hashlib.sha1(
        f"{run_seed}::unclear::{meeting_id}".encode("utf-8"),
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def sample_unclear_spans(
    *,
    turns: list[dict[str, Any]],
    unclear_rate: float,
    run_seed: int,
    meeting_id: str,
    min_words: int = 1,
    max_words: int = 3,
    severity_heavy_probability: float = 0.2,
) -> list[list[UnclearSpan]]:
    """Return a parallel list of unclear-span lists, one per turn.

    ``unclear_rate`` is the target fraction of words to obscure across the
    meeting. The sampler picks contiguous word ranges of length
    ``[min_words, max_words]`` until it reaches the target, skipping
    backchannels and very short turns (< 5 words).
    """
    per_turn: list[list[UnclearSpan]] = [[] for _ in turns]
    if unclear_rate <= 0:
        return per_turn

    rng = random.Random(_meeting_seed(run_seed, meeting_id))

    # Collect eligible turn / word-count pairs + which word indices are
    # word_truncation fragments (those must be excluded from unclear ranges —
    # a truncated word has no acoustic surface to attenuate).
    eligible: list[tuple[int, int]] = []
    truncation_indices_by_turn: dict[int, set[int]] = {}
    total_words = 0
    for idx, turn in enumerate(turns):
        if turn.get("is_backchannel"):
            continue
        text = str(turn.get("tts_text") or turn.get("text") or "")
        tokens = [(m, s, e) for m, s, e in tokenize_for_ami(text)]
        if len(tokens) < 5:
            continue
        truncation_indices: set[int] = set()
        for raw in turn.get("semantic_edit_layer", []) or []:
            edit = raw if isinstance(raw, TranscriptEdit) else TranscriptEdit.model_validate(raw)
            if edit.type != "word_truncation":
                continue
            fragment = edit.surface.strip()
            if not fragment.endswith("-"):
                continue
            tts_fragment = fragment[:-1] + "--"
            char_idx = text.find(tts_fragment)
            if char_idx < 0:
                continue
            w_start, w_end = _char_span_to_word_range(
                tokens, char_idx, char_idx + len(tts_fragment),
            )
            if w_end > w_start:
                truncation_indices.update(range(w_start, w_end))
        eligible.append((idx, len(tokens)))
        truncation_indices_by_turn[idx] = truncation_indices
        total_words += len(tokens)
    if not eligible or total_words == 0:
        return per_turn

    budget = max(1, int(round(total_words * unclear_rate)))
    used_ranges_by_turn: dict[int, list[tuple[int, int]]] = {}
    placed = 0
    # Bound iterations so we do not loop forever when most turns are saturated.
    iterations = 0
    max_iterations = budget * 20
    while placed < budget and iterations < max_iterations:
        iterations += 1
        turn_idx, word_count = rng.choice(eligible)
        span_len = rng.randint(min_words, min(max_words, word_count - 1))
        if span_len <= 0:
            continue
        start = rng.randint(0, word_count - span_len)
        end = start + span_len
        if _overlaps_existing(used_ranges_by_turn.get(turn_idx, []), start, end):
            continue
        truncation_indices = truncation_indices_by_turn.get(turn_idx) or set()
        if any(w in truncation_indices for w in range(start, end)):
            continue
        severity = "heavy" if rng.random() < severity_heavy_probability else "partial"
        per_turn[turn_idx].append(
            UnclearSpan(word_start=start, word_end=end, severity=severity),
        )
        used_ranges_by_turn.setdefault(turn_idx, []).append((start, end))
        placed += span_len

    return per_turn


def _overlaps_existing(
    ranges: list[tuple[int, int]],
    start: int,
    end: int,
) -> bool:
    for r_start, r_end in ranges:
        if not (end <= r_start or start >= r_end):
            return True
    return False
