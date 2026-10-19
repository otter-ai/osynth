"""Detection + cleanup for word-truncation fragments in TTS rewrites.

Single-letter fragments (`r--`, `b--`, `á--`) render poorly in Qwen3-TTS:
the model pronounces isolated letters as their letter name
(`"are"`, `"bee"`, `"ess"`) instead of cutting off the start of a word. We
strip them at two layers:

1. Hard delete — any ``\\p{L}--(?![\\p{L}-])`` token in ``tts_text`` (and its
   single-hyphen analog in ``semantic_verbatim_text`` / ``semantic_hints``)
   is unconditionally removed.
2. Probabilistic rate cap — even multi-character fragments (`bu--`, `re--`,
   `aparata--`) are revert-deleted above ``max_truncation_rate`` per
   100 spoken-form words, chosen by deterministic seeded sampling so
   per-meeting clusters of LLM-emitted truncations don't bring back the
   bad listening experience.

The module exposes both a stateful ``TruncationSanitizer`` (for the
batch-level rate cap across a meeting) and stateless pure-string cleaners
(``remove_single_letter_truncs``) that can also be applied to persisted
``formatted_turns.json`` files.
"""
from __future__ import annotations

import hashlib
import logging
import random
import re
from dataclasses import dataclass, field

from ..models.tts import SemanticHint, SpokenFormRewrite

logger = logging.getLogger(__name__)


# ``\p{L}`` (any Unicode letter) is not directly available in stdlib ``re``.
# The equivalent class in Python 3 (with default Unicode-aware ``\w``) is
# "word character minus digit minus underscore" — i.e. alphabetic letters
# of any script, including accented Spanish characters like ``á``, ``é``.
_LETTER = r"[^\W\d_]"
_LETTER_OR_HYPHEN = rf"(?:{_LETTER}|-)"

SINGLE_LETTER_TRUNC_TTS_RE: re.Pattern[str] = re.compile(
    rf"(?<![\w-]){_LETTER}--(?!{_LETTER_OR_HYPHEN})",
    re.UNICODE,
)
SINGLE_LETTER_TRUNC_SEMANTIC_RE: re.Pattern[str] = re.compile(
    rf"(?<![\w-]){_LETTER}-(?!{_LETTER_OR_HYPHEN})",
    re.UNICODE,
)
ANY_TRUNC_TTS_RE: re.Pattern[str] = re.compile(
    rf"(?<![\w-]){_LETTER}+--(?!{_LETTER_OR_HYPHEN})",
    re.UNICODE,
)
ANY_TRUNC_SEMANTIC_RE: re.Pattern[str] = re.compile(
    rf"(?<![\w-]){_LETTER}+-(?!{_LETTER_OR_HYPHEN})",
    re.UNICODE,
)


# ---------------------------------------------------------------------------
# Pure-string cleaners
# ---------------------------------------------------------------------------


def count_single_letter_tts(text: str) -> int:
    """Number of single-letter ``X--`` fragments in ``text``."""
    return len(SINGLE_LETTER_TRUNC_TTS_RE.findall(text or ""))


def count_word_truncations(text: str) -> int:
    """Number of word-truncation ``X+--`` fragments in ``text`` (any length)."""
    return len(ANY_TRUNC_TTS_RE.findall(text or ""))


def is_single_letter_fragment(
    fragment: str | None, *, single_hyphen: bool = False,
) -> bool:
    """True iff ``fragment`` is one alphabetic char + ``-`` or ``--``."""
    if not fragment:
        return False
    expected_suffix = "-" if single_hyphen else "--"
    if not fragment.endswith(expected_suffix):
        return False
    body = fragment[: -len(expected_suffix)]
    return len(body) == 1 and body.isalpha()


def remove_single_letter_truncs(
    text: str,
    *,
    single_hyphen: bool = False,
) -> str:
    """Strip every single-letter truncation fragment from ``text``.

    By default operates on the TTS-side double-hyphen form (``r--``, ``á--``).
    Set ``single_hyphen=True`` to clean the semantic-side single-hyphen form
    (``r-``, ``á-``).

    Drops the single adjacent whitespace (trailing preferred; leading when
    the fragment is at end-of-text). If the deleted fragment was sentence-
    initial (its first letter was uppercase), the next alphabetic character
    in the cleaned text is capitalized to preserve readable casing.
    """
    pattern = (
        SINGLE_LETTER_TRUNC_SEMANTIC_RE if single_hyphen
        else SINGLE_LETTER_TRUNC_TTS_RE
    )
    return _strip_pattern_matches(text, pattern)


def _strip_pattern_matches(text: str, pattern: re.Pattern[str]) -> str:
    if not text:
        return text
    matches = list(pattern.finditer(text))
    if not matches:
        return text
    return _delete_spans(text, [(m.start(), m.end(), m.group(0)) for m in matches])


def _delete_spans(
    text: str, spans: list[tuple[int, int, str]],
) -> str:
    """Drop each ``(start, end)`` span plus one adjacent whitespace.

    ``spans`` must be sorted by start offset and non-overlapping. Re-
    capitalizes the next alphabetic character after each deletion whose
    fragment's first character was uppercase (sentence-initial signal).
    Final result has internal whitespace runs collapsed and ends trimmed.
    """
    if not spans:
        return text
    pieces: list[str] = []
    cursor = 0
    capitalize_next = False
    for start, end, surface in spans:
        if start < cursor:
            continue
        first_char = surface[:1]

        trailing_consume = 0
        leading_consume = 0
        if end < len(text) and text[end].isspace():
            trailing_consume = 1
        elif start > 0 and text[start - 1].isspace():
            leading_consume = 1

        segment = text[cursor : start - leading_consume]
        if capitalize_next:
            segment = _capitalize_first_alpha(segment)
            capitalize_next = False
        pieces.append(segment)
        capitalize_next = bool(first_char) and first_char.isupper()
        cursor = end + trailing_consume

    tail = text[cursor:]
    if capitalize_next:
        tail = _capitalize_first_alpha(tail)
    pieces.append(tail)

    return _collapse_internal_whitespace("".join(pieces)).strip()


def _capitalize_first_alpha(text: str) -> str:
    for i, ch in enumerate(text):
        if ch.isalpha():
            upper = ch.upper()
            if upper != ch:
                return text[:i] + upper + text[i + 1 :]
            return text
    return text


_WS_COLLAPSE_RE = re.compile(r" {2,}")


def _collapse_internal_whitespace(text: str) -> str:
    return _WS_COLLAPSE_RE.sub(" ", text)


# ---------------------------------------------------------------------------
# Stateful per-meeting sanitizer
# ---------------------------------------------------------------------------


@dataclass
class SanitizerStats:
    """Per-batch summary of sanitizer activity."""

    single_letter_dropped: int = 0
    rate_capped_dropped: int = 0
    hints_dropped: int = 0


@dataclass
class TruncationSanitizer:
    """Stateful per-meeting word-truncation sanitizer.

    Tracks running totals of spoken-form words and retained
    ``word_truncation`` hints across successive turns so the probabilistic
    rate cap applies with meeting-wide context. Single-letter fragments are
    dropped unconditionally; multi-character fragments above the configured
    rate are reverted via deterministic seeded sampling keyed on
    ``(run_id, meeting_id, turn_index)``.
    """

    max_rate_per_100_words: float = 1.0
    run_id: str = ""
    meeting_id: str = ""
    running_word_count: int = 0
    running_trunc_count: int = 0
    stats: SanitizerStats = field(default_factory=SanitizerStats)

    def sanitize(
        self,
        rewrite: SpokenFormRewrite,
        *,
        turn_index: int | None = None,
    ) -> SpokenFormRewrite:
        """Return a sanitized copy of ``rewrite``; update running state.

        Pass order:

        1. Hard pass — strip every single-letter ``\\p{L}--`` fragment from
           ``tts_text``, the single-hyphen analog from
           ``semantic_verbatim_text``, the matching ``word_truncation`` hint,
           and any paired ``repair`` / ``false_start`` hint anchored on the
           deleted fragment.
        2. Rate-cap pass — count remaining multi-char ``word_truncation``
           hints; if the running rate would exceed ``max_rate_per_100_words``
           after this turn, drop hints from the current turn (chosen by
           deterministic seeded sampling) until the rate fits the budget.
           The corresponding ``\\p{L}+--`` fragment is deleted from
           ``tts_text`` and its single-hyphen analog from
           ``semantic_verbatim_text``.
        """
        tts_text = rewrite.tts_text or ""
        semantic_text = rewrite.semantic_verbatim_text or ""
        hints: list[SemanticHint] = list(rewrite.semantic_hints)

        tts_text, semantic_text, hints, hard_dropped = self._hard_strip_single_letter(
            tts_text, semantic_text, hints, turn_index=turn_index,
        )

        tts_text, semantic_text, hints = self._apply_rate_cap(
            tts_text, semantic_text, hints, turn_index=turn_index,
        )

        self.running_word_count += len(tts_text.split())

        return rewrite.model_copy(
            update={
                "tts_text": tts_text,
                "semantic_verbatim_text": semantic_text,
                "text": tts_text,
                "semantic_hints": hints,
            },
        )

    def _hard_strip_single_letter(
        self,
        tts_text: str,
        semantic_text: str,
        hints: list[SemanticHint],
        *,
        turn_index: int | None,
    ) -> tuple[str, str, list[SemanticHint], int]:
        tts_count = count_single_letter_tts(tts_text)
        sem_count = len(SINGLE_LETTER_TRUNC_SEMANTIC_RE.findall(semantic_text))

        cleaned_tts = remove_single_letter_truncs(tts_text)
        cleaned_semantic = remove_single_letter_truncs(
            semantic_text, single_hyphen=True,
        )

        kept_hints, dropped_hints = self._drop_single_letter_hints(hints)
        if not (tts_count or sem_count or dropped_hints):
            return cleaned_tts, cleaned_semantic, kept_hints, 0

        dropped = max(tts_count, sem_count, len(dropped_hints))
        self.stats.single_letter_dropped += dropped
        self.stats.hints_dropped += len(dropped_hints)
        logger.warning(
            "Sanitizer dropped %d single-letter truncation(s) (run=%r, "
            "meeting=%r, turn=%r, dropped_hint_fragments=%r)",
            dropped,
            self.run_id,
            self.meeting_id,
            turn_index,
            [h.fragment for h in dropped_hints],
        )
        return cleaned_tts, cleaned_semantic, kept_hints, dropped

    @staticmethod
    def _drop_single_letter_hints(
        hints: list[SemanticHint],
    ) -> tuple[list[SemanticHint], list[SemanticHint]]:
        kept: list[SemanticHint] = []
        dropped: list[SemanticHint] = []
        deleted_fragments: set[str] = set()
        for hint in hints:
            if hint.type == "word_truncation" and (
                is_single_letter_fragment(hint.fragment, single_hyphen=True)
                or is_single_letter_fragment(hint.fragment)
            ):
                dropped.append(hint)
                if hint.fragment:
                    deleted_fragments.add(hint.fragment.strip())
                continue
            kept.append(hint)

        # Drop paired repair/false_start hints anchored on a deleted fragment.
        final_kept: list[SemanticHint] = []
        for hint in kept:
            if hint.type in ("repair", "false_start") and _hint_anchors_deleted(
                hint, deleted_fragments,
            ):
                dropped.append(hint)
                continue
            final_kept.append(hint)
        return final_kept, dropped

    def _apply_rate_cap(
        self,
        tts_text: str,
        semantic_text: str,
        hints: list[SemanticHint],
        *,
        turn_index: int | None,
    ) -> tuple[str, str, list[SemanticHint]]:
        if self.max_rate_per_100_words <= 0:
            return tts_text, semantic_text, hints

        trunc_positions = [
            (i, h) for i, h in enumerate(hints)
            if h.type == "word_truncation" and h.fragment
        ]
        if not trunc_positions:
            return tts_text, semantic_text, hints

        words_after = self.running_word_count + len(tts_text.split())
        budget = int(words_after * self.max_rate_per_100_words / 100.0)
        room = max(0, budget - self.running_trunc_count)
        excess = max(0, len(trunc_positions) - room)

        if excess <= 0:
            self.running_trunc_count += len(trunc_positions)
            return tts_text, semantic_text, hints

        rng = _seeded_rng(self.run_id, self.meeting_id, turn_index)
        local_order = list(range(len(trunc_positions)))
        rng.shuffle(local_order)
        drop_local = set(local_order[:excess])
        dropped_hint_ids: set[int] = set()
        dropped_hints: list[SemanticHint] = []
        for local_idx, (_, hint) in enumerate(trunc_positions):
            if local_idx in drop_local:
                dropped_hint_ids.add(id(hint))
                dropped_hints.append(hint)

        tts_spans = _locate_fragment_spans(
            tts_text, dropped_hints, single_hyphen=False,
        )
        sem_spans = _locate_fragment_spans(
            semantic_text, dropped_hints, single_hyphen=True,
        )

        cleaned_tts = _delete_spans(tts_text, tts_spans) if tts_spans else tts_text
        cleaned_semantic = (
            _delete_spans(semantic_text, sem_spans) if sem_spans else semantic_text
        )

        kept_hints = [h for h in hints if id(h) not in dropped_hint_ids]

        self.stats.rate_capped_dropped += len(dropped_hints)
        self.stats.hints_dropped += len(dropped_hints)
        self.running_trunc_count += len(trunc_positions) - len(dropped_hints)
        logger.warning(
            "Sanitizer rate-capped %d multi-char truncation(s) (run=%r, "
            "meeting=%r, turn=%r, fragments=%r)",
            len(dropped_hints),
            self.run_id,
            self.meeting_id,
            turn_index,
            [h.fragment for h in dropped_hints],
        )
        return cleaned_tts, cleaned_semantic, kept_hints


def _hint_anchors_deleted(
    hint: SemanticHint, deleted_fragments: set[str],
) -> bool:
    """Decide whether a ``repair`` / ``false_start`` hint anchors on a deleted fragment."""
    candidates = [hint.reparandum, hint.surface, hint.repair]
    for cand in candidates:
        if not cand:
            continue
        token = cand.strip()
        if token in deleted_fragments:
            return True
        # Allow ``r`` to match a deleted ``r-`` fragment as well.
        if token + "-" in deleted_fragments or token + "--" in deleted_fragments:
            return True
    return False


def _locate_fragment_spans(
    text: str,
    hints: list[SemanticHint],
    *,
    single_hyphen: bool,
) -> list[tuple[int, int, str]]:
    """Find the first ``text`` span for each hint's fragment.

    The fragment in the hint is the semantic-side single-hyphen form (e.g.
    ``bu-``). For the TTS surface we look for ``bu--``; for the semantic
    surface we look for ``bu-``. Each hint contributes at most one span.
    Spans are returned sorted by start offset and de-duplicated.
    """
    if not text or not hints:
        return []
    spans: list[tuple[int, int, str]] = []
    consumed: list[tuple[int, int]] = []
    for hint in hints:
        if not hint.fragment:
            continue
        body = hint.fragment.rstrip("-")
        if not body:
            continue
        token = body + ("-" if single_hyphen else "--")
        pattern = re.compile(
            rf"(?<![\w-]){re.escape(token)}(?!{_LETTER_OR_HYPHEN})",
            re.UNICODE,
        )
        for m in pattern.finditer(text):
            span = (m.start(), m.end())
            if any(
                not (span[1] <= c[0] or span[0] >= c[1]) for c in consumed
            ):
                continue
            spans.append((m.start(), m.end(), m.group(0)))
            consumed.append(span)
            break
    spans.sort(key=lambda s: s[0])
    return spans


def _seeded_rng(run_id: str, meeting_id: str, turn_index: int | None) -> random.Random:
    """Return a deterministic RNG keyed on ``(run_id, meeting_id, turn_index)``."""
    seed_str = f"{run_id}::{meeting_id}::{turn_index if turn_index is not None else -1}"
    digest = hashlib.sha256(seed_str.encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def sanitize_truncations(
    rewrite: SpokenFormRewrite,
    *,
    max_rate_per_100_words: float = 1.0,
    run_id: str = "",
    meeting_id: str = "",
    turn_index: int | None = None,
) -> SpokenFormRewrite:
    """One-shot sanitize: hard delete single-letter; no carried rate state.

    Convenience wrapper for unit testing and one-off rewrites. For batch-
    level rate cap across a meeting, use ``TruncationSanitizer`` directly so
    the running word + truncation counters carry across turns.
    """
    sanitizer = TruncationSanitizer(
        max_rate_per_100_words=max_rate_per_100_words,
        run_id=run_id,
        meeting_id=meeting_id,
    )
    return sanitizer.sanitize(rewrite, turn_index=turn_index)
