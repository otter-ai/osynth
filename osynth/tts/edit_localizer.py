"""Deterministic transcript edit-layer localization.

The spoken-form LLM owns the rewrite. This module owns the character offsets:
given two concrete text forms, it derives edit spans that reconstruct exactly.
"""
from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from typing import Any, Iterable, Literal

from ..models.tts import SemanticHint, TTSEditType, TranscriptEdit, apply_edit_layer
from .ami_view import tokenize_for_ami

logger = logging.getLogger(__name__)


_BUILTIN_FILLERS: frozenset[str] = frozenset(
    {"uh", "um", "ah", "eh", "mm", "hmm", "er"},
)
_WORD_RE = re.compile(r"[\w]+(?:-[\w]+)*", re.UNICODE)
_EDGE_PUNCT_RE = re.compile(r"^[^\w]+|[^\w]+$", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+", re.UNICODE)


def _find_with_normalization(
    haystack: str, needle: str, start: int = 0,
) -> tuple[int, int]:
    """Locate ``needle`` in ``haystack[start:]`` tolerating cosmetic drift.

    Returns ``(match_start, match_end)`` offsets into the original
    ``haystack`` string. Returns ``(-1, -1)`` when no match is found. The
    ladder runs in order — first hit wins:

      1. exact substring,
      2. whitespace-collapsed (runs of whitespace -> single space),
      3. case-insensitive whitespace-collapsed,
      4. token-sequence walk: locate each whitespace-split needle token in
         order, allowing intervening whitespace / punctuation between them.

    For non-exact matches, ``match_end`` reflects the actual span in
    ``haystack`` (which may differ in length from ``len(needle)``).
    """
    if not needle:
        return (-1, -1)
    pos = haystack.find(needle, start)
    if pos >= 0:
        return (pos, pos + len(needle))

    span = _find_whitespace_collapsed(haystack, needle, start, casefold=False)
    if span[0] >= 0:
        return span

    span = _find_whitespace_collapsed(haystack, needle, start, casefold=True)
    if span[0] >= 0:
        return span

    return _find_token_sequence(haystack, needle, start)


def _find_whitespace_collapsed(
    haystack: str, needle: str, start: int, *, casefold: bool,
) -> tuple[int, int]:
    """Search ``needle`` in ``haystack`` after collapsing whitespace runs.

    Maps the collapsed match span back to original ``haystack`` offsets.
    """
    collapsed_haystack, mapping = _collapse_whitespace_with_mapping(haystack)
    collapsed_needle = _WHITESPACE_RE.sub(" ", needle).strip()
    if not collapsed_needle:
        return (-1, -1)

    # Translate ``start`` (an offset into the original haystack) to an offset
    # into the collapsed string.
    collapsed_start = 0
    for collapsed_idx, original_idx in enumerate(mapping):
        if original_idx >= start:
            collapsed_start = collapsed_idx
            break
    else:
        collapsed_start = len(collapsed_haystack)

    if casefold:
        idx = collapsed_haystack.casefold().find(
            collapsed_needle.casefold(), collapsed_start,
        )
    else:
        idx = collapsed_haystack.find(collapsed_needle, collapsed_start)
    if idx < 0 or idx >= len(mapping):
        return (-1, -1)
    end_idx = idx + len(collapsed_needle)
    match_start = mapping[idx]
    if end_idx - 1 < len(mapping):
        # Inclusive last index -> exclusive end in original string.
        match_end = mapping[end_idx - 1] + 1
    else:
        match_end = len(haystack)
    return (match_start, match_end)


def _collapse_whitespace_with_mapping(text: str) -> tuple[str, list[int]]:
    """Return whitespace-collapsed ``text`` plus index map back to original."""
    out_chars: list[str] = []
    mapping: list[int] = []
    in_ws = False
    for i, ch in enumerate(text):
        if ch.isspace():
            if in_ws:
                continue
            in_ws = True
            out_chars.append(" ")
            mapping.append(i)
            continue
        in_ws = False
        out_chars.append(ch)
        mapping.append(i)
    return "".join(out_chars), mapping


_MAX_INTER_TOKEN_GAP = 2


def _find_token_sequence(
    haystack: str, needle: str, start: int,
) -> tuple[int, int]:
    """Walk needle tokens through haystack, tolerating short intervening tokens.

    Matches needle tokens in order, allowing up to ``_MAX_INTER_TOKEN_GAP``
    haystack tokens (e.g. an inserted ``"uh"``) between consecutive matches.
    Returns ``(first_token_start, last_token_end)`` or ``(-1, -1)``.
    """
    needle_tokens = [
        m.group(0)
        for m in _WORD_RE.finditer(needle.casefold())
    ]
    if not needle_tokens:
        return (-1, -1)
    haystack_matches = [
        m for m in _WORD_RE.finditer(haystack.casefold()) if m.start() >= start
    ]
    if len(haystack_matches) < len(needle_tokens):
        return (-1, -1)
    for first in range(len(haystack_matches)):
        if haystack_matches[first].group(0) != needle_tokens[0]:
            continue
        cursor = first + 1
        last_matched = first
        ok = True
        for next_token in needle_tokens[1:]:
            search_end = min(cursor + _MAX_INTER_TOKEN_GAP + 1, len(haystack_matches))
            found = None
            for j in range(cursor, search_end):
                if haystack_matches[j].group(0) == next_token:
                    found = j
                    break
            if found is None:
                ok = False
                break
            last_matched = found
            cursor = found + 1
        if ok:
            return (
                haystack_matches[first].start(),
                haystack_matches[last_matched].end(),
            )
    return (-1, -1)


def localize_tts_edits(
    semantic_verbatim_text: str,
    tts_text: str,
) -> list[TranscriptEdit]:
    """Localize the TTS edit layer from final semantic and TTS forms.

    Edits are derived from token-level opcodes, with small replacements such as
    ``is`` -> ``is, um,`` refined into pure insertions when possible. Returned
    edits use ``spoken_form`` as a placeholder type; callers can refine the
    type with ``classify_tts_edit``.
    """
    return _localize_token_edits(
        semantic_verbatim_text,
        tts_text,
        edit_type="spoken_form",
        refine_prefix_insertions=True,
    )


def localize_semantic_edits(
    canonical_text: str,
    semantic_verbatim_text: str,
    hints: Iterable[SemanticHint | dict[str, Any]] | None,
) -> list[TranscriptEdit]:
    """Localize semantic edit hints against canonical and semantic text forms.

    Phrase ambiguity is resolved by first occurrence in left-to-right reading
    order. If hints are missing or incomplete, a reconstructing token diff is
    still returned and unlabeled regions default to ``reformulation``.
    """
    edits = _localize_token_edits(
        canonical_text,
        semantic_verbatim_text,
        edit_type="reformulation",
        refine_prefix_insertions=False,
    )
    hint_items = [_coerce_hint(h) for h in hints or []]
    hint_items = [h for h in hint_items if h is not None]
    if not edits or not hint_items:
        return edits

    generic_edits = list(edits)
    for hint in hint_items:
        if hint.type == "repair":
            replacement = _repair_edit_from_hint(
                canonical_text, semantic_verbatim_text, hint,
            )
            if replacement is not None:
                candidate = _replace_overlapping_edits(edits, replacement)
                if apply_edit_layer(canonical_text, candidate) == semantic_verbatim_text:
                    edits = candidate
                    continue
            edits = _label_first_overlapping_edit(
                edits, semantic_verbatim_text, hint, hint.type,
            )
            continue

        edits = _label_first_overlapping_edit(
            edits, semantic_verbatim_text, hint, hint.type,
        )

    if apply_edit_layer(canonical_text, edits) != semantic_verbatim_text:
        logger.warning(
            "Semantic hint localization did not reconstruct; using unlabeled diff",
        )
        return generic_edits
    return sorted(edits, key=lambda e: (e.base_start, e.base_end, e.span_start))


def classify_tts_edit(
    edit: TranscriptEdit,
    disfluency_profile: dict[str, Any] | Any | None = None,
    *,
    base_text: str = "",
) -> TTSEditType:
    """Classify a localized TTS edit by surface form and local context."""
    if _is_word_truncation_rendering(edit):
        return "spoken_form"
    if _is_filler(edit, disfluency_profile):
        return "filler"
    if _is_repetition(edit, base_text):
        return "repetition"
    if _is_prolongation(edit):
        return "prolongation"
    return "spoken_form"


def _localize_token_edits(
    base_text: str,
    derived_text: str,
    *,
    edit_type: Literal[
        "repair",
        "false_start",
        "reformulation",
        "word_truncation",
        "filler",
        "repetition",
        "prolongation",
        "spoken_form",
    ],
    refine_prefix_insertions: bool,
) -> list[TranscriptEdit]:
    if base_text == derived_text:
        return []

    base_tokens = tokenize_for_ami(base_text)
    derived_tokens = tokenize_for_ami(derived_text)
    if not base_tokens:
        return [
            TranscriptEdit(
                type=edit_type,
                span_start=0,
                span_end=len(derived_text),
                base_start=0,
                base_end=0,
                surface=derived_text,
                original="",
            ),
        ] if derived_text else []
    if not derived_tokens:
        return [
            TranscriptEdit(
                type=edit_type,
                span_start=0,
                span_end=0,
                base_start=0,
                base_end=len(base_text),
                surface="",
                original=base_text,
            ),
        ]

    matcher = SequenceMatcher(
        a=[t[0] for t in base_tokens],
        b=[t[0] for t in derived_tokens],
        autojunk=False,
    )
    edits: list[TranscriptEdit] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag == "insert":
            edits.append(_insert_edit(base_text, derived_text, base_tokens, derived_tokens, i1, j1, j2, edit_type))
            continue
        if tag == "delete":
            edits.append(_delete_edit(base_text, derived_text, base_tokens, derived_tokens, i1, i2, j1, edit_type))
            continue

        replacement = _replace_edit(
            base_text, derived_text, base_tokens, derived_tokens, i1, i2, j1, j2, edit_type,
        )
        if refine_prefix_insertions:
            refined = _prefix_insertion_refinement(replacement)
            if refined is not None:
                edits.append(refined)
                continue
        split = _split_replace_by_alignment(
            base_text,
            derived_text,
            base_tokens,
            derived_tokens,
            i1,
            i2,
            j1,
            j2,
            edit_type,
            refine_prefix_insertions=refine_prefix_insertions,
        )
        if split is not None:
            edits.extend(split)
            continue
        edits.append(replacement)

    edits = sorted(edits, key=lambda e: (e.base_start, e.base_end, e.span_start))
    if apply_edit_layer(base_text, edits) != derived_text:
        logger.error("Token edit localization failed reconstruction; using full substitution")
        return [
            TranscriptEdit(
                type=edit_type,
                span_start=0,
                span_end=len(derived_text),
                base_start=0,
                base_end=len(base_text),
                surface=derived_text,
                original=base_text,
            ),
        ]
    return edits


def _insert_edit(
    base_text: str,
    derived_text: str,
    base_tokens: list[tuple[str, int, int]],
    derived_tokens: list[tuple[str, int, int]],
    base_index: int,
    derived_start_token: int,
    derived_end_token: int,
    edit_type: str,
) -> TranscriptEdit:
    base_pos = _insert_position(base_text, base_tokens, base_index)
    span_start, span_end = _inserted_span(derived_text, derived_tokens, derived_start_token, derived_end_token)
    return TranscriptEdit(
        type=edit_type,
        span_start=span_start,
        span_end=span_end,
        base_start=base_pos,
        base_end=base_pos,
        surface=derived_text[span_start:span_end],
        original="",
    )


def _delete_edit(
    base_text: str,
    derived_text: str,
    base_tokens: list[tuple[str, int, int]],
    derived_tokens: list[tuple[str, int, int]],
    base_start_token: int,
    base_end_token: int,
    derived_index: int,
    edit_type: str,
) -> TranscriptEdit:
    base_start, base_end = _deleted_span(base_text, base_tokens, base_start_token, base_end_token)
    span_pos = _insert_position(derived_text, derived_tokens, derived_index)
    return TranscriptEdit(
        type=edit_type,
        span_start=span_pos,
        span_end=span_pos,
        base_start=base_start,
        base_end=base_end,
        surface="",
        original=base_text[base_start:base_end],
    )


def _replace_edit(
    base_text: str,
    derived_text: str,
    base_tokens: list[tuple[str, int, int]],
    derived_tokens: list[tuple[str, int, int]],
    base_start_token: int,
    base_end_token: int,
    derived_start_token: int,
    derived_end_token: int,
    edit_type: str,
) -> TranscriptEdit:
    base_start = base_tokens[base_start_token][1]
    base_end = base_tokens[base_end_token - 1][2]
    span_start = derived_tokens[derived_start_token][1]
    span_end = derived_tokens[derived_end_token - 1][2]
    return TranscriptEdit(
        type=edit_type,
        span_start=span_start,
        span_end=span_end,
        base_start=base_start,
        base_end=base_end,
        surface=derived_text[span_start:span_end],
        original=base_text[base_start:base_end],
    )


def _split_replace_by_alignment(
    base_text: str,
    derived_text: str,
    base_tokens: list[tuple[str, int, int]],
    derived_tokens: list[tuple[str, int, int]],
    base_start_token: int,
    base_end_token: int,
    derived_start_token: int,
    derived_end_token: int,
    edit_type: str,
    *,
    refine_prefix_insertions: bool,
) -> list[TranscriptEdit] | None:
    base_keys = [_alignment_key(tok) for tok, _, _ in base_tokens[base_start_token:base_end_token]]
    derived_keys = [
        _alignment_key(tok)
        for tok, _, _ in derived_tokens[derived_start_token:derived_end_token]
    ]
    if not any(key in set(derived_keys) for key in base_keys if key):
        return None

    matcher = SequenceMatcher(a=base_keys, b=derived_keys, autojunk=False)
    split: list[TranscriptEdit] = []
    changed = False
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        abs_i1 = base_start_token + i1
        abs_i2 = base_start_token + i2
        abs_j1 = derived_start_token + j1
        abs_j2 = derived_start_token + j2
        if tag == "equal":
            for offset in range(i2 - i1):
                b_idx = abs_i1 + offset
                d_idx = abs_j1 + offset
                if base_tokens[b_idx][0] == derived_tokens[d_idx][0]:
                    continue
                split.append(
                    _replace_edit(
                        base_text,
                        derived_text,
                        base_tokens,
                        derived_tokens,
                        b_idx,
                        b_idx + 1,
                        d_idx,
                        d_idx + 1,
                        edit_type,
                    ),
                )
                changed = True
            continue
        changed = True
        if tag == "insert":
            split.append(
                _insert_edit(
                    base_text,
                    derived_text,
                    base_tokens,
                    derived_tokens,
                    abs_i1,
                    abs_j1,
                    abs_j2,
                    edit_type,
                ),
            )
            continue
        if tag == "delete":
            split.append(
                _delete_edit(
                    base_text,
                    derived_text,
                    base_tokens,
                    derived_tokens,
                    abs_i1,
                    abs_i2,
                    abs_j1,
                    edit_type,
                ),
            )
            continue
        replacement = _replace_edit(
            base_text,
            derived_text,
            base_tokens,
            derived_tokens,
            abs_i1,
            abs_i2,
            abs_j1,
            abs_j2,
            edit_type,
        )
        if refine_prefix_insertions:
            refined = _prefix_insertion_refinement(replacement)
            if refined is not None:
                split.append(refined)
                continue
        split.append(replacement)
    return split if changed else None


def _alignment_key(token: str) -> str:
    return re.sub(r"[^\w]+", "", token.lower(), flags=re.UNICODE).strip("_")


def _insert_position(
    text: str,
    tokens: list[tuple[str, int, int]],
    token_index: int,
) -> int:
    if not tokens:
        return 0
    if token_index < len(tokens):
        return tokens[token_index][1]
    return len(text)


def _inserted_span(
    text: str,
    tokens: list[tuple[str, int, int]],
    start_token: int,
    end_token: int,
) -> tuple[int, int]:
    if not tokens or start_token >= end_token:
        return (0, 0)
    if end_token < len(tokens):
        return (tokens[start_token][1], tokens[end_token][1])
    if start_token > 0:
        return (tokens[start_token - 1][2], len(text))
    return (0, len(text))


def _deleted_span(
    text: str,
    tokens: list[tuple[str, int, int]],
    start_token: int,
    end_token: int,
) -> tuple[int, int]:
    if not tokens or start_token >= end_token:
        return (0, 0)
    if end_token < len(tokens):
        return (tokens[start_token][1], tokens[end_token][1])
    if start_token > 0:
        return (tokens[start_token - 1][2], len(text))
    return (0, len(text))


def _prefix_insertion_refinement(edit: TranscriptEdit) -> TranscriptEdit | None:
    original = edit.original
    surface = edit.surface
    if not original or not surface.startswith(original) or surface == original:
        return None
    if _is_word_truncation_rendering(edit):
        return None
    inserted = surface[len(original):]
    return TranscriptEdit(
        type=edit.type,
        span_start=edit.span_start + len(original),
        span_end=edit.span_end,
        base_start=edit.base_end,
        base_end=edit.base_end,
        surface=inserted,
        original="",
    )


def _coerce_hint(hint: SemanticHint | dict[str, Any]) -> SemanticHint | None:
    if isinstance(hint, SemanticHint):
        return hint
    if isinstance(hint, dict):
        try:
            return SemanticHint.model_validate(hint)
        except ValueError:
            return None
    return None


def _repair_edit_from_hint(
    canonical_text: str,
    semantic_verbatim_text: str,
    hint: SemanticHint,
) -> TranscriptEdit | None:
    if not hint.reparandum or not hint.repair:
        return None
    reparandum_start, reparandum_end = _find_with_normalization(
        semantic_verbatim_text, hint.reparandum,
    )
    if reparandum_start < 0:
        logger.debug("Could not localize repair reparandum %r", hint.reparandum)
        return None
    repair_start, span_end = _find_with_normalization(
        semantic_verbatim_text, hint.repair, reparandum_end,
    )
    if repair_start < 0:
        repair_start, span_end = _find_with_normalization(
            semantic_verbatim_text, hint.repair,
        )
    if repair_start < 0:
        logger.debug("Could not localize repair phrase %r", hint.repair)
        return None
    base_start, base_end = _find_with_normalization(canonical_text, hint.repair)
    if base_start < 0:
        return None
    return TranscriptEdit(
        type="repair",
        span_start=reparandum_start,
        span_end=span_end,
        base_start=base_start,
        base_end=base_end,
        surface=semantic_verbatim_text[reparandum_start:span_end],
        original=canonical_text[base_start:base_end],
        reparandum=hint.reparandum,
        interregnum=hint.interregnum,
        repair=hint.repair,
    )


def _replace_overlapping_edits(
    edits: list[TranscriptEdit],
    replacement: TranscriptEdit,
) -> list[TranscriptEdit]:
    kept = [edit for edit in edits if not _overlaps_for_replacement(edit, replacement)]
    kept.append(replacement)
    return sorted(kept, key=lambda e: (e.base_start, e.base_end, e.span_start))


def _overlaps_for_replacement(edit: TranscriptEdit, replacement: TranscriptEdit) -> bool:
    base_overlap = (
        edit.base_start < replacement.base_end
        and replacement.base_start < edit.base_end
    )
    insertion_inside = (
        edit.base_start == edit.base_end
        and replacement.base_start <= edit.base_start <= replacement.base_end
    )
    span_overlap = (
        edit.span_start < replacement.span_end
        and replacement.span_start < edit.span_end
    )
    return base_overlap or insertion_inside or span_overlap


def _label_first_overlapping_edit(
    edits: list[TranscriptEdit],
    semantic_verbatim_text: str,
    hint: SemanticHint,
    edit_type: str,
) -> list[TranscriptEdit]:
    target = _hint_surface(hint)
    if not target:
        return edits
    target_start, target_end = _find_with_normalization(
        semantic_verbatim_text, target,
    )
    if target_start < 0:
        logger.debug("Could not localize semantic hint phrase %r", target)
        return edits
    for idx, edit in enumerate(edits):
        if edit.span_start < target_end and target_start < edit.span_end:
            updates: dict[str, Any] = {"type": edit_type}
            if hint.type == "repair":
                updates.update({
                    "reparandum": hint.reparandum,
                    "interregnum": hint.interregnum,
                    "repair": hint.repair,
                })
            labeled = edit.model_copy(update=updates)
            return [*edits[:idx], labeled, *edits[idx + 1:]]
        if target_start <= edit.span_start and edit.span_end <= target_end:
            labeled = edit.model_copy(update={"type": edit_type})
            return [*edits[:idx], labeled, *edits[idx + 1:]]
    return edits


def _hint_surface(hint: SemanticHint) -> str | None:
    if hint.type == "word_truncation":
        return hint.fragment
    return hint.surface or hint.reparandum or hint.repair


def _is_word_truncation_rendering(edit: TranscriptEdit) -> bool:
    original = edit.original.strip()
    surface = edit.surface.strip()
    return (
        bool(original)
        and original.endswith("-")
        and surface.endswith("--")
        and surface[:-2] == original[:-1]
    )


def _is_filler(
    edit: TranscriptEdit,
    disfluency_profile: dict[str, Any] | Any | None,
) -> bool:
    if edit.original != "":
        return False
    surface = _normalized_phrase(edit.surface)
    if not surface:
        return False
    fillers = set(_BUILTIN_FILLERS)
    profile_words: Iterable[Any] = []
    if isinstance(disfluency_profile, dict):
        profile_words = disfluency_profile.get("filler_words") or []
    elif disfluency_profile is not None:
        profile_words = getattr(disfluency_profile, "filler_words", []) or []
    fillers.update(_normalized_phrase(str(word)) for word in profile_words)
    return surface in fillers


def _is_repetition(edit: TranscriptEdit, base_text: str) -> bool:
    if edit.original != "" or not base_text:
        return False
    surface_tokens = _normalized_tokens(edit.surface)
    if not surface_tokens:
        return False
    base_tokens_raw = tokenize_for_ami(base_text)
    base_tokens = [_normalized_phrase(tok) for tok, _, _ in base_tokens_raw]
    base_tokens = [tok for tok in base_tokens if tok]
    insertion_idx = 0
    for idx, (_, start, end) in enumerate(base_tokens_raw):
        if start <= edit.base_start < end:
            insertion_idx = idx
            break
        if edit.base_start >= end:
            insertion_idx = idx + 1
    length = len(surface_tokens)
    before = base_tokens[insertion_idx - length:insertion_idx]
    after = base_tokens[insertion_idx:insertion_idx + length]
    return before == surface_tokens or after == surface_tokens


def _is_prolongation(edit: TranscriptEdit) -> bool:
    if not edit.original or not edit.surface:
        return False
    original = _normalized_phrase(edit.original)
    surface = _normalized_phrase(edit.surface)
    if not original or not surface or original == surface:
        return False
    collapsed = re.sub(r"(.)\1{2,}", r"\1", surface)
    return collapsed == original


def _normalized_phrase(text: str) -> str:
    words = _normalized_tokens(text)
    return " ".join(words)


def _normalized_tokens(text: str) -> list[str]:
    cleaned = _EDGE_PUNCT_RE.sub("", text.strip().lower())
    return [m.group(0).strip("_") for m in _WORD_RE.finditer(cleaned) if m.group(0).strip("_")]
