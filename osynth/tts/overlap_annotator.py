"""Step-7 sub-stage: LLM-driven overlap annotation.

Picks which already-emitted turns should start while a previous speaker is still
talking, anchored to a specific word index in that earlier turn's ``tts_text``.

The annotator does *not* change transcript text, edit layers, or turn ordering;
it only attaches an optional :class:`OverlapAnchor` to each turn. Step 9's
mixer translates the anchor to an absolute timestamp using NFA word alignments.

Failure mode mirrors post-hoc jargon extraction: a bad LLM response or a turn
whose anchor would self-overlap is logged and dropped — never fails the
pipeline.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from ..models.calendar import (
    MeetingJoinSpec,
    ParticipantJoinMode,
)
from ..models.settings import OverlapSettings
from ..models.tts import OverlapAnchor
from ..utils.llm_json import extract_json
from ..utils.prompt import render_prompt
from ..utils.usage_tracker import tracker

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"\S+")


def _tokenize_words(text: str) -> list[str]:
    """Whitespace tokenization aligned with how NFA splits ``tts_text``."""
    return _WORD_RE.findall(text or "")


def _participant_join_mode(
    speaker_id: str, join_spec: MeetingJoinSpec | None,
) -> str:
    if join_spec is None:
        return ""
    info = join_spec.participant_modes.get(speaker_id)
    if info is None:
        return ""
    if isinstance(info.mode, ParticipantJoinMode):
        return info.mode.value
    return str(info.mode)


def _format_turns_block(
    turns: list[dict[str, Any]], join_spec: MeetingJoinSpec | None,
) -> str:
    lines: list[str] = []
    for idx, turn in enumerate(turns):
        speaker_id = str(turn.get("speaker_id") or "")
        speaker_name = str(turn.get("speaker") or speaker_id or "Unknown").strip()
        join_mode = _participant_join_mode(speaker_id, join_spec)
        tts_text = str(turn.get("tts_text") or turn.get("text") or "")
        words = _tokenize_words(tts_text)
        header = (
            f"### Turn {idx} — speaker_id={speaker_id} "
            f"name={speaker_name} join_mode={join_mode or 'unknown'} "
            f"n_words={len(words)}"
        )
        lines.append(header)
        if not words:
            lines.append("  (no words)")
            continue
        # 0-based numbered word list, e.g. "  0:so  1:the  2:Q3-launch".
        word_chunks = [f"{i}:{w}" for i, w in enumerate(words)]
        # Wrap to ~10 words per line for readability without misleading the
        # LLM into thinking lines are sentences.
        for start in range(0, len(word_chunks), 10):
            lines.append("  " + " ".join(word_chunks[start : start + 10]))
    return "\n".join(lines)


def _build_prompt(
    *,
    prompt_template: str,
    turns: list[dict[str, Any]],
    join_spec: MeetingJoinSpec | None,
    settings: OverlapSettings,
) -> str:
    overall_mode = ""
    if join_spec is not None:
        overall_mode = (
            join_spec.mode.value
            if hasattr(join_spec.mode, "value")
            else str(join_spec.mode)
        )
    target_per_ten = max(0, round(settings.target_rate * 10))
    full_overlap_pct = max(0, round(settings.full_overlap_rate * 100))
    return render_prompt(
        prompt_template,
        {
            "{{MEETING_MODE}}": overall_mode or "unknown",
            "{{TURN_COUNT}}": str(len(turns)),
            "{{TARGET_PER_TEN}}": str(target_per_ten),
            "{{FULL_OVERLAP_PCT}}": str(full_overlap_pct),
            "{{TURNS_BLOCK}}": _format_turns_block(turns, join_spec),
        },
    )


def _normalize_entry(entry: Any) -> dict[str, Any] | None:
    if not isinstance(entry, dict):
        return None
    try:
        turn_index = int(entry["turn_index"])
        anchor_turn_index = int(entry["anchor_turn_index"])
        anchor_word_index = int(entry["anchor_word_index"])
        mode = str(entry["mode"]).strip().lower()
    except (KeyError, TypeError, ValueError):
        return None
    if mode not in {"interrupt", "concurrent"}:
        return None
    return {
        "turn_index": turn_index,
        "anchor_turn_index": anchor_turn_index,
        "anchor_word_index": anchor_word_index,
        "mode": mode,
    }


def _validate_proposal(
    proposal: dict[str, Any],
    *,
    turns: list[dict[str, Any]],
    word_counts: list[int],
    accepted_overlap_indices: set[int],
    settings: OverlapSettings,
) -> OverlapAnchor | None:
    turn_index = proposal["turn_index"]
    anchor_idx = proposal["anchor_turn_index"]
    word_idx = proposal["anchor_word_index"]
    mode = proposal["mode"]

    if turn_index <= 0 or turn_index >= len(turns):
        return None
    # Overlap is strictly adjacent: an overlap turn must anchor on the
    # immediately previous turn. Anything further back would mean intermediate
    # turns are emitted "after" the anchor in the transcript yet "before" the
    # overlap on the wall clock — incoherent conversational ordering.
    if anchor_idx != turn_index - 1:
        return None
    if word_idx < 0 or word_idx >= word_counts[anchor_idx]:
        return None

    self_speaker = str(turns[turn_index].get("speaker_id") or "")
    anchor_speaker = str(turns[anchor_idx].get("speaker_id") or "")
    if not self_speaker or not anchor_speaker or self_speaker == anchor_speaker:
        return None

    if (
        settings.max_chain_depth <= 1
        and anchor_idx in accepted_overlap_indices
    ):
        return None

    return OverlapAnchor(
        anchor_turn_index=anchor_idx,
        anchor_word_index=word_idx,
        mode=mode,
    )


async def _propose_window(
    *,
    turns: list[dict[str, Any]],
    join_spec: MeetingJoinSpec | None,
    settings: OverlapSettings,
    prompt_template: str,
    llm_client: Any,
) -> list[dict[str, Any]]:
    """One LLM call over ``turns``; returns normalized proposals (local indices).

    Logs and returns [] on LLM/parse failure or a non-list payload.
    """
    prompt = _build_prompt(
        prompt_template=prompt_template,
        turns=turns,
        join_spec=join_spec,
        settings=settings,
    )
    raw_text: str | None = None
    try:
        response = await llm_client.create_message(
            system=prompt,
            messages=[{
                "role": "user",
                "content": (
                    "Return ONLY the JSON array of overlap annotations described "
                    "in the system prompt. Use [] if no turn should overlap."
                ),
            }],
            # On ~10k-word meetings a reasoning model can spend 4096 tokens
            # before emitting anything. Billing is per token used, so the
            # headroom is free.
            max_tokens=16384,
            temperature=0.0,
            reasoning_effort=settings.annotator_reasoning_effort,
        )
        tracker.track_llm_response(response)
        raw_text = response.text
        payload = extract_json(raw_text)
    except Exception:
        preview = raw_text[:500] if raw_text is not None else "<no response>"
        logger.exception(
            "Overlap annotator LLM call failed; falling back to no overlap; raw response prefix=%r",
            preview,
        )
        return []

    if not isinstance(payload, list):
        logger.warning(
            "Overlap annotator returned non-list payload (%s); skipping",
            type(payload).__name__,
        )
        return []
    proposals: list[dict[str, Any]] = []
    for entry in payload:
        normalized = _normalize_entry(entry)
        if normalized is not None:
            proposals.append(normalized)
    return proposals


def _window_bounds(n_turns: int, window: int) -> list[tuple[int, int, int]]:
    """``(lo, start, hi)`` per window: annotate turns ``[start, hi)``.

    Each window after the first also shows turn ``start - 1`` (``lo``) so the
    boundary turn can still anchor on its predecessor.
    """
    if window <= 0 or n_turns <= window:
        return [(0, 0, n_turns)]
    return [
        (max(0, start - 1), start, min(n_turns, start + window))
        for start in range(0, n_turns, window)
    ]


async def annotate_overlaps(
    *,
    turns: list[dict[str, Any]],
    join_spec: MeetingJoinSpec | None,
    settings: OverlapSettings,
    prompt_template: str,
    llm_client: Any,
    concurrency: int = 1,
) -> list[OverlapAnchor | None]:
    """Return one ``OverlapAnchor | None`` per turn (parallel to ``turns``).

    With ``settings.annotator_window_turns > 0`` the meeting is annotated in
    windows of that many turns (up to ``concurrency`` calls in flight): on
    long meetings a single whole-meeting call under-annotates badly (~2% vs a
    15% target on 700-turn meetings). A failed window contributes no
    overlaps; the others are kept. Never raises.
    """
    anchors: list[OverlapAnchor | None] = [None for _ in turns]
    if not settings.enabled or len(turns) < 2:
        return anchors

    word_counts = [
        len(_tokenize_words(str(t.get("tts_text") or t.get("text") or "")))
        for t in turns
    ]

    bounds = _window_bounds(len(turns), settings.annotator_window_turns)
    sem = asyncio.Semaphore(max(1, concurrency))

    async def one(lo: int, start: int, hi: int) -> list[dict[str, Any]]:
        async with sem:
            local = await _propose_window(
                turns=turns[lo:hi],
                join_spec=join_spec,
                settings=settings,
                prompt_template=prompt_template,
                llm_client=llm_client,
            )
        shifted = []
        for p in local:
            q = dict(p)
            q["turn_index"] += lo
            q["anchor_turn_index"] += lo
            if start <= q["turn_index"] < hi:
                shifted.append(q)
        return shifted

    per_window = await asyncio.gather(*(one(*b) for b in bounds))
    # Validate in global turn-index order so the chain-depth check is
    # deterministic and also holds across window boundaries.
    proposals = sorted(
        (p for window in per_window for p in window),
        key=lambda p: p["turn_index"],
    )

    accepted: set[int] = set()
    dropped = 0
    for proposal in proposals:
        anchor = _validate_proposal(
            proposal,
            turns=turns,
            word_counts=word_counts,
            accepted_overlap_indices=accepted,
            settings=settings,
        )
        if anchor is None:
            dropped += 1
            continue
        if anchors[proposal["turn_index"]] is not None:
            dropped += 1
            continue
        anchors[proposal["turn_index"]] = anchor
        accepted.add(proposal["turn_index"])

    accepted_count = sum(1 for a in anchors if a is not None)
    logger.info(
        "Overlap annotator: %d windows, %d proposed, %d accepted, %d dropped",
        len(bounds), len(proposals), accepted_count, dropped,
    )
    return anchors


def annotate_overlaps_payload(
    *,
    turns: list[dict[str, Any]],
    payload: Any,
    settings: OverlapSettings,
) -> list[OverlapAnchor | None]:
    """Synchronous variant for tests / pre-canned LLM payloads.

    Applies the same validation pipeline as :func:`annotate_overlaps` but skips
    the LLM call.
    """
    anchors: list[OverlapAnchor | None] = [None for _ in turns]
    if not isinstance(payload, list):
        return anchors
    word_counts = [
        len(_tokenize_words(str(t.get("tts_text") or t.get("text") or "")))
        for t in turns
    ]
    proposals: list[dict[str, Any]] = []
    for entry in payload:
        normalized = _normalize_entry(entry)
        if normalized is not None:
            proposals.append(normalized)
    proposals.sort(key=lambda p: p["turn_index"])
    accepted: set[int] = set()
    for proposal in proposals:
        anchor = _validate_proposal(
            proposal,
            turns=turns,
            word_counts=word_counts,
            accepted_overlap_indices=accepted,
            settings=settings,
        )
        if anchor is None or anchors[proposal["turn_index"]] is not None:
            continue
        anchors[proposal["turn_index"]] = anchor
        accepted.add(proposal["turn_index"])
    return anchors


def attach_overlap_anchors(
    turns: list[dict[str, Any]],
    anchors: list[OverlapAnchor | None],
) -> list[dict[str, Any]]:
    """Mutate ``turns`` in place to carry their resolved ``overlap_anchor``."""
    for turn, anchor in zip(turns, anchors, strict=False):
        if anchor is None:
            turn.pop("overlap_anchor", None)
        else:
            turn["overlap_anchor"] = anchor.model_dump(mode="json")
    return turns


def _json_dumps_proposals(anchors: list[OverlapAnchor | None]) -> str:
    """Helper to log accepted proposals in a stable order."""
    items = [
        {
            "turn_index": idx,
            "anchor_turn_index": anchor.anchor_turn_index,
            "anchor_word_index": anchor.anchor_word_index,
            "mode": anchor.mode,
        }
        for idx, anchor in enumerate(anchors)
        if anchor is not None
    ]
    return json.dumps(items, ensure_ascii=False)
