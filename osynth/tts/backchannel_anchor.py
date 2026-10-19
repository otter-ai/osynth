"""Step-7 sub-stage: LLM-driven backchannel anchor placement.

For each turn flagged ``is_backchannel=True``, this module asks an LLM to pick
a word index inside the BC's **semantic anchor turn** — the most recent prior
turn that is itself substantive (not a backchannel, not a substantive
overlap). The result is attached as an :class:`OverlapAnchor` with
``mode="concurrent"`` so step 9's mixer can place the BC's onset using NFA
word alignments.

Adjacency relaxation: unlike the substantive overlap annotator (which enforces
``anchor_turn_index == turn_index - 1``), BC anchors walk back past any number
of intermediate BCs and substantive overlaps to land on the most recent
semantic turn.

Failure modes mirror the overlap annotator: a bad LLM response or a per-BC
validation failure logs and is dropped — never fails the pipeline. Dropped BCs
fall back to FTO placement at mix time.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from ..models.settings import BackchannelSettings
from ..models.tts import OverlapAnchor
from ..utils.llm_json import extract_json
from ..utils.prompt import render_prompt
from ..utils.usage_tracker import tracker

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"\S+")


def _tokenize_words(text: str) -> list[str]:
    return _WORD_RE.findall(text or "")


def _is_substantive_overlap(turn: dict[str, Any]) -> bool:
    """A non-BC turn whose audio is placed inside an earlier turn's speech."""
    if turn.get("is_backchannel"):
        return False
    return turn.get("overlap_anchor") is not None


def _semantic_anchor_index(
    turns: list[dict[str, Any]], bc_idx: int,
) -> int | None:
    """Walk backwards from ``bc_idx`` to find the most recent semantic turn.

    Skips BCs and substantive overlaps. Returns None if none exist.
    """
    for j in range(bc_idx - 1, -1, -1):
        candidate = turns[j]
        if candidate.get("is_backchannel"):
            continue
        if _is_substantive_overlap(candidate):
            continue
        return j
    return None


def _format_turns_block(turns: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for idx, turn in enumerate(turns):
        speaker_id = str(turn.get("speaker_id") or "")
        speaker_name = str(turn.get("speaker") or speaker_id or "Unknown").strip()
        is_bc = bool(turn.get("is_backchannel"))
        is_overlap = _is_substantive_overlap(turn)
        tts_text = str(turn.get("tts_text") or turn.get("text") or "")
        words = _tokenize_words(tts_text)
        header = (
            f"### Turn {idx} — speaker_id={speaker_id} name={speaker_name} "
            f"is_backchannel={'true' if is_bc else 'false'} "
            f"is_substantive_overlap={'true' if is_overlap else 'false'} "
            f"n_words={len(words)}"
        )
        lines.append(header)
        if not words:
            lines.append("  (no words)")
            continue
        word_chunks = [f"{i}:{w}" for i, w in enumerate(words)]
        for start in range(0, len(word_chunks), 10):
            lines.append("  " + " ".join(word_chunks[start : start + 10]))
    return "\n".join(lines)


def _build_prompt(
    *,
    prompt_template: str,
    turns: list[dict[str, Any]],
) -> str:
    return render_prompt(
        prompt_template,
        {"{{TURNS_BLOCK}}": _format_turns_block(turns)},
    )


def _normalize_entry(entry: Any) -> dict[str, Any] | None:
    if not isinstance(entry, dict):
        return None
    try:
        turn_index = int(entry["turn_index"])
        anchor_turn_index = int(entry["anchor_turn_index"])
        anchor_word_index = int(entry["anchor_word_index"])
        mode = str(entry.get("mode") or "concurrent").strip().lower()
    except (KeyError, TypeError, ValueError):
        return None
    if mode != "concurrent":
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
) -> OverlapAnchor | None:
    turn_index = proposal["turn_index"]
    anchor_idx = proposal["anchor_turn_index"]
    word_idx = proposal["anchor_word_index"]

    if turn_index <= 0 or turn_index >= len(turns):
        return None
    bc_turn = turns[turn_index]
    if not bc_turn.get("is_backchannel"):
        return None

    expected_anchor = _semantic_anchor_index(turns, turn_index)
    if expected_anchor is None or anchor_idx != expected_anchor:
        return None

    if word_idx < 0 or word_idx >= word_counts[anchor_idx]:
        return None

    self_speaker = str(bc_turn.get("speaker_id") or "")
    anchor_speaker = str(turns[anchor_idx].get("speaker_id") or "")
    if not self_speaker or not anchor_speaker or self_speaker == anchor_speaker:
        return None

    return OverlapAnchor(
        anchor_turn_index=anchor_idx,
        anchor_word_index=word_idx,
        mode="concurrent",
    )


async def anchor_backchannels(
    *,
    turns: list[dict[str, Any]],
    settings: BackchannelSettings,
    prompt_template: str,
    llm_client: Any,
) -> list[OverlapAnchor | None]:
    """Return one ``OverlapAnchor | None`` per turn (parallel to ``turns``).

    Only BC turns can receive anchors; non-BC slots are always ``None``. On
    LLM failure returns an all-``None`` list — never raises.
    """
    anchors: list[OverlapAnchor | None] = [None for _ in turns]
    bc_indices = [i for i, t in enumerate(turns) if t.get("is_backchannel")]
    if not bc_indices:
        return anchors

    word_counts = [
        len(_tokenize_words(str(t.get("tts_text") or t.get("text") or "")))
        for t in turns
    ]

    prompt = _build_prompt(prompt_template=prompt_template, turns=turns)

    raw_text: str | None = None
    try:
        response = await llm_client.create_message(
            system=prompt,
            messages=[{
                "role": "user",
                "content": (
                    "Return ONLY the JSON array of backchannel anchor records "
                    "described in the system prompt. Use [] if no backchannel "
                    "should be anchored."
                ),
            }],
            # Whole-meeting call; see overlap_annotator for why 4096 was too
            # tight for reasoning models on long meetings.
            max_tokens=16384,
            temperature=0.0,
            reasoning_effort=settings.anchor_reasoning_effort,
        )
        tracker.track_llm_response(response)
        raw_text = response.text
        payload = extract_json(raw_text)
    except Exception:
        preview = raw_text[:500] if raw_text is not None else "<no response>"
        logger.exception(
            "Backchannel anchor LLM call failed; BCs will fall back to FTO; raw response prefix=%r",
            preview,
        )
        return anchors

    if not isinstance(payload, list):
        logger.warning(
            "Backchannel anchor returned non-list payload (%s); skipping",
            type(payload).__name__,
        )
        return anchors

    proposals: list[dict[str, Any]] = []
    for entry in payload:
        normalized = _normalize_entry(entry)
        if normalized is not None:
            proposals.append(normalized)
    proposals.sort(key=lambda p: p["turn_index"])

    dropped = 0
    for proposal in proposals:
        anchor = _validate_proposal(
            proposal,
            turns=turns,
            word_counts=word_counts,
        )
        if anchor is None:
            dropped += 1
            continue
        if anchors[proposal["turn_index"]] is not None:
            dropped += 1
            continue
        anchors[proposal["turn_index"]] = anchor

    accepted_count = sum(1 for a in anchors if a is not None)
    logger.info(
        "Backchannel anchor: %d BCs, %d proposed, %d accepted, %d dropped",
        len(bc_indices), len(proposals), accepted_count, dropped,
    )
    return anchors


def anchor_backchannels_payload(
    *,
    turns: list[dict[str, Any]],
    payload: Any,
) -> list[OverlapAnchor | None]:
    """Synchronous variant for tests / pre-canned LLM payloads."""
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
    for proposal in proposals:
        anchor = _validate_proposal(
            proposal, turns=turns, word_counts=word_counts,
        )
        if anchor is None or anchors[proposal["turn_index"]] is not None:
            continue
        anchors[proposal["turn_index"]] = anchor
    return anchors


def attach_backchannel_anchors(
    turns: list[dict[str, Any]],
    anchors: list[OverlapAnchor | None],
) -> list[dict[str, Any]]:
    """Attach BC anchors as ``overlap_anchor`` on each BC turn dict."""
    for turn, anchor in zip(turns, anchors, strict=False):
        if anchor is None:
            continue
        if not turn.get("is_backchannel"):
            continue
        turn["overlap_anchor"] = anchor.model_dump(mode="json")
    return turns
