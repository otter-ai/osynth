"""Step-7 sub-stage: LLM-driven backchannel classifier.

Identifies which turns are backchannels (phatic, semantically void
acknowledgments) and reports them. Substantive turns are never listed by the
LLM — their absence in the response is how they're marked as non-backchannels.
The classifier emits one record per **backchannel** turn with a ``bc_kind``
tag drawn from ``{agreement, acknowledgment, continuer, reaction}``.

The schema is sparse on purpose. With ~5–10% BC rate per meeting, a dense
per-turn schema scaled output tokens linearly with turn count and reliably
truncated past ~150 turns. The sparse schema cuts output by ~10× and adds an
explicit ``scanned_turn_range`` integrity field plus a provider-side
``stop_reason`` truncation check so silent failures surface loudly.

LLM payload contract::

    {
      "scanned_turn_range": [start_inclusive, end_exclusive],
      "backchannels": [
        {"turn_index": 12, "text": "yeah totally", "bc_kind": "agreement"},
        ...
      ]
    }

Failure handling:

- Provider truncation (``response.truncated``): all turns fall back to the
  per-turn heuristic and a warning logs ``stop_reason``.
- Malformed payload (not a dict, missing fields, garbage): all turns fall
  back to the heuristic and the raw response prefix is logged.
- ``scanned_turn_range`` does not cover the full input range: turns outside
  the scanned range fall back to the heuristic; turns inside are honored.
- Per-entry validation failures (bad ``bc_kind``, out-of-range
  ``turn_index``, mismatched ``text``): that entry is dropped (the turn
  reverts to ``is_backchannel=False`` if inside scanned range, or to the
  heuristic if outside).

The pipeline never raises on classifier failure.
"""
from __future__ import annotations

import logging
import re
from typing import Any, get_args

from ..models.settings import BackchannelSettings
from ..models.tts import BackchannelKind
from ..utils.llm_json import extract_json
from ..utils.prompt import render_prompt
from ..utils.usage_tracker import tracker

logger = logging.getLogger(__name__)

_VALID_KINDS: frozenset[str] = frozenset(get_args(BackchannelKind))
_WORD_RE = re.compile(r"\S+")
_TEXT_NORMALIZE_RE = re.compile(r"[^\w\s]+", re.UNICODE)


def _tokenize_words(text: str) -> list[str]:
    return _WORD_RE.findall(text or "")


def _normalize_text(text: str) -> str:
    """Lowercase + collapse whitespace + strip punctuation for sanity-check.

    Used only to compare the LLM's echoed turn text against the canonical
    turn text — light normalization tolerates punctuation/case drift while
    still catching index-mismatch bugs.
    """
    if not text:
        return ""
    cleaned = _TEXT_NORMALIZE_RE.sub(" ", text.lower())
    return " ".join(cleaned.split())


def detect_backchannel_heuristic(
    text: str, phrases: set[str] | None,
) -> bool:
    """Legacy short-utterance match against a speaker's declared BC phrases.

    Used as a fallback when the LLM classifier fails or omits a turn.
    """
    if not phrases:
        return False
    clean = text.strip().rstrip(".!?,")
    if not clean:
        return False
    if len(clean.split()) > 5:
        return False
    return clean.lower() in phrases


def _format_turns_block(turns: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for idx, turn in enumerate(turns):
        speaker_id = str(turn.get("speaker_id") or "")
        speaker_name = str(turn.get("speaker") or speaker_id or "Unknown").strip()
        tts_text = str(turn.get("tts_text") or turn.get("text") or "").strip()
        n_words = len(_tokenize_words(tts_text))
        header = (
            f"### Turn {idx} — speaker_id={speaker_id} "
            f"name={speaker_name} n_words={n_words}"
        )
        lines.append(header)
        lines.append(f"  text: {tts_text}" if tts_text else "  (empty)")
    return "\n".join(lines)


def _format_speaker_priors(turns: list[dict[str, Any]]) -> str:
    """Collect declared backchannel_phrases by speaker_id for the prompt."""
    priors: dict[str, list[str]] = {}
    for turn in turns:
        speaker_id = str(turn.get("speaker_id") or "")
        if not speaker_id or speaker_id in priors:
            continue
        profile = turn.get("disfluency_profile") or {}
        if not isinstance(profile, dict):
            continue
        phrases_raw = profile.get("backchannel_phrases") or []
        if not isinstance(phrases_raw, list):
            continue
        phrases = [str(p).strip() for p in phrases_raw if str(p).strip()]
        if phrases:
            priors[speaker_id] = phrases
    if not priors:
        return "(no declared phrases for any speaker)"
    return "\n".join(
        f"- {speaker_id}: {', '.join(phrases)}"
        for speaker_id, phrases in priors.items()
    )


def _build_prompt(
    *,
    prompt_template: str,
    turns: list[dict[str, Any]],
) -> str:
    return render_prompt(
        prompt_template,
        {
            "{{TURNS_BLOCK}}": _format_turns_block(turns),
            "{{SPEAKER_PRIORS}}": _format_speaker_priors(turns),
        },
    )


def _parse_scanned_range(
    payload: dict[str, Any], total_turns: int,
) -> tuple[int, int] | None:
    """Validate ``scanned_turn_range`` and clamp to the input range.

    Returns ``(start, end_exclusive)`` clamped into ``[0, total_turns]``,
    or ``None`` if the field is missing or malformed (payload invalid).
    """
    raw = payload.get("scanned_turn_range")
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return None
    try:
        start = int(raw[0])
        end = int(raw[1])
    except (TypeError, ValueError):
        return None
    if start < 0 or end < start:
        return None
    start = min(start, total_turns)
    end = min(end, total_turns)
    return (start, end)


def _normalize_entry(
    entry: Any,
    *,
    turns: list[dict[str, Any]],
    seen: set[int],
) -> dict[str, Any] | None:
    """Validate a single ``backchannels[]`` entry against the input turns.

    Returns a normalized decision dict on success, or ``None`` if the entry
    is malformed (bad fields, bad ``bc_kind``, out-of-range ``turn_index``,
    duplicate, or text mismatch). Mutation of ``seen`` records accepted
    indices to dedupe.
    """
    if not isinstance(entry, dict):
        return None
    try:
        turn_index = int(entry["turn_index"])
        bc_kind = str(entry["bc_kind"]).strip().lower()
    except (KeyError, TypeError, ValueError):
        return None
    if bc_kind not in _VALID_KINDS:
        return None
    if turn_index < 0 or turn_index >= len(turns) or turn_index in seen:
        return None
    actual_text = str(
        turns[turn_index].get("tts_text")
        or turns[turn_index].get("text")
        or "",
    )
    echoed_text = entry.get("text")
    if isinstance(echoed_text, str) and echoed_text.strip():
        actual_norm = _normalize_text(actual_text)
        echoed_norm = _normalize_text(echoed_text)
        # Tolerate light punctuation/case drift but reject wildly different
        # text — that's a strong signal the model lost its place. We accept
        # if either side is a substring of the other (handles the common
        # case where the LLM trims trailing punctuation differently).
        if (
            actual_norm
            and echoed_norm
            and echoed_norm not in actual_norm
            and actual_norm not in echoed_norm
        ):
            logger.warning(
                "Backchannel classifier: text mismatch at turn_index=%d "
                "(expected=%r, got=%r); dropping entry",
                turn_index, actual_text[:80], echoed_text[:80],
            )
            return None
    seen.add(turn_index)
    return {
        "turn_index": turn_index,
        "is_backchannel": True,
        "bc_kind": bc_kind,
    }


def _heuristic_decision(
    turn: dict[str, Any],
) -> tuple[bool, str | None]:
    """Best-effort heuristic when the LLM can't be trusted for a turn."""
    profile = turn.get("disfluency_profile") or {}
    phrases_raw: list[Any] = []
    if isinstance(profile, dict):
        phrases_raw = profile.get("backchannel_phrases") or []
    phrases = {
        str(p).strip().lower()
        for p in phrases_raw
        if isinstance(p, str) and str(p).strip()
    }
    text = str(turn.get("tts_text") or turn.get("text") or "")
    is_bc = detect_backchannel_heuristic(text, phrases)
    return (is_bc, "acknowledgment" if is_bc else None)


def _apply_decisions(
    turns: list[dict[str, Any]],
    decisions: list[dict[str, Any] | None],
) -> list[dict[str, Any]]:
    """Write ``is_backchannel`` + ``bc_kind`` back onto each turn dict.

    ``None`` slots fall back to the per-turn heuristic. Non-``None`` slots
    with ``is_backchannel=False`` clear any prior BC label (the LLM
    deliberately judged the turn substantive).
    """
    for idx, turn in enumerate(turns):
        decision = decisions[idx]
        if decision is None:
            is_bc, kind = _heuristic_decision(turn)
            turn["is_backchannel"] = is_bc
            turn["bc_kind"] = kind if is_bc else None
        else:
            turn["is_backchannel"] = bool(decision["is_backchannel"])
            kind = decision["bc_kind"]
            turn["bc_kind"] = kind if (turn["is_backchannel"] and kind) else None
    return turns


def _payload_to_decisions(
    *,
    turns: list[dict[str, Any]],
    payload: Any,
) -> list[dict[str, Any] | None]:
    """Map a parsed sparse LLM payload onto a per-turn decision list.

    Slots inside the LLM-reported ``scanned_turn_range`` default to a
    decision of ``is_backchannel=False`` (so they bypass the heuristic).
    Slots outside the scanned range — and the entire list when the payload
    is malformed — stay ``None`` and fall back to the heuristic.
    """
    decisions: list[dict[str, Any] | None] = [None for _ in turns]
    if not isinstance(payload, dict):
        return decisions
    scanned = _parse_scanned_range(payload, len(turns))
    if scanned is None:
        return decisions
    start, end = scanned
    for idx in range(start, end):
        decisions[idx] = {
            "turn_index": idx,
            "is_backchannel": False,
            "bc_kind": None,
        }
    raw_bcs = payload.get("backchannels")
    if not isinstance(raw_bcs, list):
        return decisions
    seen: set[int] = set()
    for entry in raw_bcs:
        normalized = _normalize_entry(entry, turns=turns, seen=seen)
        if normalized is None:
            continue
        idx = normalized["turn_index"]
        if idx < start or idx >= end:
            # The LLM listed a BC outside its own scanned range — honor it
            # anyway since it's a positive signal, but log the inconsistency.
            logger.warning(
                "Backchannel classifier: BC turn_index=%d outside "
                "scanned_turn_range=[%d, %d); accepting",
                idx, start, end,
            )
        decisions[idx] = normalized
    return decisions


async def classify_backchannels_raw(
    *,
    turns: list[dict[str, Any]],
    settings: BackchannelSettings,
    prompt_template: str,
    llm_client: Any,
) -> list[dict[str, Any] | None]:
    """Run the LLM classifier and return per-turn decisions without mutation.

    Returns a list parallel to ``turns`` where each slot is the normalized
    decision dict (``turn_index``, ``is_backchannel``, ``bc_kind``) or
    ``None`` for turns the LLM didn't scan / the payload was unrecoverable
    for. On any LLM/parse failure, returns an all-``None`` list — never
    raises. Callers must invoke :func:`apply_backchannel_decisions` to
    stamp the decisions onto turns with heuristic fallback for the
    ``None`` slots.
    """
    if not turns:
        return []

    prompt = _build_prompt(prompt_template=prompt_template, turns=turns)

    payload: Any = None
    raw_text: str | None = None
    truncated = False
    try:
        response = await llm_client.create_message(
            system=prompt,
            messages=[{
                "role": "user",
                "content": (
                    "Return ONLY the JSON object of backchannel decisions "
                    "described in the system prompt."
                ),
            }],
            max_tokens=8192,
            temperature=0.0,
            reasoning_effort=settings.classifier_reasoning_effort,
        )
        tracker.track_llm_response(response)
        raw_text = response.text
        truncated = bool(getattr(response, "truncated", False))
        if truncated:
            stop_reason = getattr(response, "stop_reason", "") or "?"
            logger.warning(
                "Backchannel classifier response truncated (stop_reason=%s, "
                "%d turns); falling back to heuristic for all turns",
                stop_reason, len(turns),
            )
        else:
            payload = extract_json(raw_text)
            if not isinstance(payload, dict):
                logger.error(
                    "Backchannel classifier returned non-dict payload "
                    "(type=%s); falling back to heuristic; raw response prefix=%r",
                    type(payload).__name__, raw_text[:500],
                )
                payload = None
    except Exception:
        preview = raw_text[:500] if raw_text is not None else "<no response>"
        logger.exception(
            "Backchannel classifier LLM call failed; falling back to heuristic; raw response prefix=%r",
            preview,
        )
        payload = None

    if truncated:
        decisions: list[dict[str, Any] | None] = [None for _ in turns]
    else:
        decisions = _payload_to_decisions(turns=turns, payload=payload)

    missing = sum(1 for d in decisions if d is None)
    if missing:
        logger.info(
            "Backchannel classifier: %d/%d turns fell back to heuristic",
            missing, len(turns),
        )
    return decisions


def apply_backchannel_decisions(
    turns: list[dict[str, Any]],
    decisions: list[dict[str, Any] | None],
) -> list[dict[str, Any]]:
    """Stamp LLM decisions (with heuristic fallback) onto ``turns`` in place.

    The ``decisions`` list must be parallel to ``turns``; ``None`` entries
    fall back to :func:`detect_backchannel_heuristic` per turn.
    """
    if not turns:
        return turns
    _apply_decisions(turns, decisions)
    bc_count = sum(1 for t in turns if t.get("is_backchannel"))
    logger.info(
        "Backchannel classifier: %d/%d turns flagged is_backchannel",
        bc_count, len(turns),
    )
    return turns


async def classify_backchannels(
    *,
    turns: list[dict[str, Any]],
    settings: BackchannelSettings,
    prompt_template: str,
    llm_client: Any,
) -> list[dict[str, Any]]:
    """LLM-classify backchannels for ``turns`` and mutate them in place.

    Thin wrapper around :func:`classify_backchannels_raw` +
    :func:`apply_backchannel_decisions` for callers that don't need to
    parallelize the LLM call with other passes.
    """
    if not turns:
        return turns
    decisions = await classify_backchannels_raw(
        turns=turns,
        settings=settings,
        prompt_template=prompt_template,
        llm_client=llm_client,
    )
    return apply_backchannel_decisions(turns, decisions)


def classify_backchannels_payload(
    *,
    turns: list[dict[str, Any]],
    payload: Any,
) -> list[dict[str, Any]]:
    """Synchronous variant for tests / pre-canned LLM payloads.

    Applies the same validation + heuristic-fallback pipeline as
    :func:`classify_backchannels` but skips the LLM call.
    """
    if not turns:
        return turns
    decisions = _payload_to_decisions(turns=turns, payload=payload)
    _apply_decisions(turns, decisions)
    return turns
