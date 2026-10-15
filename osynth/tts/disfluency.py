"""LLM-based transcript-form generation and disfluency injection.

Uses an LLM to post-process clean transcript turns into two downstream forms:
a semantic verbatim transcript for human-readable written output and a literal
TTS transcript with speakable text plus realistic disfluencies.

Supports both Anthropic and OpenAI models via the unified LLMClient.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from pydantic import ValidationError

from ..models.tts import (
    reconstructs_to,
    validate_formatted_turns,
    validate_spoken_form_rewrites,
)
from .edit_localizer import (
    classify_tts_edit,
    localize_semantic_edits,
    localize_tts_edits,
)
from .truncation_sanitizer import TruncationSanitizer
from ..utils.llm_client import LLMClient
from ..utils.usage_tracker import tracker

logger = logging.getLogger(__name__)


def _parse_rewrites_tolerant(payload: Any) -> list[Any]:
    """Validate an LLM rewrite payload, recovering from per-item failures.

    Two kinds of LLM-side errors are tolerated by dropping/scrubbing the
    offending entry rather than failing the whole batch:

    * ``semantic_hints`` entries that don't match the inventory → that field is
      cleared on the offending item.
    * Items missing required text forms (``tts_text`` /
      ``semantic_verbatim_text``) → the item is removed entirely. Downstream
      code in ``format_turns_for_speech`` already treats "no rewrite for index
      ``i``" as "leave the turn with ``canonical_text``", so dropping is a safe
      degraded-mode fallback.
    """
    try:
        return validate_spoken_form_rewrites(payload)
    except ValidationError as exc:
        if not isinstance(payload, list):
            raise
        bad_hint_indices: set[int] = set()
        drop_indices: set[int] = set()
        for err in exc.errors():
            loc = err.get("loc") or ()
            if not loc or not isinstance(loc[0], int):
                raise
            idx = loc[0]
            if "semantic_hints" in loc:
                bad_hint_indices.add(idx)
            else:
                drop_indices.add(idx)
        if not bad_hint_indices and not drop_indices:
            raise
        scrubbed = [dict(item) if isinstance(item, dict) else item for item in payload]
        for i in bad_hint_indices:
            if isinstance(scrubbed[i], dict):
                scrubbed[i]["semantic_hints"] = []
                logger.warning(
                    "Stripping semantic_hints from rewrite at position %d: "
                    "LLM emitted an invalid hint", i,
                )
        for i in sorted(drop_indices, reverse=True):
            offender = scrubbed[i]
            turn_index = (
                offender.get("index") if isinstance(offender, dict) else None
            )
            logger.warning(
                "Dropping malformed rewrite at position %d (turn index %r): "
                "downstream will fall back to canonical_text.",
                i,
                turn_index,
            )
            del scrubbed[i]
        return validate_spoken_form_rewrites(scrubbed)


async def format_turns_for_speech(
    turns: list[dict[str, Any]],
    *,
    system_prompt: str,
    model: str = "claude-sonnet-4-6",
    batch_size: int = 30,
    run_id: str = "",
    meeting_id: str = "",
    max_truncation_rate: float | None = 1.0,
    concurrency: int = 1,
    reasoning_effort: str | None = None,
) -> list[dict[str, Any]]:
    """Run an LLM pass to derive semantic-verbatim and TTS transcript forms.

    Processes turns in batches to stay within context limits. Returns a new list
    of turn dicts with ``canonical_text``, ``semantic_verbatim_text``, and
    ``tts_text`` populated. The legacy ``text`` field remains an alias of
    ``tts_text`` for older TTS code paths.

    Args:
        turns: List of dicts with at least ``text`` and ``speaker`` keys.
        system_prompt: System prompt for the LLM (loaded from config).
        model: Model to use for formatting (Anthropic or OpenAI).
        batch_size: Number of turns per LLM call.
        run_id: Run identifier; threaded into the sanitizer's seeded RNG so
            rate-cap drops are reproducible across runs.
        meeting_id: Meeting identifier; same purpose as ``run_id`` and also
            scopes the running word + truncation counters that drive the
            probabilistic rate cap.
        max_truncation_rate: Upper bound on retained ``word_truncation``
            hints per 100 spoken-form words after sanitization. ``None``
            disables the rate cap (single-letter hard delete still runs).
        concurrency: Maximum batches in flight at once. Results are applied
            in batch order regardless, so output does not depend on it.
        reasoning_effort: Forwarded to reasoning models (ignored otherwise).

    Returns:
        Copy of *turns* with the three transcript text forms populated.
    """
    client = LLMClient(model=model)
    formatted_turns = [dict(t) for t in turns]
    sanitizer = TruncationSanitizer(
        max_rate_per_100_words=(
            max_truncation_rate if max_truncation_rate is not None else 0.0
        ),
        run_id=run_id,
        meeting_id=meeting_id,
    )

    batches: list[tuple[int, int, str]] = []
    for batch_start in range(0, len(turns), batch_size):
        batch_end = min(batch_start + batch_size, len(turns))
        batch_items = []
        for i in range(batch_start, batch_end):
            canonical_text = str(
                turns[i].get("canonical_text") or turns[i]["text"]
            ).strip()
            item: dict[str, Any] = {
                "index": i,
                "speaker": turns[i].get("speaker", ""),
                "text": canonical_text,
            }
            if turns[i].get("disfluency_profile"):
                item["disfluency_profile"] = turns[i]["disfluency_profile"]
            batch_items.append(item)
        batches.append((batch_start, batch_end, json.dumps(batch_items, ensure_ascii=False)))

    # Batches are independent LLM calls, so they run concurrently; their
    # results are then applied strictly in batch order below because the
    # sanitizer's rate cap keeps running per-meeting counters (seeded RNG).
    semaphore = asyncio.Semaphore(max(1, concurrency))

    def _parse_batch(raw_text: str) -> Any:
        response_text = raw_text.strip()
        if response_text.startswith("```"):
            first_nl = response_text.index("\n")
            response_text = response_text[first_nl + 1:]
            if response_text.endswith("```"):
                response_text = response_text[:-3].strip()
        return json.loads(response_text)

    async def _format_batch(batch_start: int, batch_end: int, user_prompt: str) -> Any:
        """Call + parse one batch; retry once on malformed JSON.

        Returns the parsed payload, or ``None`` if both attempts are malformed
        (the batch's turns then keep their canonical text). One bad batch used
        to fail the whole meeting's speech prep.
        """
        for attempt in range(2):
            async with semaphore:
                logger.info(
                    "Formatting turns %d-%d/%d for transcript forms...",
                    batch_start + 1,
                    batch_end,
                    len(turns),
                )
                response = await client.create_message(
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                    max_tokens=8192,
                    temperature=0.0,
                    reasoning_effort=reasoning_effort,
                )
            tracker.track_llm_response(response)
            try:
                return _parse_batch(response.text)
            except (json.JSONDecodeError, ValueError) as exc:
                logger.warning(
                    "Formatting batch turns %d-%d returned malformed JSON "
                    "(attempt %d/2): %s",
                    batch_start + 1, batch_end, attempt + 1, exc,
                )
        return None

    payloads = await asyncio.gather(
        *(_format_batch(start, end, prompt) for start, end, prompt in batches)
    )

    for (batch_start, batch_end, _), payload in zip(batches, payloads):
        result_items = [] if payload is None else _parse_rewrites_tolerant(payload)
        result_items = [
            sanitizer.sanitize(item, turn_index=item.index)
            for item in result_items
        ]
        result_by_index = {item.index: item for item in result_items}

        for i in range(batch_start, batch_end):
            canonical_text = str(
                turns[i].get("canonical_text") or turns[i]["text"]
            ).strip()
            formatted_turns[i]["canonical_text"] = canonical_text
            formatted_turns[i]["text_original"] = canonical_text
            formatted_turns[i]["original_text"] = canonical_text
            if i in result_by_index:
                rewrite = result_by_index[i]
                semantic_text = rewrite.semantic_verbatim_text or canonical_text
                tts_text = rewrite.tts_text or canonical_text
                formatted_turns[i]["semantic_verbatim_text"] = semantic_text
                formatted_turns[i]["tts_text"] = tts_text
                formatted_turns[i]["text"] = tts_text

                semantic_edits = localize_semantic_edits(
                    canonical_text,
                    semantic_text,
                    rewrite.semantic_hints,
                )
                tts_edits = [
                    edit.model_copy(
                        update={
                            "type": classify_tts_edit(
                                edit,
                                turns[i].get("disfluency_profile"),
                                base_text=semantic_text,
                            ),
                        },
                    )
                    for edit in localize_tts_edits(semantic_text, tts_text)
                ]
                if semantic_edits and not reconstructs_to(
                    canonical_text, semantic_edits, semantic_text,
                ):
                    logger.error(
                        "Localized semantic_edit_layer for turn %d does not "
                        "reconstruct semantic_verbatim_text", i,
                    )
                    semantic_edits = []
                if tts_edits and not reconstructs_to(
                    semantic_text, tts_edits, tts_text,
                ):
                    logger.error(
                        "Localized tts_edit_layer for turn %d does not "
                        "reconstruct tts_text", i,
                    )
                    tts_edits = []
                formatted_turns[i]["semantic_edit_layer"] = [
                    e.model_dump(mode="json") for e in semantic_edits
                ]
                formatted_turns[i]["tts_edit_layer"] = [
                    e.model_dump(mode="json") for e in tts_edits
                ]
            else:
                logger.warning(
                    "LLM did not return formatted text for turn %d, keeping original", i,
                )
                formatted_turns[i]["semantic_verbatim_text"] = canonical_text
                formatted_turns[i]["tts_text"] = canonical_text
                formatted_turns[i]["text"] = canonical_text
                formatted_turns[i]["semantic_edit_layer"] = []
                formatted_turns[i]["tts_edit_layer"] = []

    return validate_formatted_turns(formatted_turns)
