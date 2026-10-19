"""Per-turn entity/jargon extraction over the semantic-verbatim transcript.

Runs an LLM pass over each meeting's ``semantic_verbatim_text`` turns and
identifies spans an ASR system would need workspace context to transcribe
correctly (proper nouns, project codenames, jargon, technical terms,
numerics, dates, etc.). Output is the per-turn reference for
contextualized-words and error-analysis metrics.

This is complementary to step 6's artifact-level jargon pass: that view
is the workspace vocabulary aggregate; this one is grounded at the
turn-level surface and tied to the transcript that downstream ASR will
be measured against.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from ..models.people import ExternalPerson, NicknameUsage, OrgStructure, Person
from ..models.state import Project
from ..models.tts import EntitySpan, TurnEntities
from ..utils.llm_client import LLMClient, ReasoningEffort
from ..utils.llm_json import extract_json
from ..utils.prompt import render_prompt
from ..utils.usage_tracker import tracker

logger = logging.getLogger(__name__)


def _format_people_context(org: OrgStructure) -> str:
    lines: list[str] = []
    for person_id, person in sorted(org.people.items()):
        role = ""
        if isinstance(person, Person) and person.role and person.role.title:
            role = f" — {person.role.title}"
        elif isinstance(person, ExternalPerson):
            role = f" — external, {person.organization or 'unknown org'}"
        lines.append(f"- {person_id}: {person.name}{role}")
        if person.nickname and person.nickname_usage != NicknameUsage.NONE:
            lines.append(f"    nickname: {person.nickname}")
    if not lines:
        return "(no people)"
    return "\n".join(lines)


def _format_projects_context(projects: dict[str, Project]) -> str:
    if not projects:
        return "(no projects)"
    lines: list[str] = []
    for project_id, project in sorted(projects.items()):
        lines.append(f"- {project_id}: {project.name}")
        for term in project.specialized_terms or []:
            cat = getattr(term, "category", None) or "term"
            expansion = getattr(term, "expansion", None)
            extra = f" ({expansion})" if expansion else ""
            lines.append(f"    [{cat}] {term.surface}{extra}")
    return "\n".join(lines)


def _format_external_context(org: OrgStructure) -> str:
    orgs: set[str] = set()
    for person in org.people.values():
        if isinstance(person, ExternalPerson) and person.organization:
            orgs.add(person.organization)
    if not orgs:
        return "(none)"
    return "\n".join(f"- {name}" for name in sorted(orgs))


async def extract_meeting_entities(
    *,
    turns: list[dict[str, Any]],
    org: OrgStructure,
    projects: dict[str, Project],
    language_name: str,
    prompt_template: str,
    llm_client: LLMClient,
    batch_size: int = 30,
    reasoning_effort: ReasoningEffort | None = None,
    concurrency: int = 1,
) -> list[TurnEntities]:
    """Extract entities from a meeting's semantic-verbatim turns.

    Returns one ``TurnEntities`` per turn that has at least one span.
    Failures (parse errors, validation errors) are logged and skipped at
    the batch level; the rest of the meeting still produces entities.
    """
    if not turns:
        return []

    system_prompt = render_prompt(
        prompt_template,
        {
            "{{LANGUAGE_NAME}}": language_name,
            "{{PEOPLE_CONTEXT}}": _format_people_context(org),
            "{{PROJECTS_CONTEXT}}": _format_projects_context(projects),
            "{{EXTERNAL_CONTEXT}}": _format_external_context(org),
        },
    )

    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def _extract_batch(batch_start: int, batch_end: int, batch: list[dict]) -> list[TurnEntities]:
        results: list[TurnEntities] = []
        raw_text: str | None = None
        try:
            async with semaphore:
                response = await llm_client.create_message(
                    system=system_prompt,
                    messages=[{
                        "role": "user",
                        "content": json.dumps(batch, ensure_ascii=False),
                    }],
                    max_tokens=8192,
                    temperature=0.0,
                    reasoning_effort=reasoning_effort,
                )
            tracker.track_llm_response(response)
            raw_text = response.text
            payload = extract_json(raw_text)
            if not isinstance(payload, list):
                logger.warning(
                    "Entity pass returned non-list payload for turns %d-%d; skipping",
                    batch_start, batch_end,
                )
                return results
            for raw in payload:
                if not isinstance(raw, dict):
                    continue
                try:
                    entry = TurnEntities.model_validate(raw)
                except Exception as exc:
                    logger.warning("Skipping invalid entity entry: %s", exc)
                    continue
                if entry.spans:
                    results.append(entry)
        except Exception:
            preview = raw_text[:500] if raw_text is not None else "<no response>"
            logger.exception(
                "Entity extraction failed for batch turns %d-%d; raw response prefix=%r",
                batch_start, batch_end, preview,
            )
        return results

    jobs = []
    for batch_start in range(0, len(turns), batch_size):
        batch_end = min(batch_start + batch_size, len(turns))
        batch = [
            {
                "turn_index": idx,
                "text": str(turns[idx].get("semantic_verbatim_text") or ""),
            }
            for idx in range(batch_start, batch_end)
            if str(turns[idx].get("semantic_verbatim_text") or "").strip()
        ]
        if batch:
            jobs.append(_extract_batch(batch_start, batch_end, batch))

    # Batches are independent; gather keeps results in batch order.
    all_results: list[TurnEntities] = []
    for batch_results in await asyncio.gather(*jobs):
        all_results.extend(batch_results)
    return all_results


def filter_entities_to_surfaces(
    entities: list[TurnEntities],
    turns: list[dict[str, Any]],
) -> list[TurnEntities]:
    """Drop spans whose `surface` does not match the turn text at the offsets.

    LLMs hallucinate offsets; this is a cheap sanity check. Mismatched spans
    are dropped silently. Returns a filtered copy.
    """
    cleaned: list[TurnEntities] = []
    for entry in entities:
        if entry.turn_index < 0 or entry.turn_index >= len(turns):
            continue
        text = str(turns[entry.turn_index].get("semantic_verbatim_text") or "")
        kept_spans: list[EntitySpan] = []
        for span in entry.spans:
            if (
                0 <= span.char_start < span.char_end <= len(text)
                and text[span.char_start : span.char_end] == span.surface
            ):
                kept_spans.append(span)
        if kept_spans:
            cleaned.append(TurnEntities(turn_index=entry.turn_index, spans=kept_spans))
    return cleaned
