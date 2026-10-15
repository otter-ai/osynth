"""Step 4: Artifact generation — orchestrate generation in dependency order.

Iterates through the topological generation order from Step 3, dispatching
to per-type generators (meeting, email, document, slides). Each artifact
is generated with a scoped workspace state reflecting its transitive
dependencies.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import re
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ..models.artifacts import Document, EmailMessage, EmailThread, MeetingArtifact, SlidePresentation
from ..models.calendar import MeetingJoinMode, MeetingJoinSpec
from ..models.llm_outputs import (
    DocumentResponse,
    EmailMessageResponse,
    MeetingEventExtractionResponse,
    MeetingTranscriptResponse,
    SlidePresentationResponse,
)
from ..models.people import ExternalPerson, OrgStructure, Person
from ..models.settings import SyntheticDataSettings
from ..models.state import WorkspaceEvent, WorkspaceState, reduce
from ..models.workspace import WorkspaceConfig
from ..tts.ami_normalize import load_ami_split
from ..utils import fileio
from ..utils.context_builder import ContextBuilder, person_reference_name
from ..utils.id_validation import IdDrift, IdRegistry
from ..utils.language import LANG_CONFIG
from ..utils.llm_json import extract_json
from ..utils.llm_client import LLMClient, LLMResponse, model_uses_reasoning
from ..utils.prompt import read_prompt, render_prompt
from ..utils.prompt_trace import PromptTraceRecorder
from ..utils.text_normalize import ascii_quotes as _ascii_quotes
from ..utils.usage_tracker import tracker

logger = logging.getLogger(__name__)

MAX_RETRIES = 2

ARTIFACT_MODELS: dict[str, type[BaseModel]] = {
    "meeting": MeetingArtifact,
    "email_thread": EmailThread,
    "document": Document,
    "slides": SlidePresentation,
}


def _validate_artifact_payload(artifact_type: str, payload: Any) -> dict[str, Any]:
    model_cls = ARTIFACT_MODELS[artifact_type]
    return model_cls.model_validate(payload).model_dump(mode="json")


def _extract_trailing_json_object(text: str) -> Any:
    """Parse the last top-level JSON object in ``text``.

    The judge prompt allows free-form reasoning followed by a JSON object on
    the final line, so we walk back from the end until we find a balanced
    ``{...}`` block and return its parsed form. Returns ``None`` if no valid
    JSON object can be recovered.
    """
    try:
        return extract_json(text)
    except json.JSONDecodeError:
        pass
    if not text:
        return None
    last_close = text.rfind("}")
    if last_close == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(last_close, -1, -1):
        ch = text[i]
        # Cheap balanced-brace scan ignoring strings is fine for the judge
        # payload format (no nested objects with stray braces in strings).
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "}":
            depth += 1
        elif ch == "{":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[i : last_close + 1])
                except json.JSONDecodeError:
                    return None
    return None


# The rewrite stage is expected to hold its length (measured 0.95x), so its
# calendar floor is capped at this fraction of the transcript it was handed
# rather than at an unreachable full-slot target. The refine stage is
# deliberately allowed to compress and is gated on how much of its input
# survives, via ``duration_targeting.refine_min_retention`` instead.
_BASELINE_FLOOR_TOLERANCE = 0.05


class Step4Artifacts:
    """Generate artifacts in dependency order."""

    def __init__(
        self,
        settings: SyntheticDataSettings,
        output_root: str,
        data: dict,
        *,
        llm_client: Any | None = None,
        realism_rewrite_client: Any | None = None,
        realism_judge_client: Any | None = None,
        realism_refine_client: Any | None = None,
    ) -> None:
        self.settings = settings
        self.data = data

        self.model: str = settings.model
        self.llm = llm_client or LLMClient(model=self.model)
        self.max_concurrent: int = settings.generation.max_concurrent
        self._realism_rewrite_client = realism_rewrite_client
        self._realism_judge_client = realism_judge_client
        self._realism_refine_client = realism_refine_client
        self.realism_rewrite_settings = settings.generation.realism_rewrite
        self._realism_rewrite_stats: dict[str, int] = {
            "attempted": 0,
            "accepted": 0,
            "fallback": 0,
            "judge_calls": 0,
            "refine_calls": 0,
            "refine_fallbacks": 0,
            "refine_window_calls": 0,
            "refine_window_fallbacks": 0,
            "early_stops": 0,
        }
        self._realism_rewrite_fallback_reasons: dict[str, int] = {}
        self._realism_iterations_histogram: dict[int, int] = {}
        # Round at which the pairwise judge was first fooled (0 = baseline,
        # which step 4 does not produce; the histogram starts at 1).
        self._realism_fooled_at_round_histogram: dict[int, int] = {}

        # Duration-targeting stats. Populated when
        # ``settings.generation.duration_targeting.enabled`` is True. We track
        # initial vs. final word counts per meeting so the eventual step-11
        # validator can correlate transcript-floor compliance with audio
        # duration fill against the calendar slot.
        self.duration_targeting_settings = settings.generation.duration_targeting
        self._duration_targeting_stats: dict[str, Any] = {
            "meetings_targeted": 0,
            "initial_below_floor": 0,
            "retries_attempted": 0,
            "final_below_floor": 0,
        }
        self._duration_targeting_per_meeting: dict[str, dict[str, Any]] = {}

        # Per-call output-token budgets. Transcript generation is the largest
        # artifact-level surface (long-form dialogue + summary + outline), so
        # configs typically give it a larger budget than the other calls.
        self.budget = settings.generation.llm.step_4

        self.output_root = Path(output_root)
        self.output_dir = self.output_root / "step_4_artifacts"
        self.prompt_tracer = PromptTraceRecorder(self.output_dir / "prompt_traces")
        self._io_lock = asyncio.Lock()

        self.workspace_config: WorkspaceConfig = data["workspace_config"]
        self.org: OrgStructure = data["org"]
        self.initial_state: WorkspaceState = data["initial_state"]
        self.generation_order: list[dict] = data["generation_order"]
        self.ctx = ContextBuilder(self.workspace_config, self.org)
        self.id_registry = IdRegistry.from_org(self.org, self.initial_state.projects)
        self.id_drift = IdDrift()

        # Build specs index
        self.specs_by_id: dict[str, dict] = {}
        for key in ("meeting_specs", "email_specs", "document_specs", "slide_specs"):
            for spec in data.get(key, []):
                self.specs_by_id[spec["id"]] = spec

        # Load prompts
        self.language: str = settings.language
        code_root = Path(__file__).resolve().parent.parent
        meetings_cfg = settings.prompts.meetings
        emails_cfg = settings.prompts.emails
        documents_cfg = settings.prompts.documents
        slides_cfg = settings.prompts.slides

        self.prompts = {
            "meetings_system": read_prompt(code_root / meetings_cfg.system, self.language),
            "meetings_overview": read_prompt(code_root / meetings_cfg.overview, self.language),
            "meetings_transcript": read_prompt(code_root / meetings_cfg.transcript, self.language),
            "meetings_event_extraction": read_prompt(code_root / meetings_cfg.event_extraction, self.language),
            "meetings_anti_exposition": read_prompt(code_root / meetings_cfg.anti_exposition, self.language),
            "emails_system": read_prompt(code_root / emails_cfg.system, self.language),
            "emails_message": read_prompt(code_root / emails_cfg.message, self.language),
            "documents_system": read_prompt(code_root / documents_cfg.system, self.language),
            "documents_document": read_prompt(code_root / documents_cfg.document, self.language),
            "slides_system": read_prompt(code_root / slides_cfg.system, self.language),
            "slides_content": read_prompt(code_root / slides_cfg.content, self.language),
        }
        self._realism_rewrite_prompt = read_prompt(
            code_root / settings.prompts.transcripts.realism_rewrite, self.language,
        )
        self._realism_judge_prompt = read_prompt(
            code_root / settings.prompts.transcripts.realism_judge, self.language,
        )
        self._realism_refine_prompt = read_prompt(
            code_root / settings.prompts.transcripts.realism_refine, self.language,
        )
        lang_cfg = LANG_CONFIG.get(self.language, LANG_CONFIG["en"])
        self._language_name = str(lang_cfg["name"])
        self._code_root = code_root

        # Track generated artifacts and their events
        self.artifacts: dict[str, dict] = {}
        self.events_by_artifact: dict[str, list[dict]] = {}

    async def run(self, force: bool = False) -> dict:
        status_path = self.output_dir / "generation_status.json"
        if status_path.exists() and not force:
            status = fileio.read_json(status_path)
            if all(s == "completed" for s in status.values()):
                logger.info("Step 4 outputs exist, loading from disk.")
                return self._load_from_disk()

        # Build dependency graph
        adj: dict[str, list[str]] = defaultdict(list)
        in_degree: dict[str, int] = {}
        for entry in self.generation_order:
            nid = entry["id"]
            in_degree[nid] = 0
        for entry in self.generation_order:
            for dep in entry.get("depends_on", []):
                if dep in in_degree:
                    adj[dep].append(entry["id"])
                    in_degree[entry["id"]] += 1

        completed: set[str] = set()
        failed: set[str] = set()
        status: dict[str, str] = {e["id"]: "pending" for e in self.generation_order}

        semaphore = asyncio.Semaphore(self.max_concurrent)
        total = len(self.generation_order)

        while len(completed) + len(failed) < total:
            # Find ready nodes (all deps satisfied)
            ready = [
                nid for nid, deg in in_degree.items()
                if deg == 0 and nid not in completed and nid not in failed
            ]

            if not ready:
                # Check for blocked nodes
                remaining = total - len(completed) - len(failed)
                logger.warning("No ready nodes but %d remaining — breaking (possible cycle)", remaining)
                break

            # Sort by timestamp for determinism
            ready.sort(key=lambda nid: (
                self.specs_by_id.get(nid, {}).get("timestamp", ""),
                self.specs_by_id.get(nid, {}).get("type", ""),
                nid,
            ))

            logger.info("Generating batch of %d artifacts (%d/%d complete)",
                        len(ready), len(completed), total)

            async def gen_one(artifact_id: str) -> tuple[str, bool]:
                async with semaphore:
                    try:
                        await self._generate_artifact(artifact_id)
                        return artifact_id, True
                    except Exception:
                        logger.exception("Failed to generate %s", artifact_id)
                        return artifact_id, False

            results = await asyncio.gather(*[gen_one(nid) for nid in ready])

            for artifact_id, success in results:
                if success:
                    completed.add(artifact_id)
                    status[artifact_id] = "completed"
                    # Decrement in-degree of dependents
                    for neighbor in adj.get(artifact_id, []):
                        in_degree[neighbor] -= 1
                else:
                    failed.add(artifact_id)
                    status[artifact_id] = "failed"

            # Save progress
            fileio.write_json(self.output_dir / "generation_status.json", status)

        # Save events
        fileio.write_json(
            self.output_root / "state" / "generated_events_by_artifact.json",
            self.events_by_artifact,
        )

        self.id_drift.persist(self.output_dir / "id_drift.json")

        if self.duration_targeting_settings.enabled:
            try:
                fileio.write_json(
                    self.output_dir / "duration_targeting_stats.json",
                    {
                        **self._duration_targeting_stats,
                        "per_meeting": self._duration_targeting_per_meeting,
                    },
                )
            except Exception:
                logger.exception("Failed persisting duration_targeting_stats.json")
            logger.info(
                "Step 4 duration targeting: %d meetings, %d initially below floor, "
                "%d retries attempted, %d still below floor after retries",
                self._duration_targeting_stats["meetings_targeted"],
                self._duration_targeting_stats["initial_below_floor"],
                self._duration_targeting_stats["retries_attempted"],
                self._duration_targeting_stats["final_below_floor"],
            )

        if self.realism_rewrite_settings.enabled:
            attempted = self._realism_rewrite_stats["attempted"]
            accepted = self._realism_rewrite_stats["accepted"]
            fallback = self._realism_rewrite_stats["fallback"]
            fallback_rate = (fallback / attempted) if attempted else 0.0
            fileio.write_json(
                self.output_dir / "realism_rewrite_stats.json",
                {
                    "attempted": attempted,
                    "accepted": accepted,
                    "fallback": fallback,
                    "fallback_rate": fallback_rate,
                    "fallback_reasons": dict(self._realism_rewrite_fallback_reasons),
                    "judge_calls": self._realism_rewrite_stats["judge_calls"],
                    "judge_pairwise_calls": self._realism_rewrite_stats["judge_calls"],
                    "refine_calls": self._realism_rewrite_stats["refine_calls"],
                    "refine_fallbacks": self._realism_rewrite_stats["refine_fallbacks"],
                    "refine_window_calls": self._realism_rewrite_stats["refine_window_calls"],
                    "refine_window_fallbacks": (
                        self._realism_rewrite_stats["refine_window_fallbacks"]
                    ),
                    "early_stops": self._realism_rewrite_stats["early_stops"],
                    "iterations_run_histogram": {
                        str(k): v
                        for k, v in sorted(self._realism_iterations_histogram.items())
                    },
                    "fooled_at_round_histogram": {
                        str(k): v
                        for k, v in sorted(self._realism_fooled_at_round_histogram.items())
                    },
                },
            )
            logger.info(
                "Step 4 realism rewrite: %d attempted, %d accepted, %d fallback (%.2f%%); "
                "%d judge calls, %d refine calls, %d early stops",
                attempted, accepted, fallback, fallback_rate * 100,
                self._realism_rewrite_stats["judge_calls"],
                self._realism_rewrite_stats["refine_calls"],
                self._realism_rewrite_stats["early_stops"],
            )

        logger.info("Step 4 complete: %d generated, %d failed", len(completed), len(failed))
        return {
            "artifacts": self.artifacts,
            "generated_events_by_artifact": self.events_by_artifact,
        }

    # ------------------------------------------------------------------
    # Single artifact generation
    # ------------------------------------------------------------------

    async def _generate_artifact(self, artifact_id: str) -> None:
        spec = self.specs_by_id.get(artifact_id)
        if not spec:
            logger.warning("No spec found for %s", artifact_id)
            return

        artifact_type = spec["type"]
        scoped_state = self._build_scoped_state(artifact_id)

        if artifact_type == "meeting":
            artifact, events = await self._generate_meeting(spec, scoped_state)
        elif artifact_type == "email_thread":
            artifact, events = await self._generate_email_thread(spec, scoped_state)
        elif artifact_type == "document":
            artifact, events = await self._generate_document(spec, scoped_state)
        elif artifact_type == "slides":
            artifact, events = await self._generate_slides(spec, scoped_state)
        else:
            logger.warning("Unknown artifact type: %s", artifact_type)
            return

        self.artifacts[artifact_id] = artifact
        self.events_by_artifact[artifact_id] = events

        # Persist artifact
        self._save_artifact(artifact_id, artifact_type, artifact)

    def _build_scoped_state(self, artifact_id: str) -> WorkspaceState:
        """Build state reflecting transitive dependencies of this artifact."""
        # Find transitive deps
        visited: set[str] = set()
        stack = list(self.specs_by_id.get(artifact_id, {}).get("depends_on", []))
        while stack:
            dep_id = stack.pop()
            if dep_id in visited:
                continue
            visited.add(dep_id)
            dep_spec = self.specs_by_id.get(dep_id, {})
            stack.extend(dep_spec.get("depends_on", []))

        # Apply events from all transitive deps
        state = self.initial_state.model_copy(deep=True)
        # Sort deps by timestamp
        dep_events: list[dict] = []
        for dep_id in visited:
            for event_dict in self.events_by_artifact.get(dep_id, []):
                dep_events.append(event_dict)
        dep_events.sort(key=lambda e: e.get("timestamp", ""))

        for event_dict in dep_events:
            try:
                event = WorkspaceEvent.model_validate(event_dict)
                state = reduce(state, event)
            except Exception:
                pass  # Skip malformed events

        return state

    _TECHNICALITY_DESCRIPTIONS: dict[str, str] = {
        "low": (
            "Routine working conversation. Specialized vocabulary surfaces only "
            "when directly relevant to a topic — most of the meeting is everyday "
            "phrasing and short status exchanges."
        ),
        "mid": (
            "Working session with real depth on at least one topic. The team's "
            "project-specific vocabulary surfaces several times across the meeting, "
            "mixed naturally with ordinary phrasing and short asides."
        ),
        "deep": (
            "Technically dense meeting (e.g., a debugging walkthrough, architecture "
            "review, deep-dive between domain experts). Specialized vocabulary is "
            "prominent throughout — terms get repeated as the discussion narrows in, "
            "introduced once and shorthanded afterwards, and occasionally replaced by "
            "casual stand-ins before resurfacing. Density should feel earned by the "
            "substance of the conversation, not as a recap of jargon."
        ),
    }

    def _sample_technicality(self, meeting_id: str) -> str:
        dist = self.settings.generation.meeting_technicality_distribution or {}
        items = [(k, v) for k, v in dist.items() if v > 0]
        if not items:
            return "low"
        seed = int(hashlib.sha1(meeting_id.encode("utf-8")).hexdigest()[:12], 16)
        rng = random.Random(seed)
        levels, weights = zip(*items)
        return rng.choices(levels, weights=weights, k=1)[0]

    def _parse_wpm_range(self) -> tuple[int, int]:
        """Return ``(wpm_min, wpm_max)`` from the active language config.

        ``language.py`` exposes ``words_per_minute_range`` as a string like
        ``"120-150"``; this helper parses it once and falls back to the
        English 120-150 default if the string is malformed.
        """
        lang_cfg = LANG_CONFIG.get(self.language, LANG_CONFIG["en"])
        raw = str(lang_cfg.get("words_per_minute_range", "120-150"))
        try:
            lo_str, hi_str = raw.split("-", 1)
            lo, hi = int(lo_str.strip()), int(hi_str.strip())
            if lo <= 0 or hi <= 0 or hi < lo:
                raise ValueError
            return lo, hi
        except (ValueError, AttributeError):
            return 120, 150

    def _calendar_turn_target(self, spec: dict) -> int:
        """Target turn count for ``spec``, or 0 when duration targeting is off.

        Turns rather than words are the length control here; see
        ``DurationTargetingSettings.target_turns_per_slot_minute``.
        """
        cfg = self.duration_targeting_settings
        if not cfg.enabled or cfg.target_turns_per_slot_minute <= 0:
            return 0
        duration = int(spec.get("duration_minutes", 30) or 30)
        if duration <= 0:
            duration = 30
        return max(1, int(round(duration * cfg.target_turns_per_slot_minute)))

    def _calendar_word_budget(self, spec: dict) -> tuple[int, int, int]:
        """Compute ``(floor_words, target_words, ceiling_words)`` for ``spec``.

        Derived from ``spec.duration_minutes`` and the language WPM range.
        When ``duration_targeting`` is disabled, returns a permissive
        ``(0, mean_words, large_ceiling)`` tuple so the prompt can still
        render the target without any enforcement downstream.
        """
        duration = int(spec.get("duration_minutes", 30) or 30)
        if duration <= 0:
            duration = 30
        lo, hi = self._parse_wpm_range()
        cfg = self.duration_targeting_settings
        mean_wpm = (lo + hi) / 2.0
        if not cfg.enabled:
            return (0, int(duration * mean_wpm), duration * hi * 4)
        floor = max(0, int(duration * lo * cfg.min_words_fraction))
        target = max(floor, int(duration * mean_wpm * cfg.target_words_fraction))
        ceiling = max(target, int(duration * hi * cfg.max_words_fraction))
        return floor, target, ceiling

    @staticmethod
    def _turn_word_count(turns: list[dict[str, Any]]) -> int:
        return sum(len(str(t.get("text") or "").split()) for t in turns)

    def _project_context_for(self, spec: dict, state: WorkspaceState) -> str:
        related_project_id = spec.get("related_project_id") or None
        if related_project_id and related_project_id in state.projects:
            return self.ctx.project_summary(related_project_id, state)
        return "No related project."

    def _person_name(self, person_id: str) -> str:
        person = self.org.get_person(person_id)
        return person.name if person else person_id

    def _person_reference_name(self, person_id: str) -> str:
        person = self.org.get_person(person_id)
        return person_reference_name(person) if person else person_id

    _JOIN_MODE_GUIDANCE: dict[MeetingJoinMode, str] = {
        MeetingJoinMode.VIRTUAL: (
            "All attendees join over video. Expect occasional check-audio "
            "phrasing (e.g. 'can you hear me?'), brief connection hiccups, "
            "muted-mic moments, and remote turn-taking friction. No one is "
            "co-located, so there is no in-room side chatter."
        ),
        MeetingJoinMode.IN_PERSON: (
            "Everyone is co-located in the same room. Speakers may reference "
            "the whiteboard, the room, or each other's reactions directly. "
            "Expect brief overlapping reactions, side comments to whoever is "
            "nearby, and physical-context cues; no remote dial-in friction."
        ),
        MeetingJoinMode.HYBRID: (
            "Some attendees are in the same room; others are remote. Remote "
            "participants periodically ask the in-room group to repeat or "
            "summarize side comments. Expect occasional 'for folks remote' "
            "framing, audio adjustments for the room, and a mix of in-room "
            "overlap and remote turn-taking. Side discussion in the room must "
            "be verbalized clearly enough for the transcript."
        ),
    }

    def _format_join_context(
        self, spec_join: dict | MeetingJoinSpec | None, attendee_ids: list[str]
    ) -> str:
        if spec_join is None:
            return "Meeting modality: unspecified."
        if isinstance(spec_join, dict):
            try:
                join_spec = MeetingJoinSpec.model_validate(spec_join)
            except Exception:
                return "Meeting modality: unspecified."
        else:
            join_spec = spec_join

        lines = [f"Meeting modality: {join_spec.mode.value}"]
        if join_spec.room_name:
            lines.append(f"Room: {join_spec.room_name}")

        per_attendee: list[str] = []
        for pid in attendee_ids:
            info = join_spec.participant_modes.get(pid)
            if info is None:
                continue
            name = self._person_reference_name(pid)
            entry = f"- {pid} ({name}): {info.mode.value}"
            if info.location_hint:
                entry += f"; location={info.location_hint}"
            if info.audio_context:
                entry += f"; audio={info.audio_context}"
            per_attendee.append(entry)
        if per_attendee:
            lines.append("Per-attendee join modes:")
            lines.extend(per_attendee)

        guidance = self._JOIN_MODE_GUIDANCE.get(join_spec.mode)
        if guidance:
            lines.append(f"Behavioral guidance: {guidance}")
        if join_spec.notes:
            lines.append(f"Notes: {join_spec.notes}")
        return "\n".join(lines)

    def _format_attendee_roster(self, attendee_ids: list[str]) -> str:
        if not attendee_ids:
            return "No attendees listed."
        lines: list[str] = []
        for pid in attendee_ids:
            person = self.org.get_person(pid)
            if not person:
                lines.append(f"- {pid}: Unknown; reference_label={pid}; nickname_usage=none")
                continue
            fields = [
                f"- {pid}: {person.name}",
                f"reference_label={person_reference_name(person)}",
            ]
            if person.nickname:
                fields.append(f"nickname={person.nickname}")
            fields.append(f"nickname_usage={person.nickname_usage.value}")
            lines.append("; ".join(fields))
        return "\n".join(lines)

    def _sanitize_transcript_turns(
        self,
        meeting_id: str,
        attendee_ids: list[str],
        turns: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        attendee_set = set(attendee_ids)
        speaker_names = {pid: self._person_reference_name(pid) for pid in attendee_ids}
        sanitized: list[dict[str, Any]] = []

        for idx, turn in enumerate(turns):
            context = f"meeting:{meeting_id}/turn:{idx}"
            speaker_id = str(turn.get("speaker_id", "") or "").strip()
            text = _ascii_quotes(str(turn.get("text", "") or ""))

            if speaker_id in attendee_set:
                sanitized.append({
                    "speaker_id": speaker_id,
                    "speaker_name": speaker_names[speaker_id],
                    "text": text,
                })
                continue

            bad_fields: dict[str, str] = {}
            if speaker_id:
                bad_fields["speaker_id"] = speaker_id
            self.id_drift.record(
                kind="person",
                field_name="transcript.turn",
                context=context,
                bad=json.dumps(bad_fields, sort_keys=True) if bad_fields else "<missing speaker>",
                action="dropped_unresolved_turn",
            )

        return sanitized

    def _sanitize_meeting_person_id(
        self,
        *,
        meeting_id: str,
        field_name: str,
        context_suffix: str,
        person_id: str,
        attendee_ids: list[str],
    ) -> str:
        candidate = str(person_id or "").strip()
        if not candidate:
            return ""
        if candidate in set(attendee_ids):
            return candidate
        self.id_drift.record(
            kind="person",
            field_name=field_name,
            context=f"meeting:{meeting_id}/{context_suffix}",
            bad=candidate,
            action="set_to_empty_non_attendee_id",
        )
        return ""

    def _sanitize_meeting_person_id_list(
        self,
        *,
        meeting_id: str,
        field_name: str,
        context_suffix: str,
        person_ids: list[str],
        attendee_ids: list[str],
    ) -> list[str]:
        attendee_set = set(attendee_ids)
        valid_ids: list[str] = []
        invalid_ids: list[str] = []
        seen: set[str] = set()
        for raw_id in person_ids:
            candidate = str(raw_id or "").strip()
            if not candidate:
                continue
            if candidate in attendee_set:
                if candidate not in seen:
                    valid_ids.append(candidate)
                    seen.add(candidate)
                continue
            invalid_ids.append(candidate)
        if invalid_ids:
            self.id_drift.record(
                kind="person",
                field_name=field_name,
                context=f"meeting:{meeting_id}/{context_suffix}",
                bad=invalid_ids,
                action="dropped_non_attendee_ids",
            )
        return valid_ids

    def _sanitize_extracted_meeting_fields(
        self,
        meeting_id: str,
        attendee_ids: list[str],
        extracted: dict[str, Any],
    ) -> dict[str, Any]:
        sanitized_action_items: list[dict[str, Any]] = []
        for idx, action_item in enumerate(extracted.get("action_items", [])):
            assignee_id = self._sanitize_meeting_person_id(
                meeting_id=meeting_id,
                field_name="action_item.assignee_id",
                context_suffix=f"action_item:{idx}",
                person_id=str(action_item.get("assignee_id", "") or ""),
                attendee_ids=attendee_ids,
            )
            sanitized_action_items.append({
                **action_item,
                "assignee_id": assignee_id,
                "assignee_name": self._person_name(assignee_id) if assignee_id else "",
            })

        sanitized_decisions: list[dict[str, Any]] = []
        for idx, decision in enumerate(extracted.get("decisions", [])):
            participant_ids = self._sanitize_meeting_person_id_list(
                meeting_id=meeting_id,
                field_name="decision.participant_ids",
                context_suffix=f"decision:{idx}",
                person_ids=decision.get("participant_ids", []) or [],
                attendee_ids=attendee_ids,
            )
            sanitized_decisions.append({
                **decision,
                "participant_ids": participant_ids,
                "participant_names": [self._person_name(pid) for pid in participant_ids],
            })

        sanitized_follow_ups: list[dict[str, Any]] = []
        for idx, follow_up in enumerate(extracted.get("follow_ups", [])):
            responsible_person_id = self._sanitize_meeting_person_id(
                meeting_id=meeting_id,
                field_name="follow_up.responsible_person_id",
                context_suffix=f"follow_up:{idx}",
                person_id=str(follow_up.get("responsible_person_id", "") or ""),
                attendee_ids=attendee_ids,
            )
            sanitized_follow_ups.append({
                **follow_up,
                "responsible_person_id": responsible_person_id,
                "responsible_person_name": (
                    self._person_name(responsible_person_id) if responsible_person_id else ""
                ),
            })

        extracted["action_items"] = sanitized_action_items
        extracted["decisions"] = sanitized_decisions
        extracted["follow_ups"] = sanitized_follow_ups
        return extracted

    def _canonical_email_routing(
        self,
        participant_ids: list[str],
        sender_id: str,
        cc_ids: list[str] | None = None,
    ) -> tuple[list[str], list[str]]:
        canonical_participants = list(dict.fromkeys(participant_ids))
        canonical_cc_ids = [
            pid for pid in list(dict.fromkeys(cc_ids or []))
            if pid != sender_id and pid not in canonical_participants
        ]
        canonical_recipient_ids = [
            pid for pid in canonical_participants
            if pid != sender_id
        ]
        return canonical_recipient_ids, canonical_cc_ids

    def _record_email_routing_drift(
        self,
        *,
        thread_id: str,
        message_id: str,
        field_name: str,
        actual: str | list[str],
        expected: str | list[str],
    ) -> None:
        self._record_canonical_field_drift(
            kind="person",
            field_name=field_name,
            context=f"email:{thread_id}/{message_id}",
            actual=actual,
            expected=expected,
        )

    def _record_canonical_field_drift(
        self,
        *,
        kind: str,
        field_name: str,
        context: str,
        actual: str | list[str] | None,
        expected: str | list[str] | None,
    ) -> None:
        if actual == expected:
            return
        self.id_drift.record(
            kind=kind,
            field_name=field_name,
            context=context,
            bad=actual if actual not in (None, "") else "<missing>",
            action=f"overwritten_with_canonical={expected}",
        )

    # ------------------------------------------------------------------
    # Realism rewrite (optional)
    # ------------------------------------------------------------------

    def _get_realism_rewrite_client(self) -> Any:
        if self._realism_rewrite_client is None:
            self._realism_rewrite_client = LLMClient(
                model=self.realism_rewrite_settings.model,
            )
        return self._realism_rewrite_client

    def _get_realism_judge_client(self) -> Any:
        if self._realism_judge_client is None:
            self._realism_judge_client = LLMClient(
                model=self.realism_rewrite_settings.judge_model,
            )
        return self._realism_judge_client

    def _get_realism_refine_client(self) -> Any:
        if self._realism_refine_client is None:
            self._realism_refine_client = LLMClient(
                model=(
                    self.realism_rewrite_settings.refine_model
                    or self.realism_rewrite_settings.model
                ),
            )
        return self._realism_refine_client

    def _lookup_speaker_name(
        self, turns: list[dict[str, Any]], speaker_id: str,
    ) -> str:
        for t in turns:
            if t.get("speaker_id") == speaker_id:
                name = str(t.get("speaker_name") or "").strip()
                if name:
                    return name
        person = self.org.get_person(speaker_id)
        return person.name if person else speaker_id

    def _brief_state_context(self, summary: str) -> str:
        lines: list[str] = []
        company = getattr(self.workspace_config, "company", None)
        if company is not None:
            lines.append(f"Company: {company.name} ({company.industry})")
        if summary:
            lines.append(f"Prior summary: {summary[:600]}")
        return "\n".join(lines) or "(no additional state context)"

    def _sample_ami_reference(self, meeting_id: str) -> str:
        ami_dir_raw = self.realism_rewrite_settings.ami_reference_dir
        if not ami_dir_raw:
            raise ValueError(
                "Realism rewrite is enabled but generation.realism_rewrite.ami_reference_dir "
                "is not set.",
            )
        # Relative paths are resolved against the working directory.
        ami_dir = Path(ami_dir_raw)
        if not ami_dir.exists():
            raise FileNotFoundError(
                f"AMI reference directory not found: {ami_dir}.",
            )
        # Only draw from the `style_ref` half of the AMI split — the other
        # half is reserved for the transcript_discriminability metric so the
        # rewrite cannot leak its style anchor into the evaluator's "real"
        # sample. See `load_ami_split` for the partition rule.
        style_ids, _eval_ids = load_ami_split(ami_dir)
        if not style_ids:
            raise RuntimeError(
                f"AMI split at {ami_dir} produced an empty style_ref half.",
            )
        digest = hashlib.sha1(meeting_id.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % len(style_ids)
        ref_path = ami_dir / f"{style_ids[idx]}.txt"
        if not ref_path.exists():
            raise FileNotFoundError(
                f"AMI style_ref file missing: {ref_path}. Materialize normalized "
                f"AMI transcripts via osynth.tts.ami_normalize.normalize_ami_meeting.",
            )
        text = ref_path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        max_lines = max(1, self.realism_rewrite_settings.ami_reference_lines)
        if len(lines) <= max_lines:
            return text
        start = random.Random(int.from_bytes(digest[4:8], "big")).randint(
            0, len(lines) - max_lines,
        )
        return "\n".join(lines[start : start + max_lines])

    def _render_length_budget_clause(
        self,
        *,
        floor_words: int | None,
        current_word_count: int,
        target_words: int | None = None,
    ) -> str:
        """Build the ``{{LENGTH_BUDGET_CLAUSE}}`` insert for realism prompts.

        When duration targeting is enabled and ``floor_words`` is known,
        always renders a length clause so the realism rewrite/refine
        prompts have positive upward steering rather than only a late
        floor-breach reject. Three regions, computed from ``target_words``
        when supplied and ``floor_words`` otherwise:

        * ``current_word_count > target_words * 1.20`` (or, without
          ``target_words``, ``> floor_words * 2``) -> empty string.
          The transcript is comfortably above target; let realism do its
          job without length steering at all.
        * Mid band -> "LENGTH TARGET" clause. Names the target and floor,
          asks the rewriter to restructure rather than delete so the
          total stays within +/- 10% of target.
        * ``current_word_count <= floor_words * 1.25`` -> "LENGTH
          CONSTRAINT" clause. Stronger language; the floor-breach reject
          in ``_validate_rewrite_payload`` is the backstop, but by this
          point the rewriter has already lost most of its headroom.
        """
        cfg = self.duration_targeting_settings
        if not cfg.enabled or not floor_words or floor_words <= 0:
            return ""

        # When ``target_words`` is unknown (legacy callers / unit tests),
        # fall back to a generous ceiling derived from the floor so we
        # still render the constraint clause near the floor while not
        # over-rendering for callers that have plenty of headroom.
        if target_words and target_words > 0:
            target_ceiling = target_words * 1.20
            target_for_prompt = target_words
        else:
            target_ceiling = floor_words * 2.0
            target_for_prompt = floor_words  # best available reference

        if current_word_count > target_ceiling:
            return ""

        floor_headroom = floor_words * 1.25
        if current_word_count <= floor_headroom:
            target_clause = (
                f" against a {target_words}-word target"
                if target_words and target_words > 0
                else ""
            )
            return (
                "LENGTH CONSTRAINT: This transcript is at the minimum word count "
                f"for its scheduled duration ({current_word_count} words against "
                f"a {floor_words}-word floor{target_clause}). You MUST NOT shorten "
                "it materially. Apply structural realism (dropped threads, "
                "abandoned clauses, derailments, blurred handoffs) by *restructuring* "
                "existing content — split fluent paragraphs into messy back-and-forth, "
                "expand interruptions into 4-5 line side conversations, let "
                "speakers talk past each other — not by deleting content. The "
                "checklist instruction to permanently delete a topic thread is "
                "OVERRIDDEN by this constraint; instead leave that thread visibly "
                "unresolved or compressed into a derailing aside that retains its "
                "words. The scheduled meeting slot must be filled."
            )

        # Mid band: soft target clause. Positive upward steering so the
        # rewriter doesn't ratchet down toward the floor across iterations.
        target_window = max(1, int(target_for_prompt * 0.10))
        return (
            "LENGTH TARGET: This transcript is sized for its scheduled duration "
            f"({current_word_count} words; target {target_for_prompt}, floor "
            f"{floor_words}). Keep the total within +/- {target_window} words of "
            "the target. Apply structural realism by *restructuring* existing "
            "content (split fluent paragraphs into messy back-and-forth, expand "
            "interruptions into 4-5 line side conversations, let speakers talk "
            "past each other, leave threads dangling) rather than deleting whole "
            "topics. If the judge's reasoning calls out a topic to drop, compress "
            "it into a derailing aside that retains its words instead of removing "
            "it. Realism must come from structural mess, not from cutting word count."
        )

    def _render_realism_prompt(
        self,
        *,
        meeting_id: str,
        title: str,
        meeting_type: str,
        technicality_level: str,
        attendee_ids: list[str],
        state: WorkspaceState,
        summary: str,
        transcript_turns: list[dict[str, Any]],
        floor_words: int | None = None,
        target_words: int | None = None,
    ) -> str:
        if not attendee_ids:
            attendee_ids = sorted({
                str(t.get("speaker_id") or "")
                for t in transcript_turns
                if t.get("speaker_id")
            })

        participants_lines: list[str] = []
        for pid in attendee_ids:
            person = self.org.get_person(pid)
            if person is None:
                participants_lines.append(f"- {pid}")
                continue
            role = ""
            if isinstance(person, Person):
                role = person.role.title or ""
            elif isinstance(person, ExternalPerson):
                role = person.role_title or person.relationship_type or "external"
            participants_lines.append(
                f"- {person.name} ({pid}): {role}".rstrip(": ")
            )
        participants_context = "\n".join(participants_lines) or "(none listed)"

        project_lines: list[str] = []
        for project in (state.projects or {}).values():
            related = getattr(project, "related_meeting_ids", None) or []
            if related and meeting_id not in related:
                continue
            name = getattr(project, "name", None) or getattr(project, "id", "")
            description = getattr(project, "description", "") or ""
            project_lines.append(f"- {name}: {description}".strip())
            terms = getattr(project, "specialized_terms", None) or []
            for term in terms[:6]:
                surface = getattr(term, "surface", "") if not isinstance(term, dict) else term.get("surface", "")
                expansion = getattr(term, "expansion", "") if not isinstance(term, dict) else term.get("expansion", "")
                if surface:
                    project_lines.append(f"    * {surface}: {expansion}".rstrip(": "))
        project_context = "\n".join(project_lines) or "(no project context available)"

        state_context = self._brief_state_context(summary)
        ami_reference = self._sample_ami_reference(meeting_id)
        synthetic_json = json.dumps(
            [
                {
                    "speaker_id": str(t.get("speaker_id") or ""),
                    "speaker_name": str(t.get("speaker_name") or ""),
                    "text": str(t.get("text") or ""),
                }
                for t in transcript_turns
            ],
            ensure_ascii=False,
            indent=2,
        )

        length_budget_clause = self._render_length_budget_clause(
            floor_words=floor_words,
            target_words=target_words,
            current_word_count=sum(
                len(str(t.get("text") or "").split()) for t in transcript_turns
            ),
        )

        replacements = {
            "{{MEETING_TITLE}}": str(title or meeting_id),
            "{{MEETING_TYPE}}": str(meeting_type or ""),
            "{{TECHNICALITY_LEVEL}}": technicality_level,
            "{{PARTICIPANTS_CONTEXT}}": participants_context,
            "{{PROJECT_CONTEXT}}": project_context,
            "{{STATE_CONTEXT}}": state_context,
            "{{AMI_REFERENCE}}": ami_reference,
            "{{SYNTHETIC_TRANSCRIPT}}": synthetic_json,
            "{{LENGTH_BUDGET_CLAUSE}}": length_budget_clause,
        }
        return render_prompt(self._realism_rewrite_prompt, replacements)

    async def _realism_rewrite_meeting(
        self,
        *,
        meeting_id: str,
        title: str,
        meeting_type: str,
        technicality_level: str,
        attendee_ids: list[str],
        state: WorkspaceState,
        summary: str,
        transcript_turns: list[dict[str, Any]],
        floor_words: int | None = None,
        target_words: int | None = None,
    ) -> tuple[list[dict[str, Any]], bool, str | None]:
        """Run the initial structural realism rewrite.

        Returns ``(turns, accepted, fallback_reason)``. On any failure or
        validation rejection, ``turns`` is the original input and
        ``accepted=False`` with a short reason string. Persists the
        pre-rewrite snapshot to ``meetings/<id>/transcript_pre_rewrite.json``
        as a side effect (only the initial pass writes this; the iterative
        loop reuses it).
        """
        cfg = self.realism_rewrite_settings
        meeting_dir = self.output_dir / "meetings" / meeting_id
        fileio.makedirs(meeting_dir)
        pre_rewrite_path = meeting_dir / "transcript_pre_rewrite.json"
        try:
            fileio.write_json(pre_rewrite_path, transcript_turns)
        except Exception:
            logger.exception("Failed persisting pre-rewrite transcript for %s", meeting_id)

        self._realism_rewrite_stats["attempted"] += 1

        original_word_count = sum(
            len(str(t.get("text") or "").split()) for t in transcript_turns
        )
        if original_word_count == 0:
            return self._record_rewrite_fallback(
                transcript_turns, meeting_id, "empty_transcript",
            )

        try:
            prompt = self._render_realism_prompt(
                meeting_id=meeting_id,
                title=title,
                meeting_type=meeting_type,
                technicality_level=technicality_level,
                attendee_ids=attendee_ids,
                state=state,
                summary=summary,
                transcript_turns=transcript_turns,
                floor_words=floor_words,
                target_words=target_words,
            )
            client = self._get_realism_rewrite_client()
            response = await client.create_message(
                system=prompt,
                messages=[{
                    "role": "user",
                    "content": "Return ONLY the rewritten JSON array of turns. No commentary.",
                }],
                max_tokens=32768,
                temperature=0.7,
            )
            tracker.track_llm_response(response)
            async with self._io_lock:
                self.prompt_tracer.record(
                    label=f"realism_rewrite_{meeting_id}",
                    system_prompt=prompt,
                    user_prompt="Return ONLY the rewritten JSON array of turns. No commentary.",
                    response_text=response.text,
                    metadata={
                        "model": cfg.model,
                        "max_tokens": 16384,
                        "stop_reason": getattr(response, "stop_reason", None),
                        "output_tokens": getattr(response, "output_tokens", None),
                    },
                )
            payload = extract_json(response.text)
        except Exception:
            logger.exception("Realism rewrite LLM call failed for %s", meeting_id)
            return self._record_rewrite_fallback(
                transcript_turns, meeting_id, "llm_call_failed",
            )

        rewritten, reason = self._validate_rewrite_payload(
            payload=payload,
            previous_turns=transcript_turns,
            baseline_word_count=original_word_count,
            floor_words=floor_words,
        )
        if reason is not None:
            logger.warning(
                "Realism rewrite for %s rejected at validation (%s); falling back",
                meeting_id, reason,
            )
            return self._record_rewrite_fallback(transcript_turns, meeting_id, reason)

        delta = (
            sum(len(t["text"].split()) for t in rewritten) - original_word_count
        ) / max(1, original_word_count)
        logger.info(
            "Realism rewrite for %s: %d -> %d turns, word delta %+.2f",
            meeting_id, len(transcript_turns), len(rewritten), delta,
        )
        self._realism_rewrite_stats["accepted"] += 1
        return rewritten, True, None

    def _validate_rewrite_payload(
        self,
        *,
        payload: Any,
        previous_turns: list[dict[str, Any]],
        baseline_word_count: int,
        floor_words: int | None = None,
        stage: str = "rewrite",
        skip_length_gates: bool = False,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Validate a rewrite/refine LLM payload. Returns ``(turns, reason)``.

        On rejection, ``turns`` is the input ``previous_turns`` and ``reason``
        is a short structured code. On acceptance, ``reason`` is ``None``.
        The speaker set must match ``previous_turns`` exactly and the word
        count must stay within the configured shrink/expand bounds relative
        to ``baseline_word_count`` (typically the original pre-rewrite
        count, so that successive iterations don't drift arbitrarily).

        When ``floor_words`` is provided and ``duration_targeting`` is
        enabled, an additional length gate is enforced, and which one
        depends on ``stage``:

        * ``stage="rewrite"`` -- calendar-anchored shrink cap. The rewrite
          may shrink no further than
          ``floor_words * (1 - max_shrink_vs_calendar)``, itself capped at
          the input length so an under-filled slot is not punished twice.
          The rewrite is expected to be length-neutral (measured 0.95x).
        * ``stage="refine"`` -- retention floor, relative to the transcript
          this stage was handed rather than to the calendar. The refine
          pass compresses by design, and the calendar floor scales with
          slot length while writer volume does not, so a fraction-of-floor
          bar rejects long meetings by construction. The result must retain
          at least ``refine_min_retention`` of its input; ``floor_words``
          is not consulted.
        """
        cfg = self.realism_rewrite_settings
        if not isinstance(payload, list) or not payload:
            return previous_turns, "non_list_payload"

        previous_speakers = {
            str(t.get("speaker_id") or "") for t in previous_turns if t.get("speaker_id")
        }

        rewritten: list[dict[str, Any]] = []
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            speaker_id = str(entry.get("speaker_id") or "").strip()
            text = _ascii_quotes(str(entry.get("text") or "").strip())
            if not speaker_id or not text:
                continue
            rewritten.append({
                "speaker_id": speaker_id,
                "speaker_name": str(entry.get("speaker_name") or "").strip()
                    or self._lookup_speaker_name(previous_turns, speaker_id),
                "text": text,
            })

        if not rewritten:
            return previous_turns, "no_valid_turns"

        rewritten_speakers = {t["speaker_id"] for t in rewritten}
        if rewritten_speakers != previous_speakers:
            return previous_turns, "speaker_set_drift"

        rewritten_word_count = sum(len(t["text"].split()) for t in rewritten)
        previous_word_count = sum(
            len(str(t.get("text") or "").split()) for t in previous_turns
        )
        signed_delta = (
            rewritten_word_count - baseline_word_count
        ) / max(1, baseline_word_count)
        # ``skip_length_gates`` is set for the *first* windowed refine pass. The
        # aggregate delta bounds are calibrated for whole-transcript,
        # single-call refinement; against a windowed pass they reject work that
        # already cleared per-window validation. From iteration 2 onward the
        # gate bounds cumulative shortening against a fixed original baseline.
        if not skip_length_gates:
            if signed_delta < -cfg.max_word_count_shrink:
                return previous_turns, "word_count_shrink"
            if signed_delta > cfg.max_word_count_expand:
                return previous_turns, "word_count_expand"

        dt_cfg = self.duration_targeting_settings
        if skip_length_gates:
            # Per-window validation already bounded turn count and speakers
            # for every window, and failed windows fell back to their original
            # turns, so the assembled transcript cannot be pathological. Note
            # the retention bar is *stricter* than the delta cap above
            # (0.45*previous ~= 0.43*baseline vs 0.40*baseline), so it has to
            # be skipped too -- otherwise it rejects exactly the passes the
            # delta-cap skip was meant to save.
            return rewritten, None
        if stage == "refine":
            # The refine stage compresses on purpose. It rewrites each turn
            # shorter to match real per-turn verbosity (measured 20.4 -> 13.4
            # words/turn against a real AMI mean of 14.2), uniformly across the
            # transcript -- not by deleting threads: turn counts are preserved
            # or grow (1.05x). Gate it on how much of its own input survives,
            # which is scale-invariant, rather than on the calendar floor,
            # which is not: measured output ran 0.54 / 0.50 / 0.44 / 0.39 of
            # floor across 15 / 30 / 45 / 60 minute slots purely because the
            # floor grew with the slot.
            if dt_cfg.enabled and previous_word_count > 0:
                retention_floor = int(
                    previous_word_count * dt_cfg.refine_min_retention
                )
                if rewritten_word_count < retention_floor:
                    return previous_turns, "word_count_below_refine_retention"
        elif dt_cfg.enabled and floor_words is not None and floor_words > 0:
            calendar_floor = int(floor_words * (1 - dt_cfg.max_shrink_vs_calendar))
            # The floor is derived from the full calendar slot
            # (``slot_minutes * wpm``), which assumes participants talk for the
            # entire booked hour. Real transcripts land near 40% of slot, so a
            # transcript typically arrives *already below* this floor -- and an
            # absolute bar then rejects every faithful rewrite. This check
            # exists to stop the rewrite from *shrinking* the transcript, not to
            # punish the writer for not filling the slot, so never demand more
            # words than the input already had.
            calendar_floor = min(
                calendar_floor,
                int(baseline_word_count * (1 - _BASELINE_FLOOR_TOLERANCE)),
            )
            if rewritten_word_count < calendar_floor:
                return previous_turns, "word_count_below_calendar_floor"

        return rewritten, None

    def _record_rewrite_fallback(
        self,
        transcript_turns: list[dict[str, Any]],
        meeting_id: str,
        reason: str,
    ) -> tuple[list[dict[str, Any]], bool, str | None]:
        self._realism_rewrite_stats["fallback"] += 1
        self._realism_rewrite_fallback_reasons[reason] = (
            self._realism_rewrite_fallback_reasons.get(reason, 0) + 1
        )
        return transcript_turns, False, reason

    @staticmethod
    def _serialize_turns(turns: list[dict[str, Any]]) -> str:
        return json.dumps(
            [
                {
                    "speaker_id": str(t.get("speaker_id") or ""),
                    "speaker_name": str(t.get("speaker_name") or ""),
                    "text": str(t.get("text") or ""),
                }
                for t in turns
            ],
            ensure_ascii=False,
            indent=2,
        )

    @staticmethod
    def _blind_speakers_in_turns(
        turns: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Replace per-side speaker labels with generic Speaker N labels.

        Used only inside the judge call so that naming-distribution shortcuts
        (AMI initials, attendee real names) cannot serve as a tell. The
        mapping is order-of-appearance, stable within a single transcript.
        """
        mapping: dict[str, str] = {}
        blinded: list[dict[str, Any]] = []
        for turn in turns:
            sid = str(turn.get("speaker_id") or "").strip()
            if not sid:
                continue
            label = mapping.get(sid)
            if label is None:
                label = f"Speaker {len(mapping) + 1}"
                mapping[sid] = label
            blinded.append({
                "speaker_id": label,
                "speaker_name": label,
                "text": str(turn.get("text") or ""),
            })
        return blinded

    @staticmethod
    def _blind_speakers_in_reference(text: str) -> str:
        """Blind speaker labels in a plain-text AMI excerpt.

        AMI normalized excerpts are rendered as ``<LABEL>: <text>`` lines.
        Replace each distinct label with a generic ``Speaker N`` in order of
        first appearance. Lines without a leading ``LABEL:`` prefix are
        passed through unchanged.
        """
        mapping: dict[str, str] = {}
        out_lines: list[str] = []
        for line in text.splitlines():
            head, sep, rest = line.partition(":")
            if not sep or not head.strip() or " " in head.strip():
                out_lines.append(line)
                continue
            label = head.strip()
            generic = mapping.get(label)
            if generic is None:
                generic = f"Speaker {len(mapping) + 1}"
                mapping[label] = generic
            out_lines.append(f"{generic}:{rest}")
        return "\n".join(out_lines)

    @staticmethod
    def _trim_serialized_turns_to_lines(
        turns: list[dict[str, Any]], max_lines: int,
    ) -> str:
        """Serialize ``turns`` as ``Speaker: text`` lines, trimmed to the head."""
        lines: list[str] = []
        for turn in turns:
            name = str(turn.get("speaker_name") or turn.get("speaker_id") or "").strip() or "Speaker"
            text = str(turn.get("text") or "").strip()
            if not text:
                continue
            lines.append(f"{name}: {text}")
            if len(lines) >= max_lines:
                break
        return "\n".join(lines)

    async def _judge_rewrite(
        self,
        *,
        meeting_id: str,
        iteration: int,
        candidate_turns: list[dict[str, Any]],
        ami_reference: str,
        rng: random.Random,
    ) -> tuple[bool, str, list[str], dict[str, Any]]:
        """Pairwise A/B judge — decide which of two transcripts is synthetic.

        Returns ``(still_synthetic, reasoning, structural_issues,
        bookkeeping)``. ``still_synthetic`` is true iff the judge correctly
        identifies the candidate as synthetic. ``bookkeeping`` carries
        ``synthetic_side`` / ``verdict`` / ``fooled`` for the trace.

        On any failure the judge defaults to ``still_synthetic=True`` with
        empty reasoning so the refine pass still runs.
        """
        cfg = self.realism_rewrite_settings

        blinded_candidate_turns = self._blind_speakers_in_turns(candidate_turns)
        candidate_serialized = self._trim_serialized_turns_to_lines(
            blinded_candidate_turns,
            max(1, cfg.judge_candidate_lines),
        )
        reference_blinded = self._blind_speakers_in_reference(ami_reference)

        if rng.random() < 0.5:
            transcript_a, transcript_b = candidate_serialized, reference_blinded
            synthetic_side = "A"
        else:
            transcript_a, transcript_b = reference_blinded, candidate_serialized
            synthetic_side = "B"

        prompt = render_prompt(self._realism_judge_prompt, {
            "{{TRANSCRIPT_A}}": transcript_a,
            "{{TRANSCRIPT_B}}": transcript_b,
        })
        try:
            client = self._get_realism_judge_client()
            response = await client.create_message(
                system=prompt,
                messages=[{
                    "role": "user",
                    "content": "Output only the JSON verdict object on the final line.",
                }],
                max_tokens=32768,
                temperature=0.0,
                reasoning_effort=cfg.judge_reasoning_effort,
            )
            tracker.track_llm_response(response)
            self._realism_rewrite_stats["judge_calls"] += 1
            async with self._io_lock:
                self.prompt_tracer.record(
                    label=f"realism_judge_{meeting_id}_iter{iteration}",
                    system_prompt=prompt,
                    user_prompt="Output only the JSON verdict object on the final line.",
                    response_text=response.text,
                    metadata={
                        "model": cfg.judge_model,
                        "iteration": iteration,
                        "synthetic_side": synthetic_side,
                        "stop_reason": getattr(response, "stop_reason", None),
                        "output_tokens": getattr(response, "output_tokens", None),
                    },
                )
            payload = _extract_trailing_json_object(response.text)
        except Exception:
            logger.exception("Realism judge LLM call failed for %s iter %d", meeting_id, iteration)
            return True, "", [], {
                "synthetic_side": synthetic_side,
                "verdict": None,
                "fooled": False,
            }

        if not isinstance(payload, dict):
            return True, "", [], {
                "synthetic_side": synthetic_side,
                "verdict": None,
                "fooled": False,
            }
        verdict_raw = str(payload.get("verdict") or "").strip().upper()
        verdict = verdict_raw if verdict_raw in ("A", "B") else None
        reasoning = str(payload.get("reasoning") or "").strip()
        issues_raw = payload.get("structural_issues") or []
        structural_issues = [
            str(x).strip() for x in issues_raw if isinstance(x, (str, int))
        ] if isinstance(issues_raw, list) else []

        if verdict is None:
            # Unparseable — treat as "still synthetic" so the refiner runs,
            # but flag the lack of a verdict in bookkeeping.
            still_synthetic = True
            fooled = False
        else:
            still_synthetic = verdict == synthetic_side
            fooled = not still_synthetic
        return still_synthetic, reasoning, structural_issues, {
            "synthetic_side": synthetic_side,
            "verdict": verdict,
            "fooled": fooled,
        }

    @staticmethod
    def _render_turns_plain(turns: list[dict[str, Any]]) -> str:
        """Render turns as cheap ``Name: text`` lines for *context* only.

        Windows that the model must rewrite are serialized as JSON so the
        output shape is unambiguous; the surrounding transcript is only there
        so globally-scoped checklist items stay evaluable, so it does not need
        to carry ids or JSON punctuation.
        """
        return "\n".join(
            f"{str(t.get('speaker_name') or t.get('speaker_id') or '?')}: "
            f"{str(t.get('text') or '')}"
            for t in turns
        )

    def _render_windowed_context(
        self, turns: list[dict[str, Any]], start: int, end: int
    ) -> str:
        """Full transcript as context, with ``turns[start:end]`` marked."""
        parts: list[str] = []
        if start > 0:
            parts.append(
                "<context_before_window>\n"
                + self._render_turns_plain(turns[:start])
                + "\n</context_before_window>"
            )
        parts.append(
            "<window_to_revise>\n"
            + self._serialize_turns(turns[start:end])
            + "\n</window_to_revise>"
        )
        if end < len(turns):
            parts.append(
                "<context_after_window>\n"
                + self._render_turns_plain(turns[end:])
                + "\n</context_after_window>"
            )
        return "\n\n".join(parts)

    def _refine_window_clause(
        self,
        *,
        window_turns: list[dict[str, Any]],
        window_index: int,
        window_count: int,
    ) -> str:
        """Window scope + a two-sided per-turn length contract.

        The target is absolute (words per turn, anchored on AMI) rather than a
        fraction of the input, so it cannot drift downward across iterations
        the way an open-ended "tighten freely" licence did.
        """
        cfg = self.realism_rewrite_settings
        n = len(window_turns)
        w = self._turn_word_count(window_turns)
        wpt = cfg.refine_target_words_per_turn
        lo, hi = max(1, int(n * 0.85)), max(2, int(n * 1.25))
        return (
            f"WINDOW {window_index} OF {window_count}. The whole meeting is "
            f"below for context, but you are revising ONLY the {n} turns "
            f"inside <window_to_revise>. Return exactly that window as a JSON "
            f"array — not the surrounding context, not the whole meeting.\n\n"
            f"LENGTH: this window holds {w} words across {n} turns "
            f"({w / max(1, n):.0f} words per turn). Return it as roughly {n} "
            f"turns averaging about {wpt:.0f} words each — about {int(n * wpt)} "
            f"words in total. Real speakers say less per turn than this draft, "
            f"so tightening is expected; but {wpt:.0f} words per turn is the "
            f"target, not a ceiling to undercut, and dropping below it makes "
            f"the transcript read as clipped rather than real. You may split "
            f"or merge a few turns, but the window must come back with between "
            f"{lo} and {hi} turns and the same speakers. Compress how things "
            f"are said, not how much is covered.\n\n"
        )

    def _validate_refine_window(
        self,
        *,
        payload: Any,
        window_turns: list[dict[str, Any]],
        all_speakers: set[str],
    ) -> tuple[list[dict[str, Any]] | None, str | None]:
        """Validate one refined window against its own input.

        Local validation is the point of windowing: a window that comes back
        wrong costs that window, not the whole meeting.
        """
        if not isinstance(payload, list) or not payload:
            return None, "non_list_payload"
        revised: list[dict[str, Any]] = []
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            speaker_id = str(entry.get("speaker_id") or "").strip()
            text = _ascii_quotes(str(entry.get("text") or "").strip())
            if not text:
                continue
            if speaker_id not in all_speakers:
                return None, "speaker_not_in_meeting"
            revised.append({
                "speaker_id": speaker_id,
                "speaker_name": str(entry.get("speaker_name") or "").strip()
                    or self._lookup_speaker_name(window_turns, speaker_id),
                "text": text,
            })
        if not revised:
            return None, "no_valid_turns"
        n = len(window_turns)
        if not (max(1, int(n * 0.85)) <= len(revised) <= max(2, int(n * 1.25))):
            return None, "window_turn_count_drift"
        return revised, None

    async def _refine_llm_call(
        self, *, prompt: str, meeting_id: str, iteration: int, label: str
    ) -> Any | None:
        """One refine LLM call. Returns the parsed payload, or None on error."""
        cfg = self.realism_rewrite_settings
        user_msg = "Return ONLY the revised JSON array of turns. No commentary."
        try:
            client = self._get_realism_refine_client()
            response = await client.create_message(
                system=prompt,
                messages=[{"role": "user", "content": user_msg}],
                max_tokens=32768,
                temperature=0.7,
                reasoning_effort=cfg.refine_reasoning_effort,
            )
            tracker.track_llm_response(response)
            self._realism_rewrite_stats["refine_window_calls"] += 1
            async with self._io_lock:
                self.prompt_tracer.record(
                    label=label,
                    system_prompt=prompt,
                    user_prompt=user_msg,
                    response_text=response.text,
                    metadata={
                        "model": cfg.refine_model or cfg.model,
                        "iteration": iteration,
                        "stop_reason": getattr(response, "stop_reason", None),
                        "output_tokens": getattr(response, "output_tokens", None),
                    },
                )
            return extract_json(response.text)
        except Exception:
            logger.exception(
                "Realism refine LLM call failed for %s iter %d (%s)",
                meeting_id, iteration, label,
            )
            return None

    def _refine_window_bounds(self, n_total: int) -> list[tuple[int, int]]:
        """Consecutive window bounds, folding a runt tail into its neighbour."""
        window = self.realism_rewrite_settings.refine_window_turns
        bounds = [
            (i, min(i + window, n_total)) for i in range(0, n_total, window)
        ]
        if len(bounds) > 1 and (bounds[-1][1] - bounds[-1][0]) < window // 2:
            start = bounds[-2][0]
            bounds[-2:] = [(start, n_total)]
        return bounds

    async def _realism_refine_windowed(
        self,
        *,
        meeting_id: str,
        iteration: int,
        previous_turns: list[dict[str, Any]],
        judge_reasoning: str,
        structural_issues: list[str],
        ami_reference: str,
    ) -> tuple[list[dict[str, Any]], int, int]:
        """Refine window-by-window. Returns ``(turns, failures, n_windows)``.

        A window that fails validation falls back to its original turns, so
        the assembled transcript degrades per window rather than all at once.
        """
        bounds = self._refine_window_bounds(len(previous_turns))
        all_speakers = {
            str(t.get("speaker_id") or "") for t in previous_turns
        }
        assembled: list[dict[str, Any]] = []
        failures = 0
        for idx, (start, end) in enumerate(bounds, start=1):
            window_turns = previous_turns[start:end]
            prompt = render_prompt(self._realism_refine_prompt, {
                "{{JUDGE_REASONING}}": (
                    judge_reasoning or "(no specific reasoning provided)"
                ),
                "{{STRUCTURAL_ISSUES}}": (
                    ", ".join(structural_issues)
                    if structural_issues else "(none specified)"
                ),
                "{{REAL_REFERENCE}}": ami_reference,
                "{{PREVIOUS_REWRITE}}": self._render_windowed_context(
                    previous_turns, start, end
                ),
                "{{LENGTH_BUDGET_CLAUSE}}": self._refine_window_clause(
                    window_turns=window_turns,
                    window_index=idx,
                    window_count=len(bounds),
                ),
            })
            payload = await self._refine_llm_call(
                prompt=prompt,
                meeting_id=meeting_id,
                iteration=iteration,
                label=f"realism_refine_{meeting_id}_iter{iteration}_w{idx}",
            )
            if payload is None:
                revised, reason = None, "llm_call_failed"
            else:
                revised, reason = self._validate_refine_window(
                    payload=payload,
                    window_turns=window_turns,
                    all_speakers=all_speakers,
                )
            if reason is not None or revised is None:
                failures += 1
                self._realism_rewrite_stats["refine_window_fallbacks"] += 1
                logger.warning(
                    "Refine window %d/%d for %s iter %d rejected (%s); "
                    "keeping original window",
                    idx, len(bounds), meeting_id, iteration, reason,
                )
                assembled.extend(window_turns)
            else:
                assembled.extend(revised)
        return assembled, failures, len(bounds)

    async def _realism_refine(
        self,
        *,
        meeting_id: str,
        iteration: int,
        previous_turns: list[dict[str, Any]],
        baseline_word_count: int,
        judge_reasoning: str,
        structural_issues: list[str],
        ami_reference: str,
        floor_words: int | None = None,
        target_words: int | None = None,
    ) -> tuple[list[dict[str, Any]], bool, str | None]:
        """Run one refine pass given a judge verdict.

        Returns ``(turns, accepted, reason)`` — same contract as
        ``_realism_rewrite_meeting``. ``baseline_word_count`` should be the
        original pre-rewrite word count so successive refinements don't
        compound drift across iterations. ``floor_words`` and
        ``target_words`` propagate the calendar-anchored length floor and
        target into the refine prompt's ``{{LENGTH_BUDGET_CLAUSE}}`` insert
        and the post-call validator's floor-breach reject.
        """
        cfg = self.realism_rewrite_settings
        dt_cfg = self.duration_targeting_settings
        self._realism_rewrite_stats["refine_calls"] += 1

        # Windowed path: emit the transcript in bounded chunks rather than in
        # one call. See ``refine_window_turns`` for why -- the refiner
        # compresses because of how much it has to write, not because it runs
        # out of room to write it.
        if (
            cfg.refine_window_turns > 0
            and len(previous_turns) > cfg.refine_window_turns
        ):
            assembled, failures, n_windows = await self._realism_refine_windowed(
                meeting_id=meeting_id,
                iteration=iteration,
                previous_turns=previous_turns,
                judge_reasoning=judge_reasoning,
                structural_issues=structural_issues,
                ami_reference=ami_reference,
            )
            if failures >= n_windows:
                self._realism_rewrite_stats["refine_fallbacks"] += 1
                logger.warning(
                    "Realism refine iter %d for %s: all %d windows failed",
                    iteration, meeting_id, n_windows,
                )
                return previous_turns, False, "all_refine_windows_failed"
            revised, reason = self._validate_rewrite_payload(
                payload=assembled,
                previous_turns=previous_turns,
                baseline_word_count=baseline_word_count,
                floor_words=floor_words,
                stage="refine",
                skip_length_gates=(iteration == 1),
            )
            if reason is not None:
                self._realism_rewrite_stats["refine_fallbacks"] += 1
                logger.warning(
                    "Realism refine iter %d for %s rejected (%s) after "
                    "windowing; keeping previous rewrite",
                    iteration, meeting_id, reason,
                )
                return previous_turns, False, reason
            before = self._turn_word_count(previous_turns)
            after = self._turn_word_count(revised)
            logger.info(
                "Realism refine iter %d for %s: %d windows (%d fell back), "
                "%d -> %d words (%.2fx), %d -> %d turns",
                iteration, meeting_id, n_windows, failures, before, after,
                (after / before) if before else 0.0,
                len(previous_turns), len(revised),
            )
            return revised, True, None

        # The refine pass is *expected* to compress: it shortens each turn to
        # match real per-turn verbosity (20.4 -> 13.4 words/turn against a real
        # AMI mean of 14.2) at a stable ~0.58x. Three separate attempts to
        # prohibit that -- a general "MUST NOT shorten", a reinvestment rule,
        # and a numeric bound at the output site -- moved the mean from -45% to
        # only -30% and never let a single refinement through the old
        # delta-based gate. So do not fight it. The writer over-generates ahead
        # of the shrink instead, and the only length signal here is the
        # retention floor the validator enforces, which sits *below* the
        # natural output and therefore acts as a collapse guard rather than
        # upward pressure. It is stated relative to this call's own input, not
        # to the calendar slot: a slot-derived number is unreachable for long
        # meetings and would just teach the model to pad.
        previous_word_count = sum(
            len(str(t.get("text") or "").split()) for t in previous_turns
        )
        if dt_cfg.enabled and previous_word_count > 0:
            retention_floor = int(
                previous_word_count * dt_cfg.refine_min_retention
            )
            length_budget_clause = (
                f"LENGTH FLOOR: your revised transcript must contain at least "
                f"{retention_floor} words (the draft you were given has "
                f"{previous_word_count}). Tightening individual turns is "
                f"expected and welcome — real speech is less verbose per turn "
                f"than this draft. Losing whole topics is not: keep every turn "
                f"that carries content, and keep the speaker set unchanged. "
                f"Below {retention_floor} words the revision is discarded.\n\n"
            )
        else:
            length_budget_clause = ""
        prompt = render_prompt(self._realism_refine_prompt, {
            "{{JUDGE_REASONING}}": judge_reasoning or "(no specific reasoning provided)",
            "{{STRUCTURAL_ISSUES}}": (
                ", ".join(structural_issues) if structural_issues else "(none specified)"
            ),
            "{{REAL_REFERENCE}}": ami_reference,
            "{{PREVIOUS_REWRITE}}": self._serialize_turns(previous_turns),
            "{{LENGTH_BUDGET_CLAUSE}}": length_budget_clause,
        })
        payload = await self._refine_llm_call(
            prompt=prompt,
            meeting_id=meeting_id,
            iteration=iteration,
            label=f"realism_refine_{meeting_id}_iter{iteration}",
        )
        if payload is None:
            self._realism_rewrite_stats["refine_fallbacks"] += 1
            return previous_turns, False, "llm_call_failed"

        revised, reason = self._validate_rewrite_payload(
            payload=payload,
            previous_turns=previous_turns,
            baseline_word_count=baseline_word_count,
            floor_words=floor_words,
            stage="refine",
        )
        if reason is not None:
            self._realism_rewrite_stats["refine_fallbacks"] += 1
            logger.warning(
                "Realism refine iter %d for %s rejected (%s); keeping previous rewrite",
                iteration, meeting_id, reason,
            )
            return previous_turns, False, reason
        return revised, True, None

    async def _refine_realism_iteratively(
        self,
        *,
        meeting_id: str,
        title: str,
        meeting_type: str,
        technicality_level: str,
        attendee_ids: list[str],
        state: WorkspaceState,
        summary: str,
        transcript_turns: list[dict[str, Any]],
        floor_words: int | None = None,
        target_words: int | None = None,
    ) -> tuple[
        list[dict[str, Any]],
        bool,
        str | None,
        int,
        bool,
        list[str],
        list[dict[str, Any]],
    ]:
        """Initial rewrite + iterative judge → refine loop.

        Returns ``(turns, accepted, fallback_reason, iterations_run,
        early_stopped, judge_reasoning, trace)``. ``trace`` is a per-iteration
        log suitable for persisting to ``realism_refine_trace.json``.

        ``floor_words`` is the calendar-anchored word floor for this meeting
        (from ``duration_targeting``); ``target_words`` is the corresponding
        calendar-anchored word target. When provided, the rewrite, judge
        loop, and refine validator all consult them so structural realism
        gets positive upward steering toward the target rather than only a
        late floor-breach reject.
        """
        cfg = self.realism_rewrite_settings
        original_word_count = sum(
            len(str(t.get("text") or "").split()) for t in transcript_turns
        )

        turns, accepted, fallback_reason = await self._realism_rewrite_meeting(
            meeting_id=meeting_id,
            title=title,
            meeting_type=meeting_type,
            technicality_level=technicality_level,
            attendee_ids=attendee_ids,
            state=state,
            summary=summary,
            transcript_turns=transcript_turns,
            floor_words=floor_words,
            target_words=target_words,
        )

        trace: list[dict[str, Any]] = [{
            "iteration": 0,
            "stage": "initial_rewrite",
            "accepted": accepted,
            "fallback_reason": fallback_reason,
            "turns": turns,
        }]
        judge_reasonings: list[str] = []
        iterations_run = 0
        early_stopped = False

        if not accepted or cfg.refine_iterations <= 0 or original_word_count == 0:
            self._realism_iterations_histogram[iterations_run] = (
                self._realism_iterations_histogram.get(iterations_run, 0) + 1
            )
            return (
                turns, accepted, fallback_reason, iterations_run,
                early_stopped, judge_reasonings, trace,
            )

        # Reuse a single AMI excerpt across iterations to avoid the rewriter
        # chasing a moving style target.
        ami_reference = self._sample_ami_reference(meeting_id)

        # Meeting-deterministic RNG seeds the per-iteration A/B side flip so
        # the judge result is reproducible per (meeting_id, iteration).
        seed_int = int(hashlib.sha1(
            f"realism_judge::{meeting_id}".encode("utf-8")
        ).hexdigest()[:12], 16)
        judge_rng = random.Random(seed_int)

        for i in range(1, cfg.refine_iterations + 1):
            (
                still_synthetic, reasoning, structural_issues, bookkeeping,
            ) = await self._judge_rewrite(
                meeting_id=meeting_id,
                iteration=i,
                candidate_turns=turns,
                ami_reference=ami_reference,
                rng=judge_rng,
            )
            trace.append({
                "iteration": i,
                "stage": "judge",
                "still_synthetic": still_synthetic,
                "verdict": bookkeeping.get("verdict"),
                "synthetic_side": bookkeeping.get("synthetic_side"),
                "fooled": bookkeeping.get("fooled", False),
                "reasoning": reasoning,
                "structural_issues": structural_issues,
            })
            if reasoning:
                judge_reasonings.append(reasoning[:500])
            if not still_synthetic and cfg.early_stop_on_real_verdict:
                early_stopped = True
                self._realism_rewrite_stats["early_stops"] += 1
                self._realism_fooled_at_round_histogram[i] = (
                    self._realism_fooled_at_round_histogram.get(i, 0) + 1
                )
                logger.info(
                    "Realism judge for %s stopped iter %d (fooled at round %d)",
                    meeting_id, i, i,
                )
                break

            revised, refine_accepted, refine_reason = await self._realism_refine(
                meeting_id=meeting_id,
                iteration=i,
                previous_turns=turns,
                baseline_word_count=original_word_count,
                judge_reasoning=reasoning,
                structural_issues=structural_issues,
                ami_reference=ami_reference,
                floor_words=floor_words,
                target_words=target_words,
            )
            trace.append({
                "iteration": i,
                "stage": "refine",
                "accepted": refine_accepted,
                "fallback_reason": refine_reason,
                "turns": revised if refine_accepted else None,
            })
            iterations_run = i
            if not refine_accepted:
                # Keep the most recent good turns; do not advance the iteration.
                break
            turns = revised

        self._realism_iterations_histogram[iterations_run] = (
            self._realism_iterations_histogram.get(iterations_run, 0) + 1
        )
        return (
            turns, accepted, fallback_reason, iterations_run,
            early_stopped, judge_reasonings, trace,
        )

    # ------------------------------------------------------------------
    # Meeting generation (overview → transcript → event extraction)
    # ------------------------------------------------------------------

    @staticmethod
    def _projected_shipped_words(turn_count: int, word_count: int) -> float:
        """Words expected to survive refinement, from ``(turns, words)``.

        Uses an empirical estimate, ``5.67*turns + 0.2525*words``, to choose
        between a retry and the previous attempt based on expected length
        after refinement.
        """
        return 5.67 * turn_count + 0.2525 * word_count

    async def _generate_transcript_with_floor(
        self,
        *,
        spec: dict,
        attendee_ids: list[str],
        transcript_prompt: str,
        target_words: int,
        floor_words: int,
        target_turns: int = 0,
    ) -> tuple[list[dict[str, Any]], str, str, int, int]:
        """Call the transcript LLM and retry on under-generation.

        Returns ``(turns, summary, outline, initial_word_count,
        retries_used)``. When ``duration_targeting`` is disabled the
        function calls the LLM once and returns ``retries_used=0``.
        """
        cfg = self.duration_targeting_settings
        meeting_id = spec["id"]

        if (
            cfg.enabled
            and cfg.segmented_generation
            and cfg.segment_above_words > 0
            and floor_words * cfg.segment_overshoot > cfg.segment_above_words
        ):
            return await self._generate_transcript_segmented(
                spec=spec,
                attendee_ids=attendee_ids,
                transcript_prompt=transcript_prompt,
                floor_words=floor_words,
            )
        if (
            cfg.enabled
            and cfg.segment_above_words > 0
            and target_words > cfg.segment_above_words
        ):
            logger.info(
                "Meeting %s target_words=%d exceeds segment_above_words=%d; "
                "using unsegmented generation (set duration_targeting."
                "segmented_generation to split it; track %s if floor "
                "compliance suffers).",
                meeting_id,
                target_words,
                cfg.segment_above_words,
                "duration_targeting_stats.json",
            )
        transcript_response = MeetingTranscriptResponse.model_validate(
            await self._call_llm_json(
                transcript_prompt,
                system_key="meetings_system",
                max_tokens=self.budget.meeting_transcript,
                trace_label=f"meeting_transcript_{meeting_id}",
                expect_dict=True,
            )
        )
        turns = self._sanitize_transcript_turns(
            meeting_id,
            attendee_ids,
            [turn.model_dump(mode="json") for turn in transcript_response.turns],
        )
        summary = transcript_response.summary
        outline = transcript_response.outline
        initial_word_count = self._turn_word_count(turns)

        if not cfg.enabled or floor_words <= 0 or cfg.short_transcript_retries <= 0:
            return turns, summary, outline, initial_word_count, 0

        retries_used = 0
        current_word_count = initial_word_count
        current_turn_count = len(turns)
        turn_floor = (
            int(target_turns * cfg.min_turns_fraction) if target_turns > 0 else 0
        )
        for attempt in range(1, cfg.short_transcript_retries + 1):
            word_short = current_word_count < floor_words
            turn_short = turn_floor > 0 and current_turn_count < turn_floor
            if not word_short and not turn_short:
                break
            deficit = max(0, floor_words - current_word_count)
            turn_deficit = max(0, turn_floor - current_turn_count)
            logger.warning(
                "Transcript %s under contract (%d/%d words, %d/%d turns), "
                "retry %d/%d (deficit=%d words, %d turns)",
                meeting_id,
                current_word_count,
                floor_words,
                current_turn_count,
                turn_floor,
                attempt,
                cfg.short_transcript_retries,
                deficit,
                turn_deficit,
            )
            # Turns are the binding constraint on shipped length, so lead with
            # the turn deficit when there is one -- but never invite the model
            # to hit it by slicing the same words finer, which is what it does
            # unprompted (w/turn fell 23.5 -> 11.6 across a rising turn ask).
            parts = ["\n\nPRIOR ATTEMPT FELL SHORT."]
            if turn_short:
                parts.append(
                    f" It produced only {current_turn_count} turns against a "
                    f"target of {target_turns}. Add roughly "
                    f"{turn_deficit} MORE TURNS by covering more ground and "
                    "more genuine back-and-forth -- new exchanges, follow-up "
                    "questions, side discussions, listener reactions on their "
                    "own turns. Do NOT reach the count by splitting existing "
                    "turns into shorter fragments; keep turns at their natural "
                    "length."
                )
            if word_short:
                parts.append(
                    f" It was also {deficit} words short of the "
                    f"{floor_words}-word minimum (produced only "
                    f"{current_word_count} words). Generate the full "
                    f"{target_words}-word transcript this time."
                )
            parts.append(
                " Expand each agenda item with substantive back-and-forth, "
                "concrete details, follow-up questions, and side discussions. "
                "Do not wrap early or compress the content."
            )
            addendum = "".join(parts)
            try:
                transcript_response = MeetingTranscriptResponse.model_validate(
                    await self._call_llm_json(
                        transcript_prompt + addendum,
                        system_key="meetings_system",
                        max_tokens=self.budget.meeting_transcript,
                        trace_label=f"meeting_transcript_{meeting_id}_retry{attempt}",
                        expect_dict=True,
                    )
                )
            except Exception:
                logger.exception(
                    "Retry %d for transcript %s failed; keeping prior turns",
                    attempt,
                    meeting_id,
                )
                retries_used = attempt
                break

            retry_turns = self._sanitize_transcript_turns(
                meeting_id,
                attendee_ids,
                [turn.model_dump(mode="json") for turn in transcript_response.turns],
            )
            retry_word_count = self._turn_word_count(retry_turns)
            retries_used = attempt
            # Keep the retry only if it improves *projected shipped* length,
            # not raw word count: a retry with more turns but slightly fewer
            # words ships longer, because refinement preserves turns and
            # discards ~half the words.
            if (
                self._projected_shipped_words(len(retry_turns), retry_word_count)
                > self._projected_shipped_words(
                    current_turn_count, current_word_count
                )
            ):
                turns = retry_turns
                summary = transcript_response.summary
                outline = transcript_response.outline
                current_word_count = retry_word_count
                current_turn_count = len(retry_turns)
            else:
                logger.warning(
                    "Retry %d for transcript %s did not improve projected "
                    "shipped length (%d words/%d turns vs %d/%d); keeping "
                    "prior turns",
                    attempt,
                    meeting_id,
                    retry_word_count,
                    len(retry_turns),
                    current_word_count,
                    current_turn_count,
                )

        return turns, summary, outline, initial_word_count, retries_used

    # Turns of prior dialogue shown verbatim to each later part.
    _SEGMENT_CONTEXT_TURNS = 40
    # Natural spoken words/turn used to translate a part's word ask into a
    # turn ask.
    _SEGMENT_WORDS_PER_TURN = 15
    # A part this far under its ask gets one retry; the longer result is kept.
    _SEGMENT_RETRY_BELOW = 0.6
    # The last part absorbs any shortfall, up to this multiple of a normal part.
    _SEGMENT_LAST_PART_MAX = 1.6
    # Per-part output cap in tokens per requested word. JSON turns run ~1.6
    # tokens/word plus reasoning; the cap bounds time spent on each part.
    _SEGMENT_TOKENS_PER_WORD = 5

    @staticmethod
    def _split_agenda(agenda: list[str], n_parts: int) -> list[list[str]]:
        """Deal agenda items into ``n_parts`` contiguous groups.

        With fewer items than parts, items repeat across adjacent parts
        (a long item spans several parts) so no part is left empty.
        """
        items = [a for a in agenda if a and a.strip()]
        if not items:
            return [[] for _ in range(n_parts)]
        if len(items) >= n_parts:
            size, extra = divmod(len(items), n_parts)
            groups, start = [], 0
            for k in range(n_parts):
                end = start + size + (1 if k < extra else 0)
                groups.append(items[start:end])
                start = end
            return groups
        return [[items[k * len(items) // n_parts]] for k in range(n_parts)]

    @staticmethod
    def _outline_sections(transcript_prompt: str) -> list[str]:
        """Top-level section headings of the overview's ``<outline>``.

        Meeting specs usually carry no ``agenda``; the overview's timed outline
        (``- **2:06–2:15 — What is in scope**``) supplies the agenda for
        segmented generation, giving each part a specific topic.
        """
        start = transcript_prompt.find("<outline>")
        end = transcript_prompt.find("</outline>", start)
        if start < 0 or end < 0:
            return []
        block = transcript_prompt[start:end]
        return [
            m.group(1).strip()
            for m in re.finditer(r"^\s*[-*]\s+\*\*(.+?)\*\*", block, re.M)
        ]

    def _segment_addendum(
        self,
        *,
        part: int,
        n_parts: int,
        agenda_group: list[str],
        part_words: int,
        prior_turns: list[dict[str, Any]],
        prior_outlines: list[str],
    ) -> str:
        part_turns = max(10, round(part_words / self._SEGMENT_WORDS_PER_TURN))
        lines = [
            "",
            "",
            f"=== PARTIAL GENERATION: PART {part} OF {n_parts} ===",
            "This meeting is too long for one response, so it is being written in "
            f"{n_parts} consecutive parts. Write ONLY part {part}. The whole-meeting "
            "word and turn targets above do NOT apply to this call; for this part "
            f"write about {part_words} words in about {part_turns} turns, keeping "
            "turns at their natural lengths.",
            "",
        ]
        if agenda_group:
            lines += [
                "This part covers these sections of the meeting outline (stay on "
                "them; other sections belong to other parts):",
                *[f"- {item}" for item in agenda_group],
            ]
        else:
            lines.append(
                f"Cover roughly part {part} of {n_parts} of the meeting overview's "
                "flow, picking up where the conversation left off."
            )
        if part == 1:
            lines += [
                "",
                "Open the meeting naturally (people joining, small talk, getting "
                "started). Do NOT summarize, assign wrap-up action items, or say "
                "goodbye: the meeting continues in the next part. End mid-flow, as "
                "the conversation moves toward the next topic.",
            ]
        else:
            covered = "\n".join(f"- Part {i + 1}: {o.strip()}" for i, o in enumerate(prior_outlines) if o.strip())
            context = "\n".join(
                f"{t['speaker_name']} ({t['speaker_id']}): {t['text']}"
                for t in prior_turns[-self._SEGMENT_CONTEXT_TURNS:]
            )
            lines += [
                "",
                "Already covered in earlier parts:",
                covered or "- (no outline available)",
                "",
                "The meeting so far ends with these turns:",
                context,
                "",
                "Continue DIRECTLY from the last turn above: same people, same "
                "room, mid-conversation. Do NOT re-open the meeting, greet, or "
                "recap what was already said.",
            ]
            if part < n_parts:
                lines.append(
                    "Do NOT wrap up, summarize, or say goodbye: the meeting "
                    "continues in the next part. End mid-flow."
                )
            else:
                lines.append(
                    "This is the FINAL part: after covering its sections, close "
                    "the meeting naturally (owners and next steps where they arise, "
                    "then goodbyes)."
                )
        lines += [
            "",
            "Output the same JSON object as specified above, with `turns` holding "
            "ONLY this part's new turns, and `summary` / `outline` describing ONLY "
            "this part.",
        ]
        return "\n".join(lines)

    async def _generate_segment(
        self,
        *,
        meeting_id: str,
        attendee_ids: list[str],
        prompt: str,
        part: int,
        part_words: int,
        attempt_label: str = "",
    ) -> tuple[list[dict[str, Any]], str, str]:
        max_tokens = min(
            self.budget.meeting_transcript,
            max(8192, part_words * self._SEGMENT_TOKENS_PER_WORD),
        )
        response = MeetingTranscriptResponse.model_validate(
            await self._call_llm_json(
                prompt,
                system_key="meetings_system",
                max_tokens=max_tokens,
                trace_label=f"meeting_transcript_{meeting_id}_part{part}{attempt_label}",
                expect_dict=True,
            )
        )
        turns = self._sanitize_transcript_turns(
            meeting_id,
            attendee_ids,
            [turn.model_dump(mode="json") for turn in response.turns],
        )
        return turns, response.summary, response.outline

    async def _generate_transcript_segmented(
        self,
        *,
        spec: dict,
        attendee_ids: list[str],
        transcript_prompt: str,
        floor_words: int,
    ) -> tuple[list[dict[str, Any]], str, str, int, int]:
        """Write a long meeting as sequential parts (see ``segmented_generation``).

        Returns the same tuple as ``_generate_transcript_with_floor``;
        ``initial_word_count`` is the total before any part retries and
        ``retries_used`` counts part retries.
        """
        cfg = self.duration_targeting_settings
        meeting_id = spec["id"]
        total_ask = int(floor_words * cfg.segment_overshoot)
        n_parts = max(2, -(-total_ask // max(1, cfg.segment_words)))
        part_ask = -(-total_ask // n_parts)
        agenda_groups = self._split_agenda(
            list(spec.get("agenda") or []) or self._outline_sections(transcript_prompt),
            n_parts,
        )
        logger.info(
            "Meeting %s: segmented generation, %d parts x ~%d words (ask %d, floor %d)",
            meeting_id, n_parts, part_ask, total_ask, floor_words,
        )

        turns: list[dict[str, Any]] = []
        summaries: list[str] = []
        outlines: list[str] = []
        initial_word_count = 0
        retries_used = 0
        for part in range(1, n_parts + 1):
            ask = part_ask
            if part == n_parts:
                # The last part makes up any shortfall from earlier parts.
                shortfall = total_ask - self._turn_word_count(turns)
                ask = int(min(max(part_ask, shortfall), part_ask * self._SEGMENT_LAST_PART_MAX))
            prompt = transcript_prompt + self._segment_addendum(
                part=part,
                n_parts=n_parts,
                agenda_group=agenda_groups[part - 1],
                part_words=ask,
                prior_turns=turns,
                prior_outlines=outlines,
            )
            try:
                part_turns, part_summary, part_outline = await self._generate_segment(
                    meeting_id=meeting_id, attendee_ids=attendee_ids, prompt=prompt,
                    part=part, part_words=ask,
                )
            except Exception:
                logger.exception("Part %d/%d for %s failed", part, n_parts, meeting_id)
                part_turns, part_summary, part_outline = [], "", ""
            part_words = self._turn_word_count(part_turns)
            initial_word_count += part_words

            if part_words < ask * self._SEGMENT_RETRY_BELOW:
                logger.warning(
                    "Part %d/%d for %s short (%d/%d words); retrying once",
                    part, n_parts, meeting_id, part_words, ask,
                )
                retries_used += 1
                retry_prompt = prompt + (
                    f"\n\nPRIOR ATTEMPT AT THIS PART FELL SHORT ({part_words} words). "
                    f"Write the full ~{ask} words for this part with substantive "
                    "back-and-forth on its sections."
                )
                try:
                    r_turns, r_summary, r_outline = await self._generate_segment(
                        meeting_id=meeting_id, attendee_ids=attendee_ids,
                        prompt=retry_prompt, part=part, part_words=ask,
                        attempt_label="_retry",
                    )
                    if self._turn_word_count(r_turns) > part_words:
                        part_turns, part_summary, part_outline = r_turns, r_summary, r_outline
                        part_words = self._turn_word_count(r_turns)
                except Exception:
                    logger.exception("Retry of part %d/%d for %s failed", part, n_parts, meeting_id)

            logger.info(
                "Meeting %s part %d/%d: %d words, %d turns (ask %d)",
                meeting_id, part, n_parts, part_words, len(part_turns), ask,
            )
            turns.extend(part_turns)
            summaries.append(part_summary.strip())
            outlines.append(part_outline.strip())

        summary = " ".join(s for s in summaries if s)
        outline = "\n".join(f"Part {i + 1}: {o}" for i, o in enumerate(outlines) if o)
        return turns, summary, outline, initial_word_count, retries_used

    async def _generate_meeting(
        self, spec: dict, state: WorkspaceState
    ) -> tuple[dict, list[dict]]:
        # Check if meeting was planned as skipped/canceled
        planned_outcome = spec.get("planned_outcome", "completed")
        if planned_outcome in ("skipped", "canceled"):
            artifact = _validate_artifact_payload("meeting", {
                "id": spec["id"],
                "type": "meeting",
                "title": spec["title"],
                "calendar_event_id": spec.get("calendar_event_id"),
                "meeting_type": spec.get("meeting_type", ""),
                "start_time": spec.get("start_time"),
                "end_time": spec.get("end_time"),
                "attendee_ids": spec.get("attendee_ids", []),
                "realized_status": planned_outcome,
                "join_spec": spec.get("join_spec", {}),
                "transcript_json": [],
                "summary": "",
            })
            events = [{
                "type": f"meeting_{planned_outcome}",
                "timestamp": spec.get("start_time", ""),
                "data": {"meeting_id": spec["id"], "title": spec["title"],
                         "attendee_ids": spec.get("attendee_ids", [])},
                "source_artifact_id": spec["id"],
            }]
            return artifact, events

        attendee_ids = spec.get("attendee_ids", [])

        # Phase 1: Overview
        company_summary = self.ctx.company_summary()
        attendees_detail = self.ctx.participants_summary(attendee_ids)
        action_items = self.ctx.action_items_for(attendee_ids, state)

        # Check for slide context
        slide_context = ""
        related_slide_id = spec.get("related_slide_id")
        if related_slide_id and related_slide_id in self.artifacts:
            slide_artifact = self.artifacts[related_slide_id]
            slides = slide_artifact.get("slides", [])
            slide_lines = [f"Slide {s.get('slide_number', i+1)}: {s.get('title', '')}" for i, s in enumerate(slides)]
            slide_context = "Slides for this meeting:\n" + "\n".join(slide_lines)

        technicality_level = self._sample_technicality(spec["id"])
        technicality_description = self._TECHNICALITY_DESCRIPTIONS.get(
            technicality_level, ""
        )

        join_context = self._format_join_context(spec.get("join_spec"), attendee_ids)

        overview_prompt = (
            self.prompts["meetings_overview"]
            .replace("{{MEETING_TITLE}}", spec.get("title", ""))
            .replace("{{MEETING_TYPE}}", spec.get("meeting_type", ""))
            .replace("{{START_TIME}}", spec.get("start_time", ""))
            .replace("{{END_TIME}}", spec.get("end_time", ""))
            .replace("{{DURATION_MINUTES}}", str(spec.get("duration_minutes", 30)))
            .replace("{{ATTENDEES_DETAIL}}", attendees_detail)
            .replace("{{COMPANY_SUMMARY}}", company_summary)
            .replace("{{PURPOSE}}", spec.get("purpose", ""))
            .replace("{{AGENDA}}", "\n".join(spec.get("agenda", ["General discussion"])))
            .replace("{{PROJECT_SUMMARIES}}", self.ctx.all_projects_summary(state))
            .replace("{{ACTION_ITEMS}}", action_items)
            .replace("{{RECENT_DECISIONS}}", "\n".join(state.recent_decisions[-5:]) if state.recent_decisions else "None")
            .replace("{{PRIOR_MEETING_SUMMARIES}}", "No prior meeting context." )
            .replace("{{DOCUMENTS_CONTEXT}}", "")
            .replace("{{SLIDES_CONTEXT}}", slide_context)
            .replace("{{EXPECTED_OUTCOMES}}", "\n".join(spec.get("expected_outcomes", [])))
            .replace("{{TECHNICALITY_LEVEL}}", technicality_level)
            .replace("{{TECHNICALITY_DESCRIPTION}}", technicality_description)
            .replace("{{MEETING_JOIN_CONTEXT}}", join_context)
        )

        logger.info(
            "Generating meeting overview: %s (technicality=%s)",
            spec.get("title", ""), technicality_level,
        )
        overview_response = await self._call_llm(
            overview_prompt, system_key="meetings_system",
            max_tokens=self.budget.meeting_overview,
            trace_label=f"meeting_overview_{spec['id']}",
        )
        if overview_response.truncated:
            logger.warning(
                "Meeting overview truncated for %s (stop_reason=%s, output_tokens=%d). "
                "Downstream transcript will be based on a partial overview.",
                spec["id"], overview_response.stop_reason, overview_response.output_tokens,
            )
        overview_text = overview_response.text

        # Phase 2: Transcript
        attendees_profiles = self.ctx.participants_speaking_profiles(attendee_ids)
        attendee_roster = self._format_attendee_roster(attendee_ids)

        slide_context_for_transcript = ""
        if slide_context:
            slide_context_for_transcript = f"Presentation slides:\n{slide_context}\n\nSpeakers should reference these slides naturally during the meeting."

        anti_exposition_block = (
            self.prompts["meetings_anti_exposition"]
            if self.settings.generation.anti_exposition
            else ""
        )
        floor_words, target_words, ceiling_words = self._calendar_word_budget(spec)
        target_turns = self._calendar_turn_target(spec)
        transcript_prompt = (
            self.prompts["meetings_transcript"]
            .replace("{{MEETING_TITLE}}", spec.get("title", ""))
            .replace("{{MEETING_TYPE}}", spec.get("meeting_type", ""))
            .replace("{{DURATION_MINUTES}}", str(spec.get("duration_minutes", 30)))
            .replace("{{TARGET_WORDS}}", str(target_words))
            .replace("{{MIN_WORDS}}", str(floor_words))
            .replace("{{TARGET_TURNS}}", str(target_turns))
            .replace("{{MEETING_OVERVIEW}}", overview_text)
            .replace("{{ATTENDEE_ROSTER}}", attendee_roster)
            .replace("{{ATTENDEES_SPEAKING_PROFILES}}", attendees_profiles)
            .replace("{{SLIDE_CONTEXT}}", slide_context_for_transcript)
            .replace("{{TECHNICALITY_LEVEL}}", technicality_level)
            .replace("{{TECHNICALITY_DESCRIPTION}}", technicality_description)
            .replace("{{MEETING_JOIN_CONTEXT}}", join_context)
            .replace("{{ANTI_EXPOSITION_BLOCK}}", anti_exposition_block)
        )

        logger.info(
            "Generating meeting transcript: %s (duration=%d min, floor=%d, target=%d words)",
            spec.get("title", ""),
            int(spec.get("duration_minutes", 30) or 30),
            floor_words,
            target_words,
        )
        turns, summary, outline, initial_word_count, retries_used = (
            await self._generate_transcript_with_floor(
                spec=spec,
                attendee_ids=attendee_ids,
                transcript_prompt=transcript_prompt,
                target_words=target_words,
                floor_words=floor_words,
                target_turns=target_turns,
            )
        )

        if self.duration_targeting_settings.enabled:
            final_word_count = self._turn_word_count(turns)
            self._duration_targeting_stats["meetings_targeted"] += 1
            if initial_word_count < floor_words:
                self._duration_targeting_stats["initial_below_floor"] += 1
            self._duration_targeting_stats["retries_attempted"] += retries_used
            if final_word_count < floor_words:
                self._duration_targeting_stats["final_below_floor"] += 1
            self._duration_targeting_per_meeting[spec["id"]] = {
                "duration_minutes": int(spec.get("duration_minutes", 30) or 30),
                "floor_words": floor_words,
                "target_words": target_words,
                "ceiling_words": ceiling_words,
                "initial_word_count": initial_word_count,
                "post_transcript_word_count": final_word_count,
                "retries_used": retries_used,
            }

        # Phase 2b: Realism rewrite + iterative judge/refine loop (optional)
        realism_rewrite_accepted = True
        realism_rewrite_fallback_reason: str | None = None
        realism_refine_iterations = 0
        realism_refine_early_stopped = False
        realism_refine_judge_reasoning: list[str] = []
        realism_refine_trace: list[dict[str, Any]] = []
        if self.realism_rewrite_settings.enabled and turns:
            (
                turns,
                realism_rewrite_accepted,
                realism_rewrite_fallback_reason,
                realism_refine_iterations,
                realism_refine_early_stopped,
                realism_refine_judge_reasoning,
                realism_refine_trace,
            ) = await self._refine_realism_iteratively(
                meeting_id=spec["id"],
                title=spec.get("title", ""),
                meeting_type=spec.get("meeting_type", ""),
                technicality_level=technicality_level,
                attendee_ids=attendee_ids,
                state=state,
                summary=summary,
                transcript_turns=turns,
                floor_words=floor_words,
                target_words=target_words,
            )

        if self.duration_targeting_settings.enabled:
            post_refine_word_count = self._turn_word_count(turns)
            entry = self._duration_targeting_per_meeting.get(spec["id"])
            if entry is not None:
                entry["post_refine_word_count"] = post_refine_word_count
                entry["realism_rewrite_accepted"] = realism_rewrite_accepted
                entry["realism_rewrite_fallback_reason"] = (
                    realism_rewrite_fallback_reason
                )

        # Persist the semantic (post-structural-refinement, pre-disfluency)
        # transcript and any per-iteration trace. The semantic file is
        # written for every completed meeting — when the realism loop is
        # disabled it equals the LLM transcript output, so downstream
        # consumers always have one stable on-disk semantic artifact.
        meeting_dir = self.output_dir / "meetings" / spec["id"]
        if turns:
            fileio.makedirs(meeting_dir)
            try:
                fileio.write_json(meeting_dir / "transcript_semantic.json", turns)
            except Exception:
                logger.exception(
                    "Failed persisting transcript_semantic.json for %s", spec["id"],
                )
            if realism_refine_trace:
                try:
                    fileio.write_json(
                        meeting_dir / "realism_refine_trace.json",
                        realism_refine_trace,
                    )
                except Exception:
                    logger.exception(
                        "Failed persisting realism_refine_trace.json for %s", spec["id"],
                    )

        # Phase 3: Event extraction
        transcript_text = "\n".join(
            f"{t.get('speaker_name', 'Unknown')} [{t.get('speaker_id', '')}]: {t.get('text', '')}"
            for t in turns
        )
        attendee_roster = self._format_attendee_roster(attendee_ids)

        extraction_prompt = (
            self.prompts["meetings_event_extraction"]
            .replace("{{MEETING_TITLE}}", spec.get("title", ""))
            .replace("{{ATTENDEE_ROSTER}}", attendee_roster)
            .replace("{{TRANSCRIPT_TEXT}}", transcript_text)
        )

        logger.info("Extracting events from meeting: %s", spec.get("title", ""))
        extracted_response = MeetingEventExtractionResponse.model_validate(await self._call_llm_json(
            extraction_prompt, system_key="meetings_system",
            max_tokens=self.budget.meeting_event_extraction,
            trace_label=f"meeting_events_{spec['id']}",
            expect_dict=True,
        ))
        extracted = extracted_response.model_dump(mode="json")
        extracted = self._sanitize_extracted_meeting_fields(spec["id"], attendee_ids, extracted)

        # Build artifact
        artifact = _validate_artifact_payload("meeting", {
            "id": spec["id"],
            "type": "meeting",
            "title": spec["title"],
            "calendar_event_id": spec.get("calendar_event_id"),
            "meeting_type": spec.get("meeting_type", ""),
            "start_time": spec.get("start_time", ""),
            "end_time": spec.get("end_time", ""),
            "attendee_ids": attendee_ids,
            "realized_status": "completed",
            "planned_attendance_changes": spec.get("planned_attendance_changes", {}),
            "join_spec": spec.get("join_spec", {}),
            "transcript_json": turns,
            "transcript_text": transcript_text,
            "summary": summary,
            "outline": outline,
            "action_items": extracted.get("action_items", []),
            "decisions": extracted.get("decisions", []),
            "project_updates": extracted.get("project_updates", []),
            "follow_ups": extracted.get("follow_ups", []),
            "meeting_summary": extracted.get("meeting_summary", summary),
            "realism_rewrite_accepted": realism_rewrite_accepted,
            "realism_rewrite_fallback_reason": realism_rewrite_fallback_reason,
            "realism_refine_iterations": realism_refine_iterations,
            "realism_refine_early_stopped": realism_refine_early_stopped,
            "realism_refine_judge_reasoning": realism_refine_judge_reasoning,
        })

        # Build events
        events: list[dict] = []
        events.append({
            "type": "meeting_completed",
            "timestamp": spec.get("start_time", ""),
            "data": {
                "meeting_id": spec["id"],
                "title": spec["title"],
                "summary": summary,
                "attendee_ids": attendee_ids,
                "join_spec": spec.get("join_spec", {}),
            },
            "source_artifact_id": spec["id"],
        })

        for ai in extracted.get("action_items", []):
            events.append({
                "type": "action_item_created",
                "timestamp": spec.get("start_time", ""),
                "data": {
                    "id": f"ai_{spec['id']}_{len(events)}",
                    "description": ai.get("description", ""),
                    "assignee_id": ai.get("assignee_id", "") or "",
                },
                "source_artifact_id": spec["id"],
            })

        for upd in extracted.get("project_updates", []):
            pid = upd.get("project_id")
            if pid:
                upd["project_id"] = self.id_drift.normalize_project_id(
                    pid,
                    self.id_registry,
                    field_name="project_updates.project_id",
                    context=f"meeting:{spec['id']}",
                )

        for dec in extracted.get("decisions", []):
            events.append({
                "type": "decision_made",
                "timestamp": spec.get("start_time", ""),
                "data": {"description": dec.get("description", "")},
                "source_artifact_id": spec["id"],
            })

        # Attendance outcome events
        for pid, status in spec.get("planned_attendance_changes", {}).items():
            if status == "missed":
                events.append({
                    "type": "participant_missed_meeting",
                    "timestamp": spec.get("start_time", ""),
                    "data": {"meeting_id": spec["id"], "person_id": pid, "title": spec["title"]},
                    "source_artifact_id": spec["id"],
                })
            elif status == "join_late":
                events.append({
                    "type": "participant_joined_late",
                    "timestamp": spec.get("start_time", ""),
                    "data": {"meeting_id": spec["id"], "person_id": pid, "title": spec["title"]},
                    "source_artifact_id": spec["id"],
                })

        return artifact, events

    # ------------------------------------------------------------------
    # Email thread generation (per-message)
    # ------------------------------------------------------------------

    async def _generate_email_thread(
        self, spec: dict, state: WorkspaceState
    ) -> tuple[dict, list[dict]]:
        num_messages = spec.get("num_messages", 2)
        participant_ids = spec.get("participant_ids", [])
        initiator_id = spec.get("initiator_id", participant_ids[0] if participant_ids else "")
        subject = spec.get("subject", "")
        thread_id = spec["id"]

        messages: list[dict] = []
        events: list[dict] = []

        # Determine message senders (alternating, starting with initiator)
        other_participants = [p for p in participant_ids if p != initiator_id]
        sender_order = [initiator_id]
        for i in range(num_messages - 1):
            if other_participants:
                sender_order.append(other_participants[i % len(other_participants)])
            else:
                sender_order.append(initiator_id)

        company_summary = self.ctx.company_summary()
        state_summary = self.ctx.state_summary(state)
        project_context = self._project_context_for(spec, state)

        # Related context from dependency artifacts
        related_context = ""
        for dep_id in spec.get("depends_on", []):
            if dep_id in self.artifacts:
                dep = self.artifacts[dep_id]
                if dep.get("type") == "meeting":
                    related_context += f"Meeting '{dep.get('title', '')}' summary: {dep.get('summary', '')}\n"

        base_time = datetime.fromisoformat(spec.get("start_time", "2025-01-01T10:00:00"))

        for msg_idx in range(num_messages):
            sender_id = sender_order[msg_idx]
            sender_profile = self.ctx.person_summary(sender_id)
            current_sender_name = self._person_name(sender_id)
            msg_time = base_time + timedelta(minutes=30 * msg_idx)

            # Build thread history
            thread_history = ""
            if messages:
                history_lines = []
                for prev in messages:
                    sender_person = self.org.get_person(prev["sender_id"])
                    history_sender_name = sender_person.name if sender_person else prev["sender_id"]
                    history_lines.append(f"From: {history_sender_name}\nSubject: {prev['subject']}\n\n{prev['body']}\n---")
                thread_history = "\n".join(history_lines)
            else:
                thread_history = "This is the first message in the thread."

            recipient_ids, cc_ids = self._canonical_email_routing(
                participant_ids,
                sender_id,
                spec.get("cc_ids", []),
            )
            recipient_names = ", ".join(self._person_name(pid) for pid in recipient_ids) or "None"
            cc_names = ", ".join(self._person_name(pid) for pid in cc_ids) or "None"

            purpose = spec.get("purpose", "")
            if msg_idx == 0:
                msg_purpose = f"Initiate thread: {purpose}"
            else:
                msg_purpose = f"Reply to thread: {purpose}"

            user_prompt = (
                self.prompts["emails_message"]
                .replace("{{COMPANY_SUMMARY}}", company_summary)
                .replace("{{THREAD_HISTORY}}", thread_history)
                .replace("{{SENDER_ID}}", sender_id)
                .replace("{{SENDER_NAME}}", current_sender_name)
                .replace("{{SENDER_PROFILE}}", sender_profile)
                .replace("{{RECIPIENT_IDS}}", json.dumps(recipient_ids))
                .replace("{{RECIPIENT_NAMES}}", recipient_names)
                .replace("{{CC_IDS}}", json.dumps(cc_ids))
                .replace("{{CC_NAMES}}", cc_names)
                .replace("{{THREAD_SUBJECT}}", subject)
                .replace("{{MESSAGE_PURPOSE}}", msg_purpose)
                .replace("{{TIMESTAMP_CONTEXT}}", msg_time.isoformat())
                .replace("{{WORKSPACE_CONTEXT}}", state_summary)
                .replace("{{PROJECT_CONTEXT}}", project_context)
                .replace("{{RELATED_CONTEXT}}", related_context)
                .replace("{{MESSAGE_ID}}", f"msg_{thread_id}_{msg_idx + 1}")
                .replace("{{THREAD_ID}}", thread_id)
            )

            msg_data = await self._call_llm_json(
                user_prompt, system_key="emails_system",
                max_tokens=self.budget.email_message,
                trace_label=f"email_msg_{thread_id}_{msg_idx + 1}",
                expect_dict=True,
            )
            message_response = EmailMessageResponse.model_validate(msg_data)
            message_payload = message_response.model_dump(mode="json")
            canonical_message_id = f"msg_{thread_id}_{msg_idx + 1}"
            self._record_email_routing_drift(
                thread_id=thread_id,
                message_id=canonical_message_id,
                field_name="email.sender_id",
                actual=message_payload.get("sender_id", ""),
                expected=sender_id,
            )
            self._record_email_routing_drift(
                thread_id=thread_id,
                message_id=canonical_message_id,
                field_name="email.recipient_ids",
                actual=message_payload.get("recipient_ids", []),
                expected=recipient_ids,
            )
            self._record_email_routing_drift(
                thread_id=thread_id,
                message_id=canonical_message_id,
                field_name="email.cc_ids",
                actual=message_payload.get("cc_ids", []),
                expected=cc_ids,
            )
            message_payload["id"] = canonical_message_id
            message_payload["thread_id"] = thread_id
            message_payload["sender_id"] = sender_id
            message_payload["recipient_ids"] = recipient_ids
            message_payload["cc_ids"] = cc_ids
            message_payload["timestamp"] = msg_time.isoformat()
            if message_payload.get("reply_to_id") is None and msg_idx > 0:
                message_payload["reply_to_id"] = messages[-1].get("id")
            message = EmailMessage.model_validate(message_payload).model_dump(mode="json")
            messages.append(message)

            events.append({
                "type": "email_sent",
                "timestamp": msg_time.isoformat(),
                "data": {
                    "thread_id": thread_id,
                    "subject": message.get("subject", subject),
                    "sender_id": message["sender_id"],
                    "recipient_ids": message["recipient_ids"],
                    "cc_ids": message["cc_ids"],
                },
                "source_artifact_id": thread_id,
            })

        artifact = _validate_artifact_payload("email_thread", {
            "id": thread_id,
            "type": "email_thread",
            "subject": subject,
            "messages": messages,
            "participant_ids": participant_ids,
        })

        return artifact, events

    # ------------------------------------------------------------------
    # Document generation
    # ------------------------------------------------------------------

    async def _generate_document(
        self, spec: dict, state: WorkspaceState
    ) -> tuple[dict, list[dict]]:
        company_summary = self.ctx.company_summary()
        canonical_author_id = spec.get("author_id", "")
        canonical_related_project = spec.get("related_project_id") or None
        author_summary = self.ctx.person_summary(canonical_author_id)

        project_context = "No related project."
        if canonical_related_project and canonical_related_project in state.projects:
            project_context = self.ctx.project_summary(canonical_related_project, state)

        related_context = ""
        for dep_id in spec.get("depends_on", []):
            if dep_id in self.artifacts:
                dep = self.artifacts[dep_id]
                if dep.get("type") == "meeting":
                    related_context += f"Meeting '{dep.get('title', '')}' summary: {dep.get('summary', '')}\n"

        shared_ids = spec.get("shared_with_ids", [])
        shared_with = self.ctx.participants_summary(shared_ids) if shared_ids else "Not specified."

        user_prompt = (
            self.prompts["documents_document"]
            .replace("{{COMPANY_SUMMARY}}", company_summary)
            .replace("{{DOCUMENT_TITLE}}", spec.get("title", ""))
            .replace("{{DOC_TYPE}}", spec.get("doc_type", "notes"))
            .replace("{{AUTHOR_SUMMARY}}", author_summary)
            .replace("{{PROJECT_CONTEXT}}", project_context)
            .replace("{{DOCUMENT_PURPOSE}}", spec.get("purpose", ""))
            .replace("{{SHARED_WITH}}", shared_with)
            .replace("{{RELATED_CONTEXT}}", related_context)
            .replace("{{DOCUMENT_ID}}", spec["id"])
            .replace("{{AUTHOR_ID}}", spec.get("author_id", ""))
            .replace("{{PROJECT_ID}}", spec.get("related_project_id", "") or "")
        )

        logger.info("Generating document: %s", spec.get("title", ""))
        doc_response = DocumentResponse.model_validate(await self._call_llm_json(
            user_prompt, system_key="documents_system",
            max_tokens=self.budget.document,
            trace_label=f"document_{spec['id']}",
            expect_dict=True,
        ))

        artifact = _validate_artifact_payload("document", {
            "id": spec["id"],
            "type": "document",
            "title": doc_response.title or spec.get("title", ""),
            "content": doc_response.content,
            "author_id": canonical_author_id,
            "doc_type": doc_response.doc_type or spec.get("doc_type", "notes"),
            "created_at": spec.get("created_at", ""),
            "related_project": canonical_related_project,
            "shared_with_ids": doc_response.shared_with_ids,
        })
        self._record_canonical_field_drift(
            kind="person",
            field_name="document.author_id",
            context=f"document:{spec['id']}",
            actual=doc_response.author_id,
            expected=canonical_author_id,
        )
        self._record_canonical_field_drift(
            kind="project",
            field_name="document.related_project",
            context=f"document:{spec['id']}",
            actual=doc_response.related_project,
            expected=canonical_related_project,
        )

        events = [{
            "type": "document_created",
            "timestamp": spec.get("created_at", spec.get("timestamp", "")),
            "data": {
                "title": artifact["title"],
                "author_id": artifact["author_id"],
                "doc_type": artifact["doc_type"],
            },
            "source_artifact_id": spec["id"],
        }]

        return artifact, events

    # ------------------------------------------------------------------
    # Slides generation
    # ------------------------------------------------------------------

    async def _generate_slides(
        self, spec: dict, state: WorkspaceState
    ) -> tuple[dict, list[dict]]:
        company_summary = self.ctx.company_summary()
        canonical_author_id = spec.get("author_id", "")
        presenter_summary = self.ctx.person_summary(canonical_author_id)
        project_context = self._project_context_for(spec, state)

        user_prompt = (
            self.prompts["slides_content"]
            .replace("{{COMPANY_SUMMARY}}", company_summary)
            .replace("{{PRESENTATION_TITLE}}", spec.get("title", ""))
            .replace("{{PRESENTER_SUMMARY}}", presenter_summary)
            .replace("{{AUDIENCE_DESCRIPTION}}", "Team members and stakeholders")
            .replace("{{PRESENTATION_PURPOSE}}", spec.get("purpose", ""))
            .replace("{{NUM_SLIDES}}", str(spec.get("num_slides", 8)))
            .replace("{{PROJECT_CONTEXT}}", project_context)
            .replace("{{RELATED_CONTEXT}}", "\n".join(spec.get("key_points", [])))
            .replace("{{PRESENTATION_ID}}", spec["id"])
            .replace("{{AUTHOR_ID}}", spec.get("author_id", ""))
        )

        logger.info("Generating slides: %s", spec.get("title", ""))
        slides_response = SlidePresentationResponse.model_validate(await self._call_llm_json(
            user_prompt, system_key="slides_system",
            max_tokens=self.budget.slides,
            trace_label=f"slides_{spec['id']}",
            expect_dict=True,
        ))

        artifact = _validate_artifact_payload("slides", {
            "id": spec["id"],
            "type": "slides",
            "title": slides_response.title or spec.get("title", ""),
            "author_id": canonical_author_id,
            "created_at": spec.get("created_at") or spec.get("timestamp"),
            "slides": [slide.model_dump(mode="json") for slide in slides_response.slides],
        })
        self._record_canonical_field_drift(
            kind="person",
            field_name="slides.author_id",
            context=f"slides:{spec['id']}",
            actual=slides_response.author_id,
            expected=canonical_author_id,
        )

        events: list[dict] = []  # Slides don't produce workspace events
        return artifact, events

    # ------------------------------------------------------------------
    # LLM API
    # ------------------------------------------------------------------

    async def _call_llm(
        self, user_prompt: str, *, system_key: str, max_tokens: int = 8192,
        trace_label: str = "call", reasoning_effort: str | None = None,
    ) -> LLMResponse:
        from ..utils.usage_tracker import tracker

        system_prompt = self.prompts[system_key]
        extra = {} if reasoning_effort is None else {"reasoning_effort": reasoning_effort}
        response = await self.llm.create_message(
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=max_tokens,
            temperature=1.0,
            **extra,
        )
        tracker.track_llm_response(response)
        async with self._io_lock:
            self.prompt_tracer.record(
                label=trace_label, system_prompt=system_prompt,
                user_prompt=user_prompt, response_text=response.text,
                metadata={
                    "model": self.model,
                    "max_tokens": max_tokens,
                    "stop_reason": response.stop_reason,
                    "output_tokens": response.output_tokens,
                },
            )
        return response

    async def _call_llm_json(
        self, user_prompt: str, *, system_key: str, max_tokens: int = 8192,
        trace_label: str = "json", expect_dict: bool = False,
    ) -> Any:
        last_error: Exception | None = None
        last_response: LLMResponse | None = None
        reasoning_effort: str | None = None
        for attempt in range(1 + MAX_RETRIES):
            if attempt > 0:
                logger.warning("Retry %d/%d", attempt, MAX_RETRIES)
            response = await self._call_llm(
                user_prompt, system_key=system_key, max_tokens=max_tokens,
                trace_label=f"{trace_label}_attempt_{attempt + 1}",
                reasoning_effort=reasoning_effort,
            )
            last_response = response
            try:
                parsed = extract_json(response.text, repair_string_newlines=True)
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                if (
                    response.truncated
                    and not (response.text or "").strip()
                    and reasoning_effort is None
                    and model_uses_reasoning(self.model)
                ):
                    # Reduce reasoning effort when it consumes the entire
                    # budget without producing visible output.
                    reasoning_effort = "low"
                    logger.warning(
                        "%s: budget exhausted by reasoning with no output; "
                        "retrying with reasoning_effort=low", trace_label,
                    )
                if response.truncated:
                    logger.warning(
                        "JSON parse failed for %s (attempt %d): output truncated by provider "
                        "(stop_reason=%s, output_tokens=%d, max_tokens=%d). "
                        "Raise the per-call max_tokens budget or decompose the request.",
                        trace_label, attempt + 1, response.stop_reason,
                        response.output_tokens, max_tokens,
                    )
                else:
                    logger.warning(
                        "JSON parse failed for %s (attempt %d, stop_reason=%s): %s",
                        trace_label, attempt + 1, response.stop_reason, exc,
                    )
                continue
            if expect_dict and not isinstance(parsed, dict):
                last_error = ValueError(
                    f"Expected JSON object but got {type(parsed).__name__}: {parsed!r}"
                )
                logger.warning(
                    "Expected JSON object for %s (attempt %d) but got %s: %r",
                    trace_label, attempt + 1, type(parsed).__name__, parsed,
                )
                continue
            return parsed
        truncation_note = ""
        if last_response is not None and last_response.truncated:
            truncation_note = (
                f" (last response truncated by provider; stop_reason="
                f"{last_response.stop_reason}, max_tokens={max_tokens})"
            )
        raise ValueError(
            f"Failed to parse JSON from LLM for {trace_label}{truncation_note}"
        ) from last_error

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_artifact(self, artifact_id: str, artifact_type: str, artifact: dict) -> None:
        artifact = _validate_artifact_payload(artifact_type, artifact)
        type_dirs = {
            "meeting": "meetings",
            "email_thread": "emails",
            "document": "documents",
            "slides": "slides",
        }
        subdir = type_dirs.get(artifact_type, "other")

        if artifact_type == "meeting":
            # Meetings get their own directory
            artifact_dir = self.output_dir / subdir / artifact_id
            fileio.makedirs(artifact_dir)
            fileio.write_json(artifact_dir / "metadata.json", artifact)
        else:
            artifact_dir = self.output_dir / subdir
            fileio.makedirs(artifact_dir)
            fileio.write_json(artifact_dir / f"{artifact_id}.json", artifact)

    def _load_from_disk(self) -> dict:
        artifacts: dict[str, dict] = {}

        # Load meetings
        meetings_dir = self.output_dir / "meetings"
        if meetings_dir.exists():
            for name in sorted(os.listdir(meetings_dir)):
                meta = meetings_dir / name / "metadata.json"
                if meta.exists():
                    artifacts[name] = _validate_artifact_payload("meeting", fileio.read_json(meta))

        # Load emails, documents
        for subdir in ("emails", "documents", "slides"):
            subdir_path = self.output_dir / subdir
            if subdir_path.exists():
                for fname in sorted(subdir_path.iterdir()):
                    if fname.suffix == ".json":
                        artifact_type = {
                            "emails": "email_thread",
                            "documents": "document",
                            "slides": "slides",
                        }[subdir]
                        data = _validate_artifact_payload(artifact_type, fileio.read_json(fname))
                        artifacts[data.get("id", fname.stem)] = data

        # Load events
        events_path = self.output_root / "state" / "generated_events_by_artifact.json"
        events_by_artifact = fileio.read_json(events_path) if events_path.exists() else {}

        logger.info("Loaded step 4: %d artifacts", len(artifacts))
        return {
            "artifacts": artifacts,
            "generated_events_by_artifact": events_by_artifact,
        }
