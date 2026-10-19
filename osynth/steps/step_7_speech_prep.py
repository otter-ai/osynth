"""Step 7: Speech preparation — derive transcript forms and TTS tasks.

Responsibilities split by gating:

**Transcript work (always runs, regardless of audio generation)**

- Run an LLM transcript-form pass over each meeting transcript and persist
  ``formatted_turns.json`` (canonical, semantic-verbatim, TTS text + edit
  layers).
- Derive an AMI-style tokenized view (``verbatim_ami.json``).
- Extract per-turn entities over the semantic-verbatim transcript
  (``entities.json``).
- Sample acoustically-unclear word ranges deterministically per meeting
  (folded into ``formatted_turns.json``).

**Audio prep (gated on ``generation.generate_audio``)**

- Build per-speaker enrollment specs that reference voice catalog inputs.
- Generate per-speaker enrollment-text candidates for the batch/local TTS
  step.
- Persist deterministic per-meeting turn task manifests for the later TTS
  batch step (step 8).

Synthesis itself is in ``step_8_tts_batch`` and finalize work
(timing/alignment/mixing/unclear attenuation) is in ``step_9_audio_finalize``.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import shutil
from pathlib import Path
from typing import Any

from ..models.calendar import MeetingJoinSpec
from ..models.people import ExternalPerson, OrgStructure, Person
from ..models.settings import SyntheticDataSettings
from ..models.state import Project
from ..models.tts import (
    EnrollmentSpec,
    EnrollmentTextBundle,
    OverlapAnchor,
    TTSTurnTask,
    TurnEntities,
    validate_ami_turns,
    validate_enrollment_text_bundle_for_run,
    validate_formatted_turns,
)
from ..models.workspace import WorkspaceConfig
from ..tts.ami_normalize import normalize_synthetic_turns
from ..tts.ami_view import derive_ami_view
from ..tts.backchannel_anchor import (
    anchor_backchannels,
    attach_backchannel_anchors,
)
from ..tts.backchannel_classifier import (
    apply_backchannel_decisions,
    classify_backchannels_raw,
)
from ..tts.disfluency import format_turns_for_speech
from ..tts.overlap_annotator import (
    annotate_overlaps,
    attach_overlap_anchors,
)
from ..tts.transcript_entities import (
    extract_meeting_entities,
    filter_entities_to_surfaces,
)
from ..tts.unclear_sampler import sample_unclear_spans
from ..utils import fileio
from ..utils.language import LANG_CONFIG
from ..utils.llm_json import extract_json
from ..utils.llm_client import LLMClient
from ..utils.prompt import read_prompt as _read_prompt_lang, render_prompt
from ..utils.text_normalize import ascii_quotes_in_turns
from ..utils.usage_tracker import tracker

logger = logging.getLogger(__name__)


def _formatter_accepts_kwarg(fn: Any, kwarg: str) -> bool:
    """Best-effort check that ``fn`` accepts a particular keyword argument.

    Test fakes may use simpler signatures (e.g. only ``turns``,
    ``system_prompt``, ``model``); avoid passing sanitizer wiring to them.
    """
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    params = signature.parameters
    if kwarg in params:
        return True
    return any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )


class Step7SpeechPrep:
    """Prepare per-meeting TTS task manifests and per-speaker enrollment specs."""

    def __init__(
        self,
        settings: SyntheticDataSettings,
        output_root: str,
        data: dict,
        *,
        llm_client: Any | None = None,
        formatter_fn: Any | None = None,
        overlap_annotator_client: Any | None = None,
        backchannel_classifier_client: Any | None = None,
        backchannel_anchor_client: Any | None = None,
    ) -> None:
        self.settings = settings
        self.data = data
        self.run_dir = Path(output_root)
        self.output_dir = self.run_dir / "step_7_speech_prep"
        self.generate_audio: bool = settings.generation.generate_audio
        self.workspace_config: WorkspaceConfig = data["workspace_config"]
        self.org: OrgStructure = data["org"]
        self.projects: dict[str, Project] = data.get("projects", {})
        self.artifacts: dict[str, dict] = data.get("artifacts", {})
        self.voice_assignments: dict[str, str] = data.get("voice_assignments", {})

        self.run_id: str = settings.run_id
        self.language: str = settings.language
        self.seed: int = int(settings.seed or 42)
        self.unclear_rate: float = float(settings.generation.unclear_rate)
        self.max_truncation_rate: float = float(settings.tts.max_truncation_rate)

        self.formatting_model: str = settings.tts.formatting_model
        self.enrollment_text_model: str = str(settings.tts.enrollment_text_model)
        self.enrollment_target_seconds: int = settings.tts.enrollment_target_seconds
        self.enrollment_min_seconds: int = settings.tts.enrollment_min_seconds
        self.enrollment_max_seconds: int = settings.tts.enrollment_max_seconds
        self.enrollment_candidate_count: int = settings.tts.enrollment_candidate_count
        self.voice_cache_dir: str = settings.tts.voice_cache_dir

        self._llm_client = llm_client
        self._formatter_fn = formatter_fn or format_turns_for_speech
        self._overlap_annotator_client = overlap_annotator_client
        self._backchannel_classifier_client = backchannel_classifier_client
        self._backchannel_anchor_client = backchannel_anchor_client
        self._voice_catalog_cache: dict[str, str] | None = None

        self.overlap_settings = settings.speech_pipeline.overlap
        self.backchannel_settings = settings.speech_pipeline.backchannel
        self.entity_reasoning_effort = (
            settings.speech_pipeline.entity_reasoning_effort
        )
        self.formatting_reasoning_effort = (
            settings.speech_pipeline.formatting_reasoning_effort
        )
        self.batch_concurrency = max(1, settings.speech_pipeline.batch_concurrency)
        self.meeting_concurrency = max(1, settings.speech_pipeline.meeting_concurrency)

        self._code_root = Path(__file__).resolve().parent.parent
        tts_prompts = settings.prompts.tts
        spoken_form_path = tts_prompts.spoken_form
        enrollment_text_path = tts_prompts.enrollment_text
        overlap_annotator_path = tts_prompts.overlap_annotator
        self._spoken_form_prompt = _read_prompt_lang(
            self._code_root / spoken_form_path, self.language,
        )
        self._enrollment_text_prompt = _read_prompt_lang(
            self._code_root / enrollment_text_path, self.language,
        )
        self._overlap_annotator_prompt = _read_prompt_lang(
            self._code_root / overlap_annotator_path, self.language,
        )
        bc_classifier_path = tts_prompts.backchannel_classifier
        bc_anchor_path = tts_prompts.backchannel_anchor
        self._backchannel_classifier_prompt = _read_prompt_lang(
            self._code_root / bc_classifier_path, self.language,
        )
        self._backchannel_anchor_prompt = _read_prompt_lang(
            self._code_root / bc_anchor_path, self.language,
        )
        entities_path = settings.prompts.transcripts.entities
        self._entities_prompt = _read_prompt_lang(
            self._code_root / entities_path, self.language,
        )

        lang_cfg = LANG_CONFIG.get(self.language, LANG_CONFIG["en"])
        self._language_name = str(lang_cfg["name"])

    async def run(self, force: bool = False) -> dict:
        status_path = self.output_dir / "prep_status.json"
        if status_path.exists() and not force:
            logger.info("Step 7 outputs exist, skipping regeneration.")
            return self._load_from_disk()

        fileio.makedirs(self.output_dir)
        fileio.makedirs(self.output_dir / "meetings")
        if self.generate_audio:
            fileio.makedirs(self.output_dir / "speakers")

        meeting_artifacts = {
            aid: art
            for aid, art in self.artifacts.items()
            if art.get("type") == "meeting"
            and art.get("realized_status") == "completed"
            and art.get("transcript_json")
        }

        logger.info(
            "Step 7 transcripts for %d meetings (audio=%s)",
            len(meeting_artifacts),
            self.generate_audio,
        )

        meetings_status: dict[str, str] = {}
        speaker_ids: set[str] = set()
        # Meetings write only to their own directories and speaker voices
        # come from the precomputed voice_assignments, so both loops can run
        # concurrently; results are collected in the original order.
        semaphore = asyncio.Semaphore(self.meeting_concurrency)

        async def prep_one(meeting_id: str, artifact: dict) -> set[str] | None:
            async with semaphore:
                try:
                    return await self._prep_meeting(meeting_id, artifact)
                except Exception:
                    logger.exception("Speech prep failed for %s", meeting_id)
                    return None

        meeting_results = await asyncio.gather(
            *(prep_one(mid, art) for mid, art in meeting_artifacts.items())
        )
        for meeting_id, speakers_in_meeting in zip(meeting_artifacts, meeting_results):
            if speakers_in_meeting is None:
                meetings_status[meeting_id] = "failed"
            else:
                speaker_ids.update(speakers_in_meeting)
                meetings_status[meeting_id] = "completed"

        speakers_status: dict[str, str] = {}
        if self.generate_audio:
            async def enroll_one(speaker_id: str) -> str:
                async with semaphore:
                    try:
                        await self._write_enrollment_artifacts(speaker_id)
                        return "completed"
                    except Exception:
                        logger.exception("Enrollment artifacts failed for speaker %s", speaker_id)
                        return "failed"

            ordered_speakers = sorted(speaker_ids)
            for speaker_id, status in zip(
                ordered_speakers,
                await asyncio.gather(*(enroll_one(s) for s in ordered_speakers)),
            ):
                speakers_status[speaker_id] = status

        fileio.write_json(
            status_path,
            {
                "total_meetings": len(meeting_artifacts),
                "completed": sum(1 for v in meetings_status.values() if v == "completed"),
                "meetings": meetings_status,
                "speakers": sorted(speaker_ids),
                "speakers_status": speakers_status,
                "audio_enabled": self.generate_audio,
            },
        )

        logger.info(
            "Step 7 complete: %d/%d meetings, %d speakers",
            sum(1 for v in meetings_status.values() if v == "completed"),
            len(meeting_artifacts),
            len(speaker_ids),
        )
        return self._load_from_disk()

    async def _prep_meeting(self, meeting_id: str, artifact: dict) -> set[str]:
        meeting_dir = self.output_dir / "meetings" / meeting_id
        fileio.makedirs(meeting_dir)

        formatted_turns_path = meeting_dir / "formatted_turns.json"
        precomputed_entities: list[TurnEntities] | None = None
        if formatted_turns_path.exists():
            formatted_turns = validate_formatted_turns(
                fileio.read_json(formatted_turns_path),
            )
        else:
            transcript_turns = artifact.get("transcript_json") or []
            formatted_turns = await self._prepare_turns(
                transcript_turns, meeting_id=meeting_id,
            )
            if not formatted_turns:
                logger.warning("No turns to synthesize for %s", meeting_id)
                fileio.write_json(formatted_turns_path, [])
                fileio.write_json(meeting_dir / "verbatim_ami.json", [])
                fileio.write_json(meeting_dir / "entities.json", [])
                if self.generate_audio:
                    fileio.write_json(meeting_dir / "tts_tasks.json", [])
                return set()

            formatted_turns = self._attach_unclear_spans(meeting_id, formatted_turns)
            # BC classify, overlap annotate, and entity extraction are mutually
            # independent after spoken_form — fan them out concurrently. BC
            # anchor depends on both BC labels and overlap anchors, so it runs
            # after the gather + apply step.
            bc_decisions, overlap_anchors, precomputed_entities = await asyncio.gather(
                self._classify_backchannels_raw(meeting_id, formatted_turns),
                self._annotate_overlaps_raw(meeting_id, artifact, formatted_turns),
                self._extract_entities(meeting_id, formatted_turns),
            )
            apply_backchannel_decisions(formatted_turns, bc_decisions)
            attach_overlap_anchors(formatted_turns, overlap_anchors)
            formatted_turns = validate_formatted_turns(formatted_turns)
            formatted_turns = await self._anchor_backchannels(
                meeting_id, formatted_turns,
            )
            # Step 7's own LLM rewrites can reintroduce typographic quotes.
            ascii_quotes_in_turns(formatted_turns)
            fileio.write_json(formatted_turns_path, formatted_turns)

        await self._write_transcript_artifacts(
            meeting_id, meeting_dir, formatted_turns,
            entities=precomputed_entities,
        )

        speakers: set[str] = set()
        if self.generate_audio:
            speakers = self._write_tts_tasks(meeting_id, meeting_dir, formatted_turns)
        else:
            for turn in formatted_turns:
                sid = str(turn.get("speaker_id") or "")
                if sid:
                    speakers.add(sid)
        return speakers

    async def _annotate_overlaps_raw(
        self,
        meeting_id: str,
        artifact: dict[str, Any],
        formatted_turns: list[dict[str, Any]],
    ) -> list[OverlapAnchor | None]:
        anchors: list[OverlapAnchor | None] = [None for _ in formatted_turns]
        if not self.overlap_settings.enabled or len(formatted_turns) < 2:
            return anchors
        join_spec = self._meeting_join_spec(artifact)
        try:
            client = self._get_overlap_annotator_client()
            return await annotate_overlaps(
                turns=formatted_turns,
                join_spec=join_spec,
                settings=self.overlap_settings,
                prompt_template=self._overlap_annotator_prompt,
                llm_client=client,
                concurrency=self.batch_concurrency,
            )
        except Exception:
            logger.exception(
                "Overlap annotation failed for %s; proceeding without overlap",
                meeting_id,
            )
            return anchors

    def _meeting_join_spec(self, artifact: dict[str, Any]) -> MeetingJoinSpec | None:
        raw = artifact.get("join_spec")
        if raw is None:
            return None
        if isinstance(raw, MeetingJoinSpec):
            return raw
        try:
            return MeetingJoinSpec.model_validate(raw)
        except Exception:
            logger.warning(
                "Could not validate join_spec for meeting; overlap annotator will skip join context",
            )
            return None

    def _get_overlap_annotator_client(self) -> Any:
        if self._overlap_annotator_client is None:
            self._overlap_annotator_client = LLMClient(
                model=self.overlap_settings.annotator_model,
            )
        return self._overlap_annotator_client

    def _get_backchannel_classifier_client(self) -> Any:
        if self._backchannel_classifier_client is None:
            self._backchannel_classifier_client = LLMClient(
                model=self.backchannel_settings.classifier_model,
            )
        return self._backchannel_classifier_client

    def _get_backchannel_anchor_client(self) -> Any:
        if self._backchannel_anchor_client is None:
            self._backchannel_anchor_client = LLMClient(
                model=self.backchannel_settings.anchor_model,
            )
        return self._backchannel_anchor_client

    async def _classify_backchannels_raw(
        self,
        meeting_id: str,
        formatted_turns: list[dict[str, Any]],
    ) -> list[dict[str, Any] | None]:
        if not self.overlap_settings.enabled or not formatted_turns:
            return [None for _ in formatted_turns]
        try:
            client = self._get_backchannel_classifier_client()
            return await classify_backchannels_raw(
                turns=formatted_turns,
                settings=self.backchannel_settings,
                prompt_template=self._backchannel_classifier_prompt,
                llm_client=client,
            )
        except Exception:
            logger.exception(
                "Backchannel classifier failed for %s; keeping prior is_backchannel flags",
                meeting_id,
            )
            return [None for _ in formatted_turns]

    async def _extract_entities(
        self,
        meeting_id: str,
        formatted_turns: list[dict[str, Any]],
    ) -> list[TurnEntities]:
        try:
            entities = await extract_meeting_entities(
                turns=formatted_turns,
                org=self.org,
                projects=self.projects,
                language_name=self._language_name,
                prompt_template=self._entities_prompt,
                llm_client=self._get_llm_client(),
                reasoning_effort=self.entity_reasoning_effort,
                concurrency=self.batch_concurrency,
            )
            return filter_entities_to_surfaces(entities, formatted_turns)
        except Exception:
            logger.exception("Entity extraction failed for %s", meeting_id)
            return []

    async def _anchor_backchannels(
        self,
        meeting_id: str,
        formatted_turns: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not self.overlap_settings.enabled or not formatted_turns:
            return formatted_turns
        if not any(t.get("is_backchannel") for t in formatted_turns):
            return formatted_turns
        try:
            client = self._get_backchannel_anchor_client()
            anchors = await anchor_backchannels(
                turns=formatted_turns,
                settings=self.backchannel_settings,
                prompt_template=self._backchannel_anchor_prompt,
                llm_client=client,
            )
        except Exception:
            logger.exception(
                "Backchannel anchor placement failed for %s; BCs fall back to FTO",
                meeting_id,
            )
            return formatted_turns
        attach_backchannel_anchors(formatted_turns, anchors)
        return validate_formatted_turns(formatted_turns)

    def _attach_unclear_spans(
        self,
        meeting_id: str,
        formatted_turns: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if self.unclear_rate <= 0:
            return formatted_turns
        per_turn = sample_unclear_spans(
            turns=formatted_turns,
            unclear_rate=self.unclear_rate,
            run_seed=self.seed,
            meeting_id=meeting_id,
        )
        for turn, spans in zip(formatted_turns, per_turn, strict=False):
            if not spans:
                continue
            turn["unclear_spans"] = [span.model_dump(mode="json") for span in spans]
        return validate_formatted_turns(formatted_turns)

    async def _write_transcript_artifacts(
        self,
        meeting_id: str,
        meeting_dir: Path,
        formatted_turns: list[dict[str, Any]],
        *,
        entities: list[TurnEntities] | None = None,
    ) -> None:
        ami_turns = derive_ami_view(formatted_turns)
        fileio.write_json(
            meeting_dir / "verbatim_ami.json",
            validate_ami_turns([t.model_dump(mode="json") for t in ami_turns]),
        )

        # Discriminator-facing plain-text view (Speaker N: <text>), aligned with
        # the AMI normalizer surface so the transcript_discriminability metric
        # can compare synthetic vs. real on a single shared format.
        fileio.write_text(
            meeting_dir / "normalized_for_realism.txt",
            normalize_synthetic_turns(formatted_turns),
        )

        entities_path = meeting_dir / "entities.json"
        if entities_path.exists():
            return
        if entities is None:
            entities = await self._extract_entities(meeting_id, formatted_turns)
        fileio.write_json(
            entities_path,
            [e.model_dump(mode="json") for e in entities],
        )

    def _write_tts_tasks(
        self,
        meeting_id: str,
        meeting_dir: Path,
        formatted_turns: list[dict[str, Any]],
    ) -> set[str]:
        tasks: list[TTSTurnTask] = []
        speakers: set[str] = set()
        for idx, turn in enumerate(formatted_turns):
            speaker_id = str(turn.get("speaker_id") or "")
            if not speaker_id:
                logger.warning(
                    "Skipping turn %d in %s: missing speaker_id", idx, meeting_id,
                )
                continue
            speakers.add(speaker_id)
            speaker_key = str(turn.get("speaker_key") or speaker_id)
            output_path = (
                self.run_dir
                / "step_8_tts_batch"
                / "meetings"
                / meeting_id
                / "turns"
                / f"turn_{idx:03d}.wav"
            )
            tasks.append(
                TTSTurnTask(
                    task_id=f"{self.run_id}::{meeting_id}::turn_{idx:03d}",
                    run_id=self.run_id,
                    run_dir=str(self.run_dir),
                    meeting_id=meeting_id,
                    turn_index=idx,
                    speaker_id=speaker_id,
                    speaker_key=speaker_key,
                    text=turn.get("tts_text") or turn.get("text", ""),
                    output_path=str(output_path),
                    language=self.language,
                ),
            )

        fileio.write_json(
            meeting_dir / "tts_tasks.json",
            [task.model_dump(mode="json") for task in tasks],
        )
        return speakers

    async def _write_enrollment_artifacts(self, speaker_id: str) -> None:
        speaker_dir = self.output_dir / "speakers" / speaker_id
        fileio.makedirs(speaker_dir)

        voice_id = self.voice_assignments.get(speaker_id) or self._person_voice_id(speaker_id)
        source_audio_path, source_transcript = self._stage_source_prompt(
            speaker_id, voice_id, speaker_dir,
        )

        spec = EnrollmentSpec(
            speaker_id=speaker_id,
            voice_id=voice_id,
            language=self.language,
            source_audio_path=source_audio_path,
            source_transcript=source_transcript,
        )
        fileio.write_json(
            speaker_dir / "enrollment_spec.json",
            spec.model_dump(mode="json"),
        )

        bundle = await self._generate_enrollment_text_bundle(
            speaker_id=speaker_id,
            source_transcript=source_transcript,
        )
        fileio.write_json(
            speaker_dir / "enrollment_text_candidates.json",
            bundle.model_dump(mode="json"),
        )

    def _person_voice_id(self, speaker_id: str) -> str:
        person = self.org.get_person(speaker_id)
        voice_id = person.voice_id if person is not None else None
        return str(voice_id or "")

    def _stage_source_prompt(
        self,
        speaker_id: str,
        voice_id: str,
        speaker_dir: Path,
    ) -> tuple[str, str | None]:
        source_audio = self._resolve_voice_sample(voice_id)
        source_transcript = self._load_voice_ref_text(voice_id)

        if source_audio is None or not source_audio.exists():
            raise FileNotFoundError(
                f"No source prompt audio found for speaker {speaker_id} "
                f"(voice_id={voice_id!r}) under voice cache "
                f"{self._voice_cache_path()}. Stage the voice cache before "
                f"running with generate_audio=true.",
            )
        staged_audio = speaker_dir / "source_prompt.wav"
        shutil.copy2(source_audio, staged_audio)
        staged_audio_path = str(staged_audio)

        fileio.write_text(
            speaker_dir / "source_prompt_transcript.txt",
            (source_transcript or "").strip(),
        )
        return staged_audio_path, source_transcript

    def _voice_cache_path(self) -> Path:
        # Relative paths are resolved against the working directory, as in
        # steps 1 and 8.
        return Path(self.voice_cache_dir)

    def _load_voice_catalog(self) -> dict[str, str]:
        if self._voice_catalog_cache is not None:
            return self._voice_catalog_cache

        catalog_path = self._voice_cache_path() / "voice_catalog.json"
        if not catalog_path.exists():
            self._voice_catalog_cache = {}
            return self._voice_catalog_cache

        try:
            catalog = fileio.read_json(catalog_path)
            speakers = catalog.get("speakers", {}) if isinstance(catalog, dict) else {}
            self._voice_catalog_cache = {
                str(speaker_id): str((meta or {}).get("text") or "").strip()
                for speaker_id, meta in speakers.items()
            }
        except Exception:
            logger.exception("Failed reading voice catalog at %s", catalog_path)
            self._voice_catalog_cache = {}
        return self._voice_catalog_cache

    def _load_voice_ref_text(self, voice_id: str) -> str | None:
        if not voice_id:
            return None
        text = self._load_voice_catalog().get(voice_id, "").strip()
        return text or None

    def _resolve_voice_sample(self, voice_id: str) -> Path | None:
        if not voice_id:
            return None

        cache_dir = self._voice_cache_path()
        for gender_dir in ("male", "female"):
            candidate = cache_dir / gender_dir / f"{voice_id}.wav"
            if candidate.exists():
                return candidate

        candidate = cache_dir / "samples" / f"{voice_id}.wav"
        if candidate.exists():
            return candidate

        candidate = cache_dir / f"{voice_id}.wav"
        if candidate.exists():
            return candidate
        return None

    async def _prepare_turns(
        self,
        transcript_json: list[dict],
        *,
        meeting_id: str = "",
    ) -> list[dict[str, Any]]:
        turns: list[dict[str, Any]] = []
        for turn in transcript_json:
            speaker_name = turn.get("speaker_name", "Unknown")
            speaker_id = turn.get("speaker_id", "")
            text = turn.get("text", "").strip()

            if not text:
                continue
            if not speaker_id:
                logger.warning("Skipping transcript turn without speaker_id during prep")
                continue

            person = self.org.get_person(speaker_id)
            entry: dict[str, Any] = {
                "speaker": speaker_name,
                "speaker_id": speaker_id,
                "speaker_key": speaker_id,
                "canonical_text": text,
                "semantic_verbatim_text": text,
                "tts_text": text,
                "text": text,
                "is_backchannel": False,
            }

            if person and person.speaking_style and person.speaking_style.disfluency:
                profile = person.speaking_style.disfluency
                entry["disfluency_profile"] = profile.model_dump(exclude_defaults=True)

            turns.append(entry)

        if not turns:
            return []

        formatter_kwargs: dict[str, Any] = {
            "system_prompt": self._spoken_form_prompt,
            "model": self.formatting_model,
        }
        # The default formatter (``format_turns_for_speech``) accepts
        # sanitizer wiring; injected fakes used in unit tests may not, so
        # only forward the kwargs when the callable's signature supports them.
        if _formatter_accepts_kwarg(self._formatter_fn, "max_truncation_rate"):
            formatter_kwargs["run_id"] = self.run_id
            formatter_kwargs["meeting_id"] = meeting_id
            formatter_kwargs["max_truncation_rate"] = self.max_truncation_rate
        if _formatter_accepts_kwarg(self._formatter_fn, "concurrency"):
            formatter_kwargs["concurrency"] = self.batch_concurrency
            formatter_kwargs["reasoning_effort"] = self.formatting_reasoning_effort
        formatted = self._formatter_fn(turns, **formatter_kwargs)
        if inspect.isawaitable(formatted):
            formatted = await formatted

        validated = validate_formatted_turns(formatted)
        for i, turn in enumerate(validated):
            canonical_text = str(turns[i]["canonical_text"])
            # `canonical_text` is step 4's post-refinement transcript text
            # (also persisted at `transcript_semantic.json`). The formatter
            # may produce a slightly different `semantic_verbatim_text` with
            # an accompanying `semantic_edit_layer` that diffs canonical →
            # semantic — keep that intact so the layer's reconstruction
            # contract holds.
            semantic_text = str(
                turn.get("semantic_verbatim_text")
                or turn.get("full_written_text")
                or canonical_text
            ).strip()
            tts_text = str(
                turn.get("tts_text")
                or turn.get("literal_spoken_text")
                or turn.get("literal_text")
                or turn.get("text")
                or canonical_text
            ).strip()
            turn["canonical_text"] = canonical_text
            turn["semantic_verbatim_text"] = semantic_text
            turn["tts_text"] = tts_text
            turn["text"] = tts_text
            turn["original_text"] = canonical_text
            turn["text_original"] = canonical_text
            turn["is_backchannel"] = turns[i]["is_backchannel"]
            turn["speaker_id"] = turns[i]["speaker_id"]
            turn["speaker_key"] = turns[i]["speaker_key"]

        return validate_formatted_turns(validated)

    async def _generate_enrollment_text_bundle(
        self,
        *,
        speaker_id: str,
        source_transcript: str | None,
    ) -> EnrollmentTextBundle:
        candidate_plan = self._candidate_blueprints()
        system_prompt = self._render_enrollment_prompt(
            speaker_id=speaker_id,
            source_transcript=source_transcript,
            candidate_plan=candidate_plan,
        )
        response = await self._get_llm_client().create_message(
            system=system_prompt,
            messages=[{
                "role": "user",
                "content": (
                    f"Generate the enrollment-text candidates for speaker {speaker_id} "
                    f"using the provided candidate plan. Return JSON only."
                ),
            }],
            max_tokens=4096,
            temperature=0.0,
        )
        tracker.track_llm_response(response)
        payload = extract_json(response.text)
        if isinstance(payload, dict):
            payload.setdefault("speaker_id", speaker_id)
            payload.setdefault("language", self.language)
            candidates = payload.get("candidates")
            if isinstance(candidates, list):
                payload["candidates"] = self._apply_candidate_defaults(
                    candidates,
                    candidate_plan=candidate_plan,
                )

        bundle = validate_enrollment_text_bundle_for_run(
            payload,
            expected_language=self.language,
            expected_candidate_count=self.enrollment_candidate_count,
        )
        return bundle

    def _get_llm_client(self) -> Any:
        if self._llm_client is None:
            self._llm_client = LLMClient(model=self.enrollment_text_model)
        return self._llm_client

    def _render_enrollment_prompt(
        self,
        *,
        speaker_id: str,
        source_transcript: str | None,
        candidate_plan: list[dict[str, str]],
    ) -> str:
        speaker_summary = self._speaker_summary(speaker_id)
        representative_utterances = self._representative_utterances(speaker_id)
        authored_snippets = self._authored_snippets(speaker_id)

        replacements = {
            "{{LANGUAGE_NAME}}": self._language_name,
            "{{SPEAKER_ID}}": speaker_id,
            "{{TARGET_SECONDS}}": str(self.enrollment_target_seconds),
            "{{MIN_SECONDS}}": str(self.enrollment_min_seconds),
            "{{MAX_SECONDS}}": str(self.enrollment_max_seconds),
            "{{CANDIDATE_COUNT}}": str(self.enrollment_candidate_count),
            "{{CANDIDATE_PLAN_JSON}}": json.dumps(candidate_plan, ensure_ascii=False, indent=2),
            "{{SPEAKER_SUMMARY}}": speaker_summary,
            "{{SOURCE_TRANSCRIPT}}": source_transcript or "Unavailable.",
            "{{REPRESENTATIVE_UTTERANCES}}": representative_utterances,
            "{{AUTHORED_SNIPPETS}}": authored_snippets,
        }
        return render_prompt(self._enrollment_text_prompt, replacements)

    def _speaker_summary(self, speaker_id: str) -> str:
        person = self.org.get_person(speaker_id)
        if person is None:
            return f"Speaker {speaker_id} has no org profile available."

        lines = [f"Name: {person.name}"]
        if isinstance(person, Person):
            team_name = ""
            if person.team_id:
                team = self.org.teams.get(person.team_id)
                team_name = team.name if team else person.team_id
            lines.extend([
                "Type: internal employee",
                f"Role: {person.role.title or 'Unknown'}",
                f"Department: {person.role.department or 'Unknown'}",
                f"Level: {person.role.level or 'Unknown'}",
                f"Team: {team_name or 'Unknown'}",
            ])
        elif isinstance(person, ExternalPerson):
            lines.extend([
                "Type: external participant",
                f"Organization: {person.organization or 'Unknown'}",
                f"Relationship: {person.relationship_type or 'Unknown'}",
                f"Role: {person.role_title or 'Unknown'}",
            ])

        if person.personality_traits:
            lines.append(f"Personality traits: {', '.join(person.personality_traits)}")
        if person.expertise:
            lines.append(f"Expertise: {', '.join(person.expertise)}")
        speaking_style = person.speaking_style
        if speaking_style is not None:
            style_parts = [
                f"pace={speaking_style.pace}" if speaking_style.pace else "",
                f"verbosity={speaking_style.verbosity}" if speaking_style.verbosity else "",
                f"accent={speaking_style.accent}" if speaking_style.accent else "",
            ]
            notable_patterns = speaking_style.notable_patterns or []
            if notable_patterns:
                style_parts.append(f"patterns={', '.join(notable_patterns)}")
            lines.append(
                "Speaking style: "
                + ", ".join(part for part in style_parts if part)
            )
            disfluency = speaking_style.disfluency
            if disfluency is not None:
                lines.append(
                    "Disfluency profile: "
                    f"fillers={disfluency.filler_words or []}; "
                    f"filler_rate={disfluency.filler_rate}; "
                    f"repetition_rate={disfluency.repetition_rate}; "
                    f"restart_rate={disfluency.restart_rate}; "
                    f"truncation_rate={disfluency.truncation_rate}; "
                    f"elongation_rate={disfluency.elongation_rate}; "
                    f"backchannels={disfluency.backchannel_phrases or []}"
                )
        return "\n".join(lines)

    def _representative_utterances(self, speaker_id: str) -> str:
        utterances: list[str] = []
        seen: set[str] = set()
        for artifact_id in sorted(self.artifacts):
            artifact = self.artifacts[artifact_id]
            if artifact.get("type") != "meeting":
                continue
            for turn in artifact.get("transcript_json", []) or []:
                if turn.get("speaker_id") != speaker_id:
                    continue
                text = " ".join(str(turn.get("text") or "").split())
                if not text or len(text.split()) < 4:
                    continue
                if text in seen:
                    continue
                seen.add(text)
                utterances.append(text)
                if len(utterances) >= 5:
                    return "\n".join(f"- {item}" for item in utterances)
        if not utterances:
            return "- No representative meeting utterances available."
        return "\n".join(f"- {item}" for item in utterances)

    def _candidate_blueprints(self) -> list[dict[str, str]]:
        topical_seeds = [
            "recent_work",
            "upcoming_priorities",
            "process_style",
            "collaboration_pattern",
            "customer_context",
            "decision_making",
        ]
        duration_targets = ["target", "short", "long"]
        count = max(1, self.enrollment_candidate_count)
        return [
            {
                "id": f"c{index + 1}",
                "topical_seed": topical_seeds[index % len(topical_seeds)],
                "duration_target": duration_targets[index % len(duration_targets)],
            }
            for index in range(count)
        ]

    def _apply_candidate_defaults(
        self,
        candidates: list[Any],
        *,
        candidate_plan: list[dict[str, str]],
    ) -> list[dict[str, Any]]:
        plan_by_id = {candidate["id"]: candidate for candidate in candidate_plan}
        normalized: list[dict[str, Any]] = []
        for index, raw_candidate in enumerate(candidates):
            if not isinstance(raw_candidate, dict):
                normalized.append(raw_candidate)
                continue
            candidate = dict(raw_candidate)
            fallback = candidate_plan[index] if index < len(candidate_plan) else None
            plan_match = plan_by_id.get(str(candidate.get("id") or "").strip())
            candidate.setdefault("id", (plan_match or fallback or {}).get("id"))
            candidate.setdefault("topical_seed", (plan_match or fallback or {}).get("topical_seed"))
            candidate.setdefault("duration_target", (plan_match or fallback or {}).get("duration_target"))
            normalized.append(candidate)
        return normalized

    def _authored_snippets(self, speaker_id: str) -> str:
        snippets: list[str] = []
        seen: set[str] = set()
        for artifact_id in sorted(self.artifacts):
            artifact = self.artifacts[artifact_id]
            artifact_type = artifact.get("type")
            if artifact_type == "email_thread":
                for message in artifact.get("messages", []) or []:
                    if message.get("sender_id") != speaker_id:
                        continue
                    body = " ".join(str(message.get("body") or "").split())
                    if body and body not in seen:
                        seen.add(body)
                        snippets.append(body)
            elif artifact_type in {"document", "slide_deck"}:
                author_id = artifact.get("author_id") or artifact.get("owner_id")
                if author_id != speaker_id:
                    continue
                content = " ".join(
                    str(
                        artifact.get("content")
                        or artifact.get("body")
                        or artifact.get("summary")
                        or "",
                    ).split(),
                )
                if content and content not in seen:
                    seen.add(content)
                    snippets.append(content)
            if len(snippets) >= 3:
                break
        if not snippets:
            return "- No authored email or document snippets available."
        return "\n".join(f"- {snippet}" for snippet in snippets[:3])

    def _load_from_disk(self) -> dict:
        formatted_turns_by_meeting: dict[str, list[dict[str, Any]]] = {}
        meetings_dir = self.output_dir / "meetings"
        if meetings_dir.exists():
            for meeting_dir in sorted(meetings_dir.iterdir()):
                if not meeting_dir.is_dir():
                    continue
                ft_path = meeting_dir / "formatted_turns.json"
                if ft_path.exists():
                    formatted_turns_by_meeting[meeting_dir.name] = fileio.read_json(ft_path)

        speaker_enrollments: dict[str, dict[str, Any]] = {}
        speakers_dir = self.output_dir / "speakers"
        if speakers_dir.exists():
            for speaker_dir in sorted(speakers_dir.iterdir()):
                if not speaker_dir.is_dir():
                    continue
                bundle_path = speaker_dir / "enrollment_text_candidates.json"
                spec_path = speaker_dir / "enrollment_spec.json"
                if bundle_path.exists() or spec_path.exists():
                    speaker_enrollments[speaker_dir.name] = {}
                if bundle_path.exists():
                    speaker_enrollments[speaker_dir.name]["bundle"] = fileio.read_json(bundle_path)
                if spec_path.exists():
                    speaker_enrollments[speaker_dir.name]["spec"] = fileio.read_json(spec_path)

        return {
            "formatted_turns_by_meeting": formatted_turns_by_meeting,
            "speaker_enrollments": speaker_enrollments,
        }
