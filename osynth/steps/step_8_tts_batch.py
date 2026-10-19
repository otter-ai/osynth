"""Step 8: TTS — synthesize enrollment clips and turn WAVs from step 7 prep.

- synthesize one reusable enrollment clip per speaker,
- synthesize each meeting turn using only the enrollment clip + enrollment text,
- persist structured per-task results and resumable status.

"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ..models.settings import SyntheticDataSettings
from ..models.tts import (
    EnrollmentSpec,
    TTSEnrollmentTask,
    TTSResultRecord,
    TTSTurnTask,
    validate_enrollment_spec,
    validate_enrollment_text_bundle_for_run,
    validate_formatted_turns,
    validate_tts_enrollment_metadata,
    validate_tts_turn_tasks,
)
from ..tts.synthesizer import TTSSynthesizer
from ..utils import fileio

logger = logging.getLogger(__name__)


class Step8TTSBatch:
    """Run TTS synthesis for prepared speakers and meetings."""

    def __init__(
        self,
        settings: SyntheticDataSettings,
        output_root: str,
        data: dict,
        *,
        synthesizer_factory: Any | None = None,
    ) -> None:
        self.settings = settings
        self.data = data
        self.run_dir = Path(output_root)
        self.prep_dir = self.run_dir / "step_7_speech_prep"
        self.output_dir = self.run_dir / "step_8_tts_batch"
        self.generate_audio: bool = settings.generation.generate_audio
        self.run_id: str = settings.run_id

        self.tts_model: str = settings.tts.model
        self.voice_cache_dir: str = settings.tts.voice_cache_dir
        self.tts_mode: str = settings.tts.mode
        self.language: str = settings.language
        self.enrollment_target_seconds: int = settings.tts.enrollment_target_seconds
        self.enrollment_min_seconds: int = settings.tts.enrollment_min_seconds
        self.enrollment_max_seconds: int = settings.tts.enrollment_max_seconds
        self.enrollment_candidate_count: int = settings.tts.enrollment_candidate_count
        self.max_turn_duration_s: float = settings.tts.max_turn_duration_s
        self.max_seconds_per_char: float = settings.tts.max_seconds_per_char
        self.max_retries: int = settings.tts.max_retries

        self._synthesizer_factory = synthesizer_factory or TTSSynthesizer

    async def run(self, force: bool = False) -> dict:
        if not self.generate_audio:
            logger.info("Audio generation disabled in config, skipping TTS batch.")
            return {}
        status_path = self.output_dir / "tts_status.json"
        if status_path.exists() and not force:
            if self._all_completed(fileio.read_json(status_path)):
                logger.info("Step 8 TTS outputs exist, skipping.")
                return self._load_from_disk()
            # Completed enrollments and turn WAVs are reused below, so this
            # only synthesizes what is missing.
            logger.info("Step 8 has incomplete speakers or meetings; resuming.")

        fileio.makedirs(self.output_dir)
        fileio.makedirs(self.output_dir / "meetings")
        fileio.makedirs(self.output_dir / "speakers")

        prep_meetings_dir = self.prep_dir / "meetings"
        if not prep_meetings_dir.exists():
            logger.warning("No prep outputs at %s, run step_7 first", prep_meetings_dir)
            return {}

        synthesizer = self._synthesizer_factory(
            model_id=self.tts_model,
            voice_cache_dir=self.voice_cache_dir,
            mode=self.tts_mode,
            language=self.language,
            enrollment_target_seconds=self.enrollment_target_seconds,
            enrollment_min_seconds=self.enrollment_min_seconds,
            enrollment_max_seconds=self.enrollment_max_seconds,
            max_turn_duration_s=self.max_turn_duration_s,
            max_seconds_per_char=self.max_seconds_per_char,
            max_retries=self.max_retries,
            short_text_config=self.settings.tts.short_text,
        )
        return self._run(synthesizer)

    @staticmethod
    def _all_completed(status: dict) -> bool:
        statuses = [
            *(status.get("speakers") or {}).values(),
            *(status.get("meetings") or {}).values(),
        ]
        return all(s == "completed" for s in statuses)

    def _run(self, synthesizer: TTSSynthesizer) -> dict:
        speaker_results = self._synthesize_enrollments(synthesizer)
        meeting_results, meeting_status = self._synthesize_turns(synthesizer, speaker_results)

        completed_turn_paths = {
            meeting_id: [
                record.output_path
                for record in records
                if record.status == "completed" and record.output_path
            ]
            for meeting_id, records in meeting_results.items()
        }
        fileio.write_json(self.output_dir / "turn_results.json", completed_turn_paths)
        fileio.write_json(
            self.output_dir / "enrollment_results.json",
            {
                speaker_id: result.model_dump(mode="json")
                for speaker_id, result in speaker_results.items()
            },
        )
        fileio.write_json(
            self.output_dir / "tts_status.json",
            {
                "mode": self.tts_mode,
                "speakers": {
                    speaker_id: result.status
                    for speaker_id, result in speaker_results.items()
                },
                "meetings": meeting_status,
            },
        )
        return {
            "turn_results": completed_turn_paths,
            "enrollment_results": {
                speaker_id: result.model_dump(mode="json")
                for speaker_id, result in speaker_results.items()
            },
        }

    def _synthesize_enrollments(
        self,
        synthesizer: TTSSynthesizer,
    ) -> dict[str, TTSResultRecord]:
        speaker_results: dict[str, TTSResultRecord] = {}
        speakers_dir = self.prep_dir / "speakers"
        if not speakers_dir.exists():
            return speaker_results

        for speaker_dir in sorted(p for p in speakers_dir.iterdir() if p.is_dir()):
            speaker_id = speaker_dir.name
            task = self._build_enrollment_task(speaker_id)
            if task is None:
                speaker_results[speaker_id] = TTSResultRecord(
                    task_id=f"{self.run_id}::{speaker_id}::enrollment",
                    task_type="enrollment",
                    status="failed",
                    run_id=self.run_id,
                    run_dir=str(self.run_dir),
                    speaker_id=speaker_id,
                    attempt_count=0,
                    error="missing enrollment spec or candidate bundle",
                )
                continue

            if Path(task.output_audio_path).exists() and Path(task.output_metadata_path).exists():
                metadata = validate_tts_enrollment_metadata(
                    fileio.read_json(task.output_metadata_path),
                )
                speaker_results[speaker_id] = TTSResultRecord(
                    task_id=task.task_id,
                    task_type="enrollment",
                    status="completed",
                    run_id=task.run_id,
                    run_dir=task.run_dir,
                    speaker_id=speaker_id,
                    output_path=task.output_audio_path,
                    output_metadata_path=task.output_metadata_path,
                    duration_s=metadata.duration_s,
                    attempt_count=metadata.attempt_count,
                    selected_candidate_id=metadata.selected_candidate_id,
                    error=metadata.duration_status
                    if metadata.duration_status not in {None, "accepted"}
                    else None,
                )
                continue

            try:
                speaker_results[speaker_id] = self._retry_enrollment_task(synthesizer, task)
            except Exception as exc:
                logger.exception("Enrollment synthesis failed for %s", speaker_id)
                speaker_results[speaker_id] = TTSResultRecord(
                    task_id=task.task_id,
                    task_type="enrollment",
                    status="failed",
                    run_id=task.run_id,
                    run_dir=task.run_dir,
                    speaker_id=speaker_id,
                    output_path=task.output_audio_path,
                    output_metadata_path=task.output_metadata_path,
                    attempt_count=task.retry_count + 1,
                    error=str(exc),
                )
        return speaker_results

    def _synthesize_turns(
        self,
        synthesizer: TTSSynthesizer,
        speaker_results: dict[str, TTSResultRecord],
    ) -> tuple[dict[str, list[TTSResultRecord]], dict[str, str]]:
        meeting_results: dict[str, list[TTSResultRecord]] = {}
        meeting_status: dict[str, str] = {}

        prep_meetings_dir = self.prep_dir / "meetings"
        meeting_ids = sorted(d.name for d in prep_meetings_dir.iterdir() if d.is_dir())
        for meeting_id in meeting_ids:
            records: list[TTSResultRecord] = []

            formatted_turns = validate_formatted_turns(
                fileio.read_json(prep_meetings_dir / meeting_id / "formatted_turns.json"),
            )
            if not formatted_turns:
                meeting_results[meeting_id] = records
                meeting_status[meeting_id] = "completed"
                fileio.write_json(
                    self.output_dir / "meetings" / meeting_id / "turn_results.json",
                    [],
                )
                continue

            tasks = validate_tts_turn_tasks(
                fileio.read_json(prep_meetings_dir / meeting_id / "tts_tasks.json"),
            )
            fileio.makedirs(self.output_dir / "meetings" / meeting_id / "turns")

            for task in tasks:
                speaker_result = speaker_results.get(task.speaker_id)
                if speaker_result is None or speaker_result.status != "completed":
                    records.append(
                        TTSResultRecord(
                            task_id=task.task_id,
                            task_type="turn",
                            status="waiting_on_enrollment_failure",
                            run_id=task.run_id,
                            run_dir=task.run_dir,
                            speaker_id=task.speaker_id,
                            meeting_id=task.meeting_id,
                            turn_index=task.turn_index,
                            output_path=task.output_path,
                            attempt_count=task.retry_count,
                            error="speaker enrollment unavailable",
                        ),
                    )
                    continue

                output_path = Path(task.output_path)
                if output_path.exists():
                    records.append(
                        TTSResultRecord(
                            task_id=task.task_id,
                            task_type="turn",
                            status="completed",
                            run_id=task.run_id,
                            run_dir=task.run_dir,
                            speaker_id=task.speaker_id,
                            meeting_id=task.meeting_id,
                            turn_index=task.turn_index,
                            output_path=task.output_path,
                            attempt_count=task.retry_count + 1,
                        ),
                    )
                    continue

                enrollment_audio_path, enrollment_text = self._load_enrollment_inputs(task.speaker_id)
                try:
                    records.append(
                        self._retry_turn_task(
                            synthesizer,
                            task,
                            enrollment_audio_path=enrollment_audio_path,
                            enrollment_text=enrollment_text,
                        ),
                    )
                except Exception as exc:
                    logger.exception(
                        "Turn synthesis failed for %s turn %03d",
                        task.meeting_id, task.turn_index,
                    )
                    records.append(
                        TTSResultRecord(
                            task_id=task.task_id,
                            task_type="turn",
                            status="failed",
                            run_id=task.run_id,
                            run_dir=task.run_dir,
                            speaker_id=task.speaker_id,
                            meeting_id=task.meeting_id,
                            turn_index=task.turn_index,
                            output_path=task.output_path,
                            attempt_count=task.retry_count + 1,
                            error=str(exc),
                        ),
                    )

            meeting_results[meeting_id] = records
            if all(record.status == "completed" for record in records):
                meeting_status[meeting_id] = "completed"
            elif any(record.status == "failed" for record in records):
                meeting_status[meeting_id] = "failed"
            else:
                meeting_status[meeting_id] = "waiting_on_enrollment_failure"

            fileio.write_json(
                self.output_dir / "meetings" / meeting_id / "turn_results.json",
                [record.model_dump(mode="json") for record in records],
            )

        return meeting_results, meeting_status

    def _build_enrollment_task(self, speaker_id: str) -> TTSEnrollmentTask | None:
        speaker_dir = self.prep_dir / "speakers" / speaker_id
        spec_path = speaker_dir / "enrollment_spec.json"
        bundle_path = speaker_dir / "enrollment_text_candidates.json"
        if not spec_path.exists() or not bundle_path.exists():
            return None

        spec: EnrollmentSpec = validate_enrollment_spec(fileio.read_json(spec_path))
        bundle = validate_enrollment_text_bundle_for_run(
            fileio.read_json(bundle_path),
            expected_language=self.language,
            expected_candidate_count=self.enrollment_candidate_count,
        )

        speaker_output_dir = self.output_dir / "speakers" / speaker_id
        return TTSEnrollmentTask(
            task_id=f"{self.run_id}::{speaker_id}::enrollment",
            run_id=self.run_id,
            run_dir=str(self.run_dir),
            speaker_id=speaker_id,
            voice_id=spec.voice_id,
            language=spec.language,
            source_audio_path=spec.source_audio_path,
            source_transcript=spec.source_transcript,
            candidate_texts=bundle.candidates,
            output_audio_path=str(speaker_output_dir / "enrollment.wav"),
            output_metadata_path=str(speaker_output_dir / "enrollment_metadata.json"),
        )

    def _load_enrollment_inputs(self, speaker_id: str) -> tuple[str, str]:
        speaker_dir = self.output_dir / "speakers" / speaker_id
        metadata_path = speaker_dir / "enrollment_metadata.json"
        audio_path = speaker_dir / "enrollment.wav"
        text_path = speaker_dir / "enrollment_text.txt"

        if not metadata_path.exists() or not audio_path.exists() or not text_path.exists():
            raise FileNotFoundError(f"Enrollment outputs missing for speaker {speaker_id}")
        return str(audio_path), text_path.read_text(encoding="utf-8").strip()

    def _retry_enrollment_task(
        self,
        synthesizer: TTSSynthesizer,
        task: TTSEnrollmentTask,
    ) -> TTSResultRecord:
        return synthesizer.synthesize_enrollment(task)

    def _retry_turn_task(
        self,
        synthesizer: TTSSynthesizer,
        task: TTSTurnTask,
        *,
        enrollment_audio_path: str,
        enrollment_text: str,
    ) -> TTSResultRecord:
        current_task = task
        while True:
            try:
                return synthesizer.synthesize_turn(
                    current_task,
                    enrollment_audio_path=enrollment_audio_path,
                    enrollment_text=enrollment_text,
                )
            except Exception:
                attempt_count = current_task.retry_count + 1
                if attempt_count >= self.max_retries:
                    raise
                current_task = current_task.model_copy(update={"retry_count": attempt_count})


    def _load_from_disk(self) -> dict:
        turn_results: dict[str, list[str]] = {}
        turn_results_path = self.output_dir / "turn_results.json"
        if turn_results_path.exists():
            turn_results = json.loads(turn_results_path.read_text())

        enrollment_results: dict[str, dict[str, Any]] = {}
        enrollment_results_path = self.output_dir / "enrollment_results.json"
        if enrollment_results_path.exists():
            enrollment_results = json.loads(enrollment_results_path.read_text())

        return {
            "turn_results": turn_results,
            "enrollment_results": enrollment_results,
        }
