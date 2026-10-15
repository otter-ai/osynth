"""Shared helpers for enrollment-candidate evaluation and selection."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..models.tts import (
    EnrollmentCandidateAttempt,
    EnrollmentTextCandidate,
    TTSEnrollmentMetadata,
    TTSEnrollmentTask,
    TTSResultRecord,
)
from ..utils import fileio


def candidate_attempt_audio_path(output_audio_path: str | Path, candidate_id: str) -> Path:
    safe_candidate_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", candidate_id.strip()) or "candidate"
    return Path(output_audio_path).parent / "candidate_attempts" / f"{safe_candidate_id}.wav"


@dataclass(frozen=True)
class EnrollmentSelection:
    candidate: EnrollmentTextCandidate
    attempt: EnrollmentCandidateAttempt
    duration_status: str


def select_enrollment_candidate(
    candidate_texts: list[EnrollmentTextCandidate],
    candidate_attempts: list[EnrollmentCandidateAttempt],
    *,
    target_seconds: int,
) -> EnrollmentSelection | None:
    candidates_by_id = {candidate.id: candidate for candidate in candidate_texts}

    successful = [
        attempt
        for attempt in candidate_attempts
        if attempt.success and not attempt.degenerate
    ]
    if not successful:
        return None

    def _distance(attempt: EnrollmentCandidateAttempt) -> float:
        if attempt.duration_distance_s is not None:
            return attempt.duration_distance_s
        if attempt.duration_s is None:
            return float("inf")
        return abs(attempt.duration_s - target_seconds)

    def _select(
        attempts: list[EnrollmentCandidateAttempt],
        *,
        duration_status: str,
    ) -> EnrollmentSelection | None:
        if not attempts:
            return None
        best = min(attempts, key=lambda attempt: (_distance(attempt), attempt.candidate_id))
        candidate = candidates_by_id.get(best.candidate_id)
        if candidate is None:
            return None
        return EnrollmentSelection(
            candidate=candidate,
            attempt=best,
            duration_status=duration_status,
        )

    clean_in_band = [
        attempt
        for attempt in successful
        if attempt.in_band and not attempt.minor_degeneration
    ]
    selection = _select(clean_in_band, duration_status="accepted")
    if selection is not None:
        return selection

    minor_in_band = [
        attempt
        for attempt in successful
        if attempt.in_band and attempt.minor_degeneration
    ]
    selection = _select(minor_in_band, duration_status="accepted_minor_degeneration")
    if selection is not None:
        return selection

    clean_out_of_band = [
        attempt
        for attempt in successful
        if not attempt.in_band and not attempt.minor_degeneration
    ]
    return _select(clean_out_of_band, duration_status="duration_out_of_band")


def write_enrollment_output(
    task: TTSEnrollmentTask,
    *,
    selected_candidate: EnrollmentTextCandidate,
    selected_attempt: EnrollmentCandidateAttempt,
    candidate_attempts: list[EnrollmentCandidateAttempt],
    audio: Any,
    sample_rate: int,
    model_name: str,
    mode_tag: str,
) -> TTSResultRecord:
    import soundfile as sf

    output_audio_path = Path(task.output_audio_path)
    output_metadata_path = Path(task.output_metadata_path)
    fileio.makedirs(output_audio_path.parent)
    sf.write(str(output_audio_path), audio, sample_rate, format="WAV")
    fileio.write_text(output_audio_path.parent / "enrollment_text.txt", selected_candidate.text)

    metadata = TTSEnrollmentMetadata(
        speaker_id=task.speaker_id,
        voice_id=task.voice_id,
        language=task.language,
        source_prompt_path=task.source_audio_path,
        source_prompt_transcript=task.source_transcript,
        selected_candidate_id=selected_candidate.id,
        selected_enrollment_text=selected_candidate.text,
        duration_s=selected_attempt.duration_s,
        attempt_count=task.retry_count + 1,
        candidate_attempts=candidate_attempts,
        model=model_name,
        mode_tag=mode_tag,
        duration_status="accepted",
    )

    duration_status = "accepted"
    if selected_attempt.minor_degeneration:
        duration_status = "accepted_minor_degeneration"
    elif not selected_attempt.in_band:
        duration_status = "duration_out_of_band"
    metadata.duration_status = duration_status

    fileio.write_json(output_metadata_path, metadata.model_dump(mode="json"))
    return TTSResultRecord(
        task_id=task.task_id,
        task_type="enrollment",
        status="completed",
        run_id=task.run_id,
        run_dir=task.run_dir,
        speaker_id=task.speaker_id,
        output_path=str(output_audio_path),
        output_metadata_path=str(output_metadata_path),
        duration_s=selected_attempt.duration_s,
        attempt_count=task.retry_count + 1,
        selected_candidate_id=selected_candidate.id,
        error=duration_status if duration_status != "accepted" else None,
    )
