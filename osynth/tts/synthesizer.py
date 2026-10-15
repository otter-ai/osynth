"""Enrollment-first TTS synthesis.

The synthesizer produces:
- one reusable enrollment clip per speaker, then
- independent turn WAVs using only the enrollment clip + enrollment text.

"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..models.tts import (
    EnrollmentCandidateAttempt,
    TTSEnrollmentTask,
    TTSResultRecord,
    TTSTurnTask,
)
from ..utils import fileio
from .degeneration import MAX_SECONDS_PER_CHAR, MAX_TURN_DURATION_S, TTSDegenerationError
from .enrollment import (
    candidate_attempt_audio_path,
    select_enrollment_candidate,
    write_enrollment_output,
)
from .short_text import (
    coerce_short_text_config,
    short_text_attempt_inputs,
)

logger = logging.getLogger(__name__)

LANG_TO_TTS: dict[str, str] = {
    "en": "English",
    "de": "German",
    "es": "Spanish",
    "fr": "French",
    "pt": "Portuguese",
    "ko": "Korean",
}


class TTSSynthesizer:
    """Orchestrates TTS synthesis for enrollments and turns.

    Runs enrollment-first synthesis in-process with the Qwen TTS model.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        voice_cache_dir: str = "data/voice_cache",
        mode: str = "local",
        language: str = "en",
        enrollment_target_seconds: int = 15,
        enrollment_min_seconds: int = 10,
        enrollment_max_seconds: int = 20,
        max_turn_duration_s: float = MAX_TURN_DURATION_S,
        max_seconds_per_char: float = MAX_SECONDS_PER_CHAR,
        max_retries: int = 3,
        short_text_config: Any | None = None,
    ):
        self.model_id = model_id
        self.voice_cache_dir = Path(voice_cache_dir)
        if mode != "local":
            raise ValueError(f"Unsupported tts.mode {mode!r}. Use 'local'.")
        self.mode = mode
        self.language_code = language
        self.tts_language = LANG_TO_TTS.get(language, "English")
        self.enrollment_target_seconds = enrollment_target_seconds
        self.enrollment_min_seconds = enrollment_min_seconds
        self.enrollment_max_seconds = enrollment_max_seconds
        self.max_turn_duration_s = max_turn_duration_s
        self.max_seconds_per_char = max_seconds_per_char
        self.max_retries = max(1, max_retries)
        self.short_text_config = coerce_short_text_config(short_text_config)

        self._model = None
        self._voice_prompts: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public API: enrollment-first local path
    # ------------------------------------------------------------------

    def synthesize_enrollment(self, task: TTSEnrollmentTask) -> TTSResultRecord:
        return self._synthesize_enrollment_local(task)

    def synthesize_turn(
        self,
        task: TTSTurnTask,
        *,
        enrollment_audio_path: str,
        enrollment_text: str,
    ) -> TTSResultRecord:
        return self._synthesize_turn_local(
            task,
            enrollment_audio_path=enrollment_audio_path,
            enrollment_text=enrollment_text,
        )

    # ------------------------------------------------------------------
    # Local mode helpers
    # ------------------------------------------------------------------

    def _get_model(self):
        if self._model is None:
            import torch
            from qwen_tts import Qwen3TTSModel

            logger.info("Loading Qwen TTS model %s ...", self.model_id)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if device == "cuda" else torch.float32

            try:
                self._model = Qwen3TTSModel.from_pretrained(
                    self.model_id,
                    device_map=device,
                    dtype=dtype,
                    attn_implementation="kernels-community/flash-attn3",
                )
            except Exception:
                logger.warning("Flash-Attn3 unavailable, falling back to standard loading.")
                self._model = Qwen3TTSModel.from_pretrained(
                    self.model_id,
                    device_map=device,
                    dtype=dtype,
                )

            logger.info("Model loaded on %s.", device)
        return self._model

    def _get_voice_prompt(self, voice_id: str):
        if voice_id not in self._voice_prompts:
            sample_path = self._resolve_voice_sample(voice_id)
            if sample_path is None:
                raise FileNotFoundError(f"No voice sample found for {voice_id}")
            model = self._get_model()
            self._voice_prompts[voice_id] = model.create_voice_clone_prompt(
                ref_audio=str(sample_path),
                x_vector_only_mode=True,
            )
        return self._voice_prompts[voice_id]

    def _resolve_voice_sample(self, voice_id: str) -> Path | None:
        for gender_dir in ("male", "female"):
            path = self.voice_cache_dir / gender_dir / f"{voice_id}.wav"
            if path.exists():
                return path
        path = self.voice_cache_dir / "samples" / f"{voice_id}.wav"
        if path.exists():
            return path
        path = self.voice_cache_dir / f"{voice_id}.wav"
        if path.exists():
            return path
        return None

    def _generate_from_reference(
        self,
        *,
        text: str,
        voice_id: str,
        language: str | None = None,
        ref_audio_path: str | None = None,
        ref_text: str | None = None,
    ) -> tuple["np.ndarray", int, float]:
        from ..utils.usage_tracker import tracker

        model = self._get_model()
        t0 = time.monotonic()
        tts_language = LANG_TO_TTS.get(language or self.language_code, self.tts_language)

        if ref_audio_path and ref_text:
            wavs, sample_rate = model.generate_voice_clone(
                text=text,
                language=tts_language,
                ref_audio=_load_audio_reference(Path(ref_audio_path)),
                ref_text=ref_text,
                x_vector_only_mode=False,
            )
        else:
            wavs, sample_rate = model.generate_voice_clone(
                text=text,
                language=tts_language,
                voice_clone_prompt=self._get_voice_prompt(voice_id),
            )

        elapsed = time.monotonic() - t0
        audio = _audio_to_numpy(wavs[0])
        tracker.track_tts_call(text)
        tracker.track_gpu_seconds(elapsed)
        return audio, sample_rate, elapsed

    def _synthesize_enrollment_local(self, task: TTSEnrollmentTask) -> TTSResultRecord:
        import soundfile as sf_local

        source_audio_path = task.source_audio_path
        if not source_audio_path:
            resolved = self._resolve_voice_sample(task.voice_id)
            source_audio_path = str(resolved) if resolved is not None else None
        if not source_audio_path:
            return self._failed_enrollment_result(
                task,
                error=f"No source audio available for speaker {task.speaker_id}",
            )

        output_audio_path = Path(task.output_audio_path)
        fileio.makedirs(output_audio_path.parent)
        if not task.candidate_texts:
            return self._failed_enrollment_result(
                task,
                error=f"No enrollment text candidates for speaker {task.speaker_id}",
            )

        candidate_attempts: list[EnrollmentCandidateAttempt] = []
        successful_audio: dict[str, tuple[Any, int]] = {}

        for candidate in task.candidate_texts:
            candidate_attempt_path = candidate_attempt_audio_path(output_audio_path, candidate.id)
            duration_s: float | None = None
            duration_distance_s: float | None = None
            in_band = False
            for candidate_retry in range(self.max_retries):
                try:
                    audio, sample_rate, elapsed = self._generate_from_reference(
                        text=candidate.text,
                        voice_id=task.voice_id,
                        language=task.language,
                        ref_audio_path=source_audio_path,
                        ref_text=(task.source_transcript or "").strip() or None,
                    )
                    duration_s = len(audio) / sample_rate
                    duration_distance_s = abs(duration_s - self.enrollment_target_seconds)
                    in_band = self.enrollment_min_seconds <= duration_s <= self.enrollment_max_seconds
                    _validate_waveform(audio)
                    fileio.makedirs(candidate_attempt_path.parent)
                    sf_local.write(str(candidate_attempt_path), audio, sample_rate, format="WAV")
                    logger.info(
                        "Enrollment candidate %s for %s -> %.1fs audio (%.1fs synth)",
                        candidate.id, task.speaker_id, duration_s, elapsed,
                    )

                    failure_reason: str | None = None
                    if self._is_enrollment_duration_degenerate(candidate.text, duration_s):
                        failure_reason = self._enrollment_duration_error(candidate.text, duration_s)

                    candidate_attempts.append(
                        EnrollmentCandidateAttempt(
                            candidate_id=candidate.id,
                            topical_seed=candidate.topical_seed,
                            duration_target=candidate.duration_target,
                            success=failure_reason is None,
                            duration_s=duration_s,
                            in_band=in_band,
                            degenerate=failure_reason is not None,
                            failure_reason=failure_reason,
                            attempt_audio_path=str(candidate_attempt_path),
                            duration_distance_s=duration_distance_s,
                        ),
                    )
                    if failure_reason is None:
                        successful_audio[candidate.id] = (audio, sample_rate)
                    break
                except TTSDegenerationError as exc:
                    candidate_attempts.append(
                        EnrollmentCandidateAttempt(
                            candidate_id=candidate.id,
                            topical_seed=candidate.topical_seed,
                            duration_target=candidate.duration_target,
                            success=False,
                            duration_s=duration_s,
                            in_band=in_band,
                            degenerate=True,
                            failure_reason=str(exc),
                            attempt_audio_path=str(candidate_attempt_path)
                            if candidate_attempt_path.exists()
                            else None,
                            duration_distance_s=duration_distance_s,
                        ),
                    )
                    break
                except Exception as exc:
                    if candidate_retry + 1 < self.max_retries:
                        logger.warning(
                            "Retrying enrollment candidate %s for %s after error: %s",
                            candidate.id,
                            task.speaker_id,
                            exc,
                        )
                        continue
                    candidate_attempts.append(
                        EnrollmentCandidateAttempt(
                            candidate_id=candidate.id,
                            topical_seed=candidate.topical_seed,
                            duration_target=candidate.duration_target,
                            success=False,
                            duration_s=duration_s,
                            in_band=in_band,
                            degenerate=False,
                            failure_reason=str(exc),
                            attempt_audio_path=str(candidate_attempt_path)
                            if candidate_attempt_path.exists()
                            else None,
                            duration_distance_s=duration_distance_s,
                        ),
                    )

        selection = select_enrollment_candidate(
            task.candidate_texts,
            candidate_attempts,
            target_seconds=self.enrollment_target_seconds,
        )
        if selection is None:
            return self._failed_enrollment_result(
                task,
                error=self._summarize_enrollment_failures(candidate_attempts),
            )

        selected_audio = successful_audio.get(selection.candidate.id)
        if selected_audio is None:
            return self._failed_enrollment_result(
                task,
                error=(
                    f"Selected candidate {selection.candidate.id} for speaker {task.speaker_id} "
                    "has no cached audio payload"
                ),
            )
        audio, sample_rate = selected_audio
        if selection.duration_status == "accepted_minor_degeneration":
            selection.attempt.minor_degeneration = True
        return write_enrollment_output(
            task,
            selected_candidate=selection.candidate,
            selected_attempt=selection.attempt,
            candidate_attempts=candidate_attempts,
            audio=audio,
            sample_rate=sample_rate,
            model_name=self.model_id,
            mode_tag=self.mode,
        )

    def _synthesize_turn_local(
        self,
        task: TTSTurnTask,
        *,
        enrollment_audio_path: str,
        enrollment_text: str,
    ) -> TTSResultRecord:
        import soundfile as sf_local

        output_path = Path(task.output_path)
        fileio.makedirs(output_path.parent)

        max_expected_s = max(self.max_seconds_per_char * len(task.text), 10.0)
        attempt_records: list[dict[str, Any]] = []
        last_error: Exception | None = None
        for synth_input in short_text_attempt_inputs(task.text, self.short_text_config):
            try:
                audio, sample_rate, elapsed = self._generate_from_reference(
                    text=synth_input.synthesis_text,
                    voice_id=task.speaker_id,
                    language=task.language,
                    ref_audio_path=enrollment_audio_path,
                    ref_text=enrollment_text,
                )
                _validate_waveform(audio)
                audio_duration_s = len(audio) / sample_rate

                logger.info(
                    "Turn %s -> %.1fs audio (%.1fs synth, short_policy=%s)",
                    task.task_id, audio_duration_s, elapsed, synth_input.policy,
                )

                if (
                    audio_duration_s > self.max_turn_duration_s
                    or audio_duration_s > max_expected_s
                ):
                    raise TTSDegenerationError(
                        f"TTS degeneration detected on {task.task_id}: "
                        f"audio={audio_duration_s:.1f}s for {len(task.text)} chars "
                        f"(limit={min(self.max_turn_duration_s, max_expected_s):.0f}s). "
                        f"Text: {task.text[:120]!r}",
                    )

                attempt_records.append({
                    "synthesis_text": synth_input.synthesis_text,
                    "policy": synth_input.policy,
                    "used_fallback": synth_input.used_fallback,
                    "status": "completed",
                    "duration_s": audio_duration_s,
                    "error": None,
                })
                sf_local.write(str(output_path), audio, sample_rate, format="WAV")
                return TTSResultRecord(
                    task_id=task.task_id,
                    task_type="turn",
                    status="completed",
                    run_id=task.run_id,
                    run_dir=task.run_dir,
                    speaker_id=task.speaker_id,
                    meeting_id=task.meeting_id,
                    turn_index=task.turn_index,
                    output_path=str(output_path),
                    duration_s=audio_duration_s,
                    attempt_count=task.retry_count + 1,
                    original_text=task.text,
                    synthesis_text=synth_input.synthesis_text,
                    short_text_policy=synth_input.policy,
                    short_text_attempts=attempt_records,
                    used_short_text_fallback=synth_input.used_fallback,
                )
            except Exception as exc:
                last_error = exc
                attempt_records.append({
                    "synthesis_text": synth_input.synthesis_text,
                    "policy": synth_input.policy,
                    "used_fallback": synth_input.used_fallback,
                    "status": "failed",
                    "duration_s": None,
                    "error": str(exc),
                })
                logger.warning(
                    "Turn synthesis attempt failed for %s (short_policy=%s): %s",
                    task.task_id, synth_input.policy, exc,
                )

        raise RuntimeError(
            f"Turn synthesis failed for {task.task_id} after "
            f"{len(attempt_records)} short-text attempt(s)"
        ) from last_error

    def _is_enrollment_duration_degenerate(self, text: str, duration_s: float) -> bool:
        max_expected_s = max(self.max_seconds_per_char * len(text), 10.0)
        return duration_s > self.max_turn_duration_s or duration_s > max_expected_s

    def _enrollment_duration_error(self, text: str, duration_s: float) -> str:
        max_expected_s = max(self.max_seconds_per_char * len(text), 10.0)
        return (
            "Enrollment degeneration detected: "
            f"audio={duration_s:.1f}s for {len(text)} chars "
            f"(limit={min(self.max_turn_duration_s, max_expected_s):.0f}s). "
            f"Text: {text[:120]!r}"
        )

    def _failed_enrollment_result(
        self,
        task: TTSEnrollmentTask,
        *,
        error: str,
    ) -> TTSResultRecord:
        return TTSResultRecord(
            task_id=task.task_id,
            task_type="enrollment",
            status="failed",
            run_id=task.run_id,
            run_dir=task.run_dir,
            speaker_id=task.speaker_id,
            output_path=task.output_audio_path,
            output_metadata_path=task.output_metadata_path,
            attempt_count=task.retry_count + 1,
            error=error,
        )

    def _summarize_enrollment_failures(
        self,
        candidate_attempts: list[EnrollmentCandidateAttempt],
    ) -> str:
        if not candidate_attempts:
            return "No enrollment candidate produced usable audio"
        summaries = [
            f"{attempt.candidate_id}: {attempt.failure_reason or 'no usable audio'}"
            for attempt in candidate_attempts
            if not attempt.success
        ]
        if not summaries:
            return "No enrollment candidate produced usable audio"
        return "; ".join(summaries)


def _load_audio_reference(path: Path) -> tuple["np.ndarray", int]:
    import soundfile as sf

    audio, sample_rate = sf.read(str(path), dtype="float32")
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sample_rate


def _validate_waveform(audio: "np.ndarray") -> None:
    if audio.size == 0:
        raise TTSDegenerationError("Generated empty audio waveform")
    if np.isnan(audio).any():
        raise TTSDegenerationError("Generated NaN audio waveform")
    if audio.shape[0] < 128:
        raise TTSDegenerationError("Generated near-empty audio waveform")


def _audio_to_numpy(audio_data) -> "np.ndarray":
    try:
        import torch

        if isinstance(audio_data, torch.Tensor):
            return audio_data.detach().to("cpu", dtype=torch.float32).numpy()
        if isinstance(audio_data, (list, tuple)):
            converted = []
            for item in audio_data:
                if isinstance(item, torch.Tensor):
                    converted.append(item.detach().to("cpu", dtype=torch.float32).numpy())
                else:
                    converted.append(item)
            return np.asarray(converted, dtype=np.float32)
    except ImportError:
        pass

    if isinstance(audio_data, np.ndarray):
        return audio_data.astype(np.float32, copy=False)

    return np.asarray(audio_data, dtype=np.float32)
