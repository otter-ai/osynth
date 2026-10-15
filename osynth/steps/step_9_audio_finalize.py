"""Step 9: Audio finalize — timing, alignment, mixing, RTTM, transcripts.

Consumes per-turn WAVs from ``step_8_tts_batch`` together with the formatted
turn metadata from ``step_7_speech_prep`` and produces:

- ``combined.wav`` per meeting
- per-speaker WAV tracks
- timed transcript text
- RTTM diarization reference
- word-level alignment artifacts

This step performs no GPU TTS work.
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

from ..models.settings import SyntheticDataSettings
from ..models.tts import UnclearSpan, validate_formatted_turns
from ..tts.alignment import (
    collect_turn_alignments,
    compute_energy_alignments,
    compute_hybrid_alignments,
    create_nfa_manifest,
    run_nfa,
    save_alignments,
)
from ..tts.audio_mixer import (
    build_aligned_timing,
    build_timing,
    create_combined_audio,
    create_speaker_audio,
    format_timestamp_ms,
    get_audio_duration_ms,
)
from ..tts.unclear_audio import attenuate_unclear_spans
from ..utils import fileio

logger = logging.getLogger(__name__)


class Step9AudioFinalize:
    """Combine per-turn WAVs into final per-meeting audio artifacts."""

    def __init__(self, settings: SyntheticDataSettings, output_root: str, data: dict) -> None:
        self.settings = settings
        self.data = data
        self.run_dir = Path(output_root)
        self.prep_dir = self.run_dir / "step_7_speech_prep"
        self.tts_dir = self.run_dir / "step_8_tts_batch"
        self.output_dir = self.run_dir / "step_9_audio_finalize"
        self.generate_audio: bool = settings.generation.generate_audio
        self.artifacts: dict[str, dict] = data.get("artifacts", {})
        self.seed: int = settings.seed or 42

        sp = settings.speech_pipeline
        self.fto_xi: float = sp.fto_xi
        self.fto_omega: float = sp.fto_omega
        self.fto_alpha: float = sp.fto_alpha
        self.overlap_jitter_ms: int = sp.overlap.onset_jitter_ms
        self.overlap_min_post_gap_ms: int = sp.overlap.min_post_overlap_gap_ms
        self.backchannel_jitter_ms: int = sp.backchannel.onset_jitter_ms

        alignment = settings.alignment
        self.alignment_enabled: bool = alignment.enabled
        self.alignment_method: str = alignment.method
        self.nfa_model: str = alignment.nfa_model
        self.nfa_script: str | None = alignment.nfa_script
        self.nfa_batch_size: int = alignment.batch_size
        self.nfa_transcribe_device: str | None = alignment.nfa_transcribe_device
        self.nfa_viterbi_device: str | None = alignment.nfa_viterbi_device

    def _check_nfa_script(self) -> None:
        """Fail once, up front, if forced alignment is configured but unusable."""
        if not self.alignment_enabled or self.alignment_method == "energy":
            return
        if not self.nfa_script:
            raise ValueError(
                "alignment.nfa_script is not set. Point it at "
                "tools/nemo_forced_aligner/align.py in a NeMo checkout, or set "
                "alignment.method to 'energy' to skip forced alignment.",
            )
        if not Path(self.nfa_script).exists():
            raise FileNotFoundError(
                f"alignment.nfa_script not found: {Path(self.nfa_script).resolve()}. "
                "Clone NeMo (see README) or set alignment.method to 'energy'.",
            )

    async def run(self, force: bool = False) -> dict:
        if not self.generate_audio:
            logger.info("Audio generation disabled, skipping finalize.")
            return {}

        results_path = self.output_dir / "audio_results.json"
        if results_path.exists() and not force:
            logger.info("Step 9 finalize outputs exist, loading from disk.")
            return self._load_from_disk()

        self._check_nfa_script()
        fileio.makedirs(self.output_dir)

        prep_meetings_dir = self.prep_dir / "meetings"
        if not prep_meetings_dir.exists():
            logger.warning("No prep outputs at %s", prep_meetings_dir)
            return {}

        rng = random.Random(self.seed)
        audio_results: dict[str, dict] = {}
        meeting_ids = sorted(
            d.name for d in prep_meetings_dir.iterdir() if d.is_dir()
        )

        for meeting_id in meeting_ids:
            try:
                result = self._finalize_meeting(meeting_id, rng, force=force)
                if not result:
                    continue
                audio_results[meeting_id] = result

                # Mirror audio paths back onto the in-memory artifact for
                # downstream steps that read from data["artifacts"].
                artifact = self.artifacts.get(meeting_id)
                if artifact is not None:
                    artifact["audio_path"] = result.get("audio_path")
                    artifact["speaker_audio_paths"] = result.get("speaker_audio_paths", {})
                    artifact["timing_data"] = result.get("timing_data")
                    artifact["rttm_path"] = result.get("rttm_path")
            except Exception:
                logger.exception("Audio finalize failed for %s", meeting_id)

        if audio_results:
            fileio.write_json(
                results_path,
                {
                    mid: {k: str(v) if isinstance(v, Path) else v for k, v in r.items()}
                    for mid, r in audio_results.items()
                },
            )

        logger.info(
            "Step 9 finalize complete: %d/%d meetings",
            len(audio_results), len(meeting_ids),
        )
        return {"audio_results": audio_results}

    def _finalize_meeting(
        self,
        meeting_id: str,
        rng: random.Random,
        *,
        force: bool = False,
    ) -> dict:
        prep_meeting_dir = self.prep_dir / "meetings" / meeting_id
        tts_meeting_dir = self.tts_dir / "meetings" / meeting_id
        out_meeting_dir = self.output_dir / meeting_id
        fileio.makedirs(out_meeting_dir)

        formatted_turns_path = prep_meeting_dir / "formatted_turns.json"
        if not formatted_turns_path.exists():
            logger.warning("Missing formatted_turns for %s", meeting_id)
            return {}

        formatted_turns = validate_formatted_turns(
            fileio.read_json(formatted_turns_path)
        )
        if not formatted_turns:
            return {}

        turns_dir = tts_meeting_dir / "turns"
        if not turns_dir.exists():
            logger.warning("Missing turn WAVs for %s at %s", meeting_id, turns_dir)
            return {}

        # Turn WAVs are named by turn index (``turn_NNN.wav``); pair them with
        # formatted turns by index, not by sorted filename.
        expected = [turns_dir / f"turn_{i:03d}.wav" for i in range(len(formatted_turns))]
        present = [p.exists() for p in expected]
        n_rendered = present.index(False) if not all(present) else len(expected)
        if n_rendered == 0:
            logger.warning("No turn WAVs found for %s", meeting_id)
            return {}
        if any(present[n_rendered:]):
            missing = [i for i, ok in enumerate(present) if not ok]
            logger.warning(
                "Skipping %s: turn WAVs missing mid-meeting (turns %s); re-run "
                "step 8 to synthesize them.", meeting_id, missing[:10],
            )
            return {}
        if n_rendered < len(formatted_turns):
            logger.warning(
                "%s: only the first %d of %d turns have audio; rendering those.",
                meeting_id, n_rendered, len(formatted_turns),
            )
            formatted_turns = formatted_turns[:n_rendered]
        turn_paths = expected[:n_rendered]

        combined_path = out_meeting_dir / "combined.wav"
        timing_path = out_meeting_dir / "timing.json"
        if not force and combined_path.exists() and timing_path.exists():
            logger.info("Reusing cached finalize output for %s", meeting_id)
            cached = self._load_meeting_audio(out_meeting_dir)
            self._write_transcript_forms(cached["timing_data"], out_meeting_dir)
            return cached

        speaker_roster = _extract_speaker_roster(formatted_turns)
        speaker_keys = [speaker_key for speaker_key, _ in speaker_roster]
        durations_ms = [get_audio_duration_ms(p) for p in turn_paths]

        initial_timing = build_timing(
            formatted_turns, durations_ms, speaker_keys,
            fto_xi=self.fto_xi,
            fto_omega=self.fto_omega,
            fto_alpha=self.fto_alpha,
            rng=rng,
        )

        short_turn_stats: dict[str, int] = {}
        synthesized_text_by_turn = self._load_turn_synthesis_texts(meeting_id)
        turn_alignments = self._run_alignment(
            meeting_id, initial_timing, turns_dir, out_meeting_dir,
            stats=short_turn_stats,
            synthesized_text_by_turn=synthesized_text_by_turn,
        )

        if turn_alignments:
            timing_data = build_aligned_timing(
                initial_timing, turn_alignments,
                fto_xi=self.fto_xi,
                fto_omega=self.fto_omega,
                fto_alpha=self.fto_alpha,
                overlap_jitter_ms=self.overlap_jitter_ms,
                overlap_min_post_gap_ms=self.overlap_min_post_gap_ms,
                backchannel_jitter_ms=self.backchannel_jitter_ms,
                rng=rng,
            )
            save_alignments(turn_alignments, out_meeting_dir / "word_alignments.json")
        else:
            timing_data = initial_timing

        fileio.write_json(timing_path, timing_data)

        turn_paths = self._apply_unclear_attenuation(
            meeting_id, turn_paths, formatted_turns, turn_alignments, out_meeting_dir,
        )

        create_combined_audio(turn_paths, timing_data, combined_path)

        speaker_audio_paths: dict[str, str] = {}
        for i, (speaker_key, _name) in enumerate(speaker_roster):
            speaker_num = i + 1
            sp_path = out_meeting_dir / f"speaker{speaker_num}.wav"
            create_speaker_audio(turn_paths, timing_data, speaker_num, sp_path)
            speaker_audio_paths[speaker_key] = str(sp_path)

        self._write_transcript_forms(timing_data, out_meeting_dir)

        rttm_path = out_meeting_dir / "reference.rttm"
        self._write_rttm(meeting_id, timing_data, rttm_path)

        if short_turn_stats:
            timing_data["short_turn_stats"] = dict(short_turn_stats)
            fileio.write_json(timing_path, timing_data)

        return {
            "audio_path": str(combined_path),
            "speaker_audio_paths": speaker_audio_paths,
            "timing_data": timing_data,
            "rttm_path": str(rttm_path),
            "word_alignments_path": (
                str(out_meeting_dir / "word_alignments.json") if turn_alignments else None
            ),
            "overlap_stats": timing_data.get("overlap_stats"),
            "short_turn_stats": dict(short_turn_stats) if short_turn_stats else None,
        }

    def _write_transcript_forms(self, timing_data: dict, meeting_dir: Path) -> None:
        self._write_timed_transcript(
            timing_data,
            meeting_dir / "transcript.txt",
            text_field="tts_text",
        )
        self._write_timed_transcript(
            timing_data,
            meeting_dir / "transcript_semantic_verbatim.txt",
            text_field="semantic_verbatim_text",
        )
        self._write_timed_transcript(
            timing_data,
            meeting_dir / "transcript_canonical.txt",
            text_field="canonical_text",
        )

    def _apply_unclear_attenuation(
        self,
        meeting_id: str,
        turn_paths: list[Path],
        formatted_turns: list[dict],
        turn_alignments: list | None,
        out_meeting_dir: Path,
    ) -> list[Path]:
        """Acoustically muffle unclear word ranges; return the path list to mix.

        For turns with no unclear spans (or no per-word alignment), the original
        WAV is reused unchanged. Attenuated copies live under
        ``<out_meeting_dir>/unclear/turn_NNN.wav`` so the step_8 originals stay
        immutable and re-runs are idempotent.
        """
        any_unclear = any(turn.get("unclear_spans") for turn in formatted_turns)
        if not any_unclear:
            return turn_paths
        if not turn_alignments:
            logger.info(
                "Skipping unclear attenuation for %s: no word alignments", meeting_id,
            )
            return turn_paths

        try:
            import numpy as np
            import soundfile as sf
        except ImportError:
            logger.warning(
                "soundfile/numpy unavailable; skipping unclear attenuation for %s",
                meeting_id,
            )
            return turn_paths

        alignments_by_index = {ta.turn_index: ta for ta in turn_alignments}
        attenuated_dir = out_meeting_dir / "unclear"
        fileio.makedirs(attenuated_dir)
        result_paths: list[Path] = []
        for idx, src_path in enumerate(turn_paths):
            spans_raw = formatted_turns[idx].get("unclear_spans") if idx < len(formatted_turns) else None
            alignment = alignments_by_index.get(idx)
            if not spans_raw or alignment is None or not alignment.word_alignments:
                result_paths.append(src_path)
                continue
            spans = [UnclearSpan.model_validate(s) for s in spans_raw]
            audio, sr = sf.read(str(src_path), always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            try:
                out = attenuate_unclear_spans(
                    np.asarray(audio, dtype=np.float32),
                    int(sr),
                    alignment.word_alignments,
                    spans,
                )
            except Exception:
                logger.exception(
                    "Unclear attenuation failed for %s turn %d; using original",
                    meeting_id, idx,
                )
                result_paths.append(src_path)
                continue
            dst_path = attenuated_dir / src_path.name
            sf.write(str(dst_path), out, sr)
            result_paths.append(dst_path)
        return result_paths

    def _run_alignment(
        self,
        meeting_id: str,
        timing_data: dict,
        turns_dir: Path,
        meeting_dir: Path,
        *,
        stats: dict[str, int] | None = None,
        synthesized_text_by_turn: dict[int, str] | None = None,
    ) -> list | None:
        if not self.alignment_enabled:
            logger.info("Alignment disabled, using raw audio durations for %s", meeting_id)
            return None

        try:
            if self.alignment_method == "nemo":
                return self._run_nemo_alignment(
                    meeting_id,
                    timing_data,
                    turns_dir,
                    meeting_dir,
                    stats=stats,
                    synthesized_text_by_turn=synthesized_text_by_turn,
                )
            if self.alignment_method == "hybrid_short":
                logger.info("Running hybrid alignment for %s", meeting_id)
                return compute_hybrid_alignments(
                    timing_data,
                    turns_dir,
                    meeting_dir,
                    nfa_model=self.nfa_model,
                    nfa_script=self.nfa_script,
                    nfa_batch_size=self.nfa_batch_size,
                    transcribe_device=self.nfa_transcribe_device,
                    viterbi_device=self.nfa_viterbi_device,
                    stats=stats,
                    synthesized_text_by_turn=synthesized_text_by_turn,
                )
            logger.info("Running energy-based alignment for %s", meeting_id)
            return compute_energy_alignments(
                timing_data,
                turns_dir,
                stats=stats,
                synthesized_text_by_turn=synthesized_text_by_turn,
            )
        except Exception:
            logger.exception(
                "Alignment failed for %s, falling back to energy detection", meeting_id,
            )
            try:
                return compute_energy_alignments(
                    timing_data,
                    turns_dir,
                    stats=stats,
                    synthesized_text_by_turn=synthesized_text_by_turn,
                )
            except Exception:
                logger.exception("Energy alignment also failed for %s", meeting_id)
                return None

    def _run_nemo_alignment(
        self,
        meeting_id: str,
        timing_data: dict,
        turns_dir: Path,
        meeting_dir: Path,
        *,
        stats: dict[str, int] | None = None,
        synthesized_text_by_turn: dict[int, str] | None = None,
    ) -> list:
        manifest_path = meeting_dir / "nfa_manifest.json"
        n_entries = create_nfa_manifest(
            timing_data,
            turns_dir,
            manifest_path,
            synthesized_text_by_turn=synthesized_text_by_turn,
        )
        logger.info(
            "Created NFA manifest with %d entries for %s", n_entries, meeting_id,
        )
        run_nfa(
            manifest_path, meeting_dir,
            model_name=self.nfa_model,
            nfa_script=self.nfa_script,
            batch_size=self.nfa_batch_size,
            transcribe_device=self.nfa_transcribe_device,
            viterbi_device=self.nfa_viterbi_device,
        )
        return collect_turn_alignments(
            timing_data,
            meeting_dir,
            turns_dir,
            stats=stats,
            synthesized_text_by_turn=synthesized_text_by_turn,
        )

    def _load_turn_synthesis_texts(self, meeting_id: str) -> dict[int, str]:
        """Load exact synthesis text recorded by step 8, if available."""
        results_path = self.tts_dir / "meetings" / meeting_id / "turn_results.json"
        if not results_path.exists():
            return {}
        try:
            records = fileio.read_json(results_path)
        except Exception:
            logger.warning(
                "Could not read turn synthesis metadata for %s", meeting_id,
                exc_info=True,
            )
            return {}
        if not isinstance(records, list):
            return {}
        by_turn: dict[int, str] = {}
        for record in records:
            if not isinstance(record, dict):
                continue
            idx = record.get("turn_index")
            synthesis_text = record.get("synthesis_text")
            if idx is None or not isinstance(synthesis_text, str) or not synthesis_text:
                continue
            try:
                by_turn[int(idx)] = synthesis_text
            except (TypeError, ValueError):
                continue
        return by_turn

    def _write_timed_transcript(
        self,
        timing_data: dict,
        output_path: Path,
        *,
        text_field: str = "text",
    ) -> None:
        # Sort by realized speech onset, not transcript-emission order. With
        # speech overlap (and rarely with negative backchannel FTO), list order
        # can be non-monotonic on the wall clock.
        ordered = sorted(
            timing_data["turns"],
            key=lambda t: t.get("speech_start_ms", t.get("start_time_ms", 0)),
        )
        lines: list[str] = []
        for i, turn in enumerate(ordered):
            ts = format_timestamp_ms(turn["start_time_ms"])
            name = turn["speaker"]
            if turn.get("is_backchannel"):
                name += " [BC]"
            lines.append(f"{name}  {ts}")
            lines.append(str(turn.get(text_field) or turn["text"]))
            if i < len(ordered) - 1:
                lines.append("")
        fileio.write_text(output_path, "\n".join(lines))

    def _write_rttm(self, meeting_id: str, timing_data: dict, output_path: Path) -> None:
        # RTTM is conventionally sorted by start time; consumers (md-eval,
        # pyannote loaders, our viewer) assume that order.
        ordered = sorted(
            timing_data["turns"],
            key=lambda t: t.get("speech_start_ms", t.get("start_time_ms", 0)),
        )
        lines: list[str] = []
        for turn in ordered:
            if "speech_start_ms" in turn and "speech_end_ms" in turn:
                start_sec = turn["speech_start_ms"] / 1000.0
                dur_sec = (turn["speech_end_ms"] - turn["speech_start_ms"]) / 1000.0
            else:
                start_sec = turn["start_time_ms"] / 1000.0
                dur_sec = turn["duration_ms"] / 1000.0
            speaker = str(turn.get("speaker_key") or turn["speaker"]).replace(" ", "_")
            lines.append(
                f"SPEAKER {meeting_id} 1 {start_sec:.3f} {dur_sec:.3f} "
                f"<NA> <NA> {speaker} <NA> <NA>"
            )
        fileio.write_text(output_path, "\n".join(lines) + "\n")

    def _load_meeting_audio(self, meeting_dir: Path) -> dict:
        timing_data = json.loads((meeting_dir / "timing.json").read_text())
        combined_path = meeting_dir / "combined.wav"
        rttm_path = meeting_dir / "reference.rttm"

        speaker_audio_paths: dict[str, str] = {}
        for speaker_slot, speaker_meta in (timing_data.get("speakers") or {}).items():
            suffix = speaker_slot.replace("speaker", "")
            sp_path = meeting_dir / f"speaker{suffix}.wav"
            if sp_path.exists():
                if isinstance(speaker_meta, dict):
                    stable_key = (
                        speaker_meta.get("speaker_key")
                        or speaker_meta.get("speaker_name")
                        or speaker_slot
                    )
                else:
                    stable_key = speaker_meta or speaker_slot
                speaker_audio_paths[str(stable_key)] = str(sp_path)

        return {
            "audio_path": str(combined_path),
            "speaker_audio_paths": speaker_audio_paths,
            "timing_data": timing_data,
            "rttm_path": str(rttm_path) if rttm_path.exists() else None,
        }

    def _load_from_disk(self) -> dict:
        audio_results: dict[str, dict] = {}
        results_path = self.output_dir / "audio_results.json"
        if results_path.exists():
            saved = json.loads(results_path.read_text())
            for meeting_id in saved:
                meeting_dir = self.output_dir / meeting_id
                if meeting_dir.exists():
                    audio_results[meeting_id] = self._load_meeting_audio(meeting_dir)
        return {"audio_results": audio_results}


def _speaker_key_for_turn(turn: dict[str, Any]) -> str:
    return str(
        turn.get("speaker_key")
        or turn.get("speaker_id")
        or turn.get("speaker_name")
        or turn.get("speaker")
        or "Unknown"
    )


def _extract_speaker_roster(turns: list[dict[str, Any]]) -> list[tuple[str, str]]:
    seen: set[str] = set()
    speakers: list[tuple[str, str]] = []
    for turn in turns:
        speaker_key = _speaker_key_for_turn(turn)
        speaker_name = str(turn.get("speaker") or turn.get("speaker_name") or speaker_key)
        if speaker_key not in seen:
            seen.add(speaker_key)
            speakers.append((speaker_key, speaker_name))
    return speakers
