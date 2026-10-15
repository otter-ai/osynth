"""Forced alignment: word-level timing from audio + text.

Provides three alignment strategies, selected via ``AlignmentSettings.method``:

  1. **NeMo NFA** (``method="nemo"``) — CTC-based forced alignment via NeMo's
     NFA tool (``tools/nemo_forced_aligner/align.py`` in a NeMo checkout, set
     as ``AlignmentSettings.nfa_script``). Produces accurate word-level
     timestamps. Requires the ``nfa`` extra (``nemo-toolkit[asr]``). Runs on
     CPU by default; pass
     ``transcribe_device='cuda'`` / ``viterbi_device='cuda'`` via the
     ``AlignmentSettings`` overrides to use a GPU when available.
  2. **Hybrid short** (``method="hybrid_short"``, the default) — runs
     NFA only on padded short-text turns (where ``pad_short_tts_text`` added
     repetitions at synthesis time) and uses cheap energy detection for the
     rest. The NFA pass is what powers the short-turn repetition trim;
     restricting NFA to those turns keeps CPU cost low while still removing the
     duplicated "yeah yeah yeah" audio at the tail of each padded utterance.
  3. **Energy-only** (``method="energy"``) — RMS speech-onset/offset detection,
     no word-level alignments. Backwards-compatible fallback when NeMo is not
     installed.

All strategies return ``TurnAlignment`` objects with speech boundaries used by
the audio mixer to place turns based on actual speech edges, not raw file length.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..utils import fileio
from .short_text import pad_short_tts_text

logger = logging.getLogger(__name__)

# Default NeMo CTC model for forced alignment
DEFAULT_NFA_MODEL = "stt_en_fastconformer_ctc_large"

def _turn_audio_filename(turn_index: int) -> str:
    """Canonical TTS turn audio filename. Must match ``step_7_speech_prep``."""
    return f"turn_{turn_index:03d}.wav"


def _turn_ctm_filename(turn_index: int) -> str:
    return f"turn_{turn_index:03d}.ctm"


def _raw_turn_text(turn: dict[str, Any]) -> str:
    return str(turn.get("tts_text") or turn.get("text") or "")


def _synthesis_text_for_turn(
    turn: dict[str, Any],
    synthesized_text_by_turn: dict[int, str] | None = None,
) -> str:
    idx = int(turn["turn_index"])
    if synthesized_text_by_turn and idx in synthesized_text_by_turn:
        return synthesized_text_by_turn[idx]
    return pad_short_tts_text(_raw_turn_text(turn))


def _turn_was_synthesis_padded(
    turn: dict[str, Any],
    synthesized_text_by_turn: dict[int, str] | None = None,
) -> bool:
    raw_text = _raw_turn_text(turn)
    synthesis_text = _synthesis_text_for_turn(turn, synthesized_text_by_turn)
    return bool(raw_text) and synthesis_text.strip() != raw_text.strip()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class WordAlignment:
    """A single word's timing within a turn's audio."""
    word: str
    start_time: float  # seconds, relative to turn audio start
    end_time: float    # seconds
    duration: float    # seconds


@dataclass
class TurnAlignment:
    """Alignment result for a single turn."""
    turn_index: int
    word_alignments: list[WordAlignment] = field(default_factory=list)
    speech_start_s: float = 0.0   # first word onset (seconds, relative to turn audio)
    speech_end_s: float = 0.0     # last word offset (seconds, relative to turn audio)
    audio_duration_s: float = 0.0  # total turn audio file duration
    clip_end_s: float | None = None  # optional mixer cut point, relative to turn audio

    @property
    def speech_duration_s(self) -> float:
        return self.speech_end_s - self.speech_start_s

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "speech_start_s": round(self.speech_start_s, 4),
            "speech_end_s": round(self.speech_end_s, 4),
            "speech_duration_s": round(self.speech_duration_s, 4),
            "audio_duration_s": round(self.audio_duration_s, 4),
            "clip_end_s": round(self.clip_end_s, 4) if self.clip_end_s is not None else None,
            "n_words": len(self.word_alignments),
            "word_alignments": [
                {
                    "word": w.word,
                    "start_time": round(w.start_time, 4),
                    "end_time": round(w.end_time, 4),
                    "duration": round(w.duration, 4),
                }
                for w in self.word_alignments
            ],
        }


# ---------------------------------------------------------------------------
# NeMo forced alignment
# ---------------------------------------------------------------------------


def create_nfa_manifest(
    timing_data: dict[str, Any],
    turns_dir: Path,
    output_path: Path,
    *,
    only_padded: bool = False,
    include_turn_indices: set[int] | None = None,
    synthesized_text_by_turn: dict[int, str] | None = None,
) -> int:
    """Write a NeMo-style JSONL manifest mapping each turn WAV to its text.

    Args:
        timing_data: Initial timing dict (must contain ``turns`` list, each
            item must carry ``turn_index``, ``text``, optionally ``tts_text``).
        turns_dir: Directory containing the per-turn WAV files
            (``turn_NNN.wav``).
        output_path: JSONL manifest output path.
        only_padded: If True, emit a manifest entry only for turns whose
            synthesis text differs from transcript text under the legacy
            short-padding fallback.
        include_turn_indices: Optional explicit set of turn indices to emit.
            Used by hybrid alignment when word-level alignment is required
            for both short-turn trims and future overlap anchors.
        synthesized_text_by_turn: Exact text sent to TTS by step 8. When
            absent, falls back to the legacy deterministic padding rule.
            This is the entry point for ``hybrid_short`` alignment, where we
            run CPU-bound NFA on the small subset of turns that actually need
            word-level timing for the repetition trim, and let energy
            detection cover the rest.

    Returns the number of entries written.
    """
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for turn in timing_data["turns"]:
            idx = int(turn["turn_index"])
            if include_turn_indices is not None and idx not in include_turn_indices:
                continue
            if only_padded and not _turn_was_synthesis_padded(
                turn,
                synthesized_text_by_turn,
            ):
                continue
            audio_path = turns_dir / _turn_audio_filename(idx)
            if not audio_path.exists():
                logger.warning("Turn audio missing for alignment: %s", audio_path)
                continue
            # Padded turns synthesize repeated copies of ``raw_text`` to clear
            # the Qwen3-TTS minimum-token threshold (see ``tts/short_text``).
            # The actual WAV contains those repetitions, so NFA must align
            # against the padded text — otherwise CTC stretches one word
            # across the whole utterance and ``speech_end_s`` ends up at the
            # tail of the last repetition. The repetition trim happens in
            # ``collect_turn_alignments``.
            entry = {
                "audio_filepath": str(audio_path.resolve()),
                "text": _synthesis_text_for_turn(turn, synthesized_text_by_turn),
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1
    return count


def run_nfa(
    manifest_path: Path,
    output_dir: Path,
    model_name: str = DEFAULT_NFA_MODEL,
    nfa_script: str | None = None,
    batch_size: int = 1,
    transcribe_device: str | None = None,
    viterbi_device: str | None = None,
) -> Path:
    """Run NeMo forced aligner as a subprocess.

    Args:
        manifest_path: Path to the JSONL manifest.
        output_dir: Directory to write alignment outputs.
        model_name: NeMo CTC model name (from NGC).
        nfa_script: Path to NeMo's ``tools/nemo_forced_aligner/align.py``,
            from a NeMo checkout matching the installed ``nemo-toolkit``.
        batch_size: Batch size for alignment.
        transcribe_device: Torch device string for the log-prob step (e.g.,
            ``"cpu"``, ``"cuda"``, ``"cuda:0"``). ``None`` lets NFA auto-pick
            CUDA when available.
        viterbi_device: Torch device string for the Viterbi decode step. Same
            semantics as ``transcribe_device``.

    Returns:
        Path to the NFA alignments directory.
    """
    alignments_dir = output_dir / "nfa_alignments"
    fileio.makedirs(alignments_dir)

    if not nfa_script:
        raise ValueError(
            "alignment.nfa_script is not set. Point it at "
            "tools/nemo_forced_aligner/align.py in a NeMo checkout, or set "
            "alignment.method to 'energy' to skip forced alignment."
        )
    script_path = Path(nfa_script).resolve()
    if not script_path.exists():
        raise FileNotFoundError(
            f"NFA align.py not found at {script_path}. Install the [nfa] extra "
            f"and check alignment.nfa_script."
        )

    # NFA's align.py uses ``from utils.X import Y`` (relative to its own
    # directory). Hydra-style overrides are bare ``key=value`` args; we *do
    # not* shell-quote the model name (Hydra parses ``pretrained_name=foo``).
    cmd = [
        sys.executable, str(script_path.name),
        f"pretrained_name={model_name}",
        f"manifest_filepath={manifest_path.resolve()}",
        f"output_dir={alignments_dir.resolve()}",
        f"batch_size={batch_size}",
        # Only emit CTM files — the ASS subtitle writer is irrelevant here and
        # was the cause of a broken upstream import on some NeMo versions.
        "save_output_file_formats=[ctm]",
    ]
    if transcribe_device:
        cmd.append(f"transcribe_device={transcribe_device}")
    if viterbi_device:
        cmd.append(f"viterbi_device={viterbi_device}")

    logger.info(
        "Running NeMo forced alignment (model=%s, devices=%s/%s)...",
        model_name,
        transcribe_device or "auto",
        viterbi_device or "auto",
    )
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        cwd=str(script_path.parent),
    )
    if result.returncode != 0:
        logger.error(
            "NFA failed (exit %d):\nstdout tail:\n%s\nstderr tail:\n%s",
            result.returncode,
            (result.stdout or "(no stdout)")[-2000:],
            (result.stderr or "(no stderr)")[-2000:],
        )
        raise RuntimeError(f"NeMo forced alignment failed (exit {result.returncode})")
    logger.info("NFA alignment complete.")
    return alignments_dir


def parse_ctm_file(ctm_path: Path) -> list[WordAlignment]:
    """Parse a CTM (Conversation Time Mark) file into word alignments.

    CTM format: file_id channel start_time duration word [confidence] [lex] [NA]
    """
    words: list[WordAlignment] = []
    if not ctm_path.exists():
        return words
    with open(ctm_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            start = float(parts[2])
            dur = float(parts[3])
            words.append(WordAlignment(
                word=parts[4],
                start_time=start,
                end_time=start + dur,
                duration=dur,
            ))
    return words


def collect_turn_alignments(
    timing_data: dict[str, Any],
    alignments_dir: Path,
    turns_dir: Path,
    stats: dict[str, int] | None = None,
    synthesized_text_by_turn: dict[int, str] | None = None,
) -> list[TurnAlignment]:
    """Parse CTM files for all turns and compute speech boundaries.

    Args:
        timing_data: Timing data dict with turns.
        alignments_dir: Root NFA output directory (contains nfa_alignments/).
        turns_dir: Directory containing turn WAV files (for duration computation).
        stats: Optional mutable dict updated with short-turn trim counters
            (keys ``padded``, ``trimmed``, ``trim_word_count_mismatch``).

    Returns:
        List of TurnAlignment objects, one per turn.
    """
    import soundfile as sf

    ctm_words_dir = alignments_dir / "nfa_alignments" / "ctm" / "words"
    results: list[TurnAlignment] = []

    padded_total = 0
    trimmed_total = 0
    mismatch_total = 0

    for turn in timing_data["turns"]:
        idx = turn["turn_index"]
        clip_end_s: float | None = None

        # Parse CTM. The NFA output stem follows the audio basename, so
        # turn_NNN.wav → turn_NNN.ctm regardless of speaker.
        word_aligns = parse_ctm_file(ctm_words_dir / _turn_ctm_filename(idx))

        # Get audio duration from the actual WAV, falling back to the timing
        # estimate only if the file is genuinely missing (degraded synthesis).
        audio_path = turns_dir / _turn_audio_filename(idx)
        if audio_path.exists():
            info = sf.info(str(audio_path))
            audio_dur = info.duration
        else:
            audio_dur = turn["duration_ms"] / 1000.0

        # Short-utterance trim: the WAV contains repetitions added by
        # ``pad_short_tts_text`` at synthesis time; keep only the alignments
        # belonging to the first repetition so ``speech_end_s`` lands at the
        # end of the transcript-faithful occurrence, not the tail of the
        # last padded copy. The padding rule is deterministic from
        # ``raw_text`` alone — see ``tts.short_text``.
        raw_text = _raw_turn_text(turn)
        synthesis_text = _synthesis_text_for_turn(turn, synthesized_text_by_turn)
        n_original_words = len(raw_text.split())
        n_words_in_audio = len(synthesis_text.split())
        if synthesis_text.strip() != raw_text.strip() and n_original_words > 0:
            padded_total += 1
            legacy_repeat_text = pad_short_tts_text(raw_text)
            min_required_words = (
                n_words_in_audio
                if synthesis_text.strip() == legacy_repeat_text.strip()
                else n_original_words
            )
            if len(word_aligns) >= min_required_words:
                first_end = word_aligns[n_original_words - 1].end_time
                if len(word_aligns) > n_original_words:
                    next_start = word_aligns[n_original_words].start_time
                    clip_end_s = (
                        (first_end + next_start) / 2.0
                        if next_start > first_end
                        else first_end
                    )
                else:
                    clip_end_s = first_end
                word_aligns = word_aligns[:n_original_words]
                trimmed_total += 1
            else:
                mismatch_total += 1
                logger.warning(
                    "Short-turn trim skipped for turn %d (%r): NFA returned "
                    "%d words, needed at least %d from synthesis text with "
                    "%d whitespace token(s). Full audio will play.",
                    idx, raw_text, len(word_aligns), min_required_words,
                    n_words_in_audio,
                )

        # Compute speech boundaries
        if word_aligns:
            speech_start = word_aligns[0].start_time
            speech_end = word_aligns[-1].end_time
        else:
            # Fallback: use energy-based detection
            if audio_path.exists():
                speech_start, speech_end = detect_speech_boundaries(audio_path)
            else:
                speech_start = 0.0
                speech_end = audio_dur

        results.append(TurnAlignment(
            turn_index=idx,
            word_alignments=word_aligns,
            speech_start_s=speech_start,
            speech_end_s=speech_end,
            audio_duration_s=audio_dur,
            clip_end_s=clip_end_s,
        ))

    if stats is not None:
        stats["padded"] = stats.get("padded", 0) + padded_total
        stats["trimmed"] = stats.get("trimmed", 0) + trimmed_total
        stats["trim_word_count_mismatch"] = (
            stats.get("trim_word_count_mismatch", 0) + mismatch_total
        )

    return results


# ---------------------------------------------------------------------------
# Energy-based fallback
# ---------------------------------------------------------------------------


def detect_speech_boundaries(
    audio_path: Path | str,
    energy_threshold_db: float = -40.0,
    frame_length_ms: int = 25,
    hop_length_ms: int = 10,
    min_speech_duration_s: float = 0.05,
) -> tuple[float, float]:
    """Detect speech onset and offset using RMS energy.

    Simple but effective for clean TTS audio with silence padding.

    Args:
        audio_path: Path to WAV file.
        energy_threshold_db: Threshold in dB below peak for speech detection.
        frame_length_ms: Analysis frame length.
        hop_length_ms: Hop between frames.
        min_speech_duration_s: Minimum speech duration to avoid noise.

    Returns:
        (speech_start_s, speech_end_s) in seconds.
    """
    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    audio_dur = len(audio) / sr

    if len(audio) == 0:
        return 0.0, 0.0

    # Compute frame-level RMS energy
    frame_len = int(sr * frame_length_ms / 1000)
    hop_len = int(sr * hop_length_ms / 1000)
    n_frames = max(1, (len(audio) - frame_len) // hop_len + 1)

    rms = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop_len
        frame = audio[start:start + frame_len]
        rms[i] = np.sqrt(np.mean(frame ** 2) + 1e-10)

    # Convert to dB relative to peak
    peak_rms = np.max(rms)
    if peak_rms < 1e-10:
        return 0.0, audio_dur

    rms_db = 20 * np.log10(rms / peak_rms + 1e-10)

    # Find frames above threshold
    above = np.where(rms_db > energy_threshold_db)[0]
    if len(above) == 0:
        return 0.0, audio_dur

    speech_start_frame = above[0]
    speech_end_frame = above[-1]

    speech_start_s = speech_start_frame * hop_len / sr
    speech_end_s = min((speech_end_frame * hop_len + frame_len) / sr, audio_dur)

    # Enforce minimum duration
    if speech_end_s - speech_start_s < min_speech_duration_s:
        return 0.0, audio_dur

    return speech_start_s, speech_end_s


def compute_energy_alignments(
    timing_data: dict[str, Any],
    turns_dir: Path,
    stats: dict[str, int] | None = None,
    synthesized_text_by_turn: dict[int, str] | None = None,
) -> list[TurnAlignment]:
    """Compute speech boundaries for all turns using energy detection.

    Fallback when NeMo is not available. Does not produce word-level alignments.

    The energy detector cannot find repetition boundaries inside a padded short
    utterance, so ``speech_end_s`` ends up at the tail of the last repetition.
    For visibility, ``stats['energy_fallback_padded']`` is incremented for each
    short-text turn aligned through this path.
    """
    import soundfile as sf

    results: list[TurnAlignment] = []
    energy_fallback_padded = 0

    for turn in timing_data["turns"]:
        idx = turn["turn_index"]
        audio_path = turns_dir / _turn_audio_filename(idx)

        if audio_path.exists():
            info = sf.info(str(audio_path))
            audio_dur = info.duration
            speech_start, speech_end = detect_speech_boundaries(audio_path)
        else:
            audio_dur = turn["duration_ms"] / 1000.0
            speech_start = 0.0
            speech_end = audio_dur

        if _turn_was_synthesis_padded(turn, synthesized_text_by_turn):
            energy_fallback_padded += 1

        results.append(TurnAlignment(
            turn_index=idx,
            word_alignments=[],
            speech_start_s=speech_start,
            speech_end_s=speech_end,
            audio_duration_s=audio_dur,
            clip_end_s=None,
        ))

    if stats is not None:
        stats["energy_fallback_padded"] = (
            stats.get("energy_fallback_padded", 0) + energy_fallback_padded
        )

    return results


# ---------------------------------------------------------------------------
# Hybrid: NFA on padded short turns, energy on everything else
# ---------------------------------------------------------------------------


def nfa_required_turn_indices(
    timing_data: dict[str, Any],
    synthesized_text_by_turn: dict[int, str] | None = None,
) -> dict[int, set[str]]:
    """Return turns that need NeMo word alignments in hybrid mode.

    Reasons:
    - ``short_padding_trim``: the turn audio contains synthesis-only padding
      that must be clipped after the first transcript-faithful occurrence.
    - ``future_overlap_anchor``: a later turn anchors to a word inside this
      turn, so overlap placement needs word-level timestamps.
    """
    turns = timing_data.get("turns", []) or []
    n_turns = len(turns)
    required: dict[int, set[str]] = {}

    for turn in turns:
        idx = int(turn["turn_index"])
        if _turn_was_synthesis_padded(turn, synthesized_text_by_turn):
            required.setdefault(idx, set()).add("short_padding_trim")

    for turn in turns:
        anchor = turn.get("overlap_anchor")
        if not isinstance(anchor, dict):
            continue
        try:
            anchor_idx = int(anchor.get("anchor_turn_index", -1))
        except (TypeError, ValueError):
            continue
        if 0 <= anchor_idx < n_turns:
            required.setdefault(anchor_idx, set()).add("future_overlap_anchor")

    return required


def compute_hybrid_alignments(
    timing_data: dict[str, Any],
    turns_dir: Path,
    meeting_dir: Path,
    *,
    nfa_model: str = DEFAULT_NFA_MODEL,
    nfa_script: str | None = None,
    nfa_batch_size: int = 1,
    transcribe_device: str | None = None,
    viterbi_device: str | None = None,
    stats: dict[str, int] | None = None,
    synthesized_text_by_turn: dict[int, str] | None = None,
) -> list[TurnAlignment]:
    """Run NFA where word alignments are required; energy on the rest.

    Word-level alignment is required for the short-turn repetition trim and
    for any turn referenced by a future overlap anchor. Other turns can use
    cheap energy-based speech boundaries.

    The result is a single list — one ``TurnAlignment`` per turn, in the same
    order as ``timing_data["turns"]`` — with word-level alignments populated
    on padded turns and empty ``word_alignments=[]`` everywhere else. The
    repetition trim inside ``collect_turn_alignments`` only fires on turns
    that have word alignments and that were padded, so this composition is
    safe.

    Args:
        timing_data: Initial timing dict from ``build_timing``.
        turns_dir: Directory with per-turn WAVs.
        meeting_dir: Per-meeting output directory (used as the NFA scratch
            root: ``nfa_manifest.json`` and ``nfa_alignments/`` live here).
        nfa_model: NeMo CTC model name.
        nfa_script: Path to NeMo's NFA ``align.py`` (see ``run_nfa``).
        nfa_batch_size: NFA batch size.
        transcribe_device / viterbi_device: Torch device strings; ``None``
            defers to NFA's auto-detect.
        stats: Optional mutable dict updated with short-turn counters
            (``padded``, ``trimmed``, ``trim_word_count_mismatch``,
            ``hybrid_nfa_entries``, ``hybrid_energy_only_turns``).
    """
    # Start from the energy result so every turn has *some* speech boundary.
    energy_results = compute_energy_alignments(
        timing_data,
        turns_dir,
        stats=None,
        synthesized_text_by_turn=synthesized_text_by_turn,
    )
    by_index: dict[int, TurnAlignment] = {ta.turn_index: ta for ta in energy_results}

    required = nfa_required_turn_indices(timing_data, synthesized_text_by_turn)
    required_indices = set(required)
    if not required_indices:
        logger.info(
            "Hybrid alignment: no NFA-required turns; skipping NFA pass for %s",
            meeting_dir.name,
        )
        if stats is not None:
            stats["hybrid_nfa_entries"] = stats.get("hybrid_nfa_entries", 0)
            stats["hybrid_energy_only_turns"] = (
                stats.get("hybrid_energy_only_turns", 0) + len(energy_results)
            )
        return energy_results

    manifest_path = meeting_dir / "nfa_manifest.json"
    n_entries = create_nfa_manifest(
        timing_data,
        turns_dir,
        manifest_path,
        include_turn_indices=required_indices,
        synthesized_text_by_turn=synthesized_text_by_turn,
    )
    n_short = sum(
        1 for reasons in required.values() if "short_padding_trim" in reasons
    )
    n_anchor = sum(
        1 for reasons in required.values() if "future_overlap_anchor" in reasons
    )
    n_both = sum(1 for reasons in required.values() if len(reasons) > 1)
    logger.info(
        "Hybrid alignment for %s: %d turn(s) → NFA "
        "(short=%d, future_anchor=%d, both=%d), %d turn(s) → energy",
        meeting_dir.name,
        n_entries,
        n_short,
        n_anchor,
        n_both,
        len(energy_results) - n_entries,
    )
    if stats is not None:
        stats["hybrid_nfa_entries"] = stats.get("hybrid_nfa_entries", 0) + n_entries
        stats["hybrid_nfa_short_padding_trim"] = (
            stats.get("hybrid_nfa_short_padding_trim", 0) + n_short
        )
        stats["hybrid_nfa_future_overlap_anchor"] = (
            stats.get("hybrid_nfa_future_overlap_anchor", 0) + n_anchor
        )
        stats["hybrid_nfa_both_reasons"] = (
            stats.get("hybrid_nfa_both_reasons", 0) + n_both
        )
        stats["hybrid_energy_only_turns"] = (
            stats.get("hybrid_energy_only_turns", 0)
            + (len(energy_results) - n_entries)
        )
    fileio.write_json(
        meeting_dir / "nfa_manifest_selection.json",
        {
            "turns": [
                {
                    "turn_index": idx,
                    "reasons": sorted(reasons),
                    "manifest_text_source": (
                        "step_8_result"
                        if synthesized_text_by_turn and idx in synthesized_text_by_turn
                        else "legacy_short_text_rule"
                    ),
                }
                for idx, reasons in sorted(required.items())
            ],
        },
    )

    if n_entries == 0:
        # All selected turns had missing audio — already warned in
        # create_nfa_manifest. Energy fallback is what we'd return anyway.
        return energy_results

    run_nfa(
        manifest_path,
        meeting_dir,
        model_name=nfa_model,
        nfa_script=nfa_script,
        batch_size=nfa_batch_size,
        transcribe_device=transcribe_device,
        viterbi_device=viterbi_device,
    )

    # Build a NFA-only timing snippet so collect_turn_alignments only walks
    # the subset with CTM files.
    nfa_timing = {
        "turns": [
            turn for turn in timing_data["turns"]
            if int(turn["turn_index"]) in required_indices
        ],
    }
    nfa_results = collect_turn_alignments(
        nfa_timing,
        meeting_dir,
        turns_dir,
        stats=stats,
        synthesized_text_by_turn=synthesized_text_by_turn,
    )
    for ta in nfa_results:
        reasons = required.get(ta.turn_index, set())
        energy_ta = by_index[ta.turn_index]
        if "short_padding_trim" in reasons:
            # The trim boundary is between synthesis-only repetitions, so it
            # intentionally comes from NeMo word timing. Ordinary start/end
            # boundaries for non-padded NFA-selected turns remain energy VAD.
            by_index[ta.turn_index] = ta
            continue
        by_index[ta.turn_index] = TurnAlignment(
            turn_index=ta.turn_index,
            word_alignments=ta.word_alignments,
            speech_start_s=energy_ta.speech_start_s,
            speech_end_s=energy_ta.speech_end_s,
            audio_duration_s=energy_ta.audio_duration_s,
            clip_end_s=ta.clip_end_s,
        )

    return [by_index[turn["turn_index"]] for turn in timing_data["turns"]]


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------


def save_alignments(
    turn_alignments: list[TurnAlignment],
    output_path: Path,
) -> None:
    """Save alignment results to JSON."""
    data = {
        "turns": [ta.to_dict() for ta in turn_alignments],
    }
    fileio.write_json(output_path, data)
    logger.info("Saved alignments for %d turns to %s", len(turn_alignments), output_path)


def load_alignments(path: Path) -> list[TurnAlignment]:
    """Load alignment results from JSON."""
    data = json.loads(path.read_text())
    results: list[TurnAlignment] = []
    for td in data["turns"]:
        words = [
            WordAlignment(
                word=w["word"],
                start_time=w["start_time"],
                end_time=w["end_time"],
                duration=w["duration"],
            )
            for w in td.get("word_alignments", [])
        ]
        results.append(TurnAlignment(
            turn_index=td["turn_index"],
            word_alignments=words,
            speech_start_s=td["speech_start_s"],
            speech_end_s=td["speech_end_s"],
            audio_duration_s=td["audio_duration_s"],
            clip_end_s=td.get("clip_end_s"),
        ))
    return results
