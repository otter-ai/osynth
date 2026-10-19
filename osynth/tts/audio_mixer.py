"""Audio mixing: FTO timing, combined audio creation, per-speaker tracks.

Handles the post-synthesis step of placing individual turn WAV files into a
realistic timeline using Floor Transfer Offset (FTO) sampling, then mixing
into combined and per-speaker audio files.
"""
from __future__ import annotations

import io
import logging
import random
from pathlib import Path
from typing import Any

from ..models.tts import FormattedTurn

logger = logging.getLogger(__name__)


def _import_audio_deps():
    """Lazy import of numpy and soundfile (optional TTS dependencies)."""
    import numpy as np
    import soundfile as sf
    return np, sf

# Default sample rate for Qwen TTS output
DEFAULT_SAMPLE_RATE = 24000

# Tail kept after ``speech_end_in_turn_s`` when clipping a segment. Preserves
# natural decay so the trim isn't perceptible as a hard cut. Used for both the
# short-text repetition trim and ordinary trailing-silence cleanup that NFA's
# speech-end already implies.
SEGMENT_TRAIL_PAD_MS = 50

# Minimum gap (ms) enforced between two consecutive utterances by the same
# speaker. Pass-1 FTO sampling can draw negative offsets that, combined with a
# short intervening other-speaker turn, would push the next same-speaker turn's
# speech_start before the previous same-speaker speech_end. We shift forward by
# at least this much to keep timelines monotonic per speaker. Matches the
# -50ms epsilon already used by the self-overlap sweep.
SELF_OVERLAP_MIN_GAP_MS = 50.0


def _segment_clip_end_samples(
    seg_len: int,
    turn: dict[str, Any],
    sample_rate: int,
) -> int:
    """Return the sample index to clip ``seg`` at based on alignment metadata.

    Falls back to ``seg_len`` (no clipping) when alignment data is missing or
    invalid — preserving today's behavior for turns aligned via the energy
    fallback or with alignment disabled.
    """
    alignment = turn.get("alignment")
    if not isinstance(alignment, dict):
        return seg_len
    clip_end_s = alignment.get("clip_end_in_turn_s")
    if isinstance(clip_end_s, (int, float)) and clip_end_s > 0:
        end_sample = int(float(clip_end_s) * sample_rate)
        return max(0, min(seg_len, end_sample))
    speech_end_s = alignment.get("speech_end_in_turn_s")
    if not isinstance(speech_end_s, (int, float)) or speech_end_s <= 0:
        return seg_len
    pad_samples = int(SEGMENT_TRAIL_PAD_MS / 1000.0 * sample_rate)
    end_sample = int(float(speech_end_s) * sample_rate) + pad_samples
    return max(0, min(seg_len, end_sample))


def _speaker_key_for_turn(turn: FormattedTurn) -> str:
    return str(turn.speaker_key or turn.speaker_id or turn.speaker or "Unknown")


def _load_audio(path: str | Path):
    """Load a WAV file as float32 mono."""
    np, sf = _import_audio_deps()
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def _load_audio_bytes(audio_bytes: bytes):
    """Load WAV audio from bytes as float32 mono."""
    np, sf = _import_audio_deps()
    buf = io.BytesIO(audio_bytes)
    audio, sr = sf.read(buf, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def _resample(audio, orig_sr: int, target_sr: int):
    """Resample audio to target sample rate using linear interpolation.

    For production quality, librosa.resample would be better, but this avoids
    the librosa dependency for the common case where all audio is already at
    the same sample rate.
    """
    np, _ = _import_audio_deps()
    if orig_sr == target_sr:
        return audio
    ratio = target_sr / orig_sr
    n_samples = int(len(audio) * ratio)
    indices = np.linspace(0, len(audio) - 1, n_samples)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def sample_fto(
    fto_xi: float = 0.05,
    fto_omega: float = 0.15,
    fto_alpha: float = 2.0,
    rng: random.Random | None = None,
) -> int:
    """Sample a Floor Transfer Offset in milliseconds using skew-normal distribution.

    The skew-normal distribution models realistic turn-taking gaps:
    - Most transitions have a small positive gap (~50ms)
    - Some have slight overlap (negative FTO)
    - Occasionally longer pauses

    Args:
        fto_xi: Location parameter (seconds).
        fto_omega: Scale parameter (seconds).
        fto_alpha: Skewness parameter (positive = right-skewed).
        rng: Optional RNG for reproducibility.

    Returns:
        FTO in milliseconds.
    """
    from scipy.stats import skewnorm

    sample_s = skewnorm.rvs(a=fto_alpha, loc=fto_xi, scale=fto_omega,
                            random_state=rng.randint(0, 2**31) if rng else None)
    return int(sample_s * 1000)


def sample_overlap_jitter(
    jitter_ms: int = 50,
    rng: random.Random | None = None,
) -> int:
    """Sample a uniform ± jitter around the LLM-chosen overlap onset."""
    _rng = rng or random.Random()
    if jitter_ms <= 0:
        return 0
    return _rng.randint(-jitter_ms, jitter_ms)


def build_timing(
    turns: list[dict[str, Any]],
    turn_durations_ms: list[int],
    speakers: list[str],
    *,
    fto_xi: float = 0.05,
    fto_omega: float = 0.15,
    fto_alpha: float = 2.0,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """Build timing.json data structure from turns and audio durations.

    Args:
        turns: List of turn dicts with keys: speaker, speaker_key, text,
               is_backchannel, original_timestamp (optional).
        turn_durations_ms: Duration of each turn's audio in milliseconds.
        speakers: Ordered list of unique stable speaker keys.
        fto_xi/fto_omega/fto_alpha: Skew-normal FTO parameters.
        rng: Optional RNG for reproducibility.

    Returns:
        Timing data dict with speakers, turns (with start/end times), and
        total_duration_ms. This is the pre-alignment pass; backchannel and
        overlap anchor placement happen in ``build_aligned_timing`` once NFA
        word alignments are available.
    """
    parsed_turns = [(FormattedTurn.model_validate(turn), turn) for turn in turns]

    speaker_names: dict[str, str] = {}
    for turn, _raw_turn in parsed_turns:
        speaker_key = _speaker_key_for_turn(turn)
        speaker_names.setdefault(speaker_key, turn.speaker or speaker_key)

    timing: dict[str, Any] = {
        "speakers": {
            f"speaker{i + 1}": {
                "speaker_key": speaker_key,
                "speaker_name": speaker_names.get(speaker_key, speaker_key),
            }
            for i, speaker_key in enumerate(speakers)
        },
        "turns": [],
        "total_duration_ms": 0,
    }

    current_time_ms = 0

    for i, ((turn, raw_turn), dur_ms) in enumerate(zip(parsed_turns, turn_durations_ms)):
        speaker = turn.speaker
        speaker_key = _speaker_key_for_turn(turn)
        is_bc = turn.is_backchannel
        canonical_text = turn.canonical_text
        semantic_text = turn.semantic_verbatim_text
        tts_text = turn.tts_text

        # Determine FTO for the gap *after* this turn. Backchannels no longer
        # get a special negative-FTO branch here — they are re-placed as
        # concurrent overlays in build_aligned_timing using an LLM-decided
        # anchor word. The initial pass treats them as ordinary turns; the
        # aligned pass corrects placement once NFA alignments are available.
        if i < len(parsed_turns) - 1:
            fto_ms = sample_fto(fto_xi, fto_omega, fto_alpha, rng)
        else:
            fto_ms = 0

        speaker_num = (speakers.index(speaker_key) + 1) if speaker_key in speakers else 1

        turn_info = {
            "turn_index": i,
            "speaker": speaker,
            "speaker_key": speaker_key,
            "speaker_num": speaker_num,
            "original_timestamp": raw_turn.get("original_timestamp", ""),
            "start_time_ms": current_time_ms,
            "duration_ms": dur_ms,
            "end_time_ms": current_time_ms + dur_ms,
            "fto_ms": fto_ms,
            "text": tts_text,
            "canonical_text": canonical_text,
            "semantic_verbatim_text": semantic_text,
            "tts_text": tts_text,
            "is_backchannel": is_bc,
        }
        if turn.overlap_anchor is not None:
            turn_info["overlap_anchor"] = turn.overlap_anchor.model_dump(mode="json")
        timing["turns"].append(turn_info)
        current_time_ms += dur_ms + fto_ms

    timing["total_duration_ms"] = current_time_ms
    return timing


def build_aligned_timing(
    timing_data: dict[str, Any],
    turn_alignments: list,  # list[TurnAlignment] — avoids circular import
    *,
    fto_xi: float = 0.05,
    fto_omega: float = 0.15,
    fto_alpha: float = 2.0,
    overlap_jitter_ms: int = 50,
    overlap_min_post_gap_ms: int = 100,
    backchannel_jitter_ms: int = 50,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """Rebuild timing using alignment-based speech boundaries.

    Two-pass placement:

    1. Place every non-anchor turn left-to-right by skew-normal FTO from the
       previous non-anchor turn's speech_end. These are the semantic backbone
       turns; backchannels and substantive overlaps are skipped here.
    2. For every turn carrying ``overlap_anchor`` metadata, place its speech
       onset at the anchor turn's word boundary plus a small uniform jitter.
       Anchor-placed turns (BCs and substantive overlaps alike) do not advance
       the cumulative ``current_speech_end`` pointer.

    Adjacency rules differ by turn flavor:

    - Substantive overlap (``is_backchannel=False`` + ``overlap_anchor`` set):
      strict ``anchor_turn_index == turn_index - 1``. Anchor must itself be a
      Pass-1 turn (max chain depth 1).
    - Backchannel (``is_backchannel=True`` + ``overlap_anchor`` set):
      ``anchor_turn_index`` must equal the most recent prior non-BC,
      non-anchor turn — i.e., walk backwards skipping BCs and overlaps until
      we hit a semantic backbone turn. Mode must be ``concurrent``.

    After both passes, a per-speaker check flags any anchor that would have
    the same speaker mid-utterance against themselves. Every anchor that
    fails (invalid, gap check, self-overlap) is demoted to a Pass-1 turn and
    placement is redone until nothing fails, so emission order and wall-clock
    order agree for every non-overlapping turn.

    Args:
        timing_data: Original timing data from build_timing().
        turn_alignments: List of TurnAlignment objects (one per turn).
        fto_*: FTO sampling parameters.
        overlap_jitter_ms: ± jitter on the LLM-chosen overlap onset (ms).
        overlap_min_post_gap_ms: Minimum gap (ms) between a substantive
            overlap's speech_end and the next non-anchor turn's speech_end.
            Annotations that can't satisfy this fall back to FTO. Backchannels
            skip this check (they are short by definition).
        backchannel_jitter_ms: ± jitter on the LLM-chosen BC onset (ms).
        rng: Optional RNG for reproducibility.

    Returns:
        New timing data dict with alignment-informed placement. Each turn
        carries ``speech_start_ms``, ``speech_end_ms``, ``alignment`` metadata,
        ``overlap_anchor_requested`` (annotator/anchor-stage intent), and,
        when placed, ``overlap_anchor_resolved`` (realized timestamp).
    """
    aligned: dict[str, Any] = {
        "speakers": timing_data["speakers"],
        "turns": [],
        "total_duration_ms": 0,
    }

    turns = timing_data["turns"]
    n = len(turns)

    speech_in_ms = [align.speech_start_s * 1000.0 for align in turn_alignments]
    speech_out_ms = [align.speech_end_s * 1000.0 for align in turn_alignments]
    audio_dur_ms = [align.audio_duration_s * 1000.0 for align in turn_alignments]

    def _has_overlap_anchor(turn: dict[str, Any]) -> bool:
        anchor = turn.get("overlap_anchor")
        return isinstance(anchor, dict) and "anchor_turn_index" in anchor

    is_bc = [bool(turn.get("is_backchannel")) for turn in turns]
    has_anchor = [_has_overlap_anchor(turn) for turn in turns]
    speaker_keys = [
        str(turn.get("speaker_key") or turn.get("speaker") or "")
        for turn in turns
    ]
    # FTO and jitter are drawn once per turn so re-placement rounds (below)
    # see the same draws and converge instead of re-rolling the timeline.
    fto_draw_ms = [sample_fto(fto_xi, fto_omega, fto_alpha, rng) for _ in range(n)]
    jitter_draw_ms = [
        sample_overlap_jitter(
            backchannel_jitter_ms if is_bc[i] else overlap_jitter_ms, rng,
        )
        for i in range(n)
    ]

    audio_start_ms: list[float] = [0.0] * n
    speech_start_abs_ms: list[float] = [0.0] * n
    speech_end_abs_ms: list[float] = [0.0] * n
    fto_used_ms: list[int] = [0] * n
    overlap_anchor_resolved: list[dict[str, Any] | None] = [None] * n

    # An anchor that can't be placed (bad index, missing alignment, gap check,
    # self-overlap) is demoted to an ordinary Pass-1 turn and the whole
    # placement is redone, so later turns make room for it. Placing it after
    # the fact instead stacks it on top of the turns Pass 1 already laid out
    # — and a per-speaker guard read after Pass 1 pushed it past all of that
    # speaker's later turns (176 s out of order in tierA meeting_7ba24f42).
    # Demotion is monotonic, so this terminates in at most ``n + 1`` rounds.
    demoted_at_placement: set[int] = set()
    demoted_self_overlap: set[int] = set()
    missing_word_alignment: set[int] = set()
    self_overlap_pass1_shifts = 0

    def _place() -> tuple[list[bool], list[int], list[int]]:
        """One placement round; returns (placed_via_anchor, failed, collided)."""
        nonlocal self_overlap_pass1_shifts
        self_overlap_pass1_shifts = 0
        demoted = demoted_at_placement | demoted_self_overlap
        # Pass-2 turns: anchored overlaps and BCs not yet demoted. BCs without
        # an anchor (e.g. the anchor sub-stage failed) are Pass-1 turns.
        placed_via_anchor = [has_anchor[i] and i not in demoted for i in range(n)]
        for i in range(n):
            overlap_anchor_resolved[i] = None

        # Pass 1: place every non-anchor turn by FTO, keeping each speaker's
        # turns monotonic so a negative FTO can't put a speaker on top of
        # their own previous turn.
        last_speech_end_per_speaker: dict[str, float] = {}
        current_speech_end_ms = 0.0
        first = True
        for i in range(n):
            if placed_via_anchor[i]:
                continue
            if first:
                audio_start_ms[i] = 0.0
                fto_used_ms[i] = 0
                first = False
            else:
                audio_start_ms[i] = current_speech_end_ms + fto_draw_ms[i] - speech_in_ms[i]
                fto_used_ms[i] = fto_draw_ms[i]
            speech_start_abs_ms[i] = audio_start_ms[i] + speech_in_ms[i]
            speech_end_abs_ms[i] = audio_start_ms[i] + speech_out_ms[i]

            speaker = speaker_keys[i]
            prior_end = last_speech_end_per_speaker.get(speaker)
            if prior_end is not None:
                min_start = prior_end + SELF_OVERLAP_MIN_GAP_MS
                if speech_start_abs_ms[i] < min_start:
                    shift = min_start - speech_start_abs_ms[i]
                    audio_start_ms[i] += shift
                    speech_start_abs_ms[i] += shift
                    speech_end_abs_ms[i] += shift
                    fto_used_ms[i] = int(fto_used_ms[i] + shift)
                    self_overlap_pass1_shifts += 1
            last_speech_end_per_speaker[speaker] = speech_end_abs_ms[i]
            current_speech_end_ms = speech_end_abs_ms[i]

        def _most_recent_pass1_idx(i: int) -> int | None:
            for j in range(i - 1, -1, -1):
                if not placed_via_anchor[j] and not is_bc[j]:
                    return j
            return None

        # Pass 2: place anchor-driven turns at the anchor's word onset. They
        # do not move the Pass-1 timeline.
        failed: list[int] = []
        for i, turn in enumerate(turns):
            if not placed_via_anchor[i]:
                continue
            anchor_meta = turn.get("overlap_anchor") or {}
            try:
                anchor_idx = int(anchor_meta.get("anchor_turn_index", -1))
                word_idx = int(anchor_meta.get("anchor_word_index", -1))
            except (TypeError, ValueError):
                anchor_idx = -1
                word_idx = -1
            mode = str(anchor_meta.get("mode") or "")

            if is_bc[i]:
                # Backchannel: anchor must be the most recent prior non-BC,
                # non-anchor turn (a Pass-1 semantic backbone turn), and the
                # mode must be concurrent.
                target = _most_recent_pass1_idx(i)
                valid_anchor = (
                    target is not None
                    and anchor_idx == target
                    and mode == "concurrent"
                )
            else:
                # Substantive overlap: strict adjacency, anchor must itself be
                # a Pass-1 turn (max chain depth 1).
                valid_anchor = (
                    anchor_idx == i - 1
                    and anchor_idx >= 0
                    and not placed_via_anchor[anchor_idx]
                )

            if valid_anchor:
                anchor_words = turn_alignments[anchor_idx].word_alignments
                if not anchor_words:
                    missing_word_alignment.add(i)
                    valid_anchor = False
                elif word_idx < 0 or word_idx >= len(anchor_words):
                    valid_anchor = False
            if not valid_anchor:
                failed.append(i)
                continue

            anchor_word_onset_s = (
                turn_alignments[anchor_idx].word_alignments[word_idx].start_time
            )
            jitter = jitter_draw_ms[i]
            speech_start = (
                speech_start_abs_ms[anchor_idx] + anchor_word_onset_s * 1000.0 + jitter
            )
            turn_speech_dur = speech_out_ms[i] - speech_in_ms[i]
            proposed_speech_end = speech_start + turn_speech_dur

            # Gap check applies only to substantive overlaps. BCs are short
            # phatic overlays and may legitimately extend past the anchor's
            # speech_end into the next turn.
            if not is_bc[i]:
                next_pass1_speech_end: float | None = None
                for j in range(anchor_idx + 1, n):
                    if not placed_via_anchor[j]:
                        next_pass1_speech_end = speech_end_abs_ms[j]
                        break
                if (
                    next_pass1_speech_end is not None
                    and proposed_speech_end
                    > next_pass1_speech_end - overlap_min_post_gap_ms
                ):
                    failed.append(i)
                    continue

            candidate_audio_start = speech_start - speech_in_ms[i]
            if candidate_audio_start < 0:
                # Anchor word is too early for this turn's silence pad.
                candidate_audio_start = 0.0
                speech_start = speech_in_ms[i]
                proposed_speech_end = speech_start + turn_speech_dur
            audio_start_ms[i] = candidate_audio_start
            speech_start_abs_ms[i] = speech_start
            speech_end_abs_ms[i] = proposed_speech_end
            fto_used_ms[i] = int(jitter)
            overlap_anchor_resolved[i] = {
                "anchor_turn_index": anchor_idx,
                "anchor_word_index": word_idx,
                "mode": mode,
                "anchor_word_onset_s": anchor_word_onset_s,
                "jitter_ms": int(jitter),
            }

        # Per-speaker check: an anchored turn must not land mid-utterance
        # against the same speaker.
        collided: set[int] = set()
        for indices in per_speaker_indices.values():
            sorted_idx = sorted(indices, key=lambda j: speech_start_abs_ms[j])
            for a, b in zip(sorted_idx, sorted_idx[1:]):
                if speech_start_abs_ms[b] >= speech_end_abs_ms[a] - 50.0:
                    continue
                if overlap_anchor_resolved[b] is not None:
                    collided.add(b)
                elif overlap_anchor_resolved[a] is not None:
                    collided.add(a)
        return placed_via_anchor, failed, sorted(collided)

    per_speaker_indices: dict[str, list[int]] = {}
    for i, key in enumerate(speaker_keys):
        per_speaker_indices.setdefault(key, []).append(i)

    for _ in range(n + 1):
        placed_via_anchor, failed, collided = _place()
        if failed:
            demoted_at_placement.update(failed)
            continue
        if collided:
            demoted_self_overlap.update(collided)
            continue
        break

    # Defense-in-depth: two Pass-1 turns of one speaker colliding should be
    # impossible after the Pass-1 guard; shift the later one forward if so.
    for _ in range(n + 1):
        shifted = False
        for indices in per_speaker_indices.values():
            sorted_idx = sorted(indices, key=lambda j: speech_start_abs_ms[j])
            for a, b in zip(sorted_idx, sorted_idx[1:]):
                shift = speech_end_abs_ms[a] + SELF_OVERLAP_MIN_GAP_MS - speech_start_abs_ms[b]
                if speech_start_abs_ms[b] >= speech_end_abs_ms[a] - 50.0 or shift <= 0:
                    continue
                audio_start_ms[b] += shift
                speech_start_abs_ms[b] += shift
                speech_end_abs_ms[b] += shift
                fto_used_ms[b] = int(fto_used_ms[b] + shift)
                self_overlap_pass1_shifts += 1
                shifted = True
                logger.warning(
                    "Self-overlap shifted: turn %d pushed by %.1fms to clear "
                    "turn %d (same speaker)", b, shift, a,
                )
        if not shifted:
            break

    dropped_overlaps = len(demoted_at_placement)
    self_overlap_drops = len(demoted_self_overlap)
    overlap_anchor_missing_word_alignment = len(missing_word_alignment)

    for i, (turn, align) in enumerate(zip(turns, turn_alignments)):
        entry: dict[str, Any] = {
            **turn,
            "start_time_ms": audio_start_ms[i],
            "duration_ms": audio_dur_ms[i],
            "end_time_ms": audio_start_ms[i] + audio_dur_ms[i],
            "speech_start_ms": speech_start_abs_ms[i],
            "speech_end_ms": speech_end_abs_ms[i],
            "fto_ms": fto_used_ms[i],
            "alignment": {
                "speech_start_in_turn_s": align.speech_start_s,
                "speech_end_in_turn_s": align.speech_end_s,
                "speech_duration_s": align.speech_duration_s,
                "audio_duration_s": align.audio_duration_s,
                "n_words": len(align.word_alignments),
                "clip_end_in_turn_s": align.clip_end_s,
            },
        }
        # timing.json carries both the annotator's intent and the realized
        # placement. requested = "annotator asked for this overlap"; resolved =
        # "mixer successfully placed it". Either may be present without the
        # other (resolved without requested only if upstream JSON was edited).
        requested = entry.pop("overlap_anchor", None)
        if requested is not None:
            entry["overlap_anchor_requested"] = requested
        if overlap_anchor_resolved[i] is not None:
            entry["overlap_anchor_resolved"] = overlap_anchor_resolved[i]
        aligned["turns"].append(entry)

    if aligned["turns"]:
        aligned["total_duration_ms"] = max(
            t["end_time_ms"] for t in aligned["turns"]
        )

    aligned["overlap_stats"] = {
        "requested": sum(1 for v in has_anchor if v),
        "applied": sum(1 for v in overlap_anchor_resolved if v is not None),
        "dropped_at_placement": dropped_overlaps,
        "overlap_anchor_missing_word_alignment": overlap_anchor_missing_word_alignment,
        "dropped_self_overlap": self_overlap_drops,
        "self_overlap_pass1_shifts": self_overlap_pass1_shifts,
        "requested_backchannel": sum(
            1 for i, v in enumerate(has_anchor) if v and is_bc[i]
        ),
        "applied_backchannel": sum(
            1 for i, v in enumerate(overlap_anchor_resolved)
            if v is not None and is_bc[i]
        ),
    }
    if (
        dropped_overlaps
        or self_overlap_drops
        or self_overlap_pass1_shifts
        or overlap_anchor_missing_word_alignment
    ):
        logger.info(
            "build_aligned_timing: %d overlap(s) dropped at placement, "
            "%d missing anchor word alignment, %d dropped by self-overlap "
            "sweep, %d Pass-1 self-overlap shift(s)",
            dropped_overlaps,
            overlap_anchor_missing_word_alignment,
            self_overlap_drops,
            self_overlap_pass1_shifts,
        )

    return aligned


def create_combined_audio(
    turn_audio_paths: list[str | Path],
    timing: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Create combined audio file mixing all turns with FTO timing.

    Args:
        turn_audio_paths: Ordered list of per-turn WAV file paths.
        timing: Timing data from build_timing().
        output_path: Path to write combined WAV.
    """
    np, sf = _import_audio_deps()
    segments = []
    sample_rate: int | None = None

    for path in turn_audio_paths:
        audio, sr = _load_audio(path)
        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            audio = _resample(audio, sr, sample_rate)
        segments.append(audio)

    if sample_rate is None:
        sample_rate = DEFAULT_SAMPLE_RATE

    # Calculate required buffer length
    max_end = 0
    for turn in timing["turns"]:
        end_ms = int(turn["start_time_ms"]) + int(turn["duration_ms"])
        max_end = max(max_end, end_ms)
    total_ms = max(timing["total_duration_ms"], max_end)
    total_samples = int((total_ms / 1000.0) * sample_rate) + sample_rate

    final_audio = np.zeros(total_samples, dtype=np.float32)

    for seg, turn in zip(segments, timing["turns"]):
        start_sample = max(0, int((turn["start_time_ms"] / 1000.0) * sample_rate))
        clip_end = _segment_clip_end_samples(len(seg), turn, sample_rate)
        seg_clip = seg[:clip_end]
        end_sample = start_sample + len(seg_clip)
        if end_sample > len(final_audio):
            extension = np.zeros(end_sample - len(final_audio) + sample_rate, dtype=np.float32)
            final_audio = np.concatenate([final_audio, extension])
        # Additive mixing (allows overlapping backchannels)
        final_audio[start_sample:end_sample] += seg_clip

    # Trim trailing silence
    nonzero = np.where(np.abs(final_audio) > 1e-6)[0]
    if len(nonzero) > 0:
        tail = int(0.1 * sample_rate)
        final_audio = final_audio[: nonzero[-1] + tail + 1]

    # Normalize to prevent clipping
    peak = np.max(np.abs(final_audio))
    if peak > 1.0:
        final_audio = final_audio / peak

    sf.write(str(output_path), final_audio, sample_rate, format="WAV")
    logger.info("Combined audio: %.1fs -> %s", len(final_audio) / sample_rate, output_path)


def create_speaker_audio(
    turn_audio_paths: list[str | Path],
    timing: dict[str, Any],
    target_speaker_num: int,
    output_path: str | Path,
) -> None:
    """Create audio file for a single speaker with silence for other speakers.

    Args:
        turn_audio_paths: Ordered list of per-turn WAV file paths.
        timing: Timing data from build_timing().
        target_speaker_num: 1-based speaker number to extract.
        output_path: Path to write speaker WAV.
    """
    np, sf = _import_audio_deps()
    segments: list = []
    sample_rate: int | None = None

    for path, turn in zip(turn_audio_paths, timing["turns"]):
        if turn["speaker_num"] == target_speaker_num:
            audio, sr = _load_audio(path)
            if sample_rate is None:
                sample_rate = sr
            elif sr != sample_rate:
                audio = _resample(audio, sr, sample_rate)
            segments.append(audio)
        else:
            segments.append(None)

    if sample_rate is None:
        sample_rate = DEFAULT_SAMPLE_RATE

    max_end = 0
    for turn in timing["turns"]:
        end_ms = int(turn["start_time_ms"]) + int(turn["duration_ms"])
        max_end = max(max_end, end_ms)
    total_ms = max(timing["total_duration_ms"], max_end)
    total_samples = int((total_ms / 1000.0) * sample_rate) + sample_rate

    final_audio = np.zeros(total_samples, dtype=np.float32)

    for seg, turn in zip(segments, timing["turns"]):
        if seg is not None and turn["speaker_num"] == target_speaker_num:
            start_sample = max(0, int((turn["start_time_ms"] / 1000.0) * sample_rate))
            clip_end = _segment_clip_end_samples(len(seg), turn, sample_rate)
            seg_clip = seg[:clip_end]
            end_sample = start_sample + len(seg_clip)
            if end_sample > len(final_audio):
                extension = np.zeros(end_sample - len(final_audio) + sample_rate, dtype=np.float32)
                final_audio = np.concatenate([final_audio, extension])
            final_audio[start_sample:end_sample] = seg_clip

    # Trim trailing silence
    nonzero = np.where(np.abs(final_audio) > 1e-6)[0]
    if len(nonzero) > 0:
        tail = int(0.1 * sample_rate)
        final_audio = final_audio[: nonzero[-1] + tail + 1]

    sf.write(str(output_path), final_audio, sample_rate, format="WAV")


def get_audio_duration_ms(path: str | Path) -> int:
    """Get the duration of a WAV file in milliseconds."""
    audio, sr = _load_audio(path)
    return int((len(audio) / sr) * 1000)


def format_timestamp_ms(ms: int) -> str:
    """Format milliseconds as M:SS timestamp."""
    seconds = ms / 1000.0
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"
