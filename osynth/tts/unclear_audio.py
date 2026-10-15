"""Acoustic attenuation of unclear word ranges within a turn's audio.

Realizes the ``UnclearSpan`` annotations sampled at step 7 by applying a
low-pass + gain reduction over the spanned word window in the turn's WAV.
Used at step 9 (audio finalize) after per-turn alignment is available.

The function is standalone (operates on numpy arrays + word-level timing)
so it can be unit-tested without the full pipeline. Caller is responsible
for loading the WAV, calling this function, and writing the result back.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.signal import butter, sosfiltfilt

from ..models.tts import UnclearSpan

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AttenuationParams:
    """Tunable knobs for unclear-region attenuation.

    Severity-aware: ``partial`` is muffled but words remain perceptible;
    ``heavy`` mostly removes intelligibility while preserving prosodic
    envelope.
    """

    partial_lowpass_hz: float = 1200.0
    heavy_lowpass_hz: float = 700.0
    partial_gain_db: float = -10.0
    heavy_gain_db: float = -18.0
    fade_ms: float = 30.0
    filter_order: int = 4


def _db_to_lin(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def _build_lowpass_sos(cutoff_hz: float, sample_rate: int, order: int):
    nyquist = 0.5 * sample_rate
    normalized = max(1e-4, min(0.99, cutoff_hz / nyquist))
    return butter(order, normalized, btype="lowpass", output="sos")


def _cosine_fade(length: int) -> np.ndarray:
    if length <= 0:
        return np.zeros(0, dtype=np.float64)
    t = np.linspace(0.0, np.pi, length, dtype=np.float64)
    return 0.5 * (1.0 - np.cos(t))


def _resolve_word_window(
    word_alignments: Sequence,
    word_start: int,
    word_end: int,
    audio_duration_s: float,
) -> tuple[float, float] | None:
    if word_end <= word_start:
        return None
    if word_start < 0 or word_end > len(word_alignments):
        logger.warning(
            "Unclear span [%d, %d) outside alignment range of %d words; skipping",
            word_start, word_end, len(word_alignments),
        )
        return None
    start_s = float(word_alignments[word_start].start_time)
    end_s = float(word_alignments[word_end - 1].end_time)
    if end_s <= start_s:
        return None
    start_s = max(0.0, start_s)
    end_s = min(audio_duration_s, end_s)
    if end_s <= start_s:
        return None
    return (start_s, end_s)


def attenuate_unclear_spans(
    audio: np.ndarray,
    sample_rate: int,
    word_alignments: Sequence,
    unclear_spans: Iterable[UnclearSpan | dict],
    params: AttenuationParams | None = None,
) -> np.ndarray:
    """Return a copy of ``audio`` with unclear word ranges acoustically muffled.

    Args:
        audio: 1-D float32/float64 mono PCM samples.
        sample_rate: Sample rate in Hz.
        word_alignments: Sequence of objects with ``start_time``, ``end_time``
            attributes (typically ``WordAlignment`` from ``tts.alignment``).
            Indexed by word; ``unclear_spans`` reference these indices.
        unclear_spans: ``UnclearSpan`` instances (or equivalent dicts) sampled
            for this turn.
        params: Optional ``AttenuationParams``; defaults to module preset.

    Returns:
        A new numpy array with the same shape/dtype as ``audio``.
    """
    if audio.ndim != 1:
        raise ValueError(f"expected mono audio, got shape {audio.shape}")
    if sample_rate <= 0:
        raise ValueError(f"invalid sample_rate {sample_rate}")

    spans = [
        s if isinstance(s, UnclearSpan) else UnclearSpan.model_validate(s)
        for s in unclear_spans
    ]
    if not spans or audio.size == 0:
        return audio.copy()

    p = params or AttenuationParams()
    audio_duration_s = audio.size / float(sample_rate)
    out = audio.astype(np.float64, copy=True)
    fade_samples = max(1, int(round(p.fade_ms * 1e-3 * sample_rate)))

    partial_sos = _build_lowpass_sos(p.partial_lowpass_hz, sample_rate, p.filter_order)
    heavy_sos = _build_lowpass_sos(p.heavy_lowpass_hz, sample_rate, p.filter_order)
    partial_gain = _db_to_lin(p.partial_gain_db)
    heavy_gain = _db_to_lin(p.heavy_gain_db)

    for span in spans:
        window = _resolve_word_window(
            word_alignments, span.word_start, span.word_end, audio_duration_s,
        )
        if window is None:
            continue
        start_s, end_s = window
        start_idx = max(0, int(round(start_s * sample_rate)))
        end_idx = min(out.size, int(round(end_s * sample_rate)))
        if end_idx <= start_idx:
            continue

        sos = heavy_sos if span.severity == "heavy" else partial_sos
        gain = heavy_gain if span.severity == "heavy" else partial_gain

        region = out[start_idx:end_idx]
        if region.size < 2 * p.filter_order:
            continue
        muffled = sosfiltfilt(sos, region) * gain

        n = muffled.size
        fade_len = min(fade_samples, n // 2)
        envelope = np.ones(n, dtype=np.float64)
        if fade_len > 0:
            ramp = _cosine_fade(fade_len)
            envelope[:fade_len] = ramp
            envelope[-fade_len:] = ramp[::-1]

        out[start_idx:end_idx] = region * (1.0 - envelope) + muffled * envelope

    return out.astype(audio.dtype, copy=False)
