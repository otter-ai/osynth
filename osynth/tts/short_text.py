"""Short-text padding for Qwen3-TTS short-utterance robustness.

The Qwen3-TTS 12Hz RVQ codec needs enough tokens to fill at least one full
``num_quantizers=16`` frame. Texts shorter than ``MIN_TTS_CHARS`` reliably
generate too few tokens and Code2Wav returns an empty waveform. Repeating
the utterance is safe for concurrent backchannels because BC turns never
advance the timeline; for non-concurrent short turns the downstream pipeline
(alignment + mixer) trims the synthesis-only repetitions back to a single
occurrence so only the transcript-faithful first copy is audible.

This module is the single source of truth for the padding rule. The synthesis
site (``synthesizer``) and the post-synthesis pipeline
(``alignment``, ``audio_mixer``) all import from here so the rule cannot
drift between sides.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

MIN_TTS_CHARS = 20

ShortTextMode = Literal["repeat", "raw_then_pad"]
ShortTextFallbackPadding = Literal["repeat", "ellipsis_separated_repeat"]


@dataclass(frozen=True)
class ShortTextSynthesisConfig:
    """Runtime short-text synthesis policy."""

    enabled: bool = True
    min_chars: int = MIN_TTS_CHARS
    mode: ShortTextMode = "repeat"
    fallback_padding: ShortTextFallbackPadding = "repeat"
    repeat_separator: str = " "


@dataclass(frozen=True)
class ShortTextSynthesisInput:
    """Concrete synthesis text plus metadata for one attempt."""

    original_text: str
    synthesis_text: str
    policy: str
    was_padded: bool
    used_fallback: bool = False


def coerce_short_text_config(value: Any | None) -> ShortTextSynthesisConfig:
    """Build a runtime config from a pydantic model, dict, or ``None``."""
    if value is None:
        return ShortTextSynthesisConfig()
    if isinstance(value, ShortTextSynthesisConfig):
        return value
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if isinstance(value, dict):
        allowed = {
            "enabled",
            "min_chars",
            "mode",
            "fallback_padding",
            "repeat_separator",
        }
        return ShortTextSynthesisConfig(
            **{key: val for key, val in value.items() if key in allowed},
        )
    raise TypeError(f"Unsupported short-text config type: {type(value)!r}")


def pad_short_tts_text(
    text: str,
    *,
    min_chars: int = MIN_TTS_CHARS,
    separator: str = " ",
) -> str:
    """Join repeats of ``text`` until it reaches ``min_chars``.

    No-op when ``text`` is already long enough. The output is always an
    integer number of copies of ``text``. With the default whitespace
    separator, the word count of the padded string is an exact multiple of
    the original word count.
    """
    if len(text) >= min_chars:
        return text
    padded = text
    while len(padded) < min_chars:
        padded = padded + separator + text
    return padded


def was_padded(
    text: str,
    *,
    min_chars: int = MIN_TTS_CHARS,
    separator: str = " ",
) -> bool:
    """True iff ``pad_short_tts_text(text) != text``."""
    return pad_short_tts_text(text, min_chars=min_chars, separator=separator) != text


def build_short_text_input(
    text: str,
    config: ShortTextSynthesisConfig | dict[str, Any] | None = None,
    *,
    use_fallback: bool = False,
) -> ShortTextSynthesisInput:
    """Return the concrete text that should be sent to TTS for one attempt."""
    cfg = coerce_short_text_config(config)
    if not cfg.enabled or len(text) >= cfg.min_chars:
        return ShortTextSynthesisInput(
            original_text=text,
            synthesis_text=text,
            policy="raw",
            was_padded=False,
            used_fallback=use_fallback,
        )

    if cfg.mode == "raw_then_pad" and not use_fallback:
        return ShortTextSynthesisInput(
            original_text=text,
            synthesis_text=text,
            policy="raw",
            was_padded=False,
            used_fallback=False,
        )

    fallback = cfg.fallback_padding if cfg.mode == "raw_then_pad" else cfg.mode
    if fallback == "ellipsis_separated_repeat":
        synthesis_text = pad_short_tts_text(
            text,
            min_chars=cfg.min_chars,
            separator=cfg.repeat_separator,
        )
    else:
        synthesis_text = pad_short_tts_text(text, min_chars=cfg.min_chars)

    return ShortTextSynthesisInput(
        original_text=text,
        synthesis_text=synthesis_text,
        policy=str(fallback),
        was_padded=synthesis_text != text,
        used_fallback=use_fallback,
    )


def short_text_attempt_inputs(
    text: str,
    config: ShortTextSynthesisConfig | dict[str, Any] | None = None,
) -> list[ShortTextSynthesisInput]:
    """Return synthesis attempts in the order they should be tried."""
    cfg = coerce_short_text_config(config)
    first = build_short_text_input(text, cfg, use_fallback=False)
    if not cfg.enabled or cfg.mode != "raw_then_pad" or len(text) >= cfg.min_chars:
        return [first]
    fallback = build_short_text_input(text, cfg, use_fallback=True)
    if fallback.synthesis_text == first.synthesis_text:
        return [first]
    return [first, fallback]
