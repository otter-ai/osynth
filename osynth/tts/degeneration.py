"""Shared TTS degeneration thresholds and exceptions."""
from __future__ import annotations

MAX_SECONDS_PER_CHAR = 0.5
MAX_TURN_DURATION_S = 180.0


class TTSDegenerationError(RuntimeError):
    """Raised when TTS output is unreasonably long or malformed."""
