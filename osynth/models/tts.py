"""Pydantic models for TTS/disfluency structured payloads."""
from __future__ import annotations

import re
from typing import Any, Literal, get_args

from pydantic import BaseModel, Field, TypeAdapter, field_validator, model_validator


# Edit-layer type inventory. Split by which boundary an edit lives at: semantic
# (canonical -> semantic_verbatim) vs. TTS (semantic_verbatim -> tts_text).
# Reformulations / repairs / false-starts are speaker-side and live only on the
# semantic side. The TTS layer is strictly additive delivery noise plus
# written->spoken normalization.
SemanticEditType = Literal[
    "repair", "false_start", "reformulation", "word_truncation",
]
TTSEditType = Literal["filler", "repetition", "prolongation", "spoken_form"]
EditType = Literal[
    "repair",
    "false_start",
    "reformulation",
    "word_truncation",
    "filler",
    "repetition",
    "prolongation",
    "spoken_form",
]

_SEMANTIC_EDIT_TYPES: frozenset[str] = frozenset(get_args(SemanticEditType))
_TTS_EDIT_TYPES: frozenset[str] = frozenset(get_args(TTSEditType))


class TranscriptEdit(BaseModel):
    """One edit applied to a base transcript text to produce a derived form.

    Edits describe a span in the derived text and the corresponding span in the
    base text. Pure insertions have ``base_start == base_end`` and an empty
    ``original``. Substitutions replace ``original`` (in the base text) with
    ``surface`` (in the derived text).
    """

    type: EditType
    # Span in the derived text.
    span_start: int
    span_end: int
    # Span in the base text.
    base_start: int
    base_end: int
    # Substring of the derived text at [span_start:span_end]. Required.
    surface: str
    # Substring of the base text at [base_start:base_end]. Empty for pure
    # insertions; populated for substitutions.
    original: str = ""
    # AMI-style repair structure (optional, only meaningful for "repair").
    reparandum: str | None = None
    interregnum: str | None = None
    repair: str | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def _validate_spans(self) -> "TranscriptEdit":
        if self.span_start < 0 or self.span_end < self.span_start:
            raise ValueError(
                f"invalid derived span [{self.span_start}, {self.span_end})",
            )
        if self.base_start < 0 or self.base_end < self.base_start:
            raise ValueError(
                f"invalid base span [{self.base_start}, {self.base_end})",
            )
        return self


def _sort_edits_by_base(edits: list[TranscriptEdit]) -> list[TranscriptEdit]:
    return sorted(edits, key=lambda e: (e.base_start, e.base_end))


def apply_edit_layer(base: str, edits: list[TranscriptEdit]) -> str:
    """Reconstruct the derived text by applying ``edits`` to ``base``.

    Edits must be non-overlapping in the base text. The function does not
    enforce overlap-freeness — it is the caller's responsibility to validate
    inputs first; here we just splice. Returns the derived text.
    """
    if not edits:
        return base
    sorted_edits = _sort_edits_by_base(edits)
    pieces: list[str] = []
    cursor = 0
    for edit in sorted_edits:
        if edit.base_start < cursor:
            raise ValueError(
                f"overlapping edits at base position {edit.base_start} (cursor={cursor})",
            )
        pieces.append(base[cursor : edit.base_start])
        pieces.append(edit.surface)
        cursor = edit.base_end
    pieces.append(base[cursor:])
    return "".join(pieces)


def strip_edit_layer(derived: str, edits: list[TranscriptEdit]) -> str:
    """Inverse of ``apply_edit_layer``: recover the base text from derived."""
    if not edits:
        return derived
    sorted_edits = sorted(edits, key=lambda e: (e.span_start, e.span_end))
    pieces: list[str] = []
    cursor = 0
    for edit in sorted_edits:
        if edit.span_start < cursor:
            raise ValueError(
                f"overlapping edits at derived position {edit.span_start} (cursor={cursor})",
            )
        pieces.append(derived[cursor : edit.span_start])
        pieces.append(edit.original or "")
        cursor = edit.span_end
    pieces.append(derived[cursor:])
    return "".join(pieces)


def reconstructs_to(
    base: str,
    edits: list[TranscriptEdit],
    expected_derived: str,
) -> bool:
    """Check whether ``edits`` applied to ``base`` produce ``expected_derived``.

    Returns True iff reconstruction succeeds and matches exactly. Catches the
    overlap exception and returns False so callers can warn-and-drop rather
    than crash.
    """
    try:
        return apply_edit_layer(base, edits) == expected_derived
    except ValueError:
        return False


def _validate_layer_types(
    edits: list[TranscriptEdit],
    allowed: frozenset[str],
    layer_name: str,
) -> list[TranscriptEdit]:
    for edit in edits:
        if edit.type not in allowed:
            raise ValueError(
                f"{layer_name} cannot contain edit type {edit.type!r}; "
                f"allowed: {sorted(allowed)}",
            )
    return edits


class UnclearSpan(BaseModel):
    """Deliberately obscured word range within a turn's ``tts_text``.

    Sampled at step 7 and applied acoustically at step 10. The word indices
    address tokens in the turn's tokenized ``tts_text`` (see AMI view).
    """

    word_start: int
    word_end: int
    severity: Literal["partial", "heavy"] = "partial"

    @model_validator(mode="after")
    def _validate_range(self) -> "UnclearSpan":
        if self.word_start < 0 or self.word_end <= self.word_start:
            raise ValueError(
                f"invalid unclear word range [{self.word_start}, {self.word_end})",
            )
        return self


OverlapMode = Literal["interrupt", "concurrent"]


# Phatic backchannel taxonomy. Backchannels are semantically void by definition;
# the kind tag is for analysis only and never affects pipeline behavior.
BackchannelKind = Literal[
    "agreement",
    "acknowledgment",
    "continuer",
    "reaction",
]


class OverlapAnchor(BaseModel):
    """Speech-overlap anchor attached to a non-first turn.

    Declares that this turn's audio onset should be placed during the speech
    of an earlier turn (``anchor_turn_index``), at the position of word
    ``anchor_word_index`` (0-based index into the anchor's tokenized
    ``tts_text``). Step 9's mixer translates the word index into an absolute
    timestamp using NFA word alignments.
    """

    anchor_turn_index: int
    anchor_word_index: int
    mode: OverlapMode

    @model_validator(mode="after")
    def _validate_indices(self) -> "OverlapAnchor":
        if self.anchor_turn_index < 0:
            raise ValueError(
                f"anchor_turn_index must be non-negative (got {self.anchor_turn_index})",
            )
        if self.anchor_word_index < 0:
            raise ValueError(
                f"anchor_word_index must be non-negative (got {self.anchor_word_index})",
            )
        return self


class SemanticHint(BaseModel):
    """LLM-emitted semantic edit cue without character offsets.

    The spoken-form prompt asks the LLM for phrase-level intent only. Character
    spans are localized deterministically after the text forms are known.
    """

    type: SemanticEditType
    fragment: str | None = None
    reparandum: str | None = None
    interregnum: str | None = None
    repair: str | None = None
    surface: str | None = None


class SpokenFormRewrite(BaseModel):
    index: int
    semantic_verbatim_text: str | None = None
    tts_text: str | None = None
    text: str | None = None
    semantic_hints: list[SemanticHint] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_text(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        payload = dict(data)
        legacy_text = str(payload.get("text") or "").strip()
        tts_text = str(
            payload.get("tts_text")
            or payload.get("literal_spoken_text")
            or payload.get("literal_text")
            or legacy_text
        ).strip()
        semantic_text = str(
            payload.get("semantic_verbatim_text")
            or payload.get("full_written_text")
            or legacy_text
            or tts_text
        ).strip()
        payload["tts_text"] = tts_text
        payload["semantic_verbatim_text"] = semantic_text
        payload["text"] = tts_text
        return payload

    @model_validator(mode="after")
    def _require_text_forms(self) -> "SpokenFormRewrite":
        if not self.tts_text:
            raise ValueError("transcript-form rewrite must include tts_text")
        if not self.semantic_verbatim_text:
            raise ValueError("transcript-form rewrite must include semantic_verbatim_text")
        self.text = self.tts_text
        return self


class FormattedTurn(BaseModel):
    speaker: str
    text: str = ""
    canonical_text: str = ""
    semantic_verbatim_text: str = ""
    tts_text: str = ""
    speaker_id: str = ""
    speaker_key: str | None = None
    is_backchannel: bool = False
    # Phatic backchannel taxonomy tag. Only meaningful when `is_backchannel` is
    # True; analysis-only field, never affects pipeline behavior.
    bc_kind: BackchannelKind | None = None
    disfluency_profile: dict[str, Any] | None = None
    text_original: str | None = None
    original_text: str | None = None
    # canonical_text -> semantic_verbatim_text: speaker-side meaning-bearing edits.
    semantic_edit_layer: list[TranscriptEdit] = Field(default_factory=list)
    # semantic_verbatim_text -> tts_text: delivery noise + written->spoken normalization.
    tts_edit_layer: list[TranscriptEdit] = Field(default_factory=list)
    # Word-range spans in tts_text that should be acoustically obscured at mix time.
    unclear_spans: list[UnclearSpan] = Field(default_factory=list)
    # Optional overlap anchor: if set, the audio mixer places this turn's
    # onset within the speech of an earlier turn rather than at the end of it.
    overlap_anchor: OverlapAnchor | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_text_forms(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        payload = dict(data)
        legacy_text = str(payload.get("text") or "").strip()
        canonical_text = str(
            payload.get("canonical_text")
            or payload.get("clean_written_text")
            or payload.get("clean_text")
            or payload.get("text_original")
            or payload.get("original_text")
            or legacy_text
        ).strip()
        tts_text = str(
            payload.get("tts_text")
            or payload.get("literal_spoken_text")
            or payload.get("literal_text")
            or legacy_text
        ).strip()
        semantic_text = str(
            payload.get("semantic_verbatim_text")
            or payload.get("full_written_text")
            or canonical_text
            or legacy_text
        ).strip()

        payload["canonical_text"] = canonical_text
        payload["semantic_verbatim_text"] = semantic_text
        payload["tts_text"] = tts_text
        payload["text"] = tts_text
        payload["text_original"] = payload.get("text_original") or canonical_text
        payload["original_text"] = payload.get("original_text") or canonical_text
        return payload

    @model_validator(mode="after")
    def _keep_legacy_aliases_in_sync(self) -> "FormattedTurn":
        self.text = self.tts_text
        self.text_original = self.text_original or self.canonical_text
        self.original_text = self.original_text or self.canonical_text
        _validate_layer_types(
            self.semantic_edit_layer, _SEMANTIC_EDIT_TYPES, "semantic_edit_layer",
        )
        _validate_layer_types(
            self.tts_edit_layer, _TTS_EDIT_TYPES, "tts_edit_layer",
        )
        return self


# ---------------------------------------------------------------------------
# AMI-style tokenized verbatim view
# ---------------------------------------------------------------------------


class AMIWord(BaseModel):
    text: str
    is_punctuation: bool = False


class AMIDisfluencySpan(BaseModel):
    """Word-indexed span of an AMI-style disfluency annotation."""

    type: EditType
    word_start: int
    word_end: int
    reparandum_word_range: tuple[int, int] | None = None
    interregnum_word_range: tuple[int, int] | None = None
    repair_word_range: tuple[int, int] | None = None

    @model_validator(mode="after")
    def _validate_range(self) -> "AMIDisfluencySpan":
        if self.word_start < 0 or self.word_end < self.word_start:
            raise ValueError(
                f"invalid AMI span [{self.word_start}, {self.word_end})",
            )
        return self


class AMITurn(BaseModel):
    speaker_id: str
    words: list[AMIWord] = Field(default_factory=list)
    disfluencies: list[AMIDisfluencySpan] = Field(default_factory=list)
    unclear_spans: list[UnclearSpan] = Field(default_factory=list)
    # AMI corpus keeps backchannels as transcribed words on each channel and
    # tags them at the dialog-act layer; we mirror that here. WER consumers can
    # filter on this flag at read time.
    is_backchannel: bool = False


# ---------------------------------------------------------------------------
# Semantic-transcript entity extraction (used by contextualization metrics)
# ---------------------------------------------------------------------------

EntityClass = Literal[
    "person",
    "organization",
    "location",
    "product",
    "project_codename",
    "industry_term",
    "internal_coinage",
    "technical_term",
    "acronym",
    "date",
    "time",
    "duration",
    "currency_amount",
    "numeric",
    "version",
    "url",
    "email",
    # Catch-all for context-bearing spans that don't fit the inventory above
    # (event names, regulatory frameworks, file paths, code identifiers, etc.).
    # Auditing how often `other` is used tells us where the inventory needs
    # to grow.
    "other",
]


class EntitySpan(BaseModel):
    surface: str
    cls: EntityClass = Field(alias="class")
    char_start: int
    char_end: int
    person_id: str | None = None
    project_id: str | None = None
    source: str | None = None
    expansion: str | None = None

    model_config = {"populate_by_name": True}


class TurnEntities(BaseModel):
    turn_index: int
    spans: list[EntitySpan] = Field(default_factory=list)


_formatted_turns_adapter = TypeAdapter(list[FormattedTurn])
_spoken_form_rewrites_adapter = TypeAdapter(list[SpokenFormRewrite])
_ami_turns_adapter = TypeAdapter(list[AMITurn])


def validate_formatted_turns(data: Any) -> list[dict[str, Any]]:
    """Validate ``formatted_turns.json`` payloads and return JSON-safe dicts."""
    turns = _formatted_turns_adapter.validate_python(data)
    return [turn.model_dump(mode="json") for turn in turns]


def validate_ami_turns(data: Any) -> list[dict[str, Any]]:
    """Validate ``verbatim_ami.json`` payloads and return JSON-safe dicts."""
    turns = _ami_turns_adapter.validate_python(data)
    return [turn.model_dump(mode="json") for turn in turns]


def validate_spoken_form_rewrites(data: Any) -> list[SpokenFormRewrite]:
    """Validate an LLM transcript-form response payload."""
    return _spoken_form_rewrites_adapter.validate_python(data)


# ---------------------------------------------------------------------------
# Speaker enrollment + TTS batch contracts (TTS batch + enrollment redesign)
# ---------------------------------------------------------------------------

DurationTarget = Literal["short", "target", "long"]

class EnrollmentTextCandidate(BaseModel):
    """A single candidate enrollment text for a speaker."""

    id: str
    topical_seed: str
    duration_target: DurationTarget
    text: str

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        candidate_id = value.strip()
        if not candidate_id:
            raise ValueError("candidate id must be non-empty")
        return candidate_id

    @field_validator("topical_seed")
    @classmethod
    def _validate_topical_seed(cls, value: str) -> str:
        topical_seed = value.strip()
        if not topical_seed:
            raise ValueError("topical_seed must be non-empty")
        return topical_seed

    @field_validator("text")
    @classmethod
    def _validate_text_shape(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("candidate text must be non-empty")
        if "```" in text:
            raise ValueError("candidate text must not contain markdown fences")
        if "\n" in text:
            raise ValueError("candidate text must be a single paragraph")
        if re.search(r"^\s*(?:[-*•]|\d+[.)])\s", text, re.MULTILINE):
            raise ValueError("candidate text must not contain list markers")
        return text

class EnrollmentTextBundle(BaseModel):
    """Bundle of enrollment text candidates for one speaker."""

    speaker_id: str
    language: str
    candidates: list[EnrollmentTextCandidate] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_candidate_ids(self) -> "EnrollmentTextBundle":
        ids = [candidate.id for candidate in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate ids must be unique within a bundle")
        topical_seeds = [candidate.topical_seed for candidate in self.candidates]
        if len(topical_seeds) != len(set(topical_seeds)):
            raise ValueError("topical_seed values must be unique within a bundle")
        return self


class EnrollmentSpec(BaseModel):
    """Per-speaker enrollment input spec produced by step 7 prep."""

    speaker_id: str
    voice_id: str
    language: str
    source_audio_path: str | None = None
    source_transcript: str | None = None


class TTSEnrollmentTask(BaseModel):
    """Enrollment synthesis task consumed by step 8."""

    task_id: str
    run_id: str
    run_dir: str
    speaker_id: str
    voice_id: str
    language: str
    source_audio_path: str | None = None
    source_transcript: str | None = None
    candidate_texts: list[EnrollmentTextCandidate] = Field(default_factory=list)
    output_audio_path: str
    output_metadata_path: str
    retry_count: int = 0


class EnrollmentCandidateAttempt(BaseModel):
    """Outcome for one synthesized enrollment-text candidate."""

    candidate_id: str
    topical_seed: str
    duration_target: DurationTarget
    success: bool
    duration_s: float | None = None
    in_band: bool = False
    degenerate: bool = False
    minor_degeneration: bool = False
    failure_reason: str | None = None
    attempt_audio_path: str | None = None
    duration_distance_s: float | None = None

    @field_validator("candidate_id", "topical_seed")
    @classmethod
    def _validate_required_strings(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("candidate attempt string fields must be non-empty")
        return text


class TTSEnrollmentMetadata(BaseModel):
    """Persisted metadata for a completed enrollment output."""

    speaker_id: str
    voice_id: str
    language: str
    source_prompt_path: str | None = None
    source_prompt_transcript: str | None = None
    selected_candidate_id: str | None = None
    selected_enrollment_text: str | None = None
    duration_s: float | None = None
    attempt_count: int = 1
    candidate_attempts: list[EnrollmentCandidateAttempt] = Field(default_factory=list)
    model: str
    mode_tag: str
    duration_status: Literal["accepted", "accepted_minor_degeneration", "duration_out_of_band"] = "accepted"


class TTSTurnTask(BaseModel):
    """Turn synthesis task consumed by step 8."""

    task_id: str
    run_id: str
    run_dir: str
    meeting_id: str
    turn_index: int
    speaker_id: str
    speaker_key: str
    text: str
    output_path: str
    language: str
    retry_count: int = 0


class TTSShortTextConfig(BaseModel):
    """Short-turn synthesis policy recorded with each turn task."""

    enabled: bool = True
    min_chars: int = 20
    mode: Literal["repeat", "raw_then_pad"] = "repeat"
    fallback_padding: Literal["repeat", "ellipsis_separated_repeat"] = "repeat"
    repeat_separator: str = " "


class TTSShortTextAttempt(BaseModel):
    """One attempted synthesis text for a turn."""

    synthesis_text: str
    policy: str
    used_fallback: bool = False
    status: Literal["completed", "failed"]
    duration_s: float | None = None
    error: str | None = None


class TTSResultRecord(BaseModel):
    """Per-task result record produced by step 8."""

    task_id: str
    task_type: Literal["enrollment", "turn"]
    status: Literal["completed", "failed", "waiting_on_enrollment_failure"]
    run_id: str | None = None
    run_dir: str | None = None
    speaker_id: str | None = None
    meeting_id: str | None = None
    turn_index: int | None = None
    output_path: str | None = None
    output_metadata_path: str | None = None
    duration_s: float | None = None
    attempt_count: int = 1
    error: str | None = None
    selected_candidate_id: str | None = None
    original_text: str | None = None
    synthesis_text: str | None = None
    short_text_policy: str | None = None
    short_text_attempts: list[TTSShortTextAttempt] = Field(default_factory=list)
    used_short_text_fallback: bool = False


_enrollment_text_bundle_adapter = TypeAdapter(EnrollmentTextBundle)
_enrollment_spec_adapter = TypeAdapter(EnrollmentSpec)
_tts_enrollment_metadata_adapter = TypeAdapter(TTSEnrollmentMetadata)
_tts_enrollment_task_adapter = TypeAdapter(TTSEnrollmentTask)
_tts_turn_task_adapter = TypeAdapter(TTSTurnTask)
_tts_turn_tasks_adapter = TypeAdapter(list[TTSTurnTask])


def validate_enrollment_text_bundle(data: Any) -> EnrollmentTextBundle:
    return _validate_enrollment_text_bundle(
        data,
        expected_language=None,
        expected_candidate_count=None,
    )


def _validate_enrollment_text_bundle(
    data: Any,
    *,
    expected_language: str | None,
    expected_candidate_count: int | None,
) -> EnrollmentTextBundle:
    bundle = _enrollment_text_bundle_adapter.validate_python(data)
    if expected_language is not None and bundle.language != expected_language:
        raise ValueError(
            f"expected enrollment_text bundle language {expected_language!r}, "
            f"got {bundle.language!r}",
        )
    if expected_candidate_count is not None and len(bundle.candidates) != expected_candidate_count:
        raise ValueError(
            f"expected enrollment_text bundle to contain exactly {expected_candidate_count} "
            f"candidate(s), got {len(bundle.candidates)}",
        )
    if not bundle.candidates:
        raise ValueError("expected enrollment_text bundle to contain at least one candidate")
    return bundle


def validate_enrollment_text_bundle_for_run(
    data: Any,
    *,
    expected_language: str,
    expected_candidate_count: int | None = None,
) -> EnrollmentTextBundle:
    return _validate_enrollment_text_bundle(
        data,
        expected_language=expected_language,
        expected_candidate_count=expected_candidate_count,
    )


def validate_enrollment_spec(data: Any) -> EnrollmentSpec:
    return _enrollment_spec_adapter.validate_python(data)


def validate_tts_enrollment_metadata(data: Any) -> TTSEnrollmentMetadata:
    return _tts_enrollment_metadata_adapter.validate_python(data)


def validate_tts_turn_tasks(data: Any) -> list[TTSTurnTask]:
    return _tts_turn_tasks_adapter.validate_python(data)
