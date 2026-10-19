"""Typed runtime settings for the synthetic data pipeline."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Literal

from dynaconf import Dynaconf
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .tts import TTSShortTextConfig
from ..utils.llm_client import LLMBudgetConfig, ReasoningEffort


class _SettingsModel(BaseModel):
    """Base model for validated synthetic-data settings."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class WorkspaceSettings(_SettingsModel):
    company_size: str
    industry: str
    domain: str
    time_period_start: date
    time_period_end: date
    max_speakers: int = 13
    working_hours: tuple[int, int] = (9, 17)
    working_days: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4])
    timezone: str = "America/Los_Angeles"



class TTSSettings(_SettingsModel):
    model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    voice_cache_dir: str = "data/voice_cache"
    voice_source: Literal["commonvoice", "catalog", "static"] = "commonvoice"
    # In-process synthesis with the qwen-tts package. The released corpus was
    # rendered with a batched GPU runtime of the same model; that runtime is
    # infrastructure-specific and not included here.
    mode: Literal["local"] = "local"
    formatting_model: str = "claude-sonnet-4-6"
    enrollment_text_model: str | None = None
    enrollment_target_seconds: int = 15
    enrollment_min_seconds: int = 10
    enrollment_max_seconds: int = 20
    enrollment_candidate_count: int = 3
    max_turn_duration_s: float = 180.0
    max_seconds_per_char: float = 0.5
    max_retries: int = 3
    # Upper bound on multi-character ``word_truncation`` hints retained per
    # 100 spoken-form words after sanitization. Single-letter fragments are
    # always dropped (the Qwen3-TTS renderer pronounces them as letter names
    # rather than mid-word cut-offs); multi-character fragments above this
    # rate are reverted to their un-truncated word form via deterministic
    # seeded sampling so per-meeting clusters can't bring back the bad
    # listening experience. Set higher to allow more truncations through.
    max_truncation_rate: float = 1.0
    short_text: TTSShortTextConfig = Field(default_factory=TTSShortTextConfig)

    @model_validator(mode="after")
    def _fill_enrollment_text_model(self) -> "TTSSettings":
        if self.enrollment_text_model is None:
            self.enrollment_text_model = self.formatting_model
        return self


class OverlapSettings(_SettingsModel):
    """Step-7 LLM overlap annotation + step-9 mixer placement settings.

    When ``enabled`` is true, step 7 runs a per-meeting LLM call that decides
    which turns should overlap with which earlier turn (anchored to a word
    index inside the anchor's ``tts_text``). Step 9 then places those turns
    using the anchor's per-word alignment rather than the usual end-of-prior-
    speech FTO.
    """

    enabled: bool = False
    annotator_model: str = "claude-sonnet-4-6"
    # Optional reasoning effort for compatible models; ignored otherwise.
    # A low budget leaves more response tokens for structured annotations.
    annotator_reasoning_effort: ReasoningEffort | None = "low"
    # Annotate in windows of this many turns (0 = one call per meeting).
    # Smaller windows can improve annotation coverage on long meetings.
    annotator_window_turns: int = 0
    # Roughly fraction of inter-speaker turn boundaries the LLM should annotate.
    target_rate: float = 0.15
    # Fraction of annotated overlaps that should be "concurrent" (fully inside
    # the anchor) vs. "interrupt" (late in the anchor).
    full_overlap_rate: float = 0.30
    # ± jitter in ms around the LLM-chosen word onset.
    onset_jitter_ms: int = 50
    # Cap on chained anchoring. 1 means an overlap turn cannot itself anchor
    # another overlap.
    max_chain_depth: int = 1
    # Minimum gap (ms) the overlap turn must end before the next non-overlap
    # speech onset; if it can't be satisfied, drop the annotation.
    min_post_overlap_gap_ms: int = 100
    # Multipliers on target_rate by join mode of the overlap turn's speaker.
    in_room_bias: float = 1.5
    remote_bias: float = 0.7


class BackchannelSettings(_SettingsModel):
    """Step-7 LLM classifier + anchor sub-stages for backchannels.

    Backchannels are short, phatic, semantically void acknowledgments that
    *concurrently* overlay an earlier semantic turn. Both sub-stages are gated
    by ``speech_pipeline.overlap.enabled`` (BC reframing is part of the overlap
    feature). The classifier replaces the legacy string-match heuristic; the
    anchor sub-stage picks a word index inside the BC's semantic anchor turn.
    """

    classifier_model: str = "claude-sonnet-4-6"
    anchor_model: str = "claude-sonnet-4-6"
    # Optional reasoning effort for compatible classifier models.
    classifier_reasoning_effort: ReasoningEffort | None = "low"
    # Optional reasoning effort for compatible anchor-placement models.
    anchor_reasoning_effort: ReasoningEffort | None = "low"
    # ± jitter in ms around the LLM-chosen anchor word onset.
    onset_jitter_ms: int = 50


class SpeechPipelineSettings(_SettingsModel):
    fto_xi: float = 0.05
    fto_omega: float = 0.15
    fto_alpha: float = 2.0
    overlap: OverlapSettings = Field(default_factory=OverlapSettings)
    backchannel: BackchannelSettings = Field(default_factory=BackchannelSettings)
    # Optional reasoning effort for compatible entity-extraction models.
    entity_reasoning_effort: ReasoningEffort | None = "low"
    # Optional reasoning effort for the step-7 spoken-form/disfluency
    # formatting call (30-turn rewrite batches). None keeps the API default.
    formatting_reasoning_effort: ReasoningEffort | None = None
    # Step-7 parallelism. All output is order-preserving, so these change
    # wall time only. Defaults are sequential; tune to your provider's rate
    # limits. Peak in-flight requests per run is roughly
    # meeting_concurrency * batch_concurrency.
    # Formatting and entity batches in flight per meeting.
    batch_concurrency: int = 1
    # Meetings prepped concurrently (also bounds concurrent speaker
    # enrollments).
    meeting_concurrency: int = 1


class AlignmentSettings(_SettingsModel):
    enabled: bool = True
    # Default to ``hybrid_short`` so runs trim padded short-turn
    # repetitions automatically without per-meeting NFA cost on the long
    # turns. Override to ``"nemo"`` for full word-level alignments on every
    # turn (slow on CPU) or ``"energy"`` for the legacy onset/offset path.
    method: Literal["energy", "nemo", "hybrid_short"] = "hybrid_short"
    nfa_model: str = "stt_en_fastconformer_ctc_large"
    # Path to ``tools/nemo_forced_aligner/align.py`` in a NeMo checkout
    # (https://github.com/NVIDIA/NeMo). Required unless ``method`` is
    # ``"energy"``.
    nfa_script: str | None = None
    batch_size: int = 1
    # Torch device strings forwarded to NFA. ``None`` lets NFA auto-detect
    # CUDA; set to ``"cpu"`` on machines without a GPU.
    nfa_transcribe_device: str | None = "cpu"
    nfa_viterbi_device: str | None = "cpu"


class RealismRewriteSettings(_SettingsModel):
    """Optional pre-formatting pass that rewrites synthetic transcripts to look
    more like real ASR-style meeting transcripts. Disabled by default;
    enabled in the two-week example config.
    """
    enabled: bool = False
    model: str = "claude-sonnet-4-6"
    max_concurrent: int = 4
    # Length budget for the AMI excerpt (in lines). Keeping this small bounds
    # the prompt cost; the excerpt is a *style* reference, not full context.
    ami_reference_lines: int = 200
    # Directory of pre-normalized AMI plain-text files (one per meeting, named
    # `<meeting_id>.txt`), relative to the working directory unless absolute.
    # Required when the rewrite is enabled.
    ami_reference_dir: str | None = None
    # Iterative judge -> refine loop count (after the initial rewrite).
    # Each iteration is one judge call + one refine call. Default 5.
    refine_iterations: int = 5
    # Refine in windows of this many turns rather than re-emitting the whole
    # transcript in one call. Windowing limits unintended shortening and
    # allows length checks per window. The refiner still sees the full
    # transcript as context. 0 disables windowing.
    refine_window_turns: int = 40
    # Per-turn verbosity target for refinement, in words, based on AMI
    # reference transcripts. Limits excessive shortening of individual turns.
    refine_target_words_per_turn: float = 14.0
    # Judge and refiner default to **different model families** so the
    # discriminator/rewriter adversarial pair sees the other's outputs as
    # out-of-distribution. The judge runs the harder pairwise A/B
    # discrimination task; the refiner does the structural rewrite.
    judge_model: str = "gemini-3.1-pro-preview"
    refine_model: str | None = "claude-opus-4-7"
    # Optional reasoning effort for compatible models. The defaults use
    # each provider's normal behavior.
    refine_reasoning_effort: ReasoningEffort | None = None
    judge_reasoning_effort: ReasoningEffort | None = None
    # When the judge says "looks real," stop iterating early.
    early_stop_on_real_verdict: bool = True
    # Length budget for the candidate transcript shown to the judge (in
    # lines). The judge already sees an AMI excerpt trimmed to
    # `ami_reference_lines` (~200); matching the candidate to the same budget
    # closes the "length is itself a discriminator" shortcut. The refiner
    # continues to see the full candidate.
    judge_candidate_lines: int = 200
    # Asymmetric word-count delta guardrails. These are extreme bounds that
    # only trip on clearly broken rewrites (catastrophic content loss or
    # runaway expansion), not normal stylistic variation. Structural realism
    # naturally shrinks word counts (dropped topics, abandoned clauses) so
    # we allow more shrinkage than expansion. If the rewrite's word count
    # falls outside [original * (1 - shrink), original * (1 + expand)] we
    # fall back to the pre-rewrite transcript.
    max_word_count_shrink: float = 0.60
    max_word_count_expand: float = 0.40


class DurationTargetingSettings(_SettingsModel):
    """Make ``duration_minutes`` an enforceable transcript-length contract.

    When ``enabled``, step 4 computes a
    ``(floor, target, ceiling)`` word budget from ``duration_minutes`` and
    the language WPM range, renders it into the transcript prompt, retries
    on under-generation, and tightens the realism rewrite shrink cap against
    the calendar floor (not just the pre-rewrite baseline).
    """

    enabled: bool = True
    # Floor as fraction of ``duration_minutes * lang.wpm_min`` (e.g.
    # 30 min * 120 wpm * 0.80 = 2880 words). Transcripts below this trigger
    # a retry with a deficit-aware addendum.
    min_words_fraction: float = 0.80
    # Target as fraction of ``duration_minutes * mean(lang.wpm_range)``
    # (e.g. 30 * 135 * 1.0 = 4050). Surfaced in the prompt as the explicit
    # word-count target.
    target_words_fraction: float = 1.0
    # Soft ceiling as fraction of ``duration_minutes * lang.wpm_max * cap``
    # (e.g. 30 * 150 * 1.15 = 5175). Used for budget bookkeeping and viewer
    # diagnostics; not strictly enforced.
    max_words_fraction: float = 1.15
    # If the initial transcript word count is below ``floor_words``, retry
    # this many times with an "you under-shot by X words" addendum appended
    # to the prompt. 1 keeps the worst-case LLM cost bounded.
    short_transcript_retries: int = 1
    # When a meeting's target_words exceeds this threshold, generate the
    # transcript in ordered per-agenda chunks then concatenate, to keep each
    # call comfortably inside ``llm.meeting_transcript`` max_tokens. 0
    # disables segmentation (single-call only).
    segment_above_words: int = 4500
    # Opt-in for the segmented path above. Off by default so existing configs
    # (whose ordinary 30-min targets already exceed 4500 words) keep their
    # single-call behavior. When on, a meeting whose word ask
    # (floor * segment_overshoot) exceeds ``segment_above_words`` is written as
    # sequential parts of ~``segment_words`` each: every part sees the
    # previous turns, and only the last may close the meeting.
    segmented_generation: bool = False
    segment_words: int = 3000
    # Total word ask for segmented meetings, as a multiple of the floor.
    segment_overshoot: float = 1.10
    # Tighter shrink cap measured against the *calendar* floor (not the
    # pre-rewrite baseline). The realism rewrite may shrink the transcript
    # no lower than ``floor_words * (1 - max_shrink_vs_calendar)``. Only
    # consulted when ``enabled`` is True; otherwise the legacy
    # ``RealismRewriteSettings.max_word_count_shrink`` is the only floor.
    # Applies to the *rewrite* stage only; the refine stage is gated by
    # ``refine_min_retention`` instead.
    max_shrink_vs_calendar: float = 0.20
    # Minimum fraction of input words retained by refinement. This is
    # measured against the transcript entering the stage, since the booked
    # calendar slot may be much longer than the generated transcript.
    refine_min_retention: float = 0.45
    # Writer target in turns per booked slot-minute. Turn counts provide a
    # length control that is less affected by refinement's wording changes.
    target_turns_per_slot_minute: float = 8.7
    # Retry threshold for the turn contract, as a fraction of
    # ``target_turns``. Shortfalls trigger the same retry used for word counts.
    min_turns_fraction: float = 0.60


class MeetingPlanSettings(_SettingsModel):
    """Pin a workspace's meeting count and slot length at decomposition time.

    Applied in step 3, before any content exists: meetings are kept
    chronologically until ``count`` *held* ones are in (planned skips and
    cancellations get no transcript, so they don't count but stay on the
    calendar), and later-dated artifacts are dropped, as if the
    time period had ended there, so no generated artifact refers to a meeting
    that was never written. Used to hit exact corpus totals (workspaces /
    meetings / hours) without deleting finished meetings afterwards.
    """

    # Exact number of held (transcribed) meetings. None leaves the planner's
    # count alone.
    # Step 3 raises if the planner produced fewer, so a quota is never silently
    # missed -- widen the time period instead.
    count: int | None = None
    # Override every kept meeting's calendar slot (end_time moves; start_time
    # stays). Step 4 derives its word floor/target from this.
    duration_minutes: int | None = None


class GenerationSettings(_SettingsModel):
    generate_audio: bool = True
    generate_slides: bool = True
    max_concurrent: int = 10
    realism_rewrite: RealismRewriteSettings = Field(default_factory=RealismRewriteSettings)
    duration_targeting: DurationTargetingSettings = Field(
        default_factory=DurationTargetingSettings
    )
    meeting_plan: MeetingPlanSettings = Field(default_factory=MeetingPlanSettings)
    # When True, the step-4 transcript prompt includes an "anti-exposition" block
    # that pushes the LLM toward fragmented, debated delivery of technical content
    # (overlapping speakers, half-named jargon, talking past each other) rather
    # than clean tutorial-style hand-offs. Off by default; enable for configs that
    # are evaluated against AMI-style references where this delivery dominates.
    anti_exposition: bool = False
    llm: LLMBudgetConfig = Field(default_factory=LLMBudgetConfig)
    # Fraction of turn words deliberately marked as acoustically unclear at
    # mix time (step 10). 0 disables the feature. Sampled deterministically
    # per-meeting from the run seed.
    unclear_rate: float = 0.0
    # Per-meeting technicality is sampled from this distribution. Values are
    # proportions (do not need to sum to 1 exactly; they are normalized).
    # "low": ordinary working conversation, occasional jargon.
    # "mid": project-specific vocabulary surfaces several times during depth-ier discussion.
    # "deep": dense technical stretches where domain vocabulary recurs naturally.
    meeting_technicality_distribution: dict[str, float] = Field(
        default_factory=lambda: {"low": 0.55, "mid": 0.30, "deep": 0.15}
    )
    # Per-meeting overall join modality, sampled deterministically per meeting.
    # Keys must be members of MeetingJoinMode ("virtual", "in_person", "hybrid").
    # Values are proportions and are normalized at sampling time.
    meeting_join_mode_distribution: dict[str, float] = Field(
        default_factory=lambda: {"virtual": 0.55, "in_person": 0.20, "hybrid": 0.25}
    )
    # Per-person work arrangement used as a fallback when no LLM-emitted value
    # is available. Keys must be members of WorkArrangement ("remote", "hybrid",
    # "in_person"). Values are proportions and are normalized at sampling time.
    work_arrangement_distribution: dict[str, float] = Field(
        default_factory=lambda: {"remote": 0.25, "hybrid": 0.55, "in_person": 0.20}
    )

    @model_validator(mode="after")
    def _validate_distributions(self) -> "GenerationSettings":
        valid_join_modes = {"virtual", "in_person", "hybrid"}
        for key, weight in self.meeting_join_mode_distribution.items():
            if key not in valid_join_modes:
                raise ValueError(
                    f"meeting_join_mode_distribution has unknown key {key!r}; "
                    f"valid keys are {sorted(valid_join_modes)}"
                )
            if weight < 0:
                raise ValueError(
                    f"meeting_join_mode_distribution[{key!r}] is negative"
                )
        valid_arrangements = {"remote", "hybrid", "in_person"}
        for key, weight in self.work_arrangement_distribution.items():
            if key not in valid_arrangements:
                raise ValueError(
                    f"work_arrangement_distribution has unknown key {key!r}; "
                    f"valid keys are {sorted(valid_arrangements)}"
                )
            if weight < 0:
                raise ValueError(
                    f"work_arrangement_distribution[{key!r}] is negative"
                )
        return self


class RoomSimulationSettings(_SettingsModel):
    """Settings for the post-finalize room acoustics simulation (step 10).

    When ``enabled`` is true and a meeting has any ``IN_ROOM`` participants, the
    in-room speakers are convolved with a synthesized conference-room RIR and
    layered with background noise, an AC-unit hum, short sound events, and a
    small white-noise floor."""

    enabled: bool = False
    sample_rate: int = 16000
    room_dims: list[float] = Field(default_factory=lambda: [7.0, 5.0, 2.7])
    # Maximum image-source order. With ``ray_tracing=True`` the ray tracer
    # handles the diffuse late tail, so the image-source pass only needs to
    # cover early reflections. 3 is the pra-recommended value for small
    # rooms with ray tracing on; higher orders (the historical default was
    # 8) make the simulation slower and add a discrete-comb coloration to
    # the late tail without improving realism.
    max_order: int = 3
    ray_tracing: bool = True
    air_absorption: bool = True
    # Ray-tracing tail controls, applied via ``room.set_ray_tracing(...)``
    # after construction when ``ray_tracing=True``. The pyroomacoustics
    # defaults (``time_thres=10s``, ``energy_thres=1e-7``) are designed for
    # concert halls and produce an unrealistically long, "echo-y" late
    # tail in conference rooms. ``time_thres`` is the dominant lever on
    # measured RT60: empirically ``time_thres=0.35`` yields ~0.35-0.45 s
    # RT60 for treated conference rooms and ~0.6-0.7 s for sparsely
    # furnished offices (matching the literature for ~95 m³ rooms). The
    # sampled wall-material profile still spans the upper end of that
    # range, so the per-meeting distribution covers reverberant rooms even
    # though the mean sits below the original ``0.4`` setting. Larger
    # values bias every meeting toward a noticeably reverberant "big
    # room" sound regardless of materials, because the tail's energy
    # decay is set by ``time_thres`` long after the materials have damped
    # the early reflections. Tighten this further (e.g. 0.3) for studio-
    # dry runs.
    ray_tracing_time_thres_s: float = 0.35
    ray_tracing_energy_thres: float = 1e-5
    # Per-meeting acoustic difficulty sampling.  For each meeting a single RNG
    # keyed by (seed, "difficulty", meeting_id) draws one value uniformly from
    # each range and applies it for that meeting only, producing natural
    # variation across the dataset without changing any global default.
    #
    # ray_tracing_time_thres_range — [min, max] seconds for the ray-tracer
    #   time threshold.  Wider rooms / sparsely furnished profiles sit toward
    #   the top; treated boardrooms toward the bottom.
    # room_size_scale_range — [min, max] multiplier applied to the room's X/Y
    #   footprint (height stays fixed).  Values > 1 also scale the table
    #   seating radius so speakers move proportionally further from the mic,
    #   increasing source distance and early-reflection density simultaneously.
    # ac_boost_db_range — [min, max] dB *reduction* of ac_unit_db_below_background.
    #   0 means the AC is at the configured default level; 4 means it is 4 dB
    #   louder (harder), simulating noisier HVAC environments.
    ray_tracing_time_thres_range: list[float] = Field(
        default_factory=lambda: [0.35, 0.40]
    )
    room_size_scale_range: list[float] = Field(default_factory=lambda: [1.0, 1.3])
    ac_boost_db_range: list[float] = Field(default_factory=lambda: [0.0, 4.0])
    # When True, sample per-surface wall/floor/ceiling materials from a
    # bank of office "material profiles" using a seed derived from
    # ``settings.seed`` and the meeting id. When False, fall back to the
    # original fixed scalar α defaults baked into ``room_simulation.py``.
    randomize_wall_materials: bool = True
    # Noise sources are looked up relative to ``output_root`` unless absolute.
    background_noise_dir: str = "../data/noise/background"
    hum_noise_dir: str = "../data/noise/hum"
    event_noise_dir: str = "../data/noise/events"
    # Subfolders of ``background_noise_dir`` to pool when selecting background
    # noise. Meeting-to-meeting variety comes from covering several distinct
    # environments here.
    background_environments: list[str] = Field(
        default_factory=lambda: ["office"]
    )
    # Mixing levels (dB).
    background_snr_db: float = 20.0
    ac_unit_db_below_background: float = 6.0
    event_snr_db: float = 10.0
    events_per_minute: float = 0.5
    white_noise_db: float = -60.0
    event_categories: list[str] = Field(
        default_factory=lambda: [
            "keyboard_typing",
            "door_wood_creak",
            "clock_tick",
            "coughing",
            "sneezing",
            "footsteps",
            "door_wood_knock",
            "drinking_sipping",
            "mouse_click",
        ]
    )


class ValidationSettings(_SettingsModel):
    fail_on_error: bool = False


class TTSPromptSettings(_SettingsModel):
    spoken_form: str = "config/prompts/tts/spoken_form.txt"
    enrollment_text: str = "config/prompts/tts/enrollment_text.txt"
    overlap_annotator: str = "config/prompts/tts/overlap_annotator.txt"
    backchannel_classifier: str = "config/prompts/tts/backchannel_classifier.txt"
    backchannel_anchor: str = "config/prompts/tts/backchannel_anchor.txt"


class TranscriptsPromptSettings(_SettingsModel):
    entities: str = "config/prompts/transcripts/transcript_entities.txt"
    realism_rewrite: str = "config/prompts/transcripts/realism_rewrite.txt"
    realism_judge: str = "config/prompts/transcripts/realism_judge.txt"
    realism_refine: str = "config/prompts/transcripts/realism_refine.txt"


class WorkspaceSamplingPromptSettings(_SettingsModel):
    system: str = "config/prompts/workspace_sampling/system.txt"
    company: str = "config/prompts/workspace_sampling/company.txt"
    org_spine: str = "config/prompts/workspace_sampling/org_spine.txt"
    team_people_sketch: str = "config/prompts/workspace_sampling/team_people_sketch.txt"
    person_detail: str = "config/prompts/workspace_sampling/person_detail.txt"
    external_person_detail: str = "config/prompts/workspace_sampling/external_person_detail.txt"
    project_index: str = "config/prompts/workspace_sampling/project_index.txt"
    project_cluster_sketch: str = "config/prompts/workspace_sampling/project_cluster_sketch.txt"
    project_detail: str = "config/prompts/workspace_sampling/project_detail.txt"
    recurring_meetings_company: str = "config/prompts/workspace_sampling/recurring_meetings_company.txt"
    recurring_meetings_team: str = "config/prompts/workspace_sampling/recurring_meetings_team.txt"
    recurring_meetings_cross: str = "config/prompts/workspace_sampling/recurring_meetings_cross.txt"


class TimePeriodPromptSettings(_SettingsModel):
    system: str = "config/prompts/time_period/system.txt"
    narrative: str = "config/prompts/time_period/narrative.txt"
    week_frame: str = "config/prompts/time_period/week_frame.txt"
    week_meetings: str = "config/prompts/time_period/week_meetings.txt"
    week_emails: str = "config/prompts/time_period/week_emails.txt"
    week_documents: str = "config/prompts/time_period/week_documents.txt"
    week_slides: str = "config/prompts/time_period/week_slides.txt"


class MeetingsPromptSettings(_SettingsModel):
    system: str = "config/prompts/meetings/system.txt"
    overview: str = "config/prompts/meetings/overview.txt"
    transcript: str = "config/prompts/meetings/transcript.txt"
    event_extraction: str = "config/prompts/meetings/event_extraction.txt"
    anti_exposition: str = "config/prompts/meetings/anti_exposition.txt"


class EmailsPromptSettings(_SettingsModel):
    system: str = "config/prompts/emails/system.txt"
    message: str = "config/prompts/emails/thread.txt"


class DocumentsPromptSettings(_SettingsModel):
    system: str = "config/prompts/documents/system.txt"
    document: str = "config/prompts/documents/document.txt"


class SlidesPromptSettings(_SettingsModel):
    system: str = "config/prompts/slides/system.txt"
    content: str = "config/prompts/slides/content.txt"


class PostprocessPromptSettings(_SettingsModel):
    jargon_extraction: str = "config/prompts/postprocess/jargon_extraction.txt"


class PromptSettings(_SettingsModel):
    tts: TTSPromptSettings = Field(default_factory=TTSPromptSettings)
    transcripts: TranscriptsPromptSettings = Field(default_factory=TranscriptsPromptSettings)
    workspace_sampling: WorkspaceSamplingPromptSettings = Field(default_factory=WorkspaceSamplingPromptSettings)
    time_period: TimePeriodPromptSettings = Field(default_factory=TimePeriodPromptSettings)
    meetings: MeetingsPromptSettings = Field(default_factory=MeetingsPromptSettings)
    emails: EmailsPromptSettings = Field(default_factory=EmailsPromptSettings)
    documents: DocumentsPromptSettings = Field(default_factory=DocumentsPromptSettings)
    slides: SlidesPromptSettings = Field(default_factory=SlidesPromptSettings)
    postprocess: PostprocessPromptSettings = Field(default_factory=PostprocessPromptSettings)


class SyntheticDataSettings(_SettingsModel):
    seed: int | None = None
    language: str = "en"
    model: str
    output_root: str
    group: str = "default"
    run_id: str
    workspace: WorkspaceSettings
    tts: TTSSettings = Field(default_factory=TTSSettings)
    speech_pipeline: SpeechPipelineSettings = Field(default_factory=SpeechPipelineSettings)
    alignment: AlignmentSettings = Field(default_factory=AlignmentSettings)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    room_simulation: RoomSimulationSettings = Field(default_factory=RoomSimulationSettings)
    validation: ValidationSettings = Field(default_factory=ValidationSettings)
    prompts: PromptSettings = Field(default_factory=PromptSettings)


def default_synthetic_config_path() -> str:
    return str(Path(__file__).resolve().parent.parent / "config" / "smoke" / "smoke_sanity_check.yaml")


def _normalize_settings_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key).lower(): _normalize_settings_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_normalize_settings_value(item) for item in value]
    return value


def validate_synthetic_data_settings(data: dict[str, Any]) -> SyntheticDataSettings:
    return SyntheticDataSettings.model_validate(_normalize_settings_value(data))


def load_synthetic_data_settings(config_path: str | None = None) -> SyntheticDataSettings:
    resolved_path = config_path or default_synthetic_config_path()
    dynasettings = Dynaconf(settings_files=[resolved_path])
    return validate_synthetic_data_settings(dynasettings.to_dict())
