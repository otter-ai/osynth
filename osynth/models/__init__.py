"""Data models for the synthetic evaluation pipeline.

All models use Pydantic v2 and are fully JSON-serializable. The model hierarchy
is designed to support both realistic data generation and ground-truth extraction
for evaluation metrics.

Key Design Decisions
--------------------

**Open-ended meeting types.** The ``meeting_type`` field on ``CalendarEvent`` and
``RecurringMeeting`` is a free-form string (e.g., "1x1", "team_standup",
"sales_discovery", "board_meeting") rather than a closed enum. This lets the LLM
generate natural meeting types without constraining the taxonomy. For structured
analysis, ``MeetingLabels`` provides orthogonal classification dimensions.

**MeetingLabels dimensions.** ``MeetingLabels`` classifies meetings along three
independent axes — ``Affiliation`` (intra_team, cross_team, org_external, ...),
``ParticipantCount`` (2, 3+, single_speaker, primary_with_audience), and
``MeetingPurpose`` (training, routine_monitoring, problem_solving,
information_giving) — plus free-form ``topic`` and ``roles`` fields. Unknown enum
values are gracefully coerced to ``None`` with the original value preserved in
``raw_labels``, so new meeting types from the LLM never cause validation errors.

**PersonBase inheritance.** ``Person`` (internal employee) and ``ExternalPerson``
(prospect, customer, partner) share a common ``PersonBase`` with id, name, gender,
nickname policy, voice_id, email, personality_traits, expertise, and
speaking_style. This allows the pipeline to handle mixed-affiliation meetings
uniformly while preserving the distinction between internal and external
participants.

**Gender.** The ``gender`` field supports "male", "female", and "non-binary" to
enable realistic voice assignment and pronoun usage in generated content.

**SpeakingStyle and DisfluencyProfile.** Each person has a stable ``SpeakingStyle``
(pace, verbosity, interruption_tendency, accent, notable_patterns) that persists
across all meetings. The nested ``DisfluencyProfile`` provides numeric rates
(filler_rate, repetition_rate, restart_rate, elongation_rate per 100 words) and
preferred filler/hedge words, used by the TTS disfluency injection pipeline to
produce natural-sounding speech.

**Immutable state with event reducer.** ``WorkspaceState`` captures a snapshot of
projects, action items, threads, decisions, and per-person context at a point in
time. The ``reduce()`` function applies a ``WorkspaceEvent`` to produce a new state
(pure function, no mutation). This pattern ensures reproducible state evolution and
makes it straightforward to reconstruct ground truth at any point in the timeline.

**Artifact action items.** ``ActionItem`` in ``artifacts.py`` is the ground-truth
record generated during meeting creation, with assignee_id, due_date, and status.
``ActionItemState`` in ``state.py`` is the corresponding state-tracker entry updated
by the event reducer. Both are available for evaluation: the artifact version provides
per-meeting ground truth, while the state version tracks lifecycle across meetings.

Model Hierarchy
---------------

::

    workspace.py    Company, CompanySize, WorkspaceConfig
    people.py       PersonBase → Person, ExternalPerson
                    NicknameUsage, Team, Role, OrgStructure
                    SpeakingStyle, DisfluencyProfile
    calendar.py     CalendarEvent, RecurringMeeting, Schedule
                    MeetingLabels (Affiliation, ParticipantCount, MeetingPurpose)
    artifacts.py    MeetingArtifact, ActionItem
                    EmailThread, EmailMessage
                    Document
                    SlidePresentation, Slide, SlideTheme
    planning.py     Narrative, WeekPlan
    llm_outputs.py  Structured response models for step-4 and TTS LLM calls
    tts.py          FormattedTurn contract for transcript-form/TTS handoff
    state.py        WorkspaceState, WorkspaceEvent, EventLog, EventType
                    reduce() — pure event reducer
"""

from .workspace import Company, CompanySize, WorkspaceConfig
from .settings import (
    SyntheticDataSettings,
    WorkspaceSettings,
    TTSSettings,
    SpeechPipelineSettings,
    AlignmentSettings,
    GenerationSettings,
    ValidationSettings,
    PromptSettings,
    load_synthetic_data_settings,
    validate_synthetic_data_settings,
)
from .people import (
    Person,
    PersonBase,
    NicknameUsage,
    ExternalPerson,
    Team,
    Role,
    OrgStructure,
    SpeakingStyle,
    DisfluencyProfile,
)
from .state import (
    Project,
    ProjectStatus,
    WorkspaceState,
    ActionItemState,
    EventType,
    WorkspaceEvent,
    EventLog,
)
from .calendar import (
    Affiliation,
    ParticipantCount,
    MeetingPurpose,
    MeetingLabels,
    RecurringMeeting,
    CalendarEvent,
    Schedule,
)
from .artifacts import (
    ActionItem,
    MeetingArtifact,
    EmailMessage,
    EmailThread,
    Document,
    Slide,
    SlideTheme,
    SlidePresentation,
)
from .planning import Narrative, WeekPlan
from .tts import FormattedTurn

__all__ = [
    "Company",
    "CompanySize",
    "WorkspaceConfig",
    "SyntheticDataSettings",
    "WorkspaceSettings",
    "TTSSettings",
    "SpeechPipelineSettings",
    "AlignmentSettings",
    "GenerationSettings",
    "ValidationSettings",
    "PromptSettings",
    "load_synthetic_data_settings",
    "validate_synthetic_data_settings",
    "Person",
    "ExternalPerson",
    "NicknameUsage",
    "Team",
    "Role",
    "OrgStructure",
    "Project",
    "ProjectStatus",
    "WorkspaceState",
    "ActionItemState",
    "EventType",
    "WorkspaceEvent",
    "EventLog",
    "PersonBase",
    "SpeakingStyle",
    "DisfluencyProfile",
    "Affiliation",
    "ParticipantCount",
    "MeetingPurpose",
    "MeetingLabels",
    "RecurringMeeting",
    "CalendarEvent",
    "Schedule",
    "ActionItem",
    "MeetingArtifact",
    "EmailMessage",
    "EmailThread",
    "Document",
    "Slide",
    "SlideTheme",
    "SlidePresentation",
    "Narrative",
    "WeekPlan",
    "FormattedTurn",
]
