"""Calendar and scheduling data models."""
from __future__ import annotations

from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Work arrangement and meeting modality
# ---------------------------------------------------------------------------


class WorkArrangement(str, Enum):
    """Where a person typically works from. Influences how they tend to join
    meetings (remotely vs. from a conference room) but does not fully determine
    it — a hybrid worker can attend a given meeting either way."""

    REMOTE = "remote"
    HYBRID = "hybrid"
    IN_PERSON = "in_person"


class MeetingJoinMode(str, Enum):
    """Overall modality of a meeting.

    ``virtual``  — all active attendees are on a video call from their own endpoints.
    ``in_person`` — all active attendees are physically co-located in a room.
    ``hybrid``    — some attendees are in a room together, others join remotely.
    """

    VIRTUAL = "virtual"
    IN_PERSON = "in_person"
    HYBRID = "hybrid"


class ParticipantJoinMode(str, Enum):
    """How an individual attendee joins a meeting."""

    REMOTE = "remote"
    IN_ROOM = "in_room"
    PHONE = "phone"


class ParticipantJoinInfo(BaseModel):
    mode: ParticipantJoinMode
    location_hint: str = ""
    audio_context: str = ""


class MeetingJoinSpec(BaseModel):
    """Ground-truth modality and per-participant join state for a meeting.

    ``mode`` is the overall modality and ``participant_modes`` maps active
    attendee IDs to their individual join info. ``room_name`` is set for
    ``in_person`` / ``hybrid`` meetings."""

    mode: MeetingJoinMode = MeetingJoinMode.VIRTUAL
    participant_modes: dict[str, ParticipantJoinInfo] = Field(default_factory=dict)
    room_name: str = ""
    notes: str = ""

    @field_validator("room_name", "notes", mode="before")
    @classmethod
    def _coerce_none_str(cls, v: object) -> str:
        return v if isinstance(v, str) else ""


# ---------------------------------------------------------------------------
# Meeting classification labels (structured dimensions)
# These provide structured analysis axes without constraining the meeting_type
# field to a closed set.
# ---------------------------------------------------------------------------


class Affiliation(str, Enum):
    """How participants relate organizationally."""

    INTRA_TEAM = "intra_team"
    CROSS_TEAM = "cross_team"
    CROSS_DEPARTMENT = "cross_department"
    ORG_EXTERNAL = "org_external"
    AMBIGUOUS = "ambiguous"


class ParticipantCount(str, Enum):
    """Coarse participant count bucket."""

    THREE_PLUS = "3+"
    TWO = "2"
    SINGLE_SPEAKER = "single_speaker"
    PRIMARY_WITH_AUDIENCE = "primary_with_audience"


class MeetingPurpose(str, Enum):
    """High-level meeting purpose."""

    TRAINING = "training"
    ROUTINE_MONITORING = "routine_monitoring"
    PROBLEM_SOLVING = "problem_solving"
    INFORMATION_GIVING = "information_giving"


class MeetingLabels(BaseModel):
    """Structured classification labels for a meeting, independent of the
    free-form ``meeting_type`` string."""

    affiliation: Affiliation | None = None
    participant_count: ParticipantCount | None = None
    purpose: MeetingPurpose | None = None
    topic: str = ""  # free-form topic category
    roles: list[str] = []  # e.g. ["salesperson", "client"]
    raw_labels: dict[str, str] = Field(default_factory=dict, exclude=False)
    """Preserves the model's original label values before enum coercion."""

    @model_validator(mode="before")
    @classmethod
    def _coerce_enum_fields(cls, data: dict) -> dict:
        if not isinstance(data, dict):
            return data
        raw = {}
        for field_name, enum_cls in [
            ("affiliation", Affiliation),
            ("participant_count", ParticipantCount),
            ("purpose", MeetingPurpose),
        ]:
            val = data.get(field_name)
            if val is None:
                continue
            raw[field_name] = val
            try:
                enum_cls(val)
            except ValueError:
                # Unknown value — store raw and set to None
                data[field_name] = None
        data.setdefault("raw_labels", {})
        data["raw_labels"].update(raw)
        return data


# ---------------------------------------------------------------------------
# Calendar models
# ---------------------------------------------------------------------------


class RecurringMeeting(BaseModel):
    id: str
    title: str
    meeting_type: str  # open-ended: "1x1", "team_standup", "sales_discovery", etc.
    organizer_id: str
    attendee_ids: list[str]
    cadence: str  # "daily", "weekly", "biweekly", "monthly"
    day_of_week: int | None = None  # 0=Monday
    time_of_day: str = "09:00"  # "HH:MM" format
    duration_minutes: int
    description: str = ""
    labels: MeetingLabels = Field(default_factory=MeetingLabels)
    join_spec: MeetingJoinSpec = Field(default_factory=MeetingJoinSpec)


class CalendarEvent(BaseModel):
    id: str
    title: str
    meeting_type: str  # open-ended
    start_time: datetime
    end_time: datetime
    organizer_id: str
    attendee_ids: list[str]
    recurring_meeting_id: str | None = None
    location: str = "Virtual"
    description: str = ""
    agenda: list[str] = []
    labels: MeetingLabels = Field(default_factory=MeetingLabels)
    join_spec: MeetingJoinSpec = Field(default_factory=MeetingJoinSpec)
    meeting_artifact_id: str | None = None
    related_document_ids: list[str] = []
    related_email_thread_ids: list[str] = []


class Schedule(BaseModel):
    events: list[CalendarEvent] = []
    recurring_meetings: list[RecurringMeeting] = []

    def events_for_person(self, person_id: str) -> list[CalendarEvent]:
        return [
            e
            for e in self.events
            if person_id in e.attendee_ids or person_id == e.organizer_id
        ]

    def events_for_date(self, d: date) -> list[CalendarEvent]:
        return [e for e in self.events if e.start_time.date() == d]

    def has_conflict(self, person_id: str, start: datetime, end: datetime) -> bool:
        for event in self.events_for_person(person_id):
            if event.start_time < end and event.end_time > start:
                return True
        return False
