"""Pydantic models for step-2 narrative and week-plan contracts."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, TypeAdapter

from .calendar import MeetingJoinSpec, MeetingLabels


class NarrativeMajorEvent(BaseModel):
    approximate_date: str
    type: str
    title: str
    description: str
    impact: str
    involved_team_ids: list[str] = Field(default_factory=list)
    involved_person_ids: list[str] = Field(default_factory=list)


class ProjectArcKeyDate(BaseModel):
    date: str
    event: str


class ProjectArc(BaseModel):
    project_id: str
    arc_summary: str
    key_dates: list[ProjectArcKeyDate] = Field(default_factory=list)


class Narrative(BaseModel):
    narrative_summary: str
    themes: list[str] = Field(default_factory=list)
    major_events: list[NarrativeMajorEvent] = Field(default_factory=list)
    project_arcs: list[ProjectArc] = Field(default_factory=list)


_project_arcs_adapter = TypeAdapter(list[ProjectArc])


def validate_project_arcs(data: Any) -> list[dict[str, Any]]:
    """Validate persisted project trajectory payloads."""
    arcs = _project_arcs_adapter.validate_python(data)
    return [arc.model_dump(mode="json") for arc in arcs]


class AdhocMeetingPlan(BaseModel):
    title: str
    meeting_type: str
    organizer_id: str = ""
    attendee_ids: list[str] = Field(default_factory=list)
    day_of_week: int = 0
    preferred_time: str | None = None
    duration_minutes: int = 30
    purpose: str = ""
    expected_outcomes: list[str] = Field(default_factory=list)
    should_generate_slides: bool = False
    related_project_id: str | None = None
    labels: MeetingLabels = Field(default_factory=MeetingLabels)
    join_spec: MeetingJoinSpec | None = None


class MeetingAttendanceChange(BaseModel):
    person_id: str
    status: str
    reason: str = ""


class MeetingOutcomePlan(BaseModel):
    meeting_title: str
    outcome: str
    attendance_changes: list[MeetingAttendanceChange] = Field(default_factory=list)
    notes: str = ""


class EmailThreadPlan(BaseModel):
    subject: str
    initiator_id: str = ""
    participant_ids: list[str] = Field(default_factory=list)
    cc_ids: list[str] = Field(default_factory=list)
    day_of_week: int = 0
    preferred_time: str | None = None
    num_messages: int = 2
    purpose: str = ""
    tone: str = ""
    related_meeting_id: str | None = None
    related_project_id: str | None = None
    is_followup: bool = False


class DocumentPlan(BaseModel):
    title: str
    author_id: str = ""
    doc_type: str = "notes"
    day_of_week: int = 0
    preferred_time: str | None = None
    purpose: str = ""
    related_project_id: str | None = None
    related_meeting_id: str | None = None
    approximate_length: str = "medium"


class SlidePresentationPlan(BaseModel):
    title: str
    author_id: str = ""
    meeting_title: str = ""
    num_slides: int = 8
    purpose: str = ""
    key_points: list[str] = Field(default_factory=list)


class ProjectUpdatePlan(BaseModel):
    project_id: str | None = None
    new_status: str = ""
    details: str = ""


class StateChanges(BaseModel):
    project_updates: list[ProjectUpdatePlan] = Field(default_factory=list)
    new_priorities: list[str] = Field(default_factory=list)
    mood_shifts: dict[str, str] = Field(default_factory=dict)


class WeekPlan(BaseModel):
    week_num: int
    week_start: str
    week_end: str
    week_summary: str
    forward_brief: str
    adhoc_meetings: list[AdhocMeetingPlan] = Field(default_factory=list)
    meeting_outcomes: list[MeetingOutcomePlan] = Field(default_factory=list)
    email_threads: list[EmailThreadPlan] = Field(default_factory=list)
    documents: list[DocumentPlan] = Field(default_factory=list)
    slide_presentations: list[SlidePresentationPlan] = Field(default_factory=list)
    action_items_resolved: list[str] = Field(default_factory=list)
    state_changes: StateChanges = Field(default_factory=StateChanges)


class WeekContentPlan(BaseModel):
    """Per-family content guidance issued by the frame to fanout calls.

    The frame is authoritative on scope. Each family call must respect these
    targets so the week stays internally coherent across families.
    """

    meetings_target: int = 0
    emails_target: int = 0
    documents_target: int = 0
    slides_target: int = 0
    priorities: list[str] = Field(default_factory=list)
    notes: str = ""


class WeekFrame(BaseModel):
    """Authoritative weekly plan produced before family-specific fanout."""

    week_summary: str
    forward_brief: str
    action_items_resolved: list[str] = Field(default_factory=list)
    state_changes: StateChanges = Field(default_factory=StateChanges)
    content_plan: WeekContentPlan = Field(default_factory=WeekContentPlan)


class WeekMeetingsOutput(BaseModel):
    adhoc_meetings: list[AdhocMeetingPlan] = Field(default_factory=list)
    meeting_outcomes: list[MeetingOutcomePlan] = Field(default_factory=list)


class WeekEmailsOutput(BaseModel):
    email_threads: list[EmailThreadPlan] = Field(default_factory=list)


class WeekDocumentsOutput(BaseModel):
    documents: list[DocumentPlan] = Field(default_factory=list)


class WeekSlidesOutput(BaseModel):
    slide_presentations: list[SlidePresentationPlan] = Field(default_factory=list)
