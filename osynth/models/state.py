"""Workspace state management with reducer pattern."""
from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ProjectStatus(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    ON_HOLD = "on_hold"


SpecializedTermCategory = Literal["industry_term", "internal_coinage"]


class SpecializedTerm(BaseModel):
    """A piece of domain jargon used on a project.

    `surface` is the canonical written form. `category` distinguishes real
    industry/domain vocabulary from internal-to-this-company coinages.
    `expansion` records what an acronym or codename stands for (None when the
    term is not an acronym/codename).
    """

    surface: str
    category: SpecializedTermCategory = "industry_term"
    expansion: str | None = None

    @field_validator("surface", mode="before")
    @classmethod
    def _coerce_surface(cls, v: object) -> str:
        return v if isinstance(v, str) else ""

    @field_validator("expansion", mode="before")
    @classmethod
    def _coerce_expansion(cls, v: object) -> str | None:
        if v is None:
            return None
        if isinstance(v, str):
            stripped = v.strip()
            return stripped or None
        return None


class Project(BaseModel):
    id: str
    name: str = ""
    description: str = ""
    status: ProjectStatus = ProjectStatus.NOT_STARTED
    owner_id: str = ""
    team_ids: list[str] = []
    start_date: date = date.min
    target_end_date: date | None = None
    milestones: list[str] = []
    current_blockers: list[str] = []
    specialized_terms: list[SpecializedTerm] = []

    @field_validator("name", "description", "owner_id", mode="before")
    @classmethod
    def _coerce_none_str(cls, v: object) -> str:
        return v if isinstance(v, str) else ""


class ActionItemState(BaseModel):
    id: str
    description: str
    assignee_id: str
    due_date: date | None = None
    status: str = "open"
    source_meeting_id: str | None = None
    created_at: datetime


class WorkspaceState(BaseModel):
    """Snapshot of workspace state at a point in time."""

    timestamp: datetime
    projects: dict[str, Project] = {}
    action_items: dict[str, ActionItemState] = {}
    active_threads: dict[str, str] = {}  # thread_id -> subject
    recent_decisions: list[str] = Field(default_factory=list)
    team_morale: dict[str, str] = {}  # team_id -> "high"/"medium"/"low"
    person_focus: dict[str, list[str]] = {}  # person_id -> current priorities
    person_context: dict[str, list[str]] = {}  # person_id -> context strings
    attendance_outcomes: dict[str, dict[str, str]] = {}  # meeting_id -> person_id -> outcome
    meeting_statuses: dict[str, str] = {}  # meeting_id -> status


# --- Event Types ---


class EventType(str, Enum):
    MEETING_COMPLETED = "meeting_completed"
    MEETING_SKIPPED = "meeting_skipped"
    MEETING_CANCELED = "meeting_canceled"
    PARTICIPANT_MISSED_MEETING = "participant_missed_meeting"
    PARTICIPANT_JOINED_LATE = "participant_joined_late"
    EMAIL_SENT = "email_sent"
    DOCUMENT_CREATED = "document_created"
    DOCUMENT_UPDATED = "document_updated"
    PROJECT_STATUS_CHANGED = "project_status_changed"
    ACTION_ITEM_CREATED = "action_item_created"
    ACTION_ITEM_COMPLETED = "action_item_completed"
    DECISION_MADE = "decision_made"
    DAY_STARTED = "day_started"
    WEEK_STARTED = "week_started"


class WorkspaceEvent(BaseModel):
    type: EventType
    timestamp: datetime
    data: dict = {}
    source_artifact_id: str | None = None


class EventLog(BaseModel):
    events: list[WorkspaceEvent] = []

    def append(self, event: WorkspaceEvent) -> None:
        self.events.append(event)

    def replay(self, initial_state: WorkspaceState) -> WorkspaceState:
        state = initial_state
        for event in self.events:
            state = reduce(state, event)
        return state


# --- Reducer ---

MAX_RECENT_DECISIONS = 20
MAX_PERSON_CONTEXT = 10


def reduce(state: WorkspaceState, event: WorkspaceEvent) -> WorkspaceState:
    """Pure function: applies an event to produce a new state."""
    new_state = state.model_copy(deep=True)
    new_state.timestamp = event.timestamp

    match event.type:
        case EventType.MEETING_COMPLETED:
            _handle_meeting_completed(new_state, event)
        case EventType.MEETING_SKIPPED:
            _handle_meeting_status(new_state, event, "skipped")
        case EventType.MEETING_CANCELED:
            _handle_meeting_status(new_state, event, "canceled")
        case EventType.PARTICIPANT_MISSED_MEETING:
            _handle_attendance_outcome(new_state, event, "missed")
        case EventType.PARTICIPANT_JOINED_LATE:
            _handle_attendance_outcome(new_state, event, "join_late")
        case EventType.EMAIL_SENT:
            _handle_email_sent(new_state, event)
        case EventType.DOCUMENT_CREATED:
            _handle_document_created(new_state, event)
        case EventType.ACTION_ITEM_CREATED:
            _handle_action_item_created(new_state, event)
        case EventType.ACTION_ITEM_COMPLETED:
            _handle_action_item_completed(new_state, event)
        case EventType.PROJECT_STATUS_CHANGED:
            _handle_project_status_changed(new_state, event)
        case EventType.DECISION_MADE:
            _handle_decision_made(new_state, event)

    return new_state


def _handle_meeting_completed(state: WorkspaceState, event: WorkspaceEvent) -> None:
    meeting_id = event.source_artifact_id or event.data.get("meeting_id", "")
    if meeting_id:
        state.meeting_statuses[meeting_id] = "completed"
        state.attendance_outcomes.setdefault(meeting_id, {})
    summary = event.data.get("summary", "")
    attendee_ids = event.data.get("attendee_ids", [])
    for pid in attendee_ids:
        if meeting_id:
            state.attendance_outcomes[meeting_id].setdefault(pid, "attended")
        if pid not in state.person_context:
            state.person_context[pid] = []
        ctx = f"Meeting: {event.data.get('title', 'Unknown')} - {summary}"
        state.person_context[pid].append(ctx)
        if len(state.person_context[pid]) > MAX_PERSON_CONTEXT:
            state.person_context[pid] = state.person_context[pid][-MAX_PERSON_CONTEXT:]


def _handle_meeting_status(
    state: WorkspaceState, event: WorkspaceEvent, status: str
) -> None:
    meeting_id = event.source_artifact_id or event.data.get("meeting_id", "")
    if not meeting_id:
        return
    state.meeting_statuses[meeting_id] = status
    state.attendance_outcomes.setdefault(meeting_id, {})
    title = event.data.get("title", "Unknown")
    affected_ids = event.data.get("attendee_ids", [])
    for pid in affected_ids:
        if pid not in state.person_context:
            state.person_context[pid] = []
        state.person_context[pid].append(f"Meeting {title} was {status}.")
        if len(state.person_context[pid]) > MAX_PERSON_CONTEXT:
            state.person_context[pid] = state.person_context[pid][-MAX_PERSON_CONTEXT:]


def _handle_attendance_outcome(
    state: WorkspaceState, event: WorkspaceEvent, outcome: str
) -> None:
    meeting_id = event.source_artifact_id or event.data.get("meeting_id", "")
    person_id = event.data.get("person_id", "")
    if not meeting_id or not person_id:
        return
    state.attendance_outcomes.setdefault(meeting_id, {})
    state.attendance_outcomes[meeting_id][person_id] = outcome
    if person_id not in state.person_context:
        state.person_context[person_id] = []
    title = event.data.get("title", "Unknown")
    context = f"Missed meeting: {title}" if outcome == "missed" else f"Joined meeting late: {title}"
    state.person_context[person_id].append(context)
    if len(state.person_context[person_id]) > MAX_PERSON_CONTEXT:
        state.person_context[person_id] = state.person_context[person_id][-MAX_PERSON_CONTEXT:]


def _handle_email_sent(state: WorkspaceState, event: WorkspaceEvent) -> None:
    thread_id = event.data.get("thread_id", "")
    subject = event.data.get("subject", "")
    if thread_id:
        state.active_threads[thread_id] = subject
    recipient_ids = event.data.get("recipient_ids", [])
    cc_ids = event.data.get("cc_ids", [])
    for pid in dict.fromkeys([*recipient_ids, *cc_ids]):
        if pid not in state.person_context:
            state.person_context[pid] = []
        state.person_context[pid].append(f"Email: {subject}")
        if len(state.person_context[pid]) > MAX_PERSON_CONTEXT:
            state.person_context[pid] = state.person_context[pid][-MAX_PERSON_CONTEXT:]


def _handle_document_created(state: WorkspaceState, event: WorkspaceEvent) -> None:
    author_id = event.data.get("author_id", "")
    title = event.data.get("title", "")
    if author_id:
        if author_id not in state.person_context:
            state.person_context[author_id] = []
        state.person_context[author_id].append(f"Created document: {title}")


def _handle_action_item_created(state: WorkspaceState, event: WorkspaceEvent) -> None:
    item_id = event.data.get("id", "")
    if item_id:
        state.action_items[item_id] = ActionItemState(
            id=item_id,
            description=event.data.get("description", ""),
            assignee_id=event.data.get("assignee_id", ""),
            due_date=event.data.get("due_date"),
            source_meeting_id=event.source_artifact_id,
            created_at=event.timestamp,
        )


def _handle_action_item_completed(state: WorkspaceState, event: WorkspaceEvent) -> None:
    item_id = event.data.get("id", "")
    if item_id in state.action_items:
        state.action_items[item_id].status = "completed"


def _handle_project_status_changed(state: WorkspaceState, event: WorkspaceEvent) -> None:
    project_id = event.data.get("project_id", "")
    new_status = event.data.get("new_status", "")
    if project_id in state.projects and new_status:
        state.projects[project_id].status = ProjectStatus(new_status)
        blockers = event.data.get("blockers", None)
        if blockers is not None:
            state.projects[project_id].current_blockers = blockers


def _handle_decision_made(state: WorkspaceState, event: WorkspaceEvent) -> None:
    description = event.data.get("description", "")
    if description:
        state.recent_decisions.append(description)
        if len(state.recent_decisions) > MAX_RECENT_DECISIONS:
            state.recent_decisions = state.recent_decisions[-MAX_RECENT_DECISIONS:]
