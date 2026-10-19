"""Artifact data models for meetings, emails, documents, slides."""
from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field

from .calendar import MeetingJoinSpec


class ActionItem(BaseModel):
    id: str
    description: str
    assignee_id: str
    due_date: date | None = None
    status: str = "open"  # "open", "in_progress", "completed"
    created_in_meeting_id: str | None = None
    completed_in_meeting_id: str | None = None


class MeetingArtifact(BaseModel):
    id: str
    type: str = "meeting"
    title: str = ""
    calendar_event_id: str | None = None
    meeting_type: str = ""
    start_time: datetime | None = None
    end_time: datetime | None = None
    attendee_ids: list[str] = Field(default_factory=list)
    realized_status: str = "completed"  # "completed", "skipped", "canceled"
    attendance_outcomes: dict[str, str] = Field(default_factory=dict)
    planned_attendance_changes: dict[str, str] = Field(default_factory=dict)
    join_spec: MeetingJoinSpec = Field(default_factory=MeetingJoinSpec)
    transcript_text: str = ""
    transcript_json: list[dict] = Field(default_factory=list)
    audio_path: str | None = None
    speaker_audio_paths: dict[str, str] = Field(default_factory=dict)
    timing_data: dict | None = None
    rttm_path: str | None = None
    summary: str = ""
    outline: str = ""
    action_items: list[dict] = Field(default_factory=list)
    decisions: list[dict] = Field(default_factory=list)
    project_updates: list[dict] = Field(default_factory=list)
    follow_ups: list[dict] = Field(default_factory=list)
    meeting_summary: str = ""
    meeting_notes: str = ""
    slide_presentation_id: str | None = None
    realism_rewrite_accepted: bool = True
    realism_rewrite_fallback_reason: str | None = None
    realism_refine_iterations: int = 0
    realism_refine_early_stopped: bool = False
    realism_refine_judge_reasoning: list[str] = Field(default_factory=list)


class EmailMessage(BaseModel):
    id: str
    thread_id: str
    sender_id: str
    recipient_ids: list[str]
    cc_ids: list[str] = []
    subject: str
    body: str
    timestamp: datetime
    reply_to_id: str | None = None
    attachment_ids: list[str] = []


class EmailThread(BaseModel):
    id: str
    type: str = "email_thread"
    subject: str
    messages: list[EmailMessage] = Field(default_factory=list)
    participant_ids: list[str] = Field(default_factory=list)


class Document(BaseModel):
    id: str
    type: str = "document"
    title: str
    content: str = ""
    author_id: str
    created_at: datetime | None = None
    last_modified: datetime | None = None
    doc_type: str = "notes"  # "spec", "proposal", "notes", "report", "wiki"
    related_project: str | None = None
    shared_with_ids: list[str] = Field(default_factory=list)


class Slide(BaseModel):
    slide_number: int
    title: str
    content: str = ""
    subtitle: str = ""
    left_content: str = ""
    right_content: str = ""
    chart_description: str = ""
    speaker_notes: str = ""
    layout: str = "title_content"


class SlideTheme(BaseModel):
    primary_color: str = "#1a73e8"
    secondary_color: str = "#ffffff"
    accent_color: str = "#34a853"
    background_color: str = "#ffffff"
    text_color: str = "#202124"
    font_family: str = "Arial"
    company_name: str = ""


class SlidePresentation(BaseModel):
    id: str
    type: str = "slides"
    title: str
    author_id: str
    created_at: datetime | None = None
    slides: list[Slide] = Field(default_factory=list)
    output_path: str | None = None
    output_format: str = "png"
    theme: SlideTheme = Field(default_factory=SlideTheme)
