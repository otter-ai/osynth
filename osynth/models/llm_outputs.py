"""Pydantic models for structured LLM response payloads."""
from __future__ import annotations

from pydantic import BaseModel, Field

from .artifacts import Slide


class MeetingTranscriptTurn(BaseModel):
    speaker_id: str
    text: str


class MeetingTranscriptActionItem(BaseModel):
    description: str
    assignee_id: str | None = None
    due_description: str | None = None


class MeetingTranscriptDecision(BaseModel):
    description: str
    participant_ids: list[str] = Field(default_factory=list)


class MeetingTranscriptResponse(BaseModel):
    turns: list[MeetingTranscriptTurn] = Field(default_factory=list)
    summary: str = ""
    outline: str = ""
    action_items: list[MeetingTranscriptActionItem] = Field(default_factory=list)
    decisions: list[MeetingTranscriptDecision] = Field(default_factory=list)


class ExtractedActionItem(BaseModel):
    description: str
    assignee_id: str | None = None
    due_description: str | None = None


class MeetingDecision(BaseModel):
    description: str
    participant_ids: list[str] = Field(default_factory=list)


class MeetingProjectUpdate(BaseModel):
    project_id: str | None = None
    project_name: str | None = None
    status_update: str | None = None


class MeetingFollowUp(BaseModel):
    description: str
    responsible_person_id: str | None = None
    type: str | None = None


class MeetingEventExtractionResponse(BaseModel):
    action_items: list[ExtractedActionItem] = Field(default_factory=list)
    decisions: list[MeetingDecision] = Field(default_factory=list)
    project_updates: list[MeetingProjectUpdate] = Field(default_factory=list)
    follow_ups: list[MeetingFollowUp] = Field(default_factory=list)
    meeting_summary: str = ""


class EmailMessageResponse(BaseModel):
    id: str
    thread_id: str
    sender_id: str
    recipient_ids: list[str] = Field(default_factory=list)
    cc_ids: list[str] = Field(default_factory=list)
    subject: str
    body: str
    reply_to_id: str | None = None


class DocumentResponse(BaseModel):
    id: str
    title: str
    content: str = ""
    author_id: str
    doc_type: str = "notes"
    related_project: str | None = None
    shared_with_ids: list[str] = Field(default_factory=list)


class SlidePresentationResponse(BaseModel):
    id: str
    title: str
    author_id: str
    slides: list[Slide] = Field(default_factory=list)


class JargonSurfaceTerm(BaseModel):
    surface: str
    occurrences: int = 1


class JargonConceptCluster(BaseModel):
    canonical: str
    variants: list[str] = Field(default_factory=list)
    category: str = "industry_term"


class JargonExtractionResponse(BaseModel):
    surface_terms: list[JargonSurfaceTerm] = Field(default_factory=list)
    concept_clusters: list[JargonConceptCluster] = Field(default_factory=list)
