"""People and organization data models."""
from __future__ import annotations

from enum import Enum
from typing import Literal, Self

from pydantic import BaseModel, field_validator, model_validator

from .calendar import WorkArrangement


class DisfluencyProfile(BaseModel):
    """Per-person disfluency rates and preferred filler words, used to inject
    realistic disfluencies during transcript generation and TTS preprocessing."""

    filler_rate: float = 0.0  # fillers per 100 words (typical range: 1-8)
    filler_words: list[str] = []  # e.g. ["um", "uh", "like", "you know", "I mean", "sort of", "kind of", "right"]
    repetition_rate: float = 0.0  # word/phrase repetitions per 100 words (e.g. "the the", "I I think")
    restart_rate: float = 0.0  # false starts / self-corrections per 100 words (e.g. "we should— we need to")
    truncation_rate: float = 0.0  # mid-word truncations per 100 content words (AMI baseline ~1.97)
    elongation_rate: float = 0.0  # vowel elongations per 100 words (e.g. "sooo", "wellll")
    hedge_words: list[str] = []  # e.g. ["maybe", "perhaps", "I think", "probably", "basically"]
    backchannel_frequency: str = ""  # "none", "low", "moderate", "high" — how often they produce backchannels while listening
    backchannel_phrases: list[str] = []  # e.g. ["mm-hmm", "yeah", "right", "got it", "okay"]


class SpeakingStyle(BaseModel):
    """Stable speaking characteristics that hold across all meetings for a person."""

    pace: str = ""  # "fast", "slow", "measured", "variable"
    verbosity: str = ""  # "terse", "concise", "average", "verbose"
    interruption_tendency: str = ""  # "none", "low", "moderate", "high"
    accent: str = ""  # free-form, e.g. "British English", "Mandarin-influenced"
    notable_patterns: list[str] = []  # e.g. ["uses sports metaphors", "trails off mid-sentence"]
    disfluency: DisfluencyProfile | None = None


class Role(BaseModel):
    title: str = ""
    department: str = ""
    level: str = ""  # "IC", "Manager", "Director", "VP", "C-Suite"
    responsibilities: list[str] = []

    @field_validator("title", "department", "level", mode="before")
    @classmethod
    def _coerce_none_str(cls, v: object) -> str:
        return v if isinstance(v, str) else ""


class NicknameUsage(str, Enum):
    NONE = "none"
    SOMETIMES = "sometimes"
    MOSTLY = "mostly"


class PersonBase(BaseModel):
    """Fields shared by internal and external people."""

    id: str
    name: str
    nickname: str = ""
    nickname_usage: NicknameUsage = NicknameUsage.NONE
    gender: str = ""  # "male", "female", "non-binary"
    voice_id: str | None = None
    email: str = ""
    personality_traits: list[str] = []
    expertise: list[str] = []
    speaking_style: SpeakingStyle | None = None
    work_arrangement: WorkArrangement = WorkArrangement.HYBRID

    @field_validator("nickname_usage", mode="before")
    @classmethod
    def _coerce_nickname_usage(cls, v: object) -> object:
        if isinstance(v, NicknameUsage):
            return v
        s = str(v).lower().strip() if v is not None else ""
        if s in ("none", "never", ""):
            return NicknameUsage.NONE
        if s in ("mostly", "always", "frequent", "frequently", "often", "usually"):
            return NicknameUsage.MOSTLY
        if s in ("sometimes", "occasionally", "occasional", "rarely", "seldom"):
            return NicknameUsage.SOMETIMES
        try:
            return NicknameUsage(s)
        except ValueError:
            return NicknameUsage.NONE

    @field_validator("work_arrangement", mode="before")
    @classmethod
    def _coerce_work_arrangement(cls, v: object) -> object:
        if v is None or v == "":
            return WorkArrangement.HYBRID
        if isinstance(v, WorkArrangement):
            return v
        try:
            return WorkArrangement(str(v).lower())
        except ValueError:
            return WorkArrangement.HYBRID

    @field_validator("id", "name", "nickname", "gender", "email", mode="before")
    @classmethod
    def _coerce_none_str(cls, v: object) -> str:
        return v if isinstance(v, str) else ""

    @model_validator(mode="after")
    def _clear_unused_nickname(self) -> Self:
        if self.nickname_usage == NicknameUsage.NONE:
            self.nickname = ""
        return self


class Person(PersonBase):
    """An employee of the simulated company."""

    role: Role = Role()
    team_id: str = ""

    @field_validator("team_id", mode="before")
    @classmethod
    def _coerce_none_str(cls, v: object) -> str:
        return v if isinstance(v, str) else ""


class ExternalPerson(PersonBase):
    """A person outside the company (prospect, customer, partner, etc.)."""

    organization: str = ""
    relationship_type: str = ""  # "prospect", "customer", "partner", etc.
    role_title: str = ""

    @field_validator("organization", "relationship_type", "role_title", mode="before")
    @classmethod
    def _coerce_none_str(cls, v: object) -> str:
        return v if isinstance(v, str) else ""


class Team(BaseModel):
    id: str
    name: str = ""
    department: str = ""
    purpose: str = ""
    manager_id: str = ""
    member_ids: list[str] = []
    parent_team_id: str | None = None

    @field_validator("name", "department", "purpose", "manager_id", mode="before")
    @classmethod
    def _coerce_none_str(cls, v: object) -> str:
        return v if isinstance(v, str) else ""


class OrgStructure(BaseModel):
    teams: dict[str, Team] = {}
    people: dict[str, Person] = {}
    external_people: dict[str, ExternalPerson] = {}
    reporting: dict[str, str] = {}  # person_id -> manager_person_id

    def get_person(self, person_id: str) -> Person | ExternalPerson | None:
        if person_id in self.people:
            return self.people[person_id]
        return self.external_people.get(person_id)

    def has_person(self, person_id: str) -> bool:
        return person_id in self.people or person_id in self.external_people

    def is_internal(self, person_id: str) -> bool:
        return person_id in self.people

    def participant_type(self, person_id: str) -> Literal["internal", "external"] | None:
        if person_id in self.people:
            return "internal"
        if person_id in self.external_people:
            return "external"
        return None

    def all_people(self) -> dict[str, Person | ExternalPerson]:
        return {**self.people, **self.external_people}
