"""Workspace and company data models."""
from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, Field


class CompanySize(str, Enum):
    STARTUP = "startup"
    SMALL_BUSINESS = "small_business"
    MIDMARKET = "midmarket"
    ENTERPRISE = "enterprise"


class Company(BaseModel):
    name: str
    domain: str
    industry: str
    size: CompanySize
    location: str
    description: str
    founding_year: int | None = None
    products: list[str] = []
    current_situation: str = ""


class WorkspaceConfig(BaseModel):
    company: Company
    time_period_start: date
    time_period_end: date
    max_speakers: int = 13
    seed: int | None = None
    working_hours: tuple[int, int] = (9, 17)
    working_days: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4])
    timezone: str = "America/Los_Angeles"
