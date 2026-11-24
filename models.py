"""
Shared Pydantic models for the Research Agent.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────────────────────

class ReportStatus(str, Enum):
    PENDING = "pending"
    RESEARCHING = "researching"
    CRITIQUING = "critiquing"
    WRITING = "writing"
    DONE = "done"
    FAILED = "failed"


class SourceType(str, Enum):
    WEB = "web"
    ARXIV = "arxiv"
    PDF = "pdf"


# ── Source ───────────────────────────────────────────────────────────────────

class SourceDoc(BaseModel):
    url: str
    title: str
    source_type: SourceType
    snippet: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    fetched_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Citation ─────────────────────────────────────────────────────────────────

class Citation(BaseModel):
    id: str                  # e.g. "[1]"
    title: str
    url: str
    source_type: SourceType
    confidence_score: float


# ── Conflict flag ─────────────────────────────────────────────────────────────

class ConflictFlag(BaseModel):
    claim: str
    source_a: str
    source_b: str
    description: str


# ── Report section ───────────────────────────────────────────────────────────

class ReportSection(BaseModel):
    heading: str
    content: str
    citations: list[str] = Field(default_factory=list)  # citation IDs


# ── Full report ──────────────────────────────────────────────────────────────

class ReportDocument(BaseModel):
    id: str | None = None
    query: str
    status: ReportStatus = ReportStatus.PENDING
    summary: str = ""
    sections: list[ReportSection] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    conflict_flags: list[ConflictFlag] = Field(default_factory=list)
    sources_searched: int = 0
    word_count: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    error: str | None = None


# ── API request/response ─────────────────────────────────────────────────────

class ResearchRequest(BaseModel):
    query: str = Field(min_length=10, max_length=500)
    max_sources: int = Field(default=15, ge=5, le=30)


class ResearchResponse(BaseModel):
    report_id: str
    status: ReportStatus
    message: str
