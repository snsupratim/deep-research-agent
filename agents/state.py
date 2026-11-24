from typing import Any, TypedDict
from models import ConflictFlag, ReportSection, ReportStatus, SourceDoc

class AgentState(TypedDict):
    query: str
    max_sources: int
    sub_queries: list[str]
    sources: list[SourceDoc]
    raw_citations: list[dict[str, Any]]
    conflict_flags: list[ConflictFlag]
    sections: list[ReportSection]
    summary: str
    status: ReportStatus
    error: str | None
    report_id: str | None
    log_callback: Any
