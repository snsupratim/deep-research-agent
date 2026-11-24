import asyncio
from datetime import datetime
from typing import Any

from bson import ObjectId
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

from agents.graph import run_research_agent
from db import get_db
from models import (
    ReportDocument,
    ReportStatus,
    ResearchRequest,
    ResearchResponse,
    Citation,
    ConflictFlag,
    ReportSection,
)
from routers.websocket import manager

router = APIRouter(tags=["reports"])
limiter = Limiter(key_func=get_remote_address, default_limits=["100/hour"])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _oid(report_id: str) -> ObjectId:
    try:
        return ObjectId(report_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid report ID format")


def _serialize(doc: dict) -> dict:
    """Convert MongoDB ObjectId to string for JSON responses."""
    doc["id"] = str(doc.pop("_id", ""))
    return doc


# ── Background task ───────────────────────────────────────────────────────────

async def _run_and_save(report_id: str, query: str, max_sources: int):
    db = get_db()
    try:
        # Update status to researching
        await db.reports.update_one(
            {"_id": ObjectId(report_id)},
            {"$set": {"status": ReportStatus.RESEARCHING}},
        )
        await manager.broadcast(report_id, {"status": ReportStatus.RESEARCHING.value})

        async def on_status_update(data: dict):
            # Broadcast the updated status or message to connected WebSockets
            await manager.broadcast(report_id, data)

        final_state = await run_research_agent(
            query, 
            max_sources, 
            report_id=report_id, 
            update_callback=on_status_update
        )

        # Compute word count
        all_text = final_state.get("summary", "") + " ".join(
            s.content for s in final_state.get("sections", [])
        )
        word_count = len(all_text.split())

        update_doc: dict[str, Any] = {
            "status": final_state["status"],
            "summary": final_state.get("summary", ""),
            "sections": [s.model_dump() for s in final_state.get("sections", [])],
            "citations": final_state.get("raw_citations", []),
            "conflict_flags": [
                f.model_dump() for f in final_state.get("conflict_flags", [])
            ],
            "sources_searched": len(final_state.get("sources", [])),
            "word_count": word_count,
            "completed_at": datetime.utcnow(),
        }

        if final_state["status"] == ReportStatus.FAILED:
            update_doc["error"] = final_state.get("error")

        await db.reports.update_one(
            {"_id": ObjectId(report_id)},
            {"$set": update_doc},
        )

    except Exception as e:
        await db.reports.update_one(
            {"_id": ObjectId(report_id)},
            {
                "$set": {
                    "status": ReportStatus.FAILED,
                    "error": str(e),
                    "completed_at": datetime.utcnow(),
                }
            },
        )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/reports", response_model=ResearchResponse, status_code=202)
@limiter.limit("10/minute")
async def create_report(
    request: Request,
    body: ResearchRequest,
    background_tasks: BackgroundTasks,
):
    """Trigger a new research agent run. Returns immediately with a report_id."""
    db = get_db()
    doc = {
        "query": body.query,
        "max_sources": body.max_sources,
        "status": ReportStatus.PENDING,
        "summary": "",
        "sections": [],
        "citations": [],
        "conflict_flags": [],
        "sources_searched": 0,
        "word_count": 0,
        "created_at": datetime.utcnow(),
        "completed_at": None,
        "error": None,
    }
    result = await db.reports.insert_one(doc)
    report_id = str(result.inserted_id)

    background_tasks.add_task(
        _run_and_save, report_id, body.query, body.max_sources
    )

    return ResearchResponse(
        report_id=report_id,
        status=ReportStatus.PENDING,
        message="Research agent started. Poll /api/v1/reports/{id} for status.",
    )


@router.get("/reports/{report_id}")
@limiter.limit("60/minute")
async def get_report(request: Request, report_id: str):
    """Fetch a single report by ID."""
    db = get_db()
    doc = await db.reports.find_one({"_id": _oid(report_id)})
    if not doc:
        raise HTTPException(status_code=404, detail="Report not found")
    return _serialize(doc)


@router.get("/reports")
@limiter.limit("30/minute")
async def list_reports(request: Request, limit: int = 20, skip: int = 0):
    """List recent reports (newest first)."""
    db = get_db()
    cursor = (
        db.reports.find({}, {"summary": 0, "sections": 0})
        .sort("created_at", -1)
        .skip(skip)
        .limit(min(limit, 50))
    )
    docs = [_serialize(d) async for d in cursor]
    return {"reports": docs, "count": len(docs)}


@router.delete("/reports/{report_id}", status_code=204)
@limiter.limit("10/minute")
async def delete_report(request: Request, report_id: str):
    """Delete a report."""
    db = get_db()
    result = await db.reports.delete_one({"_id": _oid(report_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Report not found")























