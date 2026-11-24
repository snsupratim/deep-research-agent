"""
Research tools used by the LangGraph agent nodes.

All external calls are rate-limited via tenacity retry + explicit
per-tool rate tracking. Each tool returns a list[SourceDoc].
"""
from __future__ import annotations

import asyncio
import os
import re
import time
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

import aiohttp
import arxiv
import httpx
from pypdf import PdfReader
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from models import SourceDoc, SourceType

# ── Simple in-process rate-limit tokens ──────────────────────────────────────
_tool_last_call: dict[str, float] = {}
_TOOL_MIN_INTERVAL = float(os.getenv("TOOL_MIN_INTERVAL_SECONDS", "1.0"))
_ARXIV_MIN_INTERVAL = 5.0   # ArXiv is strict, give it 5 seconds between calls


async def _throttle(tool_name: str):
    last = _tool_last_call.get(tool_name, 0)
    wait = _TOOL_MIN_INTERVAL - (time.monotonic() - last)
    if wait > 0:
        await asyncio.sleep(wait)
    _tool_last_call[tool_name] = time.monotonic()


# ── Confidence heuristics ─────────────────────────────────────────────────────

def _domain_confidence(url: str) -> float:
    """Simple heuristic: academic / gov / established outlets score higher."""
    high = [".edu", ".gov", "arxiv.org", "nature.com", "science.org",
            "pubmed", "acm.org", "ieee.org", "nejm.org", "thelancet.com"]
    medium = [".org", "wikipedia.org", "bbc.", "reuters.", "nytimes.",
              "theguardian.", "wired.", "techcrunch."]
    domain = urlparse(url).netloc.lower()
    for h in high:
        if h in domain:
            return 0.90
    for m in medium:
        if m in domain:
            return 0.75
    return 0.55


# ── Tool 1 — Tavily Web Search ────────────────────────────────────────────────

@retry(
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPError)),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(3),
)
async def web_search(query: str, max_results: int = 8) -> list[SourceDoc]:
    """Search the web via Tavily API with rate limiting and retries."""
    await _throttle("tavily")
    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        print("⚠️  TAVILY_API_KEY not set — skipping web search")
        return []

    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "advanced",
                "include_answer": False,
                "include_raw_content": False,
            },
        )
        resp.raise_for_status()
        data = resp.json()

    docs: list[SourceDoc] = []
    for r in data.get("results", []):
        url = r.get("url", "")
        docs.append(
            SourceDoc(
                url=url,
                title=r.get("title", "Untitled"),
                source_type=SourceType.WEB,
                snippet=r.get("content", "")[:800],
                confidence_score=_domain_confidence(url),
                metadata={"score": r.get("score", 0)},
            )
        )
    return docs


# ── Tool 2 — ArXiv Search ────────────────────────────────────────────────────

async def arxiv_search(query: str, max_results: int = 5) -> list[SourceDoc]:
    """Query ArXiv for academic papers with rate limiting and error handling."""
    last = _tool_last_call.get("arxiv", 0)
    wait = _ARXIV_MIN_INTERVAL - (time.monotonic() - last)
    if wait > 0:
        await asyncio.sleep(wait)
        _tool_last_call["arxiv"] = time.monotonic()
    
    # Strip any web-search operators that don't work on ArXiv
    clean_query = re.sub(r'site:\S+', '', query).strip()
    clean_query = re.sub(r'\s+', ' ', clean_query)
    
    loop = asyncio.get_event_loop()

    def _sync_search():
        try:
            client = arxiv.Client(
                page_size=max_results,
                delay_seconds=3,      # be polite to ArXiv
                num_retries=1,        # don't hammer on 429
            )
            search = arxiv.Search(
                query=clean_query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance,
            )
            return list(client.results(search))
        except Exception as e:
            print(f"⚠️  ArXiv search failed: {e}")
            return []   # return empty, don't crash the whole agent

    results = await loop.run_in_executor(None, _sync_search)

    docs: list[SourceDoc] = []
    for paper in results:
        docs.append(
            SourceDoc(
                url=paper.entry_id,
                title=paper.title,
                source_type=SourceType.ARXIV,
                snippet=paper.summary[:800],
                confidence_score=0.88,
                metadata={
                    "authors": [a.name for a in paper.authors[:5]],
                    "published": paper.published.isoformat() if paper.published else "",
                    "categories": paper.categories,
                },
            )
        )
    return docs
# ── Tool 3 — PDF Reader ──────────────────────────────────────────────────────

async def read_pdf_url(url: str, max_chars: int = 4000) -> str:
    """Download and extract text from a PDF URL."""
    await _throttle("pdf")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as r:
                if r.status != 200:
                    return ""
                content = await r.read()

        import io
        reader = PdfReader(io.BytesIO(content))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
            if len(text) >= max_chars:
                break
        return text[:max_chars]
    except Exception as e:
        print(f"⚠️  PDF read failed for {url}: {e}")
        return ""


# ── Tool 4 — Fact Checker ────────────────────────────────────────────────────

def cross_reference_claims(
    claims: list[str], sources: list[SourceDoc]
) -> dict[str, list[str]]:
    """
    Simple keyword-based cross-referencing.
    Returns {claim: [source_urls that mention it]}.
    """
    result: dict[str, list[str]] = {}
    for claim in claims:
        keywords = set(re.findall(r"\b\w{5,}\b", claim.lower()))
        supporting: list[str] = []
        for src in sources:
            src_text = (src.title + " " + src.snippet).lower()
            if sum(1 for kw in keywords if kw in src_text) >= 2:
                supporting.append(src.url)
        result[claim] = supporting
    return result


# ── Tool 5 — Citation Builder ─────────────────────────────────────────────────

def build_citations(sources: list[SourceDoc]) -> list[dict[str, Any]]:
    """Deduplicate sources and assign citation IDs."""
    seen: set[str] = set()
    citations = []
    idx = 1
    for src in sources:
        if src.url in seen:
            continue
        seen.add(src.url)
        citations.append(
            {
                "id": f"[{idx}]",
                "title": src.title,
                "url": src.url,
                "source_type": src.source_type,
                "confidence_score": src.confidence_score,
            }
        )
        idx += 1
    return citations
