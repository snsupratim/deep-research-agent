import json
import re
from agents.state import AgentState
from agents.llm import _llm
from models import ReportSection, ReportStatus

def _extract_json_block(text: str) -> dict | None:
    """Robustly extract and parse a JSON dictionary from LLM output."""
    clean = text.strip()
    # Strip markdown code blocks
    if "```" in clean:
        clean = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.MULTILINE)
        clean = re.sub(r"\s*```$", "", clean, flags=re.MULTILINE)
        clean = clean.strip()
    
    # Try direct parse
    try:
        res = json.loads(clean)
        if isinstance(res, dict):
            return res
    except Exception:
        pass

    # Extract between outer { and }
    s_idx = clean.find("{")
    e_idx = clean.rfind("}")
    if s_idx != -1 and e_idx != -1 and e_idx > s_idx:
        try:
            res = json.loads(clean[s_idx : e_idx + 1])
            if isinstance(res, dict):
                return res
        except Exception:
            pass

    return None

async def writer_node(state: AgentState) -> AgentState:
    """Synthesise all research into a structured report."""
    state["status"] = ReportStatus.WRITING
    
    msg = f"✍️ Synthesising final report and evaluating {len(state['conflict_flags'])} conflicts..."
    if state.get("log_callback"):
        await state["log_callback"]({"message": msg})

    source_context = "\n\n".join(
        f"SOURCE {c['id']} — {c['title']} ({c['source_type']}, "
        f"confidence: {c['confidence_score']:.2f})\nURL: {c['url']}\n"
        + next(
            (s.snippet for s in state["sources"] if s.url == c["url"]),
            "",
        )
        for c in state["raw_citations"]
    )

    conflict_context = (
        "\n".join(
            f"⚠ CONFLICT: {f.claim} — {f.source_a} vs {f.source_b}: {f.description}"
            for f in state["conflict_flags"]
        )
        or "No major conflicts detected."
    )

    system = (
        "You are an expert research writer. Write a comprehensive, well-structured "
        "intelligence report. Use inline citations like [1], [2] referencing the "
        "source IDs provided. Be analytical, not just descriptive. "
        "Flag conflicts inline where relevant.\n\n"
        "Output ONLY valid JSON with this structure:\n"
        '{"summary": "2-3 sentence executive summary", '
        '"sections": ['
        '{"heading": "Section Title", "content": "detailed content with [1] citations", '
        '"citations": ["[1]", "[2]"]}]}'
        "\n\nAim for 4-6 sections, ~800-1200 words total content."
    )

    user = (
        f"Research question: {state['query']}\n\n"
        f"SOURCES:\n{source_context[:6000]}\n\n"
        f"CONFLICT FLAGS:\n{conflict_context}"
    )

    raw = await _llm(system, user, max_tokens=3000)

    parsed = _extract_json_block(raw)

    if parsed and "sections" in parsed:
        state["summary"] = parsed.get("summary", "")
        sections_raw = parsed.get("sections", [])
        state["sections"] = [
            ReportSection(
                heading=s.get("heading", ""),
                content=s.get("content", ""),
                citations=s.get("citations", []),
            )
            for s in sections_raw
        ]
    else:
        # Fallback: treat raw text as single section
        state["summary"] = "Research completed."
        state["sections"] = [
            ReportSection(heading="Research Findings", content=raw, citations=[])
        ]

    state["status"] = ReportStatus.DONE
    return state
