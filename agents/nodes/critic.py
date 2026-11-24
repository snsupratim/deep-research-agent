import json
import re
from agents.state import AgentState
from agents.llm import _llm
from models import ConflictFlag, ReportStatus
from tools import cross_reference_claims

def _extract_json_dict(text: str) -> dict:
    clean = text.strip()
    if "```" in clean:
        clean = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.MULTILINE)
        clean = re.sub(r"\s*```$", "", clean, flags=re.MULTILINE)
        clean = clean.strip()
    try:
        res = json.loads(clean)
        if isinstance(res, dict):
            return res
    except Exception:
        pass
    s_idx = clean.find("{")
    e_idx = clean.rfind("}")
    if s_idx != -1 and e_idx != -1 and e_idx > s_idx:
        try:
            res = json.loads(clean[s_idx : e_idx + 1])
            if isinstance(res, dict):
                return res
        except Exception:
            pass
    return {}

async def critic_node(state: AgentState) -> AgentState:
    """Validate claims, detect conflicts, assign final confidence scores."""
    state["status"] = ReportStatus.CRITIQUING
    
    msg = f"⚖️ Fact-checking {len(state['sources'])} sources for contradictions..."
    if state.get("log_callback"):
        await state["log_callback"]({"message": msg})

    source_summaries = "\n".join(
        f"[{i+1}] ({s.source_type}) {s.title}: {s.snippet[:300]}"
        for i, s in enumerate(state["sources"][:12])
    )

    system = (
    "You are a ruthless research critic. Your job is to find DISAGREEMENTS, "
    "CONTRADICTIONS, and CONFLICTING CLAIMS between sources. Be aggressive — "
    "even subtle differences in numbers, timelines, conclusions, or emphasis count as conflicts.\n\n"
    "Look specifically for:\n"
    "- Different performance numbers or benchmarks for the same model/product\n"
    "- One source saying X is effective, another saying it is not\n"
    "- Disagreements on timelines ('will happen in 2025' vs 'already here')\n"
    "- One source praising something another source criticises\n"
    "- Different statistics cited for the same phenomenon\n"
    "- Optimistic vs pessimistic conclusions on the same topic\n\n"
    "You MUST find at least 2-3 conflicts. If sources don't directly contradict, "
    "find the TENSION — where they emphasise opposite things or lead to opposite conclusions.\n\n"
    "Output ONLY valid JSON, no markdown:\n"
    '{"key_claims": ["claim1", "claim2", "claim3"], '
    '"conflicts": ['
    '{"claim": "specific contested claim", '
    '"source_a": "exact source title", '
    '"source_b": "exact source title", '
    '"description": "Source A says X while Source B says Y — explain the tension clearly"}'
    ']}'
    )
    raw = await _llm(
        system,
        f"Research topic: {state['query']}\n\nSources:\n{source_summaries}",
        max_tokens=800,
    )

    parsed = _extract_json_dict(raw)
    key_claims = parsed.get("key_claims", [])
    raw_conflicts = parsed.get("conflicts", [])

    # Cross-reference claims against sources
    cross_ref = cross_reference_claims(key_claims, state["sources"])

    # Build conflict flags
    flags: list[ConflictFlag] = []
    for c in raw_conflicts:
        flags.append(
            ConflictFlag(
                claim=c.get("claim", ""),
                source_a=c.get("source_a", ""),
                source_b=c.get("source_b", ""),
                description=c.get("description", ""),
            )
        )

    state["conflict_flags"] = flags
    return state
