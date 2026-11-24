import json
import re
from models import ReportStatus
from agents.state import AgentState
from agents.llm import _llm

async def planner_node(state: AgentState) -> AgentState:
    """Break the query into targeted sub-queries for web + arxiv."""
    state["status"] = ReportStatus.RESEARCHING
    
    msg = f"📝 Preparing research plan for: {state['query']}"
    print(msg)
    if state.get("log_callback"):
        await state["log_callback"]({"message": msg})

    system = (
        "You are a research planner. Given a research question, output ONLY a "
        "JSON array of 4-6 specific sub-queries that together cover the topic "
        "comprehensively. Mix broad overview queries with specific technical ones. "
        "For each include both a web-search variant and an academic variant."
        "\n\nOutput format (JSON array of strings only, no markdown):"
        '\n["query 1", "query 2", ...]'
    )
    try:
        raw = await _llm(system, f"Research question: {state['query']}", max_tokens=400)
    except Exception as e:
        print(f"⚠️ Planner LLM error ({e}), applying fallback sub-queries.")
        raw = ""

    clean = raw.strip()
    if "```" in clean:
        clean = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.MULTILINE)
        clean = re.sub(r"\s*```$", "", clean, flags=re.MULTILINE)
        clean = clean.strip()

    try:
        sub_queries = json.loads(clean)
        if not isinstance(sub_queries, list):
            raise ValueError("Not a list")
    except Exception:
        # Fallback
        s_i = clean.find("[")
        e_i = clean.rfind("]")
        if s_i != -1 and e_i != -1 and e_i > s_i:
            try:
                sub_queries = json.loads(clean[s_i : e_i + 1])
            except Exception:
                sub_queries = [state["query"], state["query"] + " research", state["query"] + " overview"]
        else:
            sub_queries = [state["query"], state["query"] + " research", state["query"] + " overview"]

    state["sub_queries"] = sub_queries[:6]
    print(f"📋 Planner produced {len(state['sub_queries'])} sub-queries: {state['sub_queries']}")
    return state
