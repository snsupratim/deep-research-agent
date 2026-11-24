from agents.state import AgentState
from models import SourceDoc
from tools import arxiv_search, build_citations, web_search

async def researcher_node(state: AgentState) -> AgentState:
    """Run web + arxiv searches for every sub-query."""
    all_sources: list[SourceDoc] = []
    seen_urls: set[str] = set()

    max_per_query = max(2, state["max_sources"] // len(state["sub_queries"]))
    
    msg = f"🔍 Researcher starting. Processing {len(state['sub_queries'])} sub-queries..."
    print(msg)
    if state.get("log_callback"):
        await state["log_callback"]({"message": msg})

    for sq in state["sub_queries"]:
        msg = f"  → Web search for: {sq}"
        print(msg)
        if state.get("log_callback"):
            await state["log_callback"]({"message": msg})
            
        web_docs = await web_search(sq, max_results=max_per_query)
        for doc in web_docs:
            if doc.url not in seen_urls:
                seen_urls.add(doc.url)
                all_sources.append(doc)

        # ArXiv (academic) — run for first half of sub-queries
        idx = state["sub_queries"].index(sq)
        if idx < 3:
            msg = f"  → ArXiv search for: {sq} (Will take ~5 seconds due to strict rate limits)"
            print(msg)
            if state.get("log_callback"):
                await state["log_callback"]({"message": msg})
            arxiv_docs = await arxiv_search(sq, max_results=3)
            for doc in arxiv_docs:
                if doc.url not in seen_urls:
                    seen_urls.add(doc.url)
                    all_sources.append(doc)

    # Trim to max_sources, prioritising high confidence
    all_sources.sort(key=lambda s: s.confidence_score, reverse=True)
    state["sources"] = all_sources[: state["max_sources"]]
    state["raw_citations"] = build_citations(state["sources"])
    
    msg = f"✅ Researcher finished. Found {len(state['sources'])} high-confidence sources."
    print(msg)
    if state.get("log_callback"):
        await state["log_callback"]({"message": msg})
        
    return state
