"""
Web search tool.
Uses Tavily when WEB_SEARCH_ENABLED=true, otherwise returns mock results.
"""

from app.config import cfg


def _mock_search(query: str) -> list[dict]:
    return [{
        "text": f"[Mock web result for: '{query}'] Web search is disabled. Set WEB_SEARCH_ENABLED=true and add a TAVILY_API_KEY to enable live search.",
        "source": "mock://web-search",
        "score": 0.0,
        "metadata": {"type": "web_mock"},
    }]


def _tavily_search(query: str) -> list[dict]:
    from tavily import TavilyClient
    client = TavilyClient(api_key=cfg.TAVILY_API_KEY)
    results = client.search(query=query, max_results=5)
    hits = []
    for r in results.get("results", []):
        hits.append({
            "text": r.get("content", ""),
            "source": r.get("url", "web"),
            "score": r.get("score", 0.5),
            "metadata": {"type": "web", "title": r.get("title", "")},
        })
    return hits


def web_search(query: str) -> list[dict]:
    if cfg.WEB_SEARCH_ENABLED and cfg.TAVILY_API_KEY:
        try:
            return _tavily_search(query)
        except Exception as e:
            return [{"text": f"Web search error: {e}", "source": "web-error", "score": 0.0, "metadata": {}}]
    return _mock_search(query)
