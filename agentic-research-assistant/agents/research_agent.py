"""Retrieves evidence from the vector store (and optionally web) for each sub-question."""

from rag.retriever import retrieve
from tools.web_search import web_search
from app.config import cfg


def research(sub_questions: list[str]) -> dict[str, list[dict]]:
    evidence_map: dict[str, list[dict]] = {}

    for q in sub_questions:
        hits = retrieve(q, top_k=cfg.TOP_K_RETRIEVAL)

        if cfg.WEB_SEARCH_ENABLED and len(hits) < 2:
            # Supplement with web results when doc coverage is thin
            web_hits = web_search(q)
            hits.extend(web_hits)

        evidence_map[q] = hits

    return evidence_map
