"""High-level retriever used by agents."""

from rag.vector_store import query_similar
from app.config import cfg


def retrieve(query: str, top_k: int | None = None) -> list[dict]:
    """Retrieve relevant chunks for a query. Returns list of evidence dicts."""
    return query_similar(query, top_k=top_k or cfg.TOP_K_RETRIEVAL)
