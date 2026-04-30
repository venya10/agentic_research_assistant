"""Chroma vector store — add documents, query by similarity."""

import os
import uuid
import chromadb
from chromadb.config import Settings

from app.config import cfg
from rag.chunking import Chunk
from rag.embeddings import embed_texts, embed_query


def _get_client() -> chromadb.ClientAPI:
    os.makedirs(cfg.CHROMA_PERSIST_DIR, exist_ok=True)
    return chromadb.PersistentClient(
        path=cfg.CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False),
    )


def _get_collection(client: chromadb.ClientAPI):
    return client.get_or_create_collection(
        name=cfg.CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks(chunks: list[Chunk]) -> int:
    """Embed and store chunks. Returns number added."""
    if not chunks:
        return 0

    client = _get_client()
    col = _get_collection(client)

    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)
    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [
        {"source": c.source, "chunk_index": c.chunk_index, **c.metadata}
        for c in chunks
    ]

    col.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    return len(chunks)


def query_similar(query: str, top_k: int | None = None) -> list[dict]:
    """Return top-k similar chunks as dicts with text + metadata."""
    k = top_k or cfg.TOP_K_RETRIEVAL
    client = _get_client()
    col = _get_collection(client)

    if col.count() == 0:
        return []

    q_emb = embed_query(query)
    results = col.query(
        query_embeddings=[q_emb],
        n_results=min(k, col.count()),
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "text": doc,
            "source": meta.get("source", "unknown"),
            "chunk_index": meta.get("chunk_index", -1),
            "score": round(1 - dist, 4),   # cosine similarity
            "metadata": meta,
        })
    return hits


def clear_collection() -> None:
    client = _get_client()
    client.delete_collection(cfg.CHROMA_COLLECTION)


def collection_count() -> int:
    client = _get_client()
    col = _get_collection(client)
    return col.count()
