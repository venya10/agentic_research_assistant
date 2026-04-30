"""Local sentence-transformers embedding wrapper."""

from functools import lru_cache
from app.config import cfg


@lru_cache(maxsize=1)
def get_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(cfg.EMBEDDING_MODEL)


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()
    return model.encode(texts, show_progress_bar=False).tolist()


def embed_query(query: str) -> list[float]:
    return embed_texts([query])[0]
