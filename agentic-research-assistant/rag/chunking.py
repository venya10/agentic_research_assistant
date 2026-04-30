"""Split raw text into overlapping chunks for embedding."""

from dataclasses import dataclass
from app.config import cfg


@dataclass
class Chunk:
    text: str
    source: str          # filename or URL
    chunk_index: int
    metadata: dict       # extra fields (page, section, …)


def split_text(text: str, source: str, metadata: dict | None = None) -> list[Chunk]:
    """Sliding-window character splitter."""
    size = cfg.CHUNK_SIZE
    overlap = cfg.CHUNK_OVERLAP
    meta = metadata or {}
    chunks: list[Chunk] = []

    start = 0
    idx = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(Chunk(
                text=chunk_text,
                source=source,
                chunk_index=idx,
                metadata={**meta, "start_char": start, "end_char": end},
            ))
            idx += 1
        start += size - overlap

    return chunks
