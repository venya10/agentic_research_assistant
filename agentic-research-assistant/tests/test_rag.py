"""
Unit tests for the RAG pipeline.
Run: pytest tests/test_rag.py -v

These tests use real local embeddings (no API key needed).
They DO write to a temporary Chroma collection.
"""

import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.chunking import split_text, Chunk
from rag.embeddings import embed_texts, embed_query


# ── Chunking ──────────────────────────────────────────────────────────────────

def test_split_text_basic():
    text = "word " * 200  # ~1000 chars
    chunks = split_text(text, source="test.txt")
    assert len(chunks) > 1
    for c in chunks:
        assert isinstance(c, Chunk)
        assert c.source == "test.txt"
        assert len(c.text) > 0


def test_split_text_short():
    text = "Short text."
    chunks = split_text(text, source="short.txt")
    assert len(chunks) == 1
    assert chunks[0].text == "Short text."


def test_split_text_empty():
    chunks = split_text("", source="empty.txt")
    assert len(chunks) == 0


# ── Embeddings ────────────────────────────────────────────────────────────────

def test_embed_texts_shape():
    texts = ["hello world", "another sentence"]
    embeddings = embed_texts(texts)
    assert len(embeddings) == 2
    assert isinstance(embeddings[0], list)
    assert len(embeddings[0]) > 0   # embedding dimension > 0


def test_embed_query_type():
    emb = embed_query("What is climate change?")
    assert isinstance(emb, list)
    assert len(emb) > 0


def test_embeddings_different_for_different_texts():
    e1 = embed_query("climate change")
    e2 = embed_query("neural networks")
    # They should not be identical
    assert e1 != e2
