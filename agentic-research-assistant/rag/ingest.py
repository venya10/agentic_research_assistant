"""Document ingestion pipeline — supports PDF, TXT, and Markdown."""

import io
from pathlib import Path

from rag.chunking import split_text
from rag.vector_store import add_chunks


def _extract_text_pdf(file_bytes: bytes, filename: str) -> str:
    import pypdf
    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append(f"[Page {i+1}]\n{text}")
    return "\n\n".join(pages)


def _extract_text_plain(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="replace")


def ingest_file(file_bytes: bytes, filename: str) -> int:
    """Ingest a single file. Returns number of chunks stored."""
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        text = _extract_text_pdf(file_bytes, filename)
    elif ext in (".txt", ".md", ".markdown"):
        text = _extract_text_plain(file_bytes)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    if not text.strip():
        raise ValueError(f"No extractable text found in {filename}")

    chunks = split_text(text, source=filename)
    return add_chunks(chunks)


def ingest_directory(directory: str) -> dict[str, int]:
    """Ingest all supported files in a directory. Returns {filename: chunk_count}."""
    results = {}
    for fpath in Path(directory).rglob("*"):
        if fpath.suffix.lower() in (".pdf", ".txt", ".md", ".markdown"):
            with open(fpath, "rb") as f:
                data = f.read()
            try:
                count = ingest_file(data, fpath.name)
                results[fpath.name] = count
            except Exception as e:
                results[fpath.name] = f"ERROR: {e}"
    return results
