"""Builds a citation list from evidence maps."""

from urllib.parse import urlparse


def _is_url(source: str) -> bool:
    try:
        return urlparse(source).scheme in ("http", "https")
    except Exception:
        return False


def build_citations(evidence_map: dict[str, list[dict]]) -> list[dict]:
    """
    Deduplicate sources across all sub-questions and return a sorted citation list.
    Each entry: {id, source, type, display}
    """
    seen: dict[str, dict] = {}

    for ev_list in evidence_map.values():
        for e in ev_list:
            src = e.get("source", "unknown")
            if src not in seen:
                entry = {
                    "id": len(seen) + 1,
                    "source": src,
                    "type": "web" if _is_url(src) else "document",
                    "display": src,
                }
                seen[src] = entry

    return list(seen.values())


def format_references(citations: list[dict]) -> str:
    """Format citations as a Markdown references section."""
    if not citations:
        return ""
    lines = ["## References\n"]
    for c in citations:
        prefix = "🌐" if c["type"] == "web" else "📄"
        lines.append(f"{c['id']}. {prefix} {c['display']}")
    return "\n".join(lines)
