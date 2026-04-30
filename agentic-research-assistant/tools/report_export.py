"""Export research reports to Markdown (and optionally PDF)."""

import os
from datetime import datetime

from app.config import cfg


def _safe_filename(question: str) -> str:
    slug = "".join(c if c.isalnum() or c in " -_" else "" for c in question)
    slug = slug.strip().replace(" ", "_")[:60]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"report_{slug}_{ts}"


def export_markdown(report_md: str, question: str) -> str:
    """Write report to ./reports/ and return the file path."""
    os.makedirs(cfg.REPORTS_DIR, exist_ok=True)
    fname = _safe_filename(question) + ".md"
    fpath = os.path.join(cfg.REPORTS_DIR, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(report_md)
    return fpath


def export_pdf(report_md: str, question: str) -> str:
    """Convert Markdown to PDF via weasyprint. Returns file path."""
    try:
        import markdown as md_lib
        from weasyprint import HTML

        html_body = md_lib.markdown(report_md, extensions=["tables", "fenced_code"])
        html = f"""<!DOCTYPE html><html><head>
        <meta charset="utf-8">
        <style>
          body {{ font-family: Georgia, serif; max-width: 800px; margin: 40px auto; line-height: 1.6; }}
          h1, h2, h3 {{ color: #2c3e50; }}
          code {{ background: #f4f4f4; padding: 2px 4px; border-radius: 3px; }}
        </style></head><body>{html_body}</body></html>"""

        os.makedirs(cfg.REPORTS_DIR, exist_ok=True)
        fname = _safe_filename(question) + ".pdf"
        fpath = os.path.join(cfg.REPORTS_DIR, fname)
        HTML(string=html).write_pdf(fpath)
        return fpath
    except ImportError:
        raise RuntimeError("Install weasyprint and markdown to export PDF.")
