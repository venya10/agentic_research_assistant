"""
Agentic Research Assistant — Streamlit UI
Run: streamlit run app/streamlit_app.py
"""

import sys
import os

# Make project root importable regardless of where streamlit is launched from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from app.config import cfg
from rag.ingest import ingest_file
from rag.vector_store import collection_count, clear_collection
from app.pipeline import run_pipeline
from tools.report_export import export_markdown


# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Agentic Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🔬 Research Assistant")
    st.caption(f"Model: `{cfg.LLM_PROVIDER} / {cfg.active_model()}`")

    st.divider()

    # Config validation
    errors = cfg.validate()
    if errors:
        for e in errors:
            st.error(f"⚠️ {e}")
        st.stop()

    # Document upload
    st.header("📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or Markdown files",
        type=["pdf", "txt", "md", "markdown"],
        accept_multiple_files=True,
        help="Documents are chunked, embedded, and stored in a local Chroma vector store.",
    )

    if uploaded_files:
        if st.button("Ingest Documents", type="primary", use_container_width=True):
            with st.spinner("Ingesting documents…"):
                results = {}
                for uf in uploaded_files:
                    try:
                        n = ingest_file(uf.read(), uf.name)
                        results[uf.name] = f"✅ {n} chunks"
                    except Exception as ex:
                        results[uf.name] = f"❌ {ex}"

            for fname, status in results.items():
                st.write(f"**{fname}**: {status}")

    doc_count = collection_count()
    st.info(f"Vector store: **{doc_count}** chunks indexed")

    if doc_count > 0:
        if st.button("🗑️ Clear vector store", use_container_width=True):
            clear_collection()
            st.success("Vector store cleared.")
            st.rerun()

    st.divider()
    st.header("⚙️ Settings")
    top_k = st.slider("Top-K retrieval", 1, 20, cfg.TOP_K_RETRIEVAL)
    max_sq = st.slider("Max sub-questions", 1, 10, cfg.MAX_SUBQUESTIONS)
    cfg.TOP_K_RETRIEVAL = top_k
    cfg.MAX_SUBQUESTIONS = max_sq


# ── Main area ─────────────────────────────────────────────────────────────────

st.title("Agentic Research Assistant")
st.caption("Ask a research question → AI decomposes, retrieves, verifies, and writes a cited report.")

# Research question input
question = st.text_area(
    "Research Question",
    placeholder="e.g. What are the main causes and economic impacts of climate change?",
    height=100,
)

run_clicked = st.button("🚀 Run Research", type="primary", disabled=not question.strip())

if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ── Run pipeline ──────────────────────────────────────────────────────────────

if run_clicked and question.strip():
    if doc_count == 0:
        st.warning("No documents are indexed yet. Upload and ingest at least one document first.")
    else:
        with st.spinner("Running multi-agent pipeline… this may take 30-60 seconds."):
            try:
                result = run_pipeline(question.strip())
                st.session_state.last_result = result
            except Exception as ex:
                st.error(f"Pipeline error: {ex}")
                st.stop()

# ── Display results ───────────────────────────────────────────────────────────

result = st.session_state.last_result

if result:
    tabs = st.tabs([
        "📋 Sub-Questions",
        "🔍 Evidence",
        "✅ Verification",
        "📄 Report",
        "🔗 Sources",
    ])

    # Tab 1 — Sub-questions
    with tabs[0]:
        st.subheader("Decomposed Sub-Questions")
        for i, sq in enumerate(result.get("sub_questions", []), 1):
            st.markdown(f"**{i}.** {sq}")

    # Tab 2 — Evidence
    with tabs[1]:
        st.subheader("Retrieved Evidence")
        evidence_map = result.get("evidence_map", {})
        facts_map = result.get("facts_map", {})

        for sq, ev_list in evidence_map.items():
            with st.expander(f"**{sq}**", expanded=False):
                if ev_list:
                    st.markdown("**Raw chunks retrieved:**")
                    for e in ev_list:
                        score_badge = f"`score: {e.get('score', 0):.3f}`"
                        st.markdown(f"- {score_badge} **{e['source']}**")
                        st.text(e["text"][:400] + ("…" if len(e["text"]) > 400 else ""))
                else:
                    st.info("No evidence found for this sub-question.")

                st.markdown("**Extracted facts:**")
                st.markdown(facts_map.get(sq, "_No facts extracted._"))

    # Tab 3 — Verification
    with tabs[2]:
        st.subheader("Claim Verification")
        verification = result.get("verification", [])
        if not verification:
            st.info("No verification results.")
        else:
            verdict_colors = {
                "SUPPORTED": "🟢",
                "PARTIAL": "🟡",
                "UNSUPPORTED": "🔴",
            }
            for v in verification:
                verdict = v.get("verdict", "PARTIAL")
                icon = verdict_colors.get(verdict, "⚪")
                with st.container(border=True):
                    st.markdown(f"{icon} **{verdict}** — {v.get('claim', '')}")
                    st.caption(v.get("reason", ""))

    # Tab 4 — Report
    with tabs[3]:
        st.subheader("Generated Research Report")
        report = result.get("report", "")
        if report:
            st.markdown(report)

            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Save as Markdown"):
                    try:
                        fpath = export_markdown(report, question)
                        st.success(f"Saved to `{fpath}`")
                    except Exception as ex:
                        st.error(f"Export failed: {ex}")
            with col2:
                st.download_button(
                    label="⬇️ Download .md",
                    data=report,
                    file_name="research_report.md",
                    mime="text/markdown",
                )
        else:
            st.info("Report not yet generated.")

    # Tab 5 — Sources
    with tabs[4]:
        st.subheader("Sources")
        citations = result.get("citations", [])
        if citations:
            for c in citations:
                icon = "🌐" if c["type"] == "web" else "📄"
                st.markdown(f"{c['id']}. {icon} `{c['display']}`")
        else:
            st.info("No sources recorded.")

else:
    # Landing placeholder
    st.markdown("""
    ### How it works

    1. **Upload** your research documents (PDF, TXT, Markdown) in the sidebar
    2. Click **Ingest Documents** to embed and index them
    3. Type your **research question** above
    4. Click **Run Research** to launch the multi-agent pipeline:

    | Agent | Role |
    |---|---|
    | 🗺️ Planner | Breaks question into sub-questions |
    | 🔍 Researcher | Retrieves evidence from your documents |
    | 🧪 Evidence | Extracts structured facts |
    | ⚖️ Critic | Verifies claims against evidence |
    | ✍️ Writer | Generates a cited Markdown report |
    """)
