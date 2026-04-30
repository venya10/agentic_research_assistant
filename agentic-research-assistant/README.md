# Agentic Research Assistant

A portfolio-grade AI research assistant that decomposes research questions, retrieves evidence from your documents, verifies claims, and generates structured cited reports.

## Architecture

```
User Question
     │
     ▼
[Streamlit UI] ──uploads──▶ [RAG Pipeline (Chroma + sentence-transformers)]
     │
     ▼
[Planner Agent]       — breaks question into sub-questions
     │
     ▼
[Research Agent]      — retrieves evidence per sub-question
     │
     ▼
[Evidence Agent]      — extracts structured facts
     │
     ▼
[Critic Agent]        — verifies claims (SUPPORTED / PARTIAL / UNSUPPORTED)
     │
     ▼
[Writer Agent]        — generates Markdown report with citations
```

All agents are orchestrated by **LangGraph**. The LLM provider is configurable via `.env`.

---

## Quick Start

### 1. Clone and set up environment

```bash
cd agentic-research-assistant
python -m venv .venv
# Windows
.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env and set your API key + LLM_PROVIDER
```

Minimum required in `.env`:
```
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Run the app

```bash
streamlit run app/streamlit_app.py
```

### 4. Use the app

1. Open `http://localhost:8501`
2. Upload a PDF/TXT/Markdown file in the sidebar
3. Click **Ingest Documents**
4. Type a research question
5. Click **Run Research**

---

## Sample Documents

Two sample documents are included in `data/raw/`:
- `sample_climate_change.txt` — causes, effects, and economics of climate change
- `sample_ai_transformers.txt` — transformer architecture vs RNNs

**Sample queries to try:**
- "What are the main causes and economic impacts of climate change?"
- "How does transformer architecture differ from RNNs?"
- "What are the solutions to climate change and their effectiveness?"

---

## Project Structure

```
agentic-research-assistant/
├── app/
│   ├── config.py          # Central config from .env
│   ├── llm_factory.py     # Provider-agnostic LLM factory
│   ├── pipeline.py        # LangGraph orchestration
│   └── streamlit_app.py   # Streamlit UI
├── agents/
│   ├── planner_agent.py   # Decomposes question → sub-questions
│   ├── research_agent.py  # Retrieves evidence per sub-question
│   ├── evidence_agent.py  # Extracts facts from evidence
│   ├── critic_agent.py    # Verifies claims against evidence
│   └── writer_agent.py    # Generates Markdown report
├── rag/
│   ├── ingest.py          # PDF/TXT/MD → chunks → vector store
│   ├── chunking.py        # Sliding-window text splitter
│   ├── embeddings.py      # sentence-transformers (local, no API key)
│   ├── retriever.py       # High-level retrieval interface
│   └── vector_store.py    # Chroma CRUD operations
├── tools/
│   ├── web_search.py      # Tavily web search (mocked by default)
│   ├── citation_tool.py   # Source deduplication and formatting
│   └── report_export.py   # Save reports as .md or .pdf
├── evaluation/
│   ├── eval_questions.json
│   └── evaluate.py        # Batch evaluation harness
├── data/raw/              # Put your documents here
├── reports/               # Generated reports saved here
└── tests/
    ├── test_rag.py
    └── test_agents.py
```

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | `anthropic` / `openai` / `google` |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-6` | Claude model ID |
| `OPENAI_MODEL` | `gpt-4o` | OpenAI model ID |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local sentence-transformers model |
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `TOP_K_RETRIEVAL` | `5` | Chunks retrieved per query |
| `MAX_SUBQUESTIONS` | `5` | Max sub-questions from planner |
| `WEB_SEARCH_ENABLED` | `false` | Enable Tavily web search |
| `TAVILY_API_KEY` | — | Required if web search enabled |

---

## Running Tests

```bash
# RAG tests (no API key needed — uses local embeddings)
pytest tests/test_rag.py -v

# Agent logic tests (mocked LLM)
pytest tests/test_agents.py -v

# All tests
pytest tests/ -v
```

---

## Enabling Web Search

1. Get a free API key at [tavily.com](https://tavily.com)
2. In `.env`:
   ```
   WEB_SEARCH_ENABLED=true
   TAVILY_API_KEY=tvly-...
   ```
3. Install: `pip install tavily-python`

---

## Exporting Reports

- **Download in browser**: Use the "⬇️ Download .md" button in the Report tab
- **Save to disk**: Click "💾 Save as Markdown" — saves to `./reports/`
- **PDF export**: Install `weasyprint` and call `export_pdf()` from `tools/report_export.py`

---

## CV / Portfolio Bullets

- Built an end-to-end **agentic RAG system** with LangGraph orchestration, Chroma vector store, and local sentence-transformer embeddings
- Implemented a **5-agent pipeline** (Planner → Researcher → Evidence → Critic → Writer) with structured state management
- Designed a **claim verification layer** that classifies evidence as SUPPORTED / PARTIAL / UNSUPPORTED
- Built a **provider-agnostic LLM layer** supporting Anthropic Claude, OpenAI GPT-4o, and Google Gemini via `.env` config
- Shipped a **Streamlit UI** with live document ingestion, pipeline progress tracking, and Markdown/PDF export
