"""LangGraph multi-agent pipeline: plan → research → extract → verify → write."""

from typing import TypedDict

from langgraph.graph import StateGraph, END

from agents.planner_agent import plan
from agents.research_agent import research
from agents.evidence_agent import extract_all_facts
from agents.critic_agent import verify
from agents.writer_agent import write_report
from tools.citation_tool import build_citations, format_references


# ── State definition ─────────────────────────────────────────────────────────

class ResearchState(TypedDict):
    question: str
    sub_questions: list[str]
    evidence_map: dict[str, list[dict]]
    facts_map: dict[str, str]
    verification: list[dict]
    citations: list[dict]
    report: str
    error: str


# ── Node functions ────────────────────────────────────────────────────────────

def node_plan(state: ResearchState) -> ResearchState:
    sub_questions = plan(state["question"])
    return {**state, "sub_questions": sub_questions}


def node_research(state: ResearchState) -> ResearchState:
    evidence_map = research(state["sub_questions"])
    return {**state, "evidence_map": evidence_map}


def node_extract(state: ResearchState) -> ResearchState:
    facts_map = extract_all_facts(state["evidence_map"])
    return {**state, "facts_map": facts_map}


def node_verify(state: ResearchState) -> ResearchState:
    verification = verify(state["facts_map"], state["evidence_map"])
    citations = build_citations(state["evidence_map"])
    return {**state, "verification": verification, "citations": citations}


def node_write(state: ResearchState) -> ResearchState:
    report_body = write_report(
        question=state["question"],
        sub_questions=state["sub_questions"],
        facts_map=state["facts_map"],
        verification=state["verification"],
    )
    references_section = format_references(state["citations"])
    full_report = f"{report_body}\n\n{references_section}".strip()
    return {**state, "report": full_report}


# ── Graph assembly ────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(ResearchState)

    graph.add_node("plan", node_plan)
    graph.add_node("research", node_research)
    graph.add_node("extract", node_extract)
    graph.add_node("verify", node_verify)
    graph.add_node("write", node_write)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "research")
    graph.add_edge("research", "extract")
    graph.add_edge("extract", "verify")
    graph.add_edge("verify", "write")
    graph.add_edge("write", END)

    return graph.compile()


_graph = None


def get_pipeline():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_pipeline(question: str) -> ResearchState:
    """Run the full pipeline and return the final state."""
    pipeline = get_pipeline()
    initial_state: ResearchState = {
        "question": question,
        "sub_questions": [],
        "evidence_map": {},
        "facts_map": {},
        "verification": [],
        "citations": [],
        "report": "",
        "error": "",
    }
    result = pipeline.invoke(initial_state)
    return result
