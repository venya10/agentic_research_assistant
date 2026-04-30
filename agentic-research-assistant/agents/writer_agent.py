"""Generates a structured Markdown research report with inline citations."""

from langchain_core.prompts import ChatPromptTemplate
from app.llm_factory import get_llm

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an expert research writer. "
        "Write a well-structured, analytical research report in Markdown format. "
        "The report must:\n"
        "1. Start with an Executive Summary (2-3 sentences)\n"
        "2. Have one section per sub-question with a descriptive heading\n"
        "3. Use inline citations like [Source: filename] when referencing evidence\n"
        "4. End with a Conclusion section\n"
        "5. End with a References section listing all sources used\n"
        "Write in a clear, academic but accessible style. "
        "Do not fabricate facts — use only what is provided."
    )),
    ("human", (
        "Research Question: {question}\n\n"
        "Sub-questions and extracted facts:\n{facts_section}\n\n"
        "Verified claims summary:\n{verification_summary}"
    )),
])


def _build_facts_section(sub_questions: list[str], facts_map: dict[str, str]) -> str:
    parts = []
    for q in sub_questions:
        facts = facts_map.get(q, "No facts extracted.")
        parts.append(f"Sub-question: {q}\nFacts:\n{facts}")
    return "\n\n".join(parts)


def _build_verification_summary(verification: list[dict]) -> str:
    if not verification:
        return "No verification performed."
    lines = [
        f"- [{v['verdict']}] {v['claim']} — {v['reason']}"
        for v in verification
    ]
    return "\n".join(lines)


def write_report(
    question: str,
    sub_questions: list[str],
    facts_map: dict[str, str],
    verification: list[dict],
) -> str:
    """Return a complete Markdown report string."""
    llm = get_llm(temperature=0.4)
    chain = _PROMPT | llm

    response = chain.invoke({
        "question": question,
        "facts_section": _build_facts_section(sub_questions, facts_map),
        "verification_summary": _build_verification_summary(verification),
    })
    return response.content.strip()
