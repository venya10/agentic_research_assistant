"""Extracts structured facts from evidence chunks for each sub-question."""

from langchain_core.prompts import ChatPromptTemplate
from app.llm_factory import get_llm

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a fact-extraction assistant. "
        "Given a question and a set of evidence passages, extract the key facts that are "
        "directly relevant to answering the question. "
        "Format your output as a bullet list of concise facts. "
        "For each fact, indicate which passage it came from using [Source: <filename>]. "
        "Do not invent information not present in the passages."
    )),
    ("human", (
        "Question: {question}\n\n"
        "Evidence passages:\n{passages}"
    )),
])


def _format_passages(evidence: list[dict]) -> str:
    parts = []
    for i, e in enumerate(evidence, 1):
        parts.append(f"[{i}] Source: {e['source']}\n{e['text']}")
    return "\n\n".join(parts)


def extract_facts(question: str, evidence: list[dict]) -> str:
    """Return a bulleted string of facts extracted from evidence."""
    if not evidence:
        return "• No evidence found for this sub-question."

    llm = get_llm(temperature=0.1)
    chain = _PROMPT | llm

    response = chain.invoke({
        "question": question,
        "passages": _format_passages(evidence),
    })
    return response.content.strip()


def extract_all_facts(evidence_map: dict[str, list[dict]]) -> dict[str, str]:
    """Run extract_facts for every sub-question."""
    return {q: extract_facts(q, ev) for q, ev in evidence_map.items()}
