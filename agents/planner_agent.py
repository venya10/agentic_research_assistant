"""Decomposes a research question into focused sub-questions."""

from langchain_core.prompts import ChatPromptTemplate
from app.llm_factory import get_llm
from app.config import cfg

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a research planning assistant. "
        "Given a research question, decompose it into {max_q} specific, answerable sub-questions. "
        "Each sub-question should explore a distinct aspect needed to fully answer the main question. "
        "Return ONLY a numbered list — one sub-question per line, no extra commentary."
    )),
    ("human", "Research question: {question}"),
])


def plan(question: str) -> list[str]:
    """Return a list of sub-questions for the given research question."""
    llm = get_llm(temperature=0.3)
    chain = _PROMPT | llm

    response = chain.invoke({
        "question": question,
        "max_q": cfg.MAX_SUBQUESTIONS,
    })

    raw = response.content.strip()
    lines = [
        line.lstrip("0123456789.-) ").strip()
        for line in raw.splitlines()
        if line.strip()
    ]
    # Keep only non-empty lines that look like questions
    return [l for l in lines if len(l) > 10][:cfg.MAX_SUBQUESTIONS]
