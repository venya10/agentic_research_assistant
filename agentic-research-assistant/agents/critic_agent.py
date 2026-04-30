"""Verifies whether claims are supported, partial, or unsupported by the evidence."""

import json
from langchain_core.prompts import ChatPromptTemplate
from app.llm_factory import get_llm

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a critical fact-checker. "
        "Given a set of claims and the evidence passages that were used to derive them, "
        "assess whether each claim is:\n"
        "  - SUPPORTED: clearly backed by the evidence\n"
        "  - PARTIAL: partially supported but missing key context\n"
        "  - UNSUPPORTED: not backed by any provided evidence\n\n"
        "Return a JSON array. Each element has:\n"
        '  "claim": <the claim text>,\n'
        '  "verdict": "SUPPORTED" | "PARTIAL" | "UNSUPPORTED",\n'
        '  "reason": <one sentence explanation>\n'
        "Return ONLY valid JSON."
    )),
    ("human", (
        "Claims:\n{claims}\n\n"
        "Evidence passages:\n{passages}"
    )),
])


def _flatten_evidence(evidence_map: dict[str, list[dict]]) -> str:
    parts = []
    seen = set()
    for ev_list in evidence_map.values():
        for e in ev_list:
            key = (e["source"], e.get("chunk_index", 0))
            if key not in seen:
                parts.append(f"Source: {e['source']}\n{e['text']}")
                seen.add(key)
    return "\n\n".join(parts[:15])  # cap to avoid token overflow


def verify(facts_map: dict[str, str], evidence_map: dict[str, list[dict]]) -> list[dict]:
    all_claims = "\n".join(
        f"- {line.lstrip('•- ')}"
        for facts in facts_map.values()
        for line in facts.splitlines()
        if line.strip() and line.strip() not in ("", "•")
    )

    if not all_claims.strip():
        return []

    passages = _flatten_evidence(evidence_map)
    llm = get_llm(temperature=0.0)
    chain = _PROMPT | llm

    response = chain.invoke({"claims": all_claims, "passages": passages})

    try:
        # Strip markdown code fences if present
        raw = response.content.strip().removeprefix("```json").removesuffix("```").strip()
        return json.loads(raw)
    except json.JSONDecodeError:
        return [{"claim": all_claims, "verdict": "PARTIAL", "reason": "Could not parse critic output."}]
