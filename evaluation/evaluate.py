"""
Evaluation harness.
Run: python evaluation/evaluate.py

Runs all questions from eval_questions.json through the pipeline
and prints a simple coverage / verdict summary.
Extend with RAGAS metrics once you have ground-truth answers.
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.pipeline import run_pipeline
from rag.vector_store import collection_count


def run_evaluation(questions_path: str = "evaluation/eval_questions.json"):
    with open(questions_path) as f:
        questions = json.load(f)

    if collection_count() == 0:
        print("⚠️  No documents are indexed. Ingest documents first.")
        return

    results = []
    for item in questions:
        print(f"\n{'='*60}")
        print(f"Q{item['id']}: {item['question']}")
        print("Running pipeline…")

        try:
            state = run_pipeline(item["question"])
            n_subq = len(state["sub_questions"])
            n_ev = sum(len(v) for v in state["evidence_map"].values())
            verdicts = [v["verdict"] for v in state["verification"]]
            supported = verdicts.count("SUPPORTED")
            unsupported = verdicts.count("UNSUPPORTED")
            report_len = len(state["report"])

            print(f"  Sub-questions:   {n_subq}")
            print(f"  Evidence chunks: {n_ev}")
            print(f"  Claims SUPPORTED/UNSUPPORTED: {supported}/{unsupported}")
            print(f"  Report length:   {report_len} chars")

            results.append({
                "id": item["id"],
                "status": "ok",
                "sub_questions": n_subq,
                "evidence_chunks": n_ev,
                "supported": supported,
                "unsupported": unsupported,
            })
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"id": item["id"], "status": "error", "error": str(e)})

    print("\n\n=== SUMMARY ===")
    for r in results:
        status = r.get("status", "?")
        if status == "ok":
            print(f"[✅] Q{r['id']} — {r['sub_questions']} sub-q | {r['evidence_chunks']} chunks | {r['supported']} supported")
        else:
            print(f"[❌] Q{r['id']} — {r.get('error', 'unknown error')}")


if __name__ == "__main__":
    run_evaluation()
