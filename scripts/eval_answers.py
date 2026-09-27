"""Answer quality with the live Llama model (needs working AI Gateway access).

    set -a; source .env.local; set +a        # VERCEL_OIDC_TOKEN from `vercel env pull`
    uv run python scripts/eval_answers.py

For each of the 52 in-scope questions: run the full pipeline, then ask a judge
model (a different provider than the answering model) to score
  faithfulness 1-5  every claim is supported by the retrieved excerpts
  correctness  1-5  agrees with the hand-written reference answer
For the 16 out-of-scope questions: record whether the assistant refused
(retrieval gate or the model's NOT_IN_DOCS). Writes reports/answer_eval.json
and reports/answers_for_spot_check.jsonl (for manual review).
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from supportbot import llm, rag  # noqa: E402
from supportbot.config import CHAT_MODEL  # noqa: E402
from supportbot.retrieval import get_retriever  # noqa: E402

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "openai/gpt-5.4-mini")

JUDGE_PROMPT = """You grade a customer-support answer. Return only JSON:
{"faithfulness": 1-5, "correctness": 1-5, "reason": "<one sentence>"}
faithfulness: 5 = every factual claim is supported by the EXCERPTS; 1 = mostly unsupported or invented.
correctness: 5 = fully agrees with the REFERENCE answer and addresses the question; 3 = partially; 1 = wrong or refuses.
Judge only against the excerpts and the reference, not your own knowledge."""


def judge(question: str, reference: str, answer: str, excerpts: str) -> dict:
    out = llm.get_client().chat(JUDGE_MODEL, [
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user", "content": f"QUESTION: {question}\n\nREFERENCE: {reference}\n\nEXCERPTS:\n{excerpts}\n\nANSWER:\n{answer}"},
    ], max_tokens=600, temperature=0.0, token=os.environ.get("VERCEL_OIDC_TOKEN"))
    m = re.search(r"\{.*\}", out.text, re.S)
    return json.loads(m.group(0)) if m else {"faithfulness": None, "correctness": None, "reason": out.text[:200]}


def main() -> int:
    token = os.environ.get("VERCEL_OIDC_TOKEN")
    ins = [json.loads(l) for l in (ROOT / "eval" / "questions.jsonl").open()]
    oos = [json.loads(l) for l in (ROOT / "eval" / "out_of_scope.jsonl").open()]
    rows, lat = [], []
    for q in ins:
        t = time.perf_counter()
        a = rag.answer(q["question"], token=token)
        lat.append((time.perf_counter() - t) * 1000)
        if a.mode == "retrieval_only":
            raise SystemExit(f"Gateway unavailable ({a.llm_error}); answer eval needs the LLM")
        hits = get_retriever().search(a.query).hits
        excerpts = "\n\n".join(f"[{i + 1}] {h.chunk['text']}" for i, h in enumerate(hits))
        j = judge(q["question"], q["answer"], a.text, excerpts) if a.mode == "llm" else \
            {"faithfulness": None, "correctness": 1, "reason": "refused"}
        rows.append({"id": q["id"], "question": q["question"], "mode": a.mode, "answer": a.text,
                     "citations": [c["url"] for c in a.citations], "judge": j,
                     "generation_ms": a.timings_ms.get("generation_ms"), "usage": a.usage})
        print(q["id"], a.mode, j.get("faithfulness"), j.get("correctness"), flush=True)
    refused = []
    for q in oos:
        a = rag.answer(q["question"], token=token)
        refused.append({"id": q["id"], "kind": q["kind"], "refused": a.mode == "refusal", "answer": a.text[:300]})
    answered = [r for r in rows if r["mode"] == "llm"]
    f = [r["judge"]["faithfulness"] for r in answered if r["judge"].get("faithfulness")]
    c = [r["judge"]["correctness"] for r in rows if r["judge"].get("correctness")]
    report = {
        "summary": {
            "model": CHAT_MODEL, "judge_model": JUDGE_MODEL, "in_scope": len(ins),
            "answered": len(answered), "false_refusals": len(rows) - len(answered),
            "faithfulness_mean": round(float(np.mean(f)), 2) if f else None,
            "faithful_share_ge4": round(float(np.mean([x >= 4 for x in f])), 3) if f else None,
            "correctness_mean": round(float(np.mean(c)), 2) if c else None,
            "correct_share_ge4": round(float(np.mean([x >= 4 for x in c])), 3) if c else None,
            "answers_with_citation": round(float(np.mean([bool(r["citations"]) for r in answered])), 3) if answered else None,
            "out_of_scope_refusal_rate": round(float(np.mean([r["refused"] for r in refused])), 3),
            "pipeline_latency_ms_local": {"p50": round(float(np.percentile(lat, 50))),
                                          "p95": round(float(np.percentile(lat, 95)))},
        },
        "manual_spot_check": "pending: review reports/answers_for_spot_check.jsonl and record verdicts here",
        "out_of_scope": refused,
    }
    (ROOT / "reports" / "answer_eval.json").write_text(json.dumps(report, indent=2))
    with (ROOT / "reports" / "answers_for_spot_check.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
