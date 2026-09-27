"""Out-of-scope gate: refusal rate vs the reranker-score threshold.

The gate refuses when the best reranked passage scores below
REFUSAL_RERANK_THRESHOLD. This sweeps thresholds over the 52 in-scope and 16
out-of-scope questions. Note: the threshold is chosen on these same
questions (no separate held-out split), so treat the numbers as optimistic.

    uv run python scripts/eval_refusal.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from supportbot.config import REFUSAL_RERANK_THRESHOLD  # noqa: E402
from supportbot.retrieval import get_retriever  # noqa: E402


def top_scores(name: str) -> list[tuple[str, str, float]]:
    r = get_retriever()
    out = []
    for line in (ROOT / "eval" / name).open():
        q = json.loads(line)
        hits = r.search(q["question"], k=1).hits
        out.append((q["id"], q.get("kind", "in_scope"), hits[0].rerank))
    return out


def main() -> int:
    ins, oos = top_scores("questions.jsonl"), top_scores("out_of_scope.jsonl")
    sweep = []
    for t in [-4.0, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 1.0]:
        sweep.append({
            "threshold": t,
            "out_of_scope_refused": round(sum(s < t for *_, s in oos) / len(oos), 3),
            "in_scope_refused": round(sum(s < t for *_, s in ins) / len(ins), 3),
        })
    t = REFUSAL_RERANK_THRESHOLD
    chosen = next(x for x in sweep if x["threshold"] == t)
    by_kind: dict[str, list[bool]] = {}
    for _, kind, s in oos:
        by_kind.setdefault(kind, []).append(s < t)
    report = {
        "summary": {"threshold": t, "in_scope_questions": len(ins), "out_of_scope_questions": len(oos),
                    "out_of_scope_refusal_rate": chosen["out_of_scope_refused"],
                    "in_scope_false_refusal_rate": chosen["in_scope_refused"]},
        "out_of_scope_by_kind": {k: f"{sum(v)}/{len(v)} refused" for k, v in by_kind.items()},
        "passed_gate": [{"id": i, "kind": k, "score": round(s, 2)} for i, k, s in oos if s >= t],
        "false_refusals": [{"id": i, "score": round(s, 2)} for i, _, s in ins if s < t],
        "sweep": sweep,
        "note": ("Gate only. When the LLM is available, questions that pass the gate can still be refused by "
                 "the model (NOT_IN_DOCS); that combined rate is measured in answer_eval.json."),
    }
    (ROOT / "reports" / "refusal_eval.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
