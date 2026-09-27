"""Retrieval quality on the hand-written eval set (eval/questions.jsonl).

A question's gold set is every chunk from the named page whose text contains
the gold phrase (so the labels survive re-chunking). Metrics:
  recall@k  share of questions with at least one gold chunk in the top k
  MRR@10    mean reciprocal rank of the first gold chunk (0 if not in top 10)

    uv run --group build python scripts/eval_retrieval.py
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from supportbot.config import EMBED_MODEL_HF, QUERY_PREFIX  # noqa: E402
from supportbot.retrieval import Retriever  # noqa: E402

KS = (1, 3, 5, 10)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).lower()


def load_questions(chunks: list[dict]) -> list[dict]:
    qs = []
    for line in (ROOT / "eval" / "questions.jsonl").open():
        q = json.loads(line)
        gold = {
            c["id"]
            for g in q["gold"]
            for c in chunks
            if c["page"] == g["page"] and norm(g["contains"]) in norm(c["text"])
        }
        assert gold, f"{q['id']} has no gold chunk"
        q["gold_ids"] = gold
        qs.append(q)
    return qs


def score(ranked_ids: list[list[int]], qs: list[dict]) -> dict:
    out = {}
    for k in KS:
        out[f"recall@{k}"] = float(np.mean([bool(set(r[:k]) & q["gold_ids"]) for r, q in zip(ranked_ids, qs)]))
    rr = []
    for r, q in zip(ranked_ids, qs):
        rank = next((i + 1 for i, cid in enumerate(r[:10]) if cid in q["gold_ids"]), None)
        rr.append(1.0 / rank if rank else 0.0)
    out["mrr@10"] = float(np.mean(rr))
    return {k: round(v, 3) for k, v in out.items()}


def main() -> int:
    results = {}
    parity = None
    for quantized in (True, False):
        r = Retriever(quantized=quantized)
        qs = load_questions(r.chunks)
        if quantized:
            from sentence_transformers import SentenceTransformer

            st = SentenceTransformer(EMBED_MODEL_HF)
            ref = st.encode([QUERY_PREFIX + q["question"] for q in qs], normalize_embeddings=True)
            got = np.vstack([r.embed_query(q["question"]) for q in qs])
            cos = (ref * got).sum(1)
            parity = {"min_cosine": round(float(cos.min()), 4), "mean_cosine": round(float(cos.mean()), 4)}
        tag = "int8" if quantized else "fp32"
        for mode in ("vector", "bm25", "hybrid"):
            for rerank in (False, True):
                ranked, lat = [], []
                for q in qs:
                    t = time.perf_counter()
                    res = r.search(q["question"], k=10, mode=mode, rerank=rerank)
                    lat.append((time.perf_counter() - t) * 1000)
                    ranked.append([h.chunk["id"] for h in res.hits])
                m = score(ranked, qs)
                m["p50_ms"] = round(float(np.percentile(lat, 50)), 1)
                m["p95_ms"] = round(float(np.percentile(lat, 95)), 1)
                name = f"{mode}{'+rerank' if rerank else ''} [{tag} ONNX query encoder/reranker]"
                results[name] = m
                print(f"{name:55s} {m}")
    prod = results["hybrid+rerank [int8 ONNX query encoder/reranker]"]
    base = results["hybrid [int8 ONNX query encoder/reranker]"]
    report = {
        "summary": {"questions": len(qs), "production_config": "hybrid (FAISS + BM25, RRF) + cross-encoder rerank",
                    "with_rerank": {k: prod[k] for k in ("recall@1", "recall@3", "recall@5", "mrr@10")},
                    "without_rerank": {k: base[k] for k in ("recall@1", "recall@3", "recall@5", "mrr@10")}},
        "questions": len(qs),
        "definition": "recall@k = share of questions with >=1 gold chunk in top k; MRR over top 10",
        "latency_note": "local CPU (Apple Silicon), warm process, per query",
        "query_encoder_parity_vs_sentence_transformers_fp32": parity,
        "results": results,
    }
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "retrieval_eval.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(parity))
    return 0


if __name__ == "__main__":
    sys.exit(main())
