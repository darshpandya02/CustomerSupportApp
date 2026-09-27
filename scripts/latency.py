"""Measure live latency of POST /api/chat on the deployment.

    uv run python scripts/latency.py https://<deployment> [pause_seconds]

Sends every eval question once (new conversation each, unique Idempotency-Key),
paced to stay under the per-client rate limit. Records the client round trip
and the server-side timings the API returns (retrieval, generation, total).
The first request is reported separately as a possible cold start.
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def pct(xs: list[float]) -> dict:
    return {"p50": round(float(np.percentile(xs, 50))), "p95": round(float(np.percentile(xs, 95))),
            "n": len(xs)}


def main(base: str, pause: float) -> int:
    qs = [json.loads(l)["question"] for l in (ROOT / "eval" / "questions.jsonl").open()]
    qs += [json.loads(l)["question"] for l in (ROOT / "eval" / "out_of_scope.jsonl").open()]
    rows = []
    with httpx.Client(timeout=60) as c:
        for i, q in enumerate(qs):
            t0 = time.perf_counter()
            r = c.post(f"{base}/api/chat", json={"message": q}, headers={"Idempotency-Key": uuid.uuid4().hex})
            rtt = (time.perf_counter() - t0) * 1000
            if r.status_code == 429:
                time.sleep(int(r.headers.get("Retry-After", "60")) + 1)
                continue
            j = r.json()
            rows.append({"q": q, "status": r.status_code, "mode": j.get("mode"), "rtt_ms": rtt,
                         **{k: v for k, v in (j.get("timings_ms") or {}).items()}})
            print(i, r.status_code, j.get("mode"), round(rtt), j.get("timings_ms", {}).get("retrieval_ms"), flush=True)
            time.sleep(pause)
    warm = rows[1:]
    report = {
        "base_url": base,
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "requests": len(rows),
        "modes": {m: sum(1 for r in rows if r["mode"] == m) for m in {r["mode"] for r in rows}},
        "first_request_rtt_ms": round(rows[0]["rtt_ms"]),
        "summary": {
            "retrieval_ms": pct([r["retrieval_ms"] for r in warm]),
            "rerank_ms": pct([r["rerank_ms"] for r in warm if "rerank_ms" in r]),
            "server_total_ms": pct([r["total_ms"] for r in warm]),
            "end_to_end_client_ms": pct([r["rtt_ms"] for r in warm]),
        },
        "note": "Client in Boston, function in iad1. End-to-end includes Postgres writes, rate-limit and idempotency checks.",
        "rows": [{k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()} for r in rows],
    }
    (ROOT / "reports" / "latency_live.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1].rstrip("/"), float(sys.argv[2]) if len(sys.argv) > 2 else 31))
