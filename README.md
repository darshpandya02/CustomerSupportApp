# CustomerSupportApp

A customer support assistant that answers questions from a product manual, cites the sections it used, refuses when the manual has no answer, and hands off to a FastAPI ticketing backend that routes, prioritises and tracks escalations.

**Live:** https://customer-support-app-woad.vercel.app
- Chat widget: `/`
- Agent dashboard: [`/dashboard.html`](https://customer-support-app-woad.vercel.app/dashboard.html). The demo password is **`support-demo-2026`**. It is shared on purpose so reviewers can try the agent workflow. Please don't put real data in it.
- API docs (OpenAPI): [`/api/docs`](https://customer-support-app-woad.vercel.app/api/docs), spec at `/api/openapi.json`

> **History of this repository.** Until 2026 this repo held only a two-line placeholder README. The original 2023 chatbot project described on my resume (May to Dec 2023) was not preserved, so none of that code is here. Everything in this repo was built in 2026 as a working, deployed version of that project. It keeps only the parts I could actually build and measure; the resume's percentage claims are not reproduced (see [Real stack vs resume stack](#real-stack-vs-resume-stack)).

> **Current status of LLM answers.** Generation goes through the Vercel AI Gateway, which will not serve requests until the account has a credit card on file (`customer_verification_required`). Until then, the chat endpoint falls back to **retrieval-only answers**: the three best reranked manual sections, with links and the label "LLM answers are temporarily unavailable; here are the most relevant manual sections." The LLM path is fully implemented and unit-tested against a mocked gateway. It turns on automatically once the gateway accepts requests, with no code change. Because of this, the answer-quality evaluation is **pending**.

## Architecture

```
Browser (public/: chat widget, agent dashboard; static on the Vercel CDN)
   │
   ▼
FastAPI on Vercel Functions (Python 3.12, app.py -> supportbot/api.py)
   ├── POST /api/chat
   │     idempotency check ─ rate limit (per client + daily LLM cap) ─ load memory
   │     condense follow-up (Llama 3.1 8B) → hybrid retrieval → refusal gate
   │     → grounded, cited generation (Llama 4 Maverick) → persist turn
   │     └─ gateway error? → retrieval-only answer (top 3 passages + links)
   ├── POST /api/tickets        escalation with transcript, routing, SLA
   ├── GET  /api/tickets/{ref}  customer tracking (token)
   └── /api/agent/*             login, list/filter, detail, status workflow, stats
   │
   ├── Retrieval, in-process: FAISS IndexFlatIP (bge-small, 384-d) + BM25,
   │   reciprocal rank fusion, MiniLM cross-encoder rerank (int8 ONNX)
   ├── Vercel AI Gateway (OpenAI-compatible Chat Completions, project OIDC token)
   └── Neon Postgres, schema `support` only:
         conversations, messages, tickets, ticket_events, idempotency_keys, rate_limits
```

### Retrieval
- The index is built offline (`scripts/build_index.py`) with sentence-transformers `BAAI/bge-small-en-v1.5` into a FAISS `IndexFlatIP` over normalised vectors: 457 chunks from 88 pages, 92 words per chunk on average.
- At request time the query is embedded by an int8 ONNX export of the same model (`scripts/export_onnx.py`). PyTorch does not fit in a function bundle, so the runtime is onnxruntime, tokenizers, faiss and numpy only.
  - Measured on the 52 eval questions, the int8 query vectors have a mean cosine of 0.982 to the sentence-transformers fp32 vectors (minimum 0.966).
- First stage: FAISS top 50 and BM25 top 50, fused with reciprocal rank fusion (k=60).
- Second stage: the top 15 candidates are reranked by `cross-encoder/ms-marco-MiniLM-L-6-v2` (int8 ONNX, pairs truncated to 192 tokens), and the top 5 go into the prompt.

### Generation (Llama through the AI Gateway)
- **Model:** `meta/llama-4-maverick` for answers and `meta/llama-3.1-8b` for condensing follow-ups and summarising history. Both are configurable via `CHAT_MODEL` and `SUMMARY_MODEL`. Both IDs are from the gateway's live `/v1/models` catalog.
- **Auth:** the per-request OIDC token Vercel forwards (`x-vercel-oidc-token`), with `VERCEL_OIDC_TOKEN` or `AI_GATEWAY_API_KEY` as fallbacks. No provider keys are stored.
- **Grounding:**
  - The system prompt requires a `[n]` citation on every factual sentence and forbids outside knowledge.
  - The excerpts are marked as reference text, not instructions (prompt-injection hardening).
  - The model must reply `NOT_IN_DOCS` when the excerpts don't answer the question; that reply is turned into a refusal.
  - Only citation numbers that point to real excerpts are kept.
- **Refusal gate:** before any LLM call, if the best reranked passage scores below a cross-encoder logit of -1.5, the assistant refuses and suggests escalation. Off-topic questions never reach the model.
- **Multi-turn memory:**
  - The last 4 turns are kept verbatim.
  - Older turns are folded into a running summary stored on the conversation row. Llama 3.1 8B writes it, with an extractive fallback when the gateway is down.
  - Follow-ups like "what about on iOS?" are rewritten into standalone search queries.
- **Degradation:** the gateway client opens a 5-minute circuit breaker on account-level errors, so while the card is missing each request doesn't waste a round-trip to the gateway.
- LangChain was not used. The pipeline is a thin custom module (`supportbot/rag.py`, about 200 lines): the chain is linear, and owning the prompt, citation parsing and fallback logic made them easier to test than a framework abstraction would.

### Ticketing backend
- **Escalation:** `POST /api/tickets` needs an `Idempotency-Key`. It copies the full chat transcript into the ticket, routes it, sets a priority and a first-response SLA (urgent 1 h, high 4 h, normal 24 h, low 72 h), and returns a ref (`CS-1001`) plus a tracking token.
- **Routing (`supportbot/routing.py`):**
  - High-precision keyword rules run first. They decide a ticket only when exactly one queue's rules match.
  - Otherwise a multinomial logistic regression over the same bge-small embeddings decides. That classifier is trained offline and shipped as a 3×384 weight matrix, so no extra model is loaded at runtime.
  - Priority is rule-based (security/data-loss language is urgent; lockouts, duplicate charges and multi-day outages are high).
  - The routing method, the matched rule and the classifier probabilities are stored with the ticket and shown in the dashboard.
- **Status workflow:** `open → in_progress / waiting_on_customer → resolved → closed`, with reopen from `resolved`. Invalid transitions return 409.
  - Every change (status, priority, queue, assignee, internal note) writes a `ticket_events` row with the actor.
  - First response and resolution times are stamped automatically. SLA breaches are computed in SQL.
- **Idempotency:** keys are claimed atomically with `INSERT ... ON CONFLICT DO NOTHING` in `support.idempotency_keys` and store the response. A retry with the same key replays it (`Idempotent-Replayed: true`) and never calls the LLM or creates a second ticket.
  - The same key with a different body returns 409.
  - A failed request releases the key so the client can retry.
- **Rate limits:** fixed-window counters in Postgres, updated atomically with an upsert:
  - Chat: 20 requests per client per 10 minutes.
  - Tickets: 5 per client per 10 minutes.
  - Agent login: 10 per client per 10 minutes.
  - A global cap of 1,500 LLM-backed answers per day, after which chat degrades to retrieval-only instead of spending more.
  - Client IPs are stored only as HMAC hashes.
- **Validation:** Pydantic models (message 1 to 1,000 chars, UUID conversation IDs, email pattern, enum-typed status/priority/queue, bounded pagination). Replies also have a capped `max_tokens`.
- **Agent auth:** the shared password yields an HMAC-signed 8-hour session in an HttpOnly, SameSite=Strict cookie. A bearer token is also accepted for API use.

## Data source and license

- **Source:** the [Nextcloud user manual](https://docs.nextcloud.com/server/stable/user_manual/en/) (stable), © Nextcloud GmbH and the Nextcloud documentation contributors, licensed [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/). See the [license statement](https://github.com/nextcloud/documentation#license). The processed text in `data/pages.jsonl` and `artifacts/chunks.json` is an adaptation made by chunking; every answer links back to the original section. This project is not affiliated with Nextcloud.
- **Ingestion (`scripts/scrape.py`):** requests + BeautifulSoup. The manual is static Sphinx HTML, so a browser (Selenium) was not needed.
  - The crawler reads `robots.txt` first; it only allows `/server/stable/`, and the crawler stays inside `/server/stable/user_manual/en/`.
  - It waits 0.75 s between requests, sends an identifying User-Agent, and caches raw HTML locally.
  - Chunking follows the Sphinx section tree: each chunk keeps its page, heading path and anchor URL. Sections over 220 words are split with a 40-word overlap, and tiny sections are merged forward.
- **Demo framing:** the assistant acts as the help desk of a Nextcloud-based workspace. Billing questions (plans, invoices, refunds) are deliberately not in the manual. They are refused and go to the billing queue when escalated.

## Evaluation (all numbers measured; reports in `reports/`)

### Retrieval: 52 hand-written questions with gold source chunks
- Each question names a page and a phrase that must appear in the retrieved chunk, so the labels survive re-chunking (`eval/questions.jsonl`).
- recall@k is the share of questions with at least one gold chunk in the top k. MRR is computed over the top 10.
- Run with `scripts/eval_retrieval.py`; results are in `reports/retrieval_eval.json`.

| Method | R@1 | R@3 | R@5 | R@10 | MRR@10 |
|---|---|---|---|---|---|
| FAISS vector | 0.654 | 0.827 | 0.865 | 0.981 | 0.751 |
| FAISS vector + rerank | 0.750 | 0.827 | 0.942 | 0.981 | 0.815 |
| BM25 | 0.558 | 0.750 | 0.808 | 0.827 | 0.663 |
| BM25 + rerank | 0.750 | 0.827 | 0.865 | 0.865 | 0.794 |
| Hybrid (RRF) | 0.615 | 0.788 | 0.885 | 0.962 | 0.714 |
| **Hybrid + rerank (deployed)** | **0.750** | **0.846** | **0.904** | **0.981** | **0.817** |

- The cross-encoder is what helps: R@1 goes from 0.615 to 0.750 and MRR from 0.714 to 0.817 on the hybrid candidates.
- Hybrid fusion by itself did **not** beat plain vector search on this set. It is kept because BM25 catches exact strings such as error codes (`0x800700DF`) and config keys. On this set, vector + rerank is as good or slightly better at R@5.
- The reranker settings (15 candidates, 192 tokens) were chosen to cut latency on the single-vCPU Hobby function. They were picked on this same eval set, and quality was unchanged versus 20 candidates at 512 tokens.

### Out-of-scope refusal (retrieval gate only, LLM not involved)
Run with `scripts/eval_refusal.py`; results are in `reports/refusal_eval.json`. The threshold is -1.5 (cross-encoder logit of the best passage).

| Set | Refused |
|---|---|
| 16 out-of-scope questions | **62.5%** (10/16) |
| 52 in-scope questions (false refusals) | **3.8%** (2/52) |

- **By kind:**
  - billing: 3/3 refused
  - unrelated: 4/4
  - prompt injection: 1/1
  - other products: 1/2
  - "unanswered Nextcloud facts": 1/3
  - server-administration: **0/3**
- The admin questions (install with Docker, LDAP, php.ini) pass the gate because the manual has closely related text.
- With the LLM on, those are supposed to be caught by the `NOT_IN_DOCS` instruction. That combined rate is part of the pending answer eval.
- The threshold was chosen from a sweep on these same questions, so these rates are optimistic.

### Ticket routing: 60 held-out tickets
- The test tickets (20 per queue, with priority labels) were written separately from the 120 training tickets.
- Run with `scripts/train_router.py`; results are in `reports/routing_eval.json`.

| Router | Queue accuracy |
|---|---|
| Rules only (unmatched go to `technical`) | 0.717 (rules fired on 55% of tickets, 100% precision when they fired) |
| Classifier only | 0.983 |
| **Rules, then classifier (deployed)** | **0.983** (59/60; 33 decided by rules, 27 by the classifier) |

- Priority rules: accuracy 0.933 (56/60); 90% recall on urgent/high tickets.
- The single queue error is "The Linux client asks for my password on every boot", which is technical but was predicted as account.

### Answer quality (LLM): **pending**
`scripts/eval_answers.py` is ready. It runs the full pipeline on all 68 questions and scores each in-scope answer with an LLM judge from a different provider (`openai/gpt-5.4-mini` by default). The judge gives faithfulness 1 to 5 against the retrieved excerpts, and correctness 1 to 5 against the hand-written reference answers. The script also measures the combined refusal rate and writes all answers to `reports/answers_for_spot_check.jsonl` for manual review. It needs the AI Gateway to accept requests (see status above), so no answer-quality numbers are claimed yet.

### Latency (live deployment)
See `reports/latency_live.json`, measured by `scripts/latency.py` against production from a client in Boston (function in `iad1`, Hobby plan, 1 vCPU). 68 requests (every eval question once, each in a new conversation), measured 2026-09-27. The first request was 1.4 s; it hit a warm instance, so no cold start was captured. The p50/p95 below are over the other 67 requests.

| Stage | p50 | p95 |
|---|---|---|
| Retrieval (embed + FAISS + BM25 + rerank), server-side | 974 ms | 1,029 ms |
| of which cross-encoder rerank | 961 ms | 1,018 ms |
| Server total per chat request | 976 ms | 1,143 ms |
| End-to-end from the client (HTTPS, idempotency, rate limit, Postgres writes) | 1,243 ms | 1,443 ms |

These are **retrieval-only** answers (56 of 68) and refusals (12 of 68), because the gateway is pending. LLM generation time is not included and will be measured once the gateway is on. The reranker dominates the time: first-stage retrieval takes about 10 to 15 ms on the function, and the same reranking takes about 140 ms on a laptop CPU.

### Browser verification
`scripts/e2e.py` (Playwright, headless Chromium) runs against the live URL:
- It asks 3 in-scope questions. All come back with cited manual links; they are retrieval-only while the gateway is pending.
- It asks 1 billing question, which is refused.
- It clicks "Escalate to a human", which creates a ticket.
- It logs into the dashboard and checks that the ticket is in the right queue with the transcript attached.

The last run created ticket CS-1002, routed to **billing** with high priority by a rule (matched "charged"; classifier 96% billing), with 8 transcript messages attached. The screenshots are in `reports/screens/`.

## Real stack vs resume stack

| Resume (2023) | This build (2026) |
|---|---|
| RAG with LangChain | RAG with a custom pipeline (condense → hybrid retrieve → rerank → gate → cited generation). LangChain was not used. |
| Meta LLaMA via Hugging Face | Meta Llama 4 Maverick (answers) and Llama 3.1 8B (condense/summary) via the Vercel AI Gateway. Pending account verification, with retrieval-only fallback. |
| Prompt engineering, multi-turn memory | Grounding prompt with mandatory citations and a refusal token; last 4 turns verbatim plus an LLM-written running summary persisted in Postgres. |
| Ticket creation, routing, tracking via backend APIs | FastAPI + Postgres: escalation with transcript, rules + embedding classifier routing, priorities, SLA, status workflow, audit log, customer tracking, agent dashboard. |
| FAISS embedding search via FastAPI | FAISS (bge-small embeddings) + BM25 hybrid with cross-encoder rerank, served in-process by FastAPI. |
| Selenium scraping | requests + BeautifulSoup (the source is static HTML), robots.txt respected. |
| Docker | `Dockerfile` included. It was **not built or run** here (no Docker daemon available); the deployed target is Vercel Functions. |
| AWS Comprehend, Azure OpenAI trials | **Dropped.** Not used. |
| 25% resolution, 30% handling time, 15% CSAT | **Dropped.** These are user metrics from a production deployment that I can't verify. |

## API

Full interactive docs are at [`/api/docs`](https://customer-support-app-woad.vercel.app/api/docs).

| Method | Path | Notes |
|---|---|---|
| POST | `/api/chat` | `{message, conversation_id?}`. Optional `Idempotency-Key` header, which the web client always sends. Returns `mode` (`llm`, `retrieval_only` or `refusal`), `answer`, `citations[]` (title, section, url, snippet, score), `timings_ms`. Rate-limit headers are included. |
| POST | `/api/tickets` | `{description, subject?, conversation_id?, name?, email?}`. **Requires** `Idempotency-Key`. Returns ref, tracking token, queue, priority, SLA and routing details. |
| GET | `/api/tickets/{ref}?token=` | Customer-facing status and history. |
| POST | `/api/agent/login` | `{password}` sets the session cookie and also returns a bearer token. |
| GET | `/api/agent/tickets` | Filters: `queue`, `status`, `priority`, `q`; sorted by priority, then newest. |
| GET | `/api/agent/tickets/{id}` | Ticket, transcript, events, allowed transitions. |
| PATCH | `/api/agent/tickets/{id}` | `{status?, priority?, queue?, assignee?, note?}`. Invalid transitions return 409. |
| GET | `/api/agent/stats` | Counts by queue and status, SLA breaches, routing methods, answer modes. |
| GET | `/api/health`, `/api/meta` | Health (DB, gateway breaker state); knowledge-base, model and eval summary. |

```bash
curl -s https://customer-support-app-woad.vercel.app/api/chat \
  -H 'Content-Type: application/json' -H "Idempotency-Key: $(uuidgen)" \
  -d '{"message":"How do I restore a deleted folder?"}'
```

## Running it

```bash
uv sync --group dev
# Tests: unit tests plus API tests against a throwaway Postgres (CI uses a postgres:17 service)
TEST_DATABASE_URL=postgresql://postgres@localhost:5432/supporttest uv run pytest -q

# Local server (serves public/ too); needs DATABASE_URL, AGENT_PASSWORD, SESSION_SECRET
uv run python scripts/migrate.py
uv run uvicorn app:app --reload

# Rebuild the knowledge base and models (build group pulls in sentence-transformers/torch)
uv sync --group build
uv run python scripts/scrape.py && uv run python scripts/build_index.py
uv run python scripts/export_onnx.py && uv run python scripts/train_router.py
uv run python scripts/eval_retrieval.py && uv run python scripts/eval_refusal.py
```

- CI (`.github/workflows/ci.yml`) runs the 35 tests against a Postgres service container.
- Those tests cover retrieval, routing, the gateway client (mocked HTTP), the RAG pipeline with a mocked gateway (cited answers, `NOT_IN_DOCS`, fallback, follow-up condensing, summaries), idempotent replay and conflicts, rate limits, the daily cap, validation, the ticket lifecycle and agent auth.
- Deployment is `vercel deploy --prod`. Postgres comes from a Neon database connected through the Vercel Marketplace, and everything lives in the `support` schema.

## Limitations

- **LLM answers are off until the AI Gateway account is verified**, and the answer-quality evaluation (faithfulness, correctness, combined refusal rate) is still pending. Until then the product returns retrieval-only answers.
- **The eval questions were written by me while looking at the chunked manual**, and the refusal threshold and reranker settings were tuned on those same questions. Real users phrase things differently and ask about sections I didn't sample. Treat the retrieval and refusal numbers as optimistic, in-distribution estimates rather than production performance.
- The routing and priority sets (180 tickets) are also hand-written by one author, so the 98% queue accuracy reflects clean, single-topic tickets.
- The retrieval gate misses server-administration questions (0/3 refused) because the user manual contains related text. With no LLM, those questions get retrieval-only passages that don't answer them.
- The reranker costs about 1 s per request on the Hobby plan's single vCPU. Cold starts also install dependencies at runtime (the bundle exceeds the standard size), which adds a few seconds to the first request on a new instance.
- The dashboard uses one shared demo password and has no per-agent accounts, roles or CSRF tokens beyond SameSite=Strict cookies. Anyone with the README can triage demo tickets.
- Rate limits are keyed on the client IP from `x-forwarded-for`, so users behind the same NAT share a quota. The limits are fixed-window, not sliding.
- The knowledge base is a snapshot of the stable manual taken on 2026-09-26. Re-run the scraper and index build to refresh it.
- The Dockerfile has not been built or tested.
