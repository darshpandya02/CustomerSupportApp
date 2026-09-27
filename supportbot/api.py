"""FastAPI app: chat assistant, ticket intake/tracking, and the agent API."""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Path, Query, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from . import auth, db, guards, llm, rag, tickets
from .config import (ARTIFACTS, CHAT_LIMIT_GLOBAL_DAY, CHAT_LIMIT_PER_IP, CHAT_MODEL, CHAT_WINDOW_SECONDS,
                     MAX_MESSAGE_CHARS, REFUSAL_RERANK_THRESHOLD, ROOT, SUMMARY_MODEL)
from .retrieval import get_retriever

app = FastAPI(
    title="Customer Support Assistant API",
    version="1.0.0",
    description=(
        "Retrieval-augmented support assistant over the Nextcloud user manual (CC BY 3.0) with a "
        "ticketing backend: escalation with transcript, rule + classifier routing, priorities, SLA "
        "targets, status workflow, idempotency keys and rate limits. Agent endpoints need a session "
        "from `POST /api/agent/login`."
    ),
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

Queue = Literal["billing", "technical", "account"]
Priority = Literal["low", "normal", "high", "urgent"]
Status = Literal["open", "in_progress", "waiting_on_customer", "resolved", "closed"]
_KEY = re.compile(r"^[A-Za-z0-9_\-:.]{8,128}$")
_EMAIL = r"^[^@\s]{1,64}@[^@\s]{1,190}\.[A-Za-z]{2,24}$"


# ------------------------------------------------------------------ models

class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    conversation_id: uuid.UUID | None = None

    @field_validator("message")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message must not be blank")
        return v.strip()


class Citation(BaseModel):
    n: int
    chunk_id: int
    title: str
    section: str
    url: str
    snippet: str
    score: float


class ChatOut(BaseModel):
    conversation_id: uuid.UUID
    message_id: int
    answer: str
    mode: Literal["llm", "retrieval_only", "refusal"]
    notice: str | None
    citations: list[Citation]
    search_query: str
    timings_ms: dict[str, float]
    model: str | None


class TicketIn(BaseModel):
    description: str = Field(min_length=10, max_length=4000)
    subject: str | None = Field(default=None, max_length=120)
    conversation_id: uuid.UUID | None = None
    name: str | None = Field(default=None, max_length=80)
    email: str | None = Field(default=None, max_length=254, pattern=_EMAIL)


class TicketCreated(BaseModel):
    ref: str
    tracking_token: str
    subject: str
    queue: Queue
    priority: Priority
    status: Status
    first_response_due: str
    created_at: str
    routing: dict
    transcript_messages: int


class LoginIn(BaseModel):
    password: str = Field(min_length=1, max_length=200)


class TicketPatch(BaseModel):
    status: Status | None = None
    priority: Priority | None = None
    queue: Queue | None = None
    assignee: str | None = Field(default=None, max_length=80)
    note: str | None = Field(default=None, max_length=2000)


# ------------------------------------------------------------------ helpers

def client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.headers.get("x-real-ip") or (request.client.host if request.client else "unknown")


def require_db() -> None:
    if not db.database_url():
        raise HTTPException(503, "database is not configured")


def idem_key(value: str | None) -> str | None:
    if value is not None and not _KEY.match(value):
        raise HTTPException(422, "Idempotency-Key must be 8-128 characters of [A-Za-z0-9_-:.]")
    return value


def limited(limit: guards.Limit) -> JSONResponse:
    return JSONResponse({"detail": "rate limit exceeded", "retry_after_s": limit.reset_in}, 429,
                        headers=limit.headers())


def replay(stored: tuple[int, dict]) -> JSONResponse:
    status, body = stored
    return JSONResponse(body, status, headers={"Idempotent-Replayed": "true"})


def current_agent(request: Request, authorization: Annotated[str | None, Header()] = None) -> str:
    token = request.cookies.get(auth.COOKIE)
    if not token and authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:]
    agent = auth.verify(token)
    if not agent:
        raise HTTPException(401, "agent login required")
    return agent


def _eval_summary() -> dict:
    out = {}
    for name in ("retrieval_eval", "refusal_eval", "routing_eval", "answer_eval"):
        p = ROOT / "reports" / f"{name}.json"
        if p.exists():
            out[name] = json.loads(p.read_text()).get("summary")
    return out


# ------------------------------------------------------------------ public

@app.get("/api/health", tags=["meta"])
def health() -> dict:
    db_ok = False
    if db.database_url():
        try:
            with db.conn() as c:
                db_ok = c.execute("SELECT 1").fetchone()[0] == 1
        except Exception:
            db_ok = False
    status = getattr(llm.get_client(), "status", {"available": True})
    return {"ok": True, "database": db_ok, "llm": status, "chunks": len(get_retriever().chunks)}


@app.get("/api/meta", tags=["meta"])
def meta() -> dict:
    manifest = json.loads((ARTIFACTS / "manifest.json").read_text())
    return {
        "knowledge_base": {k: manifest[k] for k in ("source", "source_url", "license", "attribution", "pages",
                                                     "chunks")},
        "models": {"chat": CHAT_MODEL, "summary_and_condense": SUMMARY_MODEL,
                   "embedding": manifest["embedding_model"], "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2"},
        "refusal_threshold": REFUSAL_RERANK_THRESHOLD,
        "limits": {"chat_per_client": CHAT_LIMIT_PER_IP, "window_s": CHAT_WINDOW_SECONDS,
                   "llm_chats_per_day": CHAT_LIMIT_GLOBAL_DAY, "max_message_chars": MAX_MESSAGE_CHARS},
        "eval": _eval_summary(),
    }


@app.post("/api/chat", response_model=ChatOut, tags=["chat"],
          responses={429: {"description": "rate limited"}, 409: {"description": "idempotency conflict"}})
def chat(body: ChatIn, request: Request, response: Response,
         idempotency_key: Annotated[str | None, Header()] = None,
         _: None = Depends(require_db)):
    """Ask the assistant. Send an `Idempotency-Key` header so retries never re-bill the LLM."""
    key = idem_key(idempotency_key)
    client = guards.client_hash(client_ip(request))
    payload = body.model_dump(mode="json")
    if key:
        try:
            stored = guards.begin("chat", key, payload)
        except guards.IdempotencyConflict:
            raise HTTPException(409, "Idempotency-Key was already used with a different request")
        except guards.IdempotencyInFlight:
            raise HTTPException(409, "a request with this Idempotency-Key is still in progress")
        if stored:
            return replay(stored)
    try:
        limit = guards.hit(f"chat:{client}", CHAT_LIMIT_PER_IP, CHAT_WINDOW_SECONDS)
        if not limit.allowed:
            if key:
                guards.release("chat", key)
            return limited(limit)
        # Global daily cap on LLM-backed answers; beyond it the assistant degrades to retrieval-only.
        day = guards.hit("chat:global", CHAT_LIMIT_GLOBAL_DAY, 86400)
        token = request.headers.get("x-vercel-oidc-token")

        cid = tickets.ensure_conversation(str(body.conversation_id) if body.conversation_id else None, client)
        memory, seq, stale = tickets.load_memory(cid)
        if stale:
            memory.summary = rag.summarize(memory.summary, stale, token)
            tickets.save_summary(cid, memory.summary, stale[-1]["seq"])
        ans = rag.answer(body.message, memory, token=token, use_llm=day.allowed)
        mid = tickets.save_turn(cid, seq, body.message, ans.text, ans.mode, ans.citations, {
            "query": ans.query, "timings_ms": ans.timings_ms, "model": ans.model, "usage": ans.usage,
            "llm_error": ans.llm_error, "top_score": ans.top_score,
        })
        out = ChatOut(conversation_id=cid, message_id=mid, answer=ans.text, mode=ans.mode, notice=ans.notice,
                      citations=ans.citations, search_query=ans.query, timings_ms=ans.timings_ms,
                      model=ans.model).model_dump(mode="json")
        if key:
            guards.finish("chat", key, 200, out)
    except Exception:
        if key:
            guards.release("chat", key)
        raise
    for k, v in limit.headers().items():
        response.headers[k] = v
    response.headers["Server-Timing"] = ", ".join(f"{k.removesuffix('_ms')};dur={v}"
                                                  for k, v in ans.timings_ms.items())
    return out


@app.post("/api/tickets", response_model=TicketCreated, status_code=201, tags=["tickets"],
          responses={429: {"description": "rate limited"}, 409: {"description": "idempotency conflict"}})
def create_ticket(body: TicketIn, request: Request,
                  idempotency_key: Annotated[str, Header(description="Required; retries return the same ticket")],
                  _: None = Depends(require_db)):
    """Escalate to a human. Attaches the chat transcript and routes the ticket to a queue."""
    key = idem_key(idempotency_key)
    payload = body.model_dump(mode="json")
    try:
        stored = guards.begin("ticket", key, payload)
    except guards.IdempotencyConflict:
        raise HTTPException(409, "Idempotency-Key was already used with a different request")
    except guards.IdempotencyInFlight:
        raise HTTPException(409, "a request with this Idempotency-Key is still in progress")
    if stored:
        return replay(stored)
    try:
        limit = guards.hit(f"ticket:{guards.client_hash(client_ip(request))}", 5, 600)
        if not limit.allowed:
            guards.release("ticket", key)
            return limited(limit)
        try:
            t = tickets.create_ticket(description=body.description, subject=body.subject,
                                      conversation_id=str(body.conversation_id) if body.conversation_id else None,
                                      name=body.name, email=body.email)
        except tickets.NotFound:
            guards.release("ticket", key)
            raise HTTPException(404, "conversation not found")
        guards.finish("ticket", key, 201, t)
    except HTTPException:
        raise
    except Exception:
        guards.release("ticket", key)
        raise
    return JSONResponse(t, 201)


@app.get("/api/tickets/{ref}", tags=["tickets"])
def track_ticket(ref: Annotated[str, Path(pattern=r"^CS-\d{1,10}$")],
                 token: Annotated[str, Query(min_length=10, max_length=64)], _: None = Depends(require_db)):
    """Customer-facing status lookup with the tracking token returned at creation."""
    try:
        return tickets.public_ticket(ref, token)
    except tickets.NotFound:
        raise HTTPException(404, "ticket not found")


# ------------------------------------------------------------------ agent

@app.post("/api/agent/login", tags=["agent"])
def login(body: LoginIn, request: Request, response: Response, _: None = Depends(require_db)):
    limit = guards.hit(f"login:{guards.client_hash(client_ip(request))}", 10, 600)
    if not limit.allowed:
        return limited(limit)
    if not auth.check_password(body.password):
        time.sleep(0.3)
        raise HTTPException(401, "wrong password")
    token = auth.issue("agent")
    response.set_cookie(auth.COOKIE, token, max_age=auth.SESSION_TTL, httponly=True, samesite="strict",
                        secure=os.environ.get("VERCEL") == "1", path="/api")
    return {"ok": True, "agent": "agent", "token": token, "expires_in": auth.SESSION_TTL}


@app.post("/api/agent/logout", tags=["agent"])
def logout(response: Response):
    response.delete_cookie(auth.COOKIE, path="/api")
    return {"ok": True}


@app.get("/api/agent/me", tags=["agent"])
def me(agent: str = Depends(current_agent)):
    return {"agent": agent}


@app.get("/api/agent/tickets", tags=["agent"])
def agent_list(agent: str = Depends(current_agent), queue: Queue | None = None, status: Status | None = None,
               priority: Priority | None = None, q: Annotated[str | None, Query(max_length=100)] = None,
               limit: Annotated[int, Query(ge=1, le=100)] = 50, offset: Annotated[int, Query(ge=0)] = 0):
    return tickets.list_tickets(queue, status, priority, q, limit, offset)


@app.get("/api/agent/tickets/{ticket_id}", tags=["agent"])
def agent_get(ticket_id: int, agent: str = Depends(current_agent)):
    try:
        return tickets.get_ticket(ticket_id)
    except tickets.NotFound:
        raise HTTPException(404, "ticket not found")


@app.patch("/api/agent/tickets/{ticket_id}", tags=["agent"])
def agent_update(ticket_id: int, body: TicketPatch, agent: str = Depends(current_agent)):
    try:
        return tickets.update_ticket(ticket_id, agent, **body.model_dump(exclude_unset=True))
    except tickets.NotFound:
        raise HTTPException(404, "ticket not found")
    except tickets.InvalidTransition as e:
        raise HTTPException(409, str(e))


@app.get("/api/agent/stats", tags=["agent"])
def agent_stats(agent: str = Depends(current_agent)):
    return tickets.stats()
