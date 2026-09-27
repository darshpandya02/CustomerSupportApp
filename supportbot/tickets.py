"""Conversation persistence and the ticket lifecycle (schema `support`)."""

from __future__ import annotations

import secrets
import uuid
from datetime import datetime, timezone

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from . import db
from .config import HISTORY_TURNS_VERBATIM
from .rag import Memory
from .routing import SLA, route

# Allowed status transitions (the state machine agents work through).
TRANSITIONS: dict[str, set[str]] = {
    "open": {"in_progress", "waiting_on_customer", "resolved", "closed"},
    "in_progress": {"open", "waiting_on_customer", "resolved"},
    "waiting_on_customer": {"in_progress", "resolved", "closed"},
    "resolved": {"open", "closed"},
    "closed": set(),
}


class NotFound(Exception):
    pass


class InvalidTransition(Exception):
    def __init__(self, current: str, target: str):
        super().__init__(f"cannot move a ticket from {current} to {target}")
        self.current, self.target = current, target


# ---------------------------------------------------------------- conversations

def ensure_conversation(conversation_id: str | None, client: str) -> str:
    cid = conversation_id or str(uuid.uuid4())
    with db.conn() as c:
        c.execute(
            "INSERT INTO support.conversations (id, client_hash) VALUES (%s, %s) ON CONFLICT (id) DO NOTHING",
            (cid, client),
        )
    return cid


def load_memory(cid: str) -> tuple[Memory, int, list[dict]]:
    """Return memory for the prompt, the next seq number, and turns due for summarising."""
    with db.conn() as c:
        c.row_factory = dict_row
        conv = c.execute("SELECT summary, summarized_upto FROM support.conversations WHERE id=%s",
                         (cid,)).fetchone()
        rows = c.execute(
            "SELECT seq, role, content FROM support.messages WHERE conversation_id=%s ORDER BY seq", (cid,)
        ).fetchall()
    upto = conv["summarized_upto"] if conv else 0
    turns = [{"role": r["role"], "content": r["content"], "seq": r["seq"]} for r in rows]
    keep = HISTORY_TURNS_VERBATIM * 2  # user+assistant messages kept verbatim
    stale = [t for t in turns[:-keep] if t["seq"] > upto] if len(turns) > keep else []
    recent = turns[-keep:]
    next_seq = (turns[-1]["seq"] + 1) if turns else 1
    return Memory(conv["summary"] if conv else "", recent), next_seq, stale


def save_summary(cid: str, summary: str, upto: int) -> None:
    with db.conn() as c:
        c.execute("UPDATE support.conversations SET summary=%s, summarized_upto=%s WHERE id=%s",
                  (summary, upto, cid))


def save_turn(cid: str, seq: int, user_text: str, answer_text: str, mode: str, citations: list[dict],
              meta: dict) -> int:
    with db.tx() as c:
        c.execute(
            "INSERT INTO support.messages (conversation_id, seq, role, content) VALUES (%s, %s, 'user', %s)",
            (cid, seq, user_text),
        )
        mid = c.execute(
            "INSERT INTO support.messages (conversation_id, seq, role, content, mode, citations, meta) "
            "VALUES (%s, %s, 'assistant', %s, %s, %s, %s) RETURNING id",
            (cid, seq + 1, answer_text, mode, Jsonb(citations), Jsonb(meta)),
        ).fetchone()[0]
        c.execute("UPDATE support.conversations SET updated_at=now() WHERE id=%s", (cid,))
    return mid


def transcript(cid: str) -> list[dict]:
    with db.conn() as c:
        c.row_factory = dict_row
        rows = c.execute(
            "SELECT role, content, mode, citations, created_at FROM support.messages "
            "WHERE conversation_id=%s ORDER BY seq", (cid,),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"], "mode": r["mode"],
             "citations": [{"n": x["n"], "url": x["url"], "section": x["section"]} for x in r["citations"]],
             "at": r["created_at"].isoformat()} for r in rows]


# ---------------------------------------------------------------- tickets

PUBLIC_FIELDS = "ref, subject, queue, priority, status, created_at, updated_at, first_response_due, resolved_at"


def _iso(row: dict) -> dict:
    return {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in row.items()}


def create_ticket(*, description: str, subject: str | None, conversation_id: str | None,
                  name: str | None, email: str | None) -> dict:
    convo = transcript(conversation_id) if conversation_id else []
    if conversation_id and not convo:
        raise NotFound("conversation not found or empty")
    # Route on what the customer wrote in the escalation form; only borrow the last chat
    # message when the form text is too short to classify on its own.
    text = f"{subject or ''} {description}".strip()
    last_user = next((t["content"] for t in reversed(convo) if t["role"] == "user"), "")
    if len(text.split()) < 6 and last_user:
        text = f"{text} {last_user}"
    r = route(text)
    subject = (subject or description.split("\n")[0])[:120]
    token = secrets.token_urlsafe(18)
    due = datetime.now(timezone.utc) + SLA[r.priority]
    with db.tx() as c:
        c.row_factory = dict_row
        t = c.execute(
            f"""
            INSERT INTO support.tickets (tracking_token, conversation_id, subject, description, customer_name,
                customer_email, queue, priority, routing, transcript, first_response_due)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id, tracking_token, {PUBLIC_FIELDS}
            """,
            (token, conversation_id, subject, description, name, email, r.queue, r.priority,
             Jsonb(r.as_dict()), Jsonb(convo), due),
        ).fetchone()
        c.execute(
            "INSERT INTO support.ticket_events (ticket_id, actor, kind, to_value, note) "
            "VALUES (%s, 'system', 'created', %s, %s)",
            (t["id"], f"{r.queue}/{r.priority}",
             f"routed by {r.method}" + (f" (matched '{r.rule}')" if r.rule else "")),
        )
    out = _iso(t)
    out.pop("id")
    out["routing"] = r.as_dict()
    out["transcript_messages"] = len(convo)
    return out


def public_ticket(ref: str, token: str) -> dict:
    with db.conn() as c:
        c.row_factory = dict_row
        t = c.execute(f"SELECT id, tracking_token, {PUBLIC_FIELDS} FROM support.tickets WHERE ref=%s",
                      (ref,)).fetchone()
        if not t or not secrets.compare_digest(t["tracking_token"], token):
            raise NotFound(ref)
        events = c.execute(
            "SELECT at, kind, to_value FROM support.ticket_events WHERE ticket_id=%s AND NOT internal "
            "AND kind IN ('created', 'status') ORDER BY at", (t["id"],),
        ).fetchall()
    out = _iso({k: v for k, v in t.items() if k not in ("id", "tracking_token")})
    out["history"] = [_iso(e) for e in events]
    return out


def list_tickets(queue: str | None, status: str | None, priority: str | None, q: str | None,
                 limit: int, offset: int) -> dict:
    where, args = [], []
    for col, val in (("queue", queue), ("status", status), ("priority", priority)):
        if val:
            where.append(f"{col} = %s")
            args.append(val)
    if q:
        where.append("(subject ILIKE %s OR description ILIKE %s OR ref ILIKE %s)")
        args += [f"%{q}%"] * 3
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    with db.conn() as c:
        c.row_factory = dict_row
        total = c.execute(f"SELECT count(*) AS n FROM support.tickets {clause}", args).fetchone()["n"]
        rows = c.execute(
            f"""SELECT id, {PUBLIC_FIELDS}, assignee, customer_email, routing->>'method' AS routed_by,
                       (status NOT IN ('resolved','closed') AND first_response_at IS NULL
                        AND first_response_due < now()) AS sla_breached
                FROM support.tickets {clause}
                ORDER BY CASE priority WHEN 'urgent' THEN 0 WHEN 'high' THEN 1 WHEN 'normal' THEN 2 ELSE 3 END,
                         created_at DESC
                LIMIT %s OFFSET %s""",
            args + [limit, offset],
        ).fetchall()
    return {"total": total, "items": [_iso(r) for r in rows]}


def get_ticket(ticket_id: int) -> dict:
    with db.conn() as c:
        c.row_factory = dict_row
        t = c.execute("SELECT * FROM support.tickets WHERE id=%s", (ticket_id,)).fetchone()
        if not t:
            raise NotFound(str(ticket_id))
        events = c.execute("SELECT * FROM support.ticket_events WHERE ticket_id=%s ORDER BY at, id",
                           (ticket_id,)).fetchall()
    t.pop("tracking_token")
    out = _iso(t)
    out["events"] = [_iso(e) for e in events]
    out["allowed_transitions"] = sorted(TRANSITIONS[t["status"]])
    return out


def update_ticket(ticket_id: int, actor: str, *, status: str | None = None, priority: str | None = None,
                  queue: str | None = None, assignee: str | None = None, note: str | None = None) -> dict:
    with db.tx() as c:
        c.row_factory = dict_row
        t = c.execute("SELECT id, status, priority, queue, assignee, first_response_at FROM support.tickets "
                      "WHERE id=%s FOR UPDATE", (ticket_id,)).fetchone()
        if not t:
            raise NotFound(str(ticket_id))
        events = []
        sets: list[str] = []
        args: list = []
        if status and status != t["status"]:
            if status not in TRANSITIONS[t["status"]]:
                raise InvalidTransition(t["status"], status)
            sets.append("status=%s")
            args.append(status)
            if status in ("resolved", "closed"):
                sets.append("resolved_at=COALESCE(resolved_at, now())")
            elif t["status"] in ("resolved",):
                sets.append("resolved_at=NULL")
            events.append(("status", t["status"], status, None, False))
        for field_, val in (("priority", priority), ("queue", queue), ("assignee", assignee)):
            if val is not None and val != t[field_]:
                sets.append(f"{field_}=%s")
                args.append(val or None)
                events.append((field_, t[field_], val or None, None, False))
        if note:
            events.append(("note", None, None, note, True))
        if not events:
            return get_ticket(ticket_id)
        if t["first_response_at"] is None:
            sets.append("first_response_at=now()")
        sets.append("updated_at=now()")
        c.execute(f"UPDATE support.tickets SET {', '.join(sets)} WHERE id=%s", args + [ticket_id])
        for kind, frm, to, nt, internal in events:
            c.execute(
                "INSERT INTO support.ticket_events (ticket_id, actor, kind, from_value, to_value, note, internal) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)", (ticket_id, actor, kind, frm, to, nt, internal),
            )
    return get_ticket(ticket_id)


def stats() -> dict:
    with db.conn() as c:
        c.row_factory = dict_row
        by = c.execute(
            """SELECT queue, status, count(*) AS n FROM support.tickets GROUP BY queue, status"""
        ).fetchall()
        breach = c.execute(
            """SELECT count(*) AS n FROM support.tickets WHERE status NOT IN ('resolved','closed')
               AND first_response_at IS NULL AND first_response_due < now()"""
        ).fetchone()["n"]
        methods = c.execute(
            "SELECT routing->>'method' AS method, count(*) AS n FROM support.tickets GROUP BY 1"
        ).fetchall()
        chats = c.execute(
            "SELECT mode, count(*) AS n FROM support.messages WHERE role='assistant' GROUP BY mode"
        ).fetchall()
    return {
        "by_queue_status": [dict(r) for r in by],
        "sla_breached_open": breach,
        "routed_by": {r["method"]: r["n"] for r in methods},
        "assistant_answers_by_mode": {r["mode"]: r["n"] for r in chats},
    }
