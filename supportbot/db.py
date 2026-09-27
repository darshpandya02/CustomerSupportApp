"""Postgres access. Every object lives in the `support` schema.

The production database is a shared Neon instance, so nothing here reads or
writes outside `support`. All statements are schema-qualified.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from functools import lru_cache

from psycopg.rows import tuple_row

SCHEMA = "support"

SCHEMA_SQL = """
CREATE SCHEMA IF NOT EXISTS support;

CREATE TABLE IF NOT EXISTS support.conversations (
    id            uuid PRIMARY KEY,
    created_at    timestamptz NOT NULL DEFAULT now(),
    updated_at    timestamptz NOT NULL DEFAULT now(),
    summary       text NOT NULL DEFAULT '',
    summarized_upto integer NOT NULL DEFAULT 0,
    client_hash   text
);

CREATE TABLE IF NOT EXISTS support.messages (
    id              bigserial PRIMARY KEY,
    conversation_id uuid NOT NULL REFERENCES support.conversations(id) ON DELETE CASCADE,
    seq             integer NOT NULL,
    role            text NOT NULL CHECK (role IN ('user', 'assistant')),
    content         text NOT NULL,
    mode            text CHECK (mode IN ('llm', 'retrieval_only', 'refusal')),
    citations       jsonb NOT NULL DEFAULT '[]'::jsonb,
    meta            jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at      timestamptz NOT NULL DEFAULT now(),
    UNIQUE (conversation_id, seq)
);

CREATE SEQUENCE IF NOT EXISTS support.ticket_ref_seq START 1001;

CREATE TABLE IF NOT EXISTS support.tickets (
    id                 bigserial PRIMARY KEY,
    ref                text NOT NULL UNIQUE DEFAULT ('CS-' || nextval('support.ticket_ref_seq')),
    tracking_token     text NOT NULL,
    conversation_id    uuid REFERENCES support.conversations(id) ON DELETE SET NULL,
    subject            text NOT NULL,
    description        text NOT NULL,
    customer_name      text,
    customer_email     text,
    queue              text NOT NULL CHECK (queue IN ('billing', 'technical', 'account')),
    priority           text NOT NULL CHECK (priority IN ('low', 'normal', 'high', 'urgent')),
    status             text NOT NULL DEFAULT 'open'
                       CHECK (status IN ('open', 'in_progress', 'waiting_on_customer', 'resolved', 'closed')),
    assignee           text,
    routing            jsonb NOT NULL,
    transcript         jsonb NOT NULL DEFAULT '[]'::jsonb,
    first_response_due timestamptz NOT NULL,
    first_response_at  timestamptz,
    resolved_at        timestamptz,
    created_at         timestamptz NOT NULL DEFAULT now(),
    updated_at         timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS tickets_queue_status_idx ON support.tickets (queue, status, created_at DESC);
CREATE INDEX IF NOT EXISTS tickets_created_idx ON support.tickets (created_at DESC);

CREATE TABLE IF NOT EXISTS support.ticket_events (
    id         bigserial PRIMARY KEY,
    ticket_id  bigint NOT NULL REFERENCES support.tickets(id) ON DELETE CASCADE,
    at         timestamptz NOT NULL DEFAULT now(),
    actor      text NOT NULL,
    kind       text NOT NULL CHECK (kind IN ('created', 'status', 'priority', 'queue', 'assignee', 'note')),
    from_value text,
    to_value   text,
    note       text,
    internal   boolean NOT NULL DEFAULT false
);
CREATE INDEX IF NOT EXISTS ticket_events_ticket_idx ON support.ticket_events (ticket_id, at);

CREATE TABLE IF NOT EXISTS support.idempotency_keys (
    scope        text NOT NULL,
    key          text NOT NULL,
    request_hash text NOT NULL,
    status_code  integer,
    response     jsonb,
    created_at   timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (scope, key)
);

CREATE TABLE IF NOT EXISTS support.rate_limits (
    bucket       text NOT NULL,
    window_start timestamptz NOT NULL,
    count        integer NOT NULL DEFAULT 0,
    PRIMARY KEY (bucket, window_start)
);
"""


def database_url() -> str | None:
    for key in ("DATABASE_URL", "POSTGRES_URL"):
        if os.environ.get(key):
            return os.environ[key]
    return None


@lru_cache(maxsize=1)
def _pool():
    from psycopg_pool import ConnectionPool

    url = database_url()
    if not url:
        raise RuntimeError("DATABASE_URL is not configured")
    # Small pool: serverless instances handle few concurrent requests, and the
    # shared Neon database has a connection budget.
    return ConnectionPool(url, min_size=0, max_size=4, open=True, timeout=10,
                          kwargs={"autocommit": True, "connect_timeout": 5},
                          check=ConnectionPool.check_connection)


@contextmanager
def conn():
    with _pool().connection() as c:
        c.row_factory = tuple_row  # callers may switch to dict_row; reset for the next borrower
        yield c


@contextmanager
def tx():
    with _pool().connection() as c:
        c.row_factory = tuple_row
        with c.transaction():
            yield c


def migrate() -> None:
    with conn() as c:
        c.execute(SCHEMA_SQL)


def reset_pool() -> None:
    if _pool.cache_info().currsize:
        _pool().close()
    _pool.cache_clear()
