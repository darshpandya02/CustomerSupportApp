"""Cost and abuse controls: Postgres-backed rate limits and idempotency keys."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass

from psycopg.types.json import Jsonb

from . import db


def client_hash(ip: str) -> str:
    """Hash client IPs so the database never stores raw addresses."""
    salt = os.environ.get("SESSION_SECRET", "dev-salt").encode()
    return hmac.new(salt, ip.encode(), hashlib.sha256).hexdigest()[:24]


@dataclass
class Limit:
    allowed: bool
    limit: int
    remaining: int
    reset_in: int  # seconds until the window resets

    def headers(self) -> dict[str, str]:
        h = {"X-RateLimit-Limit": str(self.limit), "X-RateLimit-Remaining": str(max(self.remaining, 0)),
             "X-RateLimit-Reset": str(self.reset_in)}
        if not self.allowed:
            h["Retry-After"] = str(self.reset_in)
        return h


def hit(bucket: str, limit: int, window_seconds: int, cost: int = 1) -> Limit:
    """Fixed-window counter. Atomic via INSERT ... ON CONFLICT DO UPDATE."""
    now = int(time.time())
    start = now - now % window_seconds
    with db.conn() as c:
        count = c.execute(
            """
            INSERT INTO support.rate_limits (bucket, window_start, count)
            VALUES (%s, to_timestamp(%s), %s)
            ON CONFLICT (bucket, window_start) DO UPDATE SET count = support.rate_limits.count + EXCLUDED.count
            RETURNING count
            """,
            (bucket, start, cost),
        ).fetchone()[0]
        if now % 50 == 0:  # opportunistic cleanup of old windows
            c.execute("DELETE FROM support.rate_limits WHERE window_start < now() - interval '2 days'")
    return Limit(count <= limit, limit, limit - count, start + window_seconds - now)


def request_hash(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


class IdempotencyConflict(Exception):
    """Same key reused with a different request body."""


class IdempotencyInFlight(Exception):
    """The first request with this key is still being processed."""


def begin(scope: str, key: str, payload: dict) -> tuple[int, dict] | None:
    """Claim a key. Returns a stored (status, body) to replay, or None to proceed."""
    h = request_hash(payload)
    with db.conn() as c:
        row = c.execute(
            """
            INSERT INTO support.idempotency_keys (scope, key, request_hash)
            VALUES (%s, %s, %s)
            ON CONFLICT (scope, key) DO NOTHING
            RETURNING key
            """,
            (scope, key, h),
        ).fetchone()
        if row:
            return None
        stored = c.execute(
            "SELECT request_hash, status_code, response FROM support.idempotency_keys WHERE scope=%s AND key=%s",
            (scope, key),
        ).fetchone()
    if stored[0] != h:
        raise IdempotencyConflict
    if stored[1] is None:
        raise IdempotencyInFlight
    return stored[1], stored[2]


def finish(scope: str, key: str, status: int, body: dict) -> None:
    with db.conn() as c:
        c.execute(
            "UPDATE support.idempotency_keys SET status_code=%s, response=%s WHERE scope=%s AND key=%s",
            (status, Jsonb(body), scope, key),
        )


def release(scope: str, key: str) -> None:
    """Drop a claimed key when processing failed, so the client can retry."""
    with db.conn() as c:
        c.execute("DELETE FROM support.idempotency_keys WHERE scope=%s AND key=%s AND status_code IS NULL",
                  (scope, key))
