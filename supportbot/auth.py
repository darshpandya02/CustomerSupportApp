"""Agent dashboard auth: one shared demo password -> signed, expiring session token."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time

SESSION_TTL = 8 * 3600
COOKIE = "agent_session"


def _secret() -> bytes:
    s = os.environ.get("SESSION_SECRET")
    if not s:
        if os.environ.get("VERCEL_ENV") == "production":
            raise RuntimeError("SESSION_SECRET must be set in production")
        s = "dev-only-secret"
    return s.encode()


def check_password(candidate: str) -> bool:
    expected = os.environ.get("AGENT_PASSWORD", "")
    return bool(expected) and secrets.compare_digest(candidate.encode(), expected.encode())


def issue(agent: str) -> str:
    payload = base64.urlsafe_b64encode(json.dumps({"sub": agent, "exp": int(time.time()) + SESSION_TTL})
                                       .encode()).decode().rstrip("=")
    sig = hmac.new(_secret(), payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}.{sig}"


def verify(token: str | None) -> str | None:
    if not token or "." not in token:
        return None
    payload, sig = token.rsplit(".", 1)
    good = hmac.new(_secret(), payload.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, good):
        return None
    try:
        data = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    except ValueError:
        return None
    if data.get("exp", 0) < time.time():
        return None
    return data.get("sub")
