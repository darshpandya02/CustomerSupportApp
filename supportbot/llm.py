"""Vercel AI Gateway client (OpenAI-compatible Chat Completions).

Authentication, in order: an explicit AI_GATEWAY_API_KEY, the per-request
OIDC token Vercel forwards in the `x-vercel-oidc-token` header, then the
VERCEL_OIDC_TOKEN environment variable (local `vercel env pull`).

When the Gateway refuses service for account reasons (for example
`customer_verification_required` before a card is on file), the client opens
a short circuit breaker so each chat request does not pay for a failing
round-trip. Callers fall back to retrieval-only answers.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Protocol

import httpx

from .config import GATEWAY_URL

ACCOUNT_ERRORS = {"customer_verification_required", "no_credentials", "insufficient_funds", "budget_exceeded",
                  "authentication_error", "forbidden"}


class GatewayError(Exception):
    def __init__(self, kind: str, message: str, status: int | None = None):
        super().__init__(f"{kind}: {message}")
        self.kind, self.message, self.status = kind, message, status


@dataclass
class Completion:
    text: str
    model: str
    usage: dict
    latency_ms: float


class ChatClient(Protocol):
    def chat(self, model: str, messages: list[dict], max_tokens: int, temperature: float = 0.1,
             token: str | None = None) -> Completion: ...


class GatewayClient:
    def __init__(self, base_url: str = GATEWAY_URL, timeout: float = 25.0, breaker_seconds: float = 300.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.breaker_seconds = breaker_seconds
        self._open_until = 0.0
        self._last_error: GatewayError | None = None
        self._lock = threading.Lock()
        self._http = httpx.Client(timeout=timeout)

    @property
    def status(self) -> dict:
        open_ = time.time() < self._open_until
        return {"available": not open_,
                "last_error": self._last_error.kind if self._last_error else None,
                "retry_in_s": max(0, int(self._open_until - time.time())) if open_ else 0}

    def _credential(self, token: str | None) -> str:
        cred = os.environ.get("AI_GATEWAY_API_KEY") or token or os.environ.get("VERCEL_OIDC_TOKEN")
        if not cred:
            raise GatewayError("no_credentials", "no AI Gateway API key or OIDC token available")
        return cred

    def chat(self, model: str, messages: list[dict], max_tokens: int, temperature: float = 0.1,
             token: str | None = None) -> Completion:
        if time.time() < self._open_until and self._last_error:
            raise self._last_error
        t0 = time.perf_counter()
        try:
            r = self._http.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self._credential(token)}"},
                json={"model": model, "messages": messages, "max_tokens": max_tokens,
                      "temperature": temperature},
            )
        except httpx.HTTPError as exc:
            raise GatewayError("network", str(exc)) from exc
        if r.status_code >= 400:
            try:
                err = r.json().get("error", {})
            except ValueError:
                err = {}
            e = GatewayError(err.get("type") or f"http_{r.status_code}", err.get("message") or r.text[:200],
                             r.status_code)
            if e.kind in ACCOUNT_ERRORS or r.status_code in (401, 402, 403):
                with self._lock:
                    self._open_until = time.time() + self.breaker_seconds
                    self._last_error = e
            raise e
        body = r.json()
        with self._lock:
            self._open_until, self._last_error = 0.0, None
        return Completion(
            text=(body["choices"][0]["message"].get("content") or "").strip(),
            model=body.get("model", model),
            usage=body.get("usage", {}),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )


_client: ChatClient | None = None


def get_client() -> ChatClient:
    global _client
    if _client is None:
        _client = GatewayClient()
    return _client


def set_client(client: ChatClient | None) -> None:
    """Swap the client (tests inject a fake)."""
    global _client
    _client = client
