import os

import pytest

TEST_DB = os.environ.get("TEST_DATABASE_URL")
if TEST_DB:
    os.environ["DATABASE_URL"] = TEST_DB
os.environ.setdefault("AGENT_PASSWORD", "test-password")
os.environ.setdefault("SESSION_SECRET", "test-secret")

from supportbot import db, llm  # noqa: E402
from supportbot.llm import Completion, GatewayError  # noqa: E402


class FakeLLM:
    """Stands in for the AI Gateway. `script` maps model -> callable(messages) -> text or exception."""

    def __init__(self, answer="Open Deleted files in the Files app and restore it [1].", fail=None):
        self.answer, self.fail, self.calls = answer, fail, []

    def chat(self, model, messages, max_tokens, temperature=0.1, token=None):
        self.calls.append({"model": model, "messages": messages, "token": token})
        if self.fail:
            raise self.fail
        text = self.answer(model, messages) if callable(self.answer) else self.answer
        return Completion(text=text, model=model, usage={"prompt_tokens": 10, "completion_tokens": 5},
                          latency_ms=1.0)


@pytest.fixture
def fake_llm():
    fake = FakeLLM()
    llm.set_client(fake)
    yield fake
    llm.set_client(None)


@pytest.fixture
def unavailable_llm():
    fake = FakeLLM(fail=GatewayError("customer_verification_required", "card required", 403))
    llm.set_client(fake)
    yield fake
    llm.set_client(None)


needs_db = pytest.mark.skipif(not TEST_DB, reason="TEST_DATABASE_URL not set")


@pytest.fixture
def client():
    if not TEST_DB:
        pytest.skip("TEST_DATABASE_URL not set")
    from fastapi.testclient import TestClient

    from supportbot.api import app

    db.migrate()
    with db.conn() as c:
        c.execute("TRUNCATE support.ticket_events, support.tickets, support.messages, support.conversations, "
                  "support.idempotency_keys, support.rate_limits RESTART IDENTITY CASCADE")
    return TestClient(app)
