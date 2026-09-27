import uuid

from supportbot import api, rag


def ask(client, message, cid=None, key=None):
    headers = {"Idempotency-Key": key or uuid.uuid4().hex}
    body = {"message": message}
    if cid:
        body["conversation_id"] = cid
    return client.post("/api/chat", json=body, headers=headers)


def login(client):
    r = client.post("/api/agent/login", json={"password": "test-password"})
    assert r.status_code == 200
    return {"Authorization": f"Bearer {r.json()['token']}"}


# --------------------------------------------------------------- chat

def test_chat_retrieval_only_when_gateway_unavailable(client, unavailable_llm):
    r = ask(client, "How do I roll back a file to an older version?")
    assert r.status_code == 200
    j = r.json()
    assert j["mode"] == "retrieval_only" and j["notice"] == rag.FALLBACK_NOTICE
    assert len(j["citations"]) == 3 and all(c["url"].startswith("https://") for c in j["citations"])
    assert "X-RateLimit-Remaining" in r.headers


def test_chat_llm_answer_with_citations(client, fake_llm):
    fake_llm.answer = "Use the Versions tab in the Details sidebar [1]."
    j = ask(client, "How do I roll back a file to an older version?").json()
    assert j["mode"] == "llm" and j["citations"][0]["n"] == 1


def test_chat_refuses_out_of_scope(client, fake_llm):
    j = ask(client, "Write me a short poem about cats.").json()
    assert j["mode"] == "refusal" and j["citations"] == []
    assert fake_llm.calls == []


def test_chat_idempotent_replay_and_conflict(client, fake_llm):
    key = uuid.uuid4().hex
    a = ask(client, "How do I enable two-factor authentication?", key=key)
    b = ask(client, "How do I enable two-factor authentication?", key=key)
    assert b.headers.get("Idempotent-Replayed") == "true"
    assert a.json()["message_id"] == b.json()["message_id"]
    gen_calls = [c for c in fake_llm.calls if c["model"] == rag.CHAT_MODEL]
    assert len(gen_calls) == 1  # the retry did not call the LLM again
    c = ask(client, "something else entirely", key=key)
    assert c.status_code == 409


def test_chat_validation(client, fake_llm):
    assert client.post("/api/chat", json={"message": ""}).status_code == 422
    assert client.post("/api/chat", json={"message": "   "}).status_code == 422
    assert client.post("/api/chat", json={"message": "x" * 1001}).status_code == 422
    assert client.post("/api/chat", json={"message": "hi", "conversation_id": "nope"}).status_code == 422
    assert client.post("/api/chat", json={"message": "hi"}, headers={"Idempotency-Key": "bad key!"}).status_code == 422


def test_chat_rate_limit(client, fake_llm, monkeypatch):
    monkeypatch.setattr(api, "CHAT_LIMIT_PER_IP", 2)
    codes = [ask(client, "How do I enable two-factor authentication?").status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    r = ask(client, "How do I enable two-factor authentication?")
    assert r.status_code == 429 and int(r.headers["Retry-After"]) > 0


def test_global_daily_cap_degrades_to_retrieval_only(client, fake_llm, monkeypatch):
    monkeypatch.setattr(api, "CHAT_LIMIT_GLOBAL_DAY", 0)
    j = ask(client, "How do I roll back a file to an older version?").json()
    assert j["mode"] == "retrieval_only"
    assert not [c for c in fake_llm.calls if c["model"] == rag.CHAT_MODEL]


def test_multi_turn_memory_summarises_old_turns(client, fake_llm):
    def script(model, messages):
        if "running summary" in messages[0]["content"]:
            return "User is asking about Talk features."
        if model == rag.SUMMARY_MODEL:
            return messages[-1]["content"].split("user (latest): ")[-1]
        return "Answer [1]."
    fake_llm.answer = script
    cid = None
    for q in ["How do I record a Talk call?", "Can guests join a Talk call?", "How long can I edit messages?",
              "Can messages expire automatically?", "How do I pin a message?", "Can I schedule a message?"]:
        j = ask(client, q, cid).json()
        cid = j["conversation_id"]
    from supportbot import db
    with db.conn() as c:
        summary, upto = c.execute("SELECT summary, summarized_upto FROM support.conversations WHERE id=%s",
                                  (cid,)).fetchone()
    assert summary == "User is asking about Talk features." and upto >= 2
    last = [c for c in fake_llm.calls if c["model"] == rag.CHAT_MODEL][-1]["messages"]
    assert any("Summary of earlier conversation" in m["content"] for m in last)
    assert sum(1 for m in last if m["role"] == "assistant") <= 4


# --------------------------------------------------------------- tickets

def test_escalation_creates_routed_ticket_with_transcript(client, unavailable_llm):
    cid = ask(client, "I was charged twice for my subscription this month").json()["conversation_id"]
    key = uuid.uuid4().hex
    body = {"conversation_id": cid, "description": "Please refund the duplicate charge", "email": "a@b.co"}
    r = client.post("/api/tickets", json=body, headers={"Idempotency-Key": key})
    assert r.status_code == 201
    t = r.json()
    assert t["ref"].startswith("CS-") and t["queue"] == "billing" and t["priority"] == "high"
    assert t["transcript_messages"] == 2 and t["routing"]["method"] in ("rule", "classifier")
    again = client.post("/api/tickets", json=body, headers={"Idempotency-Key": key})
    assert again.status_code == 201 and again.json()["ref"] == t["ref"]
    assert again.headers.get("Idempotent-Replayed") == "true"

    track = client.get(f"/api/tickets/{t['ref']}", params={"token": t["tracking_token"]})
    assert track.status_code == 200 and track.json()["status"] == "open"
    assert client.get(f"/api/tickets/{t['ref']}", params={"token": "wrong-token-123"}).status_code == 404

    h = login(client)
    items = client.get("/api/agent/tickets", params={"queue": "billing"}, headers=h).json()["items"]
    detail = client.get(f"/api/agent/tickets/{items[0]['id']}", headers=h).json()
    assert detail["transcript"][0]["content"].startswith("I was charged twice")


def test_ticket_requires_idempotency_key_and_valid_input(client):
    assert client.post("/api/tickets", json={"description": "help me please"}).status_code == 422
    h = {"Idempotency-Key": uuid.uuid4().hex}
    assert client.post("/api/tickets", json={"description": "short"}, headers=h).status_code == 422
    assert client.post("/api/tickets", json={"description": "long enough text", "email": "not-an-email"},
                       headers=h).status_code == 422
    r = client.post("/api/tickets", json={"description": "long enough text",
                                          "conversation_id": str(uuid.uuid4())}, headers=h)
    assert r.status_code == 404


def test_agent_auth_required(client):
    assert client.get("/api/agent/tickets").status_code == 401
    assert client.post("/api/agent/login", json={"password": "nope"}).status_code == 401
    assert client.get("/api/agent/tickets", headers={"Authorization": "Bearer forged.sig"}).status_code == 401


def test_status_workflow_and_audit_trail(client):
    t = client.post("/api/tickets", json={"description": "The desktop client is stuck syncing for days"},
                    headers={"Idempotency-Key": uuid.uuid4().hex}).json()
    assert t["queue"] == "technical"
    h = login(client)
    tid = client.get("/api/agent/tickets", headers=h).json()["items"][0]["id"]
    ok = client.patch(f"/api/agent/tickets/{tid}", json={"status": "in_progress", "assignee": "sam",
                                                          "note": "looking"}, headers=h)
    assert ok.status_code == 200 and ok.json()["status"] == "in_progress"
    assert client.patch(f"/api/agent/tickets/{tid}", json={"status": "resolved"}, headers=h).status_code == 200
    assert client.patch(f"/api/agent/tickets/{tid}", json={"status": "closed"}, headers=h).status_code == 200
    bad = client.patch(f"/api/agent/tickets/{tid}", json={"status": "open"}, headers=h)
    assert bad.status_code == 409
    d = client.get(f"/api/agent/tickets/{tid}", headers=h).json()
    kinds = [e["kind"] for e in d["events"]]
    assert kinds == ["created", "status", "assignee", "note", "status", "status"]
    assert d["resolved_at"] and d["first_response_at"]
    assert client.patch(f"/api/agent/tickets/{tid}", json={"status": "bogus"}, headers=h).status_code == 422
    stats = client.get("/api/agent/stats", headers=h).json()
    assert stats["by_queue_status"][0]["queue"] == "technical"


def test_openapi_and_health(client):
    spec = client.get("/api/openapi.json").json()
    assert "/api/chat" in spec["paths"] and "/api/agent/tickets/{ticket_id}" in spec["paths"]
    h = client.get("/api/health").json()
    assert h["database"] is True and h["chunks"] > 400
