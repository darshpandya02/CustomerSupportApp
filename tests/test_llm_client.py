import httpx
import pytest

from supportbot.llm import GatewayClient, GatewayError


def make(handler):
    c = GatewayClient(base_url="https://gw.test/v1")
    c._http = httpx.Client(transport=httpx.MockTransport(handler))
    return c


def test_parses_completion_and_sends_token(monkeypatch):
    monkeypatch.delenv("AI_GATEWAY_API_KEY", raising=False)
    seen = {}

    def handler(req):
        seen["auth"] = req.headers["authorization"]
        seen["body"] = req.read()
        return httpx.Response(200, json={"model": "meta/llama-4-maverick",
                                          "choices": [{"message": {"content": " hi [1] "}}],
                                          "usage": {"total_tokens": 3}})

    out = make(handler).chat("meta/llama-4-maverick", [{"role": "user", "content": "x"}], 10, token="oidc123")
    assert out.text == "hi [1]" and out.usage == {"total_tokens": 3}
    assert seen["auth"] == "Bearer oidc123"
    assert b'"max_tokens":10' in seen["body"]


def test_card_required_opens_breaker(monkeypatch):
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "k")
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(403, json={"error": {"type": "customer_verification_required",
                                                    "message": "add a card"}})

    c = make(handler)
    with pytest.raises(GatewayError) as e:
        c.chat("m", [], 5)
    assert e.value.kind == "customer_verification_required"
    with pytest.raises(GatewayError):
        c.chat("m", [], 5)
    assert len(calls) == 1  # second call short-circuited
    assert c.status["available"] is False


def test_server_error_does_not_open_breaker(monkeypatch):
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "k")
    c = make(lambda req: httpx.Response(502, text="bad gateway"))
    with pytest.raises(GatewayError):
        c.chat("m", [], 5)
    assert c.status["available"] is True


def test_missing_credentials(monkeypatch):
    monkeypatch.delenv("AI_GATEWAY_API_KEY", raising=False)
    monkeypatch.delenv("VERCEL_OIDC_TOKEN", raising=False)
    with pytest.raises(GatewayError) as e:
        make(lambda r: httpx.Response(200)).chat("m", [], 5)
    assert e.value.kind == "no_credentials"
