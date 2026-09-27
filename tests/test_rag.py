"""The generation path, exercised with a mocked Gateway client."""

from supportbot import rag
from supportbot.llm import GatewayError


def test_llm_answer_keeps_only_cited_sources(fake_llm):
    fake_llm.answer = "Go to Deleted files and click restore [1]. Items stay 30 days [3]. Bogus [9]."
    a = rag.answer("I deleted a folder by mistake, how do I get it back?")
    assert a.mode == "llm"
    assert [c["n"] for c in a.citations] == [1, 3]
    assert a.citations[0]["url"].startswith("https://docs.nextcloud.com/")
    system = fake_llm.calls[-1]["messages"][0]["content"]
    assert "NOT_IN_DOCS" in system and "cite" in system.lower()


def test_prompt_contains_numbered_excerpts(fake_llm):
    rag.answer("How do I enable two-factor authentication?")
    user_msg = fake_llm.calls[-1]["messages"][-1]["content"]
    assert "[1] (" in user_msg and "Question: How do I enable two-factor authentication?" in user_msg


def test_model_refusal_token_becomes_refusal(fake_llm):
    fake_llm.answer = "NOT_IN_DOCS"
    a = rag.answer("How do I enable two-factor authentication?")
    assert a.mode == "refusal" and a.citations == []


def test_out_of_scope_is_refused_without_calling_the_llm(fake_llm):
    a = rag.answer("Who won the 2022 FIFA World Cup?")
    assert a.mode == "refusal"
    assert fake_llm.calls == []


def test_gateway_unavailable_falls_back_to_retrieval_only(unavailable_llm):
    a = rag.answer("How long are deleted files kept in the trash?")
    assert a.mode == "retrieval_only"
    assert a.notice == rag.FALLBACK_NOTICE
    assert a.text.startswith(rag.FALLBACK_NOTICE)
    assert len(a.citations) == 3
    assert a.llm_error == "customer_verification_required"


def test_transient_gateway_error_also_falls_back(fake_llm):
    fake_llm.fail = GatewayError("network", "timeout")
    a = rag.answer("How long are deleted files kept in the trash?")
    assert a.mode == "retrieval_only"


def test_follow_up_is_condensed_with_history(fake_llm):
    def script(model, messages):
        if model == rag.SUMMARY_MODEL:
            return "Can guests join a Talk call without an account?"
        return "Answer [1]."
    fake_llm.answer = script
    mem = rag.Memory(turns=[{"role": "user", "content": "How long can I edit a Talk message?"},
                            {"role": "assistant", "content": "Up to 6 hours [1]."}])
    a = rag.answer("can guests join it?", mem)
    assert a.query == "Can guests join a Talk call without an account?"
    gen = [c for c in fake_llm.calls if c["model"] == rag.CHAT_MODEL][-1]["messages"]
    assert any(m["role"] == "assistant" and "6 hours" in m["content"] for m in gen)


def test_heuristic_condense_without_llm():
    mem = rag.Memory(turns=[{"role": "user", "content": "How do I set up 2FA?"}])
    assert rag.heuristic_condense("what about on iOS?", mem) == "How do I set up 2FA? what about on iOS?"
    assert rag.heuristic_condense("How do I share a calendar with my team members?", mem).startswith("How do I share")


def test_summarize_uses_llm_then_falls_back(fake_llm):
    fake_llm.answer = "User is fixing 2FA on Android."
    assert rag.summarize("", [{"role": "user", "content": "2FA help"}], None) == "User is fixing 2FA on Android."
    fake_llm.fail = GatewayError("customer_verification_required", "x")
    s = rag.summarize("", [{"role": "user", "content": "2FA help"}], None)
    assert "User asked: 2FA help" in s


def test_heuristic_rewrite_does_not_hijack_a_new_topic(unavailable_llm):
    mem = rag.Memory(turns=[{"role": "user", "content": "I deleted a folder by mistake. How do I get it back?"},
                            {"role": "assistant", "content": "..."}])
    a = rag.answer("My 2FA code keeps getting rejected even though I type it correctly. Why?", mem)
    assert a.query.startswith("My 2FA code")
    assert all("deleted" not in c["url"] for c in a.citations)
