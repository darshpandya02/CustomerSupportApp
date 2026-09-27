from supportbot import routing


def test_rule_routes_clear_billing_text():
    r = routing.route("Please send me the invoice for last month")
    assert r.queue == "billing" and r.method == "rule"


def test_ambiguous_text_goes_to_classifier():
    r = routing.route("my payment page shows a sync error")  # billing + technical rules both fire
    assert r.method == "classifier"
    assert set(r.scores) == {"billing", "technical", "account"}
    assert abs(sum(r.scores.values()) - 1) < 1e-3


def test_classifier_handles_text_without_keywords():
    assert routing.route("Someone else is using my identity to sign my documents").queue == "account"


def test_priority_rules():
    assert routing.priority_for("I think I was hacked") == "urgent"
    assert routing.priority_for("I am locked out of my account") == "high"
    assert routing.priority_for("just curious about themes") == "low"
    assert routing.priority_for("How do I share a folder?") == "normal"
