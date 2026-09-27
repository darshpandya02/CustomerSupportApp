"""Ticket routing: high-precision keyword rules first, then an embedding classifier.

Queues: billing, technical, account.
Classifier: multinomial logistic regression over the same bge-small sentence
embeddings the retriever uses (trained by scripts/train_router.py, weights in
artifacts/router.json), so routing adds no model to the bundle.
Priority is rule-based; SLA targets follow from it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import timedelta
from functools import lru_cache

import numpy as np

from .config import ARTIFACTS

QUEUES = ("billing", "technical", "account")
PRIORITIES = ("low", "normal", "high", "urgent")
SLA = {
    "urgent": timedelta(hours=1),
    "high": timedelta(hours=4),
    "normal": timedelta(hours=24),
    "low": timedelta(hours=72),
}

_R = lambda *pats: [re.compile(p, re.I) for p in pats]  # noqa: E731

QUEUE_RULES: dict[str, list[re.Pattern]] = {
    "billing": _R(
        r"\b(invoice|invoices|invoiced|receipt|refund|refunds|billing|billed|charged?|charges|prorat\w*)\b",
        r"\b(credit card|payment|payments|paypal|bank transfer|vat|coupon|promo code|pricing|price|discount)\b",
        r"\b(subscription|renewal|auto-renew\w*|downgrade|seats?|licen[cs]es?)\b",
    ),
    "account": _R(
        r"\b(forgot|reset) (my )?password\b",
        r"\b(locked out|account (is )?locked|unlock my account|hacked|compromised)\b",
        r"\b(2fa|two-factor|second factor|backup codes?|recovery codes?|passkey|security key)\b",
        r"\b(delete|close|deactivate) (my )?account\b",
        r"\b(log ?in|sign ?in)\b.*\b(can'?t|cannot|unable|fails?|error|loop\w*)\b",
        r"\b(can'?t|cannot|unable to)\b.*\b(log ?in|sign ?in)\b",
        r"\b(profile|display name|username|login name)\b",
    ),
    "technical": _R(
        r"\b(sync|syncing|synchroni[sz]\w*|webdav|caldav|carddav)\b",
        r"\b(error|crash\w*|bug|404|500|timeout|timed out|slow|stuck|blank page)\b",
        r"\b(upload\w*|download\w*)\b.*\b(fail\w*|won'?t|doesn'?t|not)\b",
        r"\b(desktop client|android app|ios app|mobile app|thunderbird|finder|screen ?shar\w*)\b",
    ),
}

PRIORITY_RULES: list[tuple[str, re.Pattern]] = [
    ("urgent", re.compile(r"\b(hacked|compromised|breach|data loss|lost (all|everything|a week|my (files|data|work))|disappeared|someone (else )?(logged|changed)|unknown sessions?)\b", re.I)),
    ("urgent", re.compile(r"\b(for everyone|whole (team|office|company)|all users|outage|production)\b.*\b(down|error|500|broken)\b", re.I)),
    ("high", re.compile(r"\b(locked out|can'?t log ?in|cannot log ?in|can'?t (access|open) (my )?account|no 2fa)\b", re.I)),
    ("high", re.compile(r"\b(charged (me )?(twice|three times|multiple times|\d+ times)|read-only|declined)\b", re.I)),
    ("high", re.compile(r"\b(for everyone|in our office|500 error|stuck .* (days?|hours)|for (two|several|\d+) days|keeps? (dropping|crashing)|drop every)\b", re.I)),
    ("low", re.compile(r"\b(just curious|wondering|feature request|nice to have|no rush|when you get a chance|what does .+ do)\b", re.I)),
]


@dataclass
class Route:
    queue: str
    priority: str
    method: str  # "rule" | "classifier"
    rule: str | None
    scores: dict[str, float]

    def as_dict(self) -> dict:
        return {"queue": self.queue, "priority": self.priority, "method": self.method, "rule": self.rule,
                "scores": self.scores}


def rule_queue(text: str) -> tuple[str | None, str | None]:
    """Return (queue, matched pattern) when exactly one queue's rules fire."""
    hits = {}
    for q, pats in QUEUE_RULES.items():
        for p in pats:
            m = p.search(text)
            if m:
                hits[q] = m.group(0)
                break
    if len(hits) == 1:
        (q, m), = hits.items()
        return q, m
    return None, None  # none or ambiguous: defer to the classifier


def priority_for(text: str) -> str:
    for level, pat in PRIORITY_RULES:
        if pat.search(text):
            return level
    return "normal"


class RouterModel:
    def __init__(self, weights: dict):
        self.classes: list[str] = weights["classes"]
        self.W = np.asarray(weights["coef"], dtype=np.float32)
        self.b = np.asarray(weights["intercept"], dtype=np.float32)

    def proba(self, emb: np.ndarray) -> np.ndarray:
        z = emb @ self.W.T + self.b
        z = z - z.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)


@lru_cache(maxsize=1)
def get_model() -> RouterModel:
    return RouterModel(json.loads((ARTIFACTS / "router.json").read_text()))


def embed(texts: list[str]) -> np.ndarray:
    from .retrieval import get_retriever

    return get_retriever().embedder.encode(texts)


def classify(text: str) -> dict[str, float]:
    model = get_model()
    p = model.proba(embed([text]))[0]
    return {c: round(float(v), 4) for c, v in zip(model.classes, p)}


def route(text: str, use_rules: bool = True, use_classifier: bool = True) -> Route:
    text = text.strip()
    scores = classify(text) if use_classifier else {}
    queue, rule = rule_queue(text) if use_rules else (None, None)
    if queue:
        method = "rule"
    elif scores:
        queue, method = max(scores, key=scores.get), "classifier"
    else:
        queue, method = "technical", "default"
    return Route(queue, priority_for(text), method, rule, scores)
