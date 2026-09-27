"""Train the queue classifier and evaluate routing on the held-out test set.

Train: eval/routing_train.jsonl (120 hand-written tickets, 40 per queue).
Test:  eval/routing_test.jsonl (60 hand-written tickets, written separately,
       with queue and priority labels).

    uv run --group build python scripts/train_router.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from supportbot import routing  # noqa: E402


def load(name: str) -> list[dict]:
    return [json.loads(l) for l in (ROOT / "eval" / name).open()]


def main() -> int:
    train, test = load("routing_train.jsonl"), load("routing_test.jsonl")
    Xtr = routing.embed([r["text"] for r in train])
    ytr = [r["queue"] for r in train]
    clf = LogisticRegression(C=4.0, max_iter=2000)
    clf.fit(Xtr, ytr)
    (ROOT / "artifacts" / "router.json").write_text(json.dumps({
        "classes": list(clf.classes_),
        "coef": clf.coef_.round(6).tolist(),
        "intercept": clf.intercept_.round(6).tolist(),
        "embedding": "BAAI/bge-small-en-v1.5 (int8 ONNX), no query prefix",
        "train_examples": len(train),
    }))
    routing.get_model.cache_clear()

    truth = [r["queue"] for r in test]
    report: dict = {"train": len(train), "test": len(test), "queues": dict(Counter(truth))}
    for name, kw in {
        "rules_only": dict(use_rules=True, use_classifier=False),
        "classifier_only": dict(use_rules=False, use_classifier=True),
        "rules_then_classifier": dict(use_rules=True, use_classifier=True),
    }.items():
        routes = [routing.route(r["text"], **kw) for r in test]
        pred = [x.queue for x in routes]
        entry = {"accuracy": round(float(np.mean([p == t for p, t in zip(pred, truth)])), 3)}
        if name == "rules_only":
            fired = [x.method == "rule" for x in routes]
            entry["coverage"] = round(float(np.mean(fired)), 3)
            entry["precision_when_fired"] = round(
                float(np.mean([p == t for p, t, f in zip(pred, truth, fired) if f])), 3)
            entry["note"] = "unmatched tickets fall back to the 'technical' queue"
        if name == "rules_then_classifier":
            entry["decided_by"] = dict(Counter(x.method for x in routes))
            labels = list(routing.QUEUES)
            entry["confusion"] = {"labels": labels,
                                  "matrix": confusion_matrix(truth, pred, labels=labels).tolist()}
            entry["errors"] = [{"text": r["text"], "truth": t, "pred": p, "method": x.method}
                               for r, t, p, x in zip(test, truth, pred, routes) if p != t]
        report[name] = entry
    pr_truth = [r["priority"] for r in test]
    pr_pred = [routing.priority_for(r["text"]) for r in test]
    report["priority_rules"] = {
        "accuracy": round(float(np.mean([p == t for p, t in zip(pr_pred, pr_truth)])), 3),
        "labels": dict(Counter(pr_truth)),
        "urgent_or_high_recall": round(float(np.mean(
            [p in ("urgent", "high") for p, t in zip(pr_pred, pr_truth) if t in ("urgent", "high")])), 3),
        "errors": [{"text": r["text"], "truth": t, "pred": p}
                   for r, t, p in zip(test, pr_truth, pr_pred) if p != t],
    }
    report = {"summary": {"test_tickets": len(test),
                          "queue_accuracy": report["rules_then_classifier"]["accuracy"],
                          "rules_only_accuracy": report["rules_only"]["accuracy"],
                          "rule_coverage": report["rules_only"]["coverage"],
                          "priority_accuracy": report["priority_rules"]["accuracy"]}, **report}
    (ROOT / "reports" / "routing_eval.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items()}, indent=1)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
