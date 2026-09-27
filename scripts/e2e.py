"""Browser check of the deployed app (Playwright, headless Chromium).

    uv run --with playwright --python 3.12 python scripts/e2e.py https://<deployment> <agent-password>

Asks three in-scope questions (expects cited answers), one out-of-scope
question (expects a refusal), escalates it to a ticket, then logs into the
agent dashboard and checks the ticket is there with its queue and transcript.
Writes screenshots to reports/screens/ and a summary to reports/e2e.json.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
SHOTS = ROOT / "reports" / "screens"

IN_SCOPE = [
    "I deleted a folder by mistake. How do I get it back?",
    "My 2FA code keeps getting rejected even though I type it correctly. Why?",
    "What does a 'conflicted copy' file from the desktop client mean?",
]
OUT_OF_SCOPE = "I was charged twice for my subscription this month, can I get a refund?"


def main(base: str, password: str) -> int:
    SHOTS.mkdir(parents=True, exist_ok=True)
    report: dict = {"base_url": base, "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "chat": []}
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        resp = page.goto(base + "/")
        report["home_status"] = resp.status
        assert resp.status == 200, resp.status

        def ask(q: str) -> dict:
            n = page.locator(".msg.bot").count()
            page.fill("#msg", q)
            t0 = time.perf_counter()
            with page.expect_response(lambda r: r.url.endswith("/api/chat")) as info:
                page.click("#send")
            r = info.value
            body = r.json()
            bot = page.locator(".msg.bot").nth(n)
            expect(bot.locator(".badge")).to_be_visible(timeout=60000)
            links = bot.locator("a[href^='https://docs.nextcloud.com']").count()
            return {"question": q, "http": r.status, "mode": body.get("mode"),
                    "badge": bot.locator(".badge").inner_text(), "citation_links": links,
                    "sources": [c["url"] for c in body.get("citations", [])],
                    "client_ms": round((time.perf_counter() - t0) * 1000),
                    "server_timings_ms": body.get("timings_ms")}

        for q in IN_SCOPE:
            res = ask(q)
            assert res["http"] == 200 and res["mode"] in ("llm", "retrieval_only") and res["citation_links"] > 0, res
            report["chat"].append(res)
        page.screenshot(path=str(SHOTS / "1-cited-answers.png"), full_page=True)

        res = ask(OUT_OF_SCOPE)
        assert res["mode"] == "refusal", res
        report["chat"].append(res)

        page.locator(".msg.bot").last.locator("[data-escalate]").click()
        page.fill("#t-desc", "I was charged twice for my subscription this month and need a refund for one payment.")
        page.fill("#t-email", "e2e-check@example.com")
        with page.expect_response(lambda r: r.url.endswith("/api/tickets")) as info:
            page.click("#ticket-form button[type=submit]")
        ticket = info.value.json()
        expect(page.locator("#ticket-result")).to_contain_text(ticket["ref"])
        page.screenshot(path=str(SHOTS / "2-refusal-and-ticket.png"), full_page=True)
        report["ticket"] = {k: ticket[k] for k in ("ref", "queue", "priority", "status", "transcript_messages")}
        report["ticket"]["routing_method"] = ticket["routing"]["method"]

        dash = browser.new_page(viewport={"width": 1400, "height": 1000})
        assert dash.goto(base + "/dashboard.html").status == 200
        dash.fill("#pw", password)
        dash.click("#login button[type=submit]")
        row = dash.locator("#rows tr", has_text=ticket["ref"])
        expect(row).to_be_visible(timeout=20000)
        report["dashboard_row_queue"] = row.locator(".pill").first.inner_text()
        row.click()
        expect(dash.locator("#detail")).to_contain_text("Chat transcript", timeout=20000)
        report["dashboard_transcript_messages"] = dash.locator(".tmsg").count()
        report["dashboard_routing_line"] = dash.locator(".routing").inner_text()
        dash.screenshot(path=str(SHOTS / "3-dashboard.png"), full_page=True)
        assert report["dashboard_row_queue"] == ticket["queue"]
        browser.close()
    report["passed"] = True
    (ROOT / "reports" / "e2e.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1].rstrip("/"), sys.argv[2]))
