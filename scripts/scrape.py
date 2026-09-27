"""Crawl the Nextcloud user manual (stable) and write one JSON line per page.

The manual is a static Sphinx site, so plain HTTP + BeautifulSoup is enough;
no browser automation is needed. The crawler:

* reads robots.txt first and only fetches URLs it allows,
* stays inside /server/stable/user_manual/en/,
* waits DELAY seconds between requests and sends an identifying User-Agent,
* caches raw HTML under data/raw/ so re-runs do not hit the site again.

Content: Nextcloud user manual, (c) Nextcloud GmbH and contributors,
licensed CC BY 3.0 (https://creativecommons.org/licenses/by/3.0/).

    uv run --group build python scripts/scrape.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import deque
from pathlib import Path
from urllib.parse import urldefrag, urljoin
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "pages.jsonl"

BASE = "https://docs.nextcloud.com/server/stable/user_manual/en/"
UA = "CustomerSupportApp-docs-ingester/1.0 (+https://github.com/darshpandya02/CustomerSupportApp)"
DELAY = 0.75
SKIP = {"genindex.html", "search.html", "contents.html"}
MAX_PAGES = 400


def allowed_by_robots() -> RobotFileParser:
    rp = RobotFileParser()
    rp.set_url("https://docs.nextcloud.com/robots.txt")
    rp.read()
    return rp


def fetch(session: requests.Session, url: str) -> str:
    RAW.mkdir(parents=True, exist_ok=True)
    cache = RAW / (hashlib.sha1(url.encode()).hexdigest() + ".html")
    if cache.exists():
        return cache.read_text()
    time.sleep(DELAY)
    r = session.get(url, timeout=20)
    r.raise_for_status()
    cache.write_text(r.text)
    return r.text


def main() -> int:
    rp = allowed_by_robots()
    session = requests.Session()
    session.headers["User-Agent"] = UA

    queue: deque[str] = deque([BASE])
    seen: set[str] = {BASE}
    pages = []
    blocked = 0
    while queue and len(pages) < MAX_PAGES:
        url = queue.popleft()
        if not rp.can_fetch(UA, url):
            blocked += 1
            continue
        html = fetch(session, url)
        soup = BeautifulSoup(html, "html.parser")
        body = soup.select_one('div[itemprop="articleBody"]')
        if body is None:
            continue
        title = soup.title.get_text(strip=True).split(" — ")[0] if soup.title else url
        pages.append({"url": url, "title": title, "html": str(body)})

        for a in soup.select("a[href]"):
            href = urldefrag(urljoin(url, a["href"]))[0]
            if not href.startswith(BASE) or not href.endswith(".html") and not href.endswith("/"):
                continue
            if href.rsplit("/", 1)[-1] in SKIP or "/_" in href[len(BASE):]:
                continue
            if href not in seen:
                seen.add(href)
                queue.append(href)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for p in pages:
            f.write(json.dumps(p) + "\n")
    print(f"pages={len(pages)} blocked_by_robots={blocked} -> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
