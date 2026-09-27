"""Chunk the scraped manual, embed it with sentence-transformers, build FAISS.

Outputs (committed, shipped with the function):
    artifacts/chunks.json     chunk text + source URL/anchor + heading path
    artifacts/faiss.index     IndexFlatIP over L2-normalised embeddings
    artifacts/manifest.json   model names, counts, source + licence

    uv run --group build python scripts/build_index.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import faiss
import numpy as np
from bs4 import BeautifulSoup, NavigableString, Tag

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from supportbot.config import EMBED_MODEL_HF, QUERY_PREFIX  # noqa: E402

PAGES = ROOT / "data" / "pages.jsonl"
ART = ROOT / "artifacts"
BASE = "https://docs.nextcloud.com/server/stable/user_manual/en/"

MAX_WORDS = 220
OVERLAP = 40
MIN_WORDS = 25


def _text_of(node: Tag) -> str:
    """Text of a section excluding nested sections."""
    parts: list[str] = []
    for child in node.children:
        if isinstance(child, NavigableString):
            s = str(child).strip()
            if s:
                parts.append(s)
            continue
        if not isinstance(child, Tag):
            continue
        if child.name == "section" or child.name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            continue
        for a in child.select("a.headerlink"):
            a.decompose()
        if child.name == "pre" or "highlight" in " ".join(child.get("class", [])):
            parts.append(child.get_text("\n", strip=False).strip())
        else:
            parts.append(child.get_text(" ", strip=True))
    text = "\n".join(p for p in parts if p)
    return re.sub(r"[ \t]+", " ", text).strip()


def _heading(sec: Tag) -> str:
    h = sec.find(re.compile(r"^h[1-6]$"), recursive=False)
    if not h:
        return ""
    for a in h.select("a.headerlink"):
        a.decompose()
    return h.get_text(" ", strip=True)


def sections(page: dict) -> list[dict]:
    soup = BeautifulSoup(page["html"], "html.parser")
    out = []

    def walk(sec: Tag, path: list[str]) -> None:
        head = _heading(sec)
        here = path + ([head] if head else [])
        text = _text_of(sec)
        if text:
            out.append({"anchor": sec.get("id", ""), "path": here, "text": text})
        for child in sec.find_all("section", recursive=False):
            walk(child, here)

    for top in soup.find_all("section", recursive=False) or soup.select("section")[:1]:
        walk(top, [])
    return out


def split_words(text: str) -> list[str]:
    words = text.split()
    if len(words) <= MAX_WORDS:
        return [text]
    pieces, start = [], 0
    while start < len(words):
        end = min(len(words), start + MAX_WORDS)
        pieces.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - OVERLAP
    return pieces


def build_chunks() -> list[dict]:
    chunks: list[dict] = []
    for line in PAGES.open():
        page = json.loads(line)
        rel = page["url"][len(BASE):] or "index.html"
        if rel == "index.html" and any(c["page"] == "index.html" for c in chunks):
            continue
        rel = rel if rel.endswith(".html") else rel + "index.html"
        pending = None  # tiny sections get merged forward into the next one
        for s in sections(page):
            if pending is not None:
                s = {**s, "text": pending["text"] + "\n" + s["text"]}
                pending = None
            if len(s["text"].split()) < MIN_WORDS:
                pending = s
                continue
            for i, piece in enumerate(split_words(s["text"])):
                chunks.append(
                    {
                        "page": rel,
                        "anchor": s["anchor"],
                        "url": BASE + rel + (f"#{s['anchor']}" if s["anchor"] else ""),
                        "title": page["title"],
                        "path": s["path"],
                        "part": i,
                        "text": piece,
                    }
                )
        if pending is not None:
            if chunks and chunks[-1]["page"] == rel:
                chunks[-1]["text"] += "\n" + pending["text"]
            elif len(pending["text"].split()) >= 8:
                chunks.append(
                    {
                        "page": rel,
                        "anchor": pending["anchor"],
                        "url": BASE + rel + (f"#{pending['anchor']}" if pending["anchor"] else ""),
                        "title": page["title"],
                        "path": pending["path"],
                        "part": 0,
                        "text": pending["text"],
                    }
                )
    for i, c in enumerate(chunks):
        c["id"] = i
    return chunks


def embed_text(c: dict) -> str:
    return " > ".join(c["path"]) + "\n" + c["text"]


def main() -> int:
    from sentence_transformers import SentenceTransformer

    chunks = build_chunks()
    model = SentenceTransformer(EMBED_MODEL_HF)
    emb = model.encode([embed_text(c) for c in chunks], batch_size=32, normalize_embeddings=True, show_progress_bar=True)
    emb = np.asarray(emb, dtype="float32")
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    ART.mkdir(exist_ok=True)
    faiss.write_index(index, str(ART / "faiss.index"))
    (ART / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False))
    words = [len(c["text"].split()) for c in chunks]
    manifest = {
        "source": "Nextcloud user manual (stable)",
        "source_url": BASE,
        "license": "CC BY 3.0 (https://creativecommons.org/licenses/by/3.0/)",
        "attribution": "Nextcloud GmbH and the Nextcloud documentation contributors",
        "pages": sum(1 for _ in PAGES.open()),
        "chunks": len(chunks),
        "words_per_chunk": {"mean": round(float(np.mean(words)), 1), "max": int(max(words))},
        "embedding_model": EMBED_MODEL_HF,
        "query_prefix": QUERY_PREFIX,
        "dim": int(emb.shape[1]),
        "index": "faiss.IndexFlatIP (cosine on normalised vectors)",
    }
    (ART / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
