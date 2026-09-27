"""Hybrid retrieval: FAISS vector search + BM25, fused with RRF, then reranked.

Runtime dependencies are deliberately small (numpy, faiss, onnxruntime,
tokenizers) so the whole retriever fits inside a serverless function.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import ARTIFACTS, CANDIDATES, QUERY_PREFIX, TOP_K

_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an and are as at be but by can do does for from how i if in into is it its me my no not of on or so "
    "that the their then there these this to was what when where which who why will with you your".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _WORD.findall(text.lower()) if t not in _STOP]


class OnnxModel:
    """A BERT-style ONNX model plus its Hugging Face fast tokenizer."""

    def __init__(self, folder: Path, filename: str, max_len: int = 512):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(folder / filename), opts, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.session.get_inputs()}
        self.tok = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self.tok.enable_truncation(max_length=max_len)
        self.tok.enable_padding(pad_id=0, pad_token="[PAD]")

    def run(self, encodings) -> np.ndarray:
        feed = {
            "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
        }
        if "token_type_ids" in self.inputs:
            feed["token_type_ids"] = np.array([e.type_ids for e in encodings], dtype=np.int64)
        return self.session.run(None, feed)[0]


class Embedder(OnnxModel):
    """bge-small-en-v1.5: CLS pooling, L2 normalised."""

    def encode(self, texts: list[str]) -> np.ndarray:
        hidden = self.run(self.tok.encode_batch(texts))
        cls = hidden[:, 0, :]
        return (cls / np.linalg.norm(cls, axis=1, keepdims=True)).astype("float32")


class Reranker(OnnxModel):
    """ms-marco MiniLM cross-encoder: one relevance logit per (query, passage)."""

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        logits = self.run(self.tok.encode_batch([(query, p) for p in passages]))
        return logits.reshape(-1)


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = np.array([len(d) for d in docs], dtype=np.float32)
        self.avg = float(self.len.mean())
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(len(self.tf), dtype=np.float32)
        for t in set(query):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(t)
                if f:
                    out[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return out


@dataclass
class Hit:
    chunk: dict
    score: float
    rerank: float | None = None


@dataclass
class Result:
    hits: list[Hit]
    timings_ms: dict = field(default_factory=dict)


def chunk_text_for_model(c: dict) -> str:
    return " > ".join(c["path"]) + "\n" + c["text"]


class Retriever:
    def __init__(self, artifacts: Path = ARTIFACTS, quantized: bool = True):
        import faiss

        self.chunks: list[dict] = json.loads((artifacts / "chunks.json").read_text())
        self.index = faiss.read_index(str(artifacts / "faiss.index"))
        name = "model.int8.onnx" if quantized else "model.onnx"
        self.embedder = Embedder(artifacts / "onnx" / "embed", name)
        self.reranker = Reranker(artifacts / "onnx" / "rerank", name, max_len=192)
        self.bm25 = BM25([tokenize(chunk_text_for_model(c)) for c in self.chunks])

    def embed_query(self, query: str) -> np.ndarray:
        return self.embedder.encode([QUERY_PREFIX + query])

    def vector(self, query: str, k: int) -> list[tuple[int, float]]:
        scores, ids = self.index.search(self.embed_query(query), k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]

    def lexical(self, query: str, k: int) -> list[tuple[int, float]]:
        s = self.bm25.scores(tokenize(query))
        top = np.argsort(-s)[:k]
        return [(int(i), float(s[i])) for i in top if s[i] > 0]

    def search(self, query: str, k: int = TOP_K, mode: str = "hybrid", rerank: bool = True,
               candidates: int = CANDIDATES) -> Result:
        t0 = time.perf_counter()
        timings: dict[str, float] = {}
        pool = max(candidates, k)
        if mode == "vector":
            ranked = self.vector(query, pool)
        elif mode == "bm25":
            ranked = self.lexical(query, pool)
        else:
            vec = self.vector(query, 50)
            timings["vector_ms"] = (time.perf_counter() - t0) * 1000
            lex = self.lexical(query, 50)
            fused: dict[int, float] = {}
            for lst in (vec, lex):  # reciprocal rank fusion
                for rank, (i, _) in enumerate(lst):
                    fused[i] = fused.get(i, 0.0) + 1.0 / (60 + rank + 1)
            ranked = sorted(fused.items(), key=lambda x: -x[1])[:pool]
        timings["first_stage_ms"] = (time.perf_counter() - t0) * 1000

        hits = [Hit(self.chunks[i], s) for i, s in ranked]
        if rerank and hits:
            t1 = time.perf_counter()
            r = self.reranker.score(query, [chunk_text_for_model(h.chunk) for h in hits])
            for h, v in zip(hits, r):
                h.rerank = float(v)
            hits.sort(key=lambda h: -h.rerank)
            timings["rerank_ms"] = (time.perf_counter() - t1) * 1000
        timings["retrieval_ms"] = (time.perf_counter() - t0) * 1000
        return Result(hits[:k], {k_: round(v, 1) for k_, v in timings.items()})


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever()
