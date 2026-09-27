"""Static configuration shared by the offline build scripts and the API."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"

# Embeddings: chunk vectors are computed offline with sentence-transformers;
# at request time the same model runs as an ONNX export (see scripts/export_onnx.py).
EMBED_MODEL_HF = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RERANK_MODEL_HF = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Generation through the Vercel AI Gateway (OpenAI-compatible endpoint).
GATEWAY_URL = os.environ.get("AI_GATEWAY_BASE_URL", "https://ai-gateway.vercel.sh/v1")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "meta/llama-4-maverick")
SUMMARY_MODEL = os.environ.get("SUMMARY_MODEL", "meta/llama-3.1-8b")

# Retrieval knobs.
CANDIDATES = 15  # hybrid candidates passed to the reranker
TOP_K = 5  # chunks placed in the prompt

# Cost controls.
MAX_MESSAGE_CHARS = 1000
MAX_ANSWER_TOKENS = 450
CHAT_LIMIT_PER_IP = int(os.environ.get("CHAT_LIMIT_PER_IP", "20"))  # per window
CHAT_WINDOW_SECONDS = 600
CHAT_LIMIT_GLOBAL_DAY = int(os.environ.get("CHAT_LIMIT_GLOBAL_DAY", "1500"))
HISTORY_TURNS_VERBATIM = 4  # most recent turns kept word for word; older ones are summarised

# Out-of-scope gate: if the best reranked passage scores below this
# cross-encoder logit, the assistant refuses instead of answering (and no LLM
# call is made). Chosen from the sweep in reports/refusal_eval.json.
REFUSAL_RERANK_THRESHOLD = float(os.environ.get("REFUSAL_RERANK_THRESHOLD", "-1.5"))
RETRIEVAL_ONLY_PASSAGES = 3
