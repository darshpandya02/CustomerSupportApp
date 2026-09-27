"""Export the embedding model and the reranker to ONNX for the serverless runtime.

The API cannot ship PyTorch (too large for a function bundle), so the query
side runs the same weights through onnxruntime. scripts/check_parity.py
verifies the ONNX embeddings match sentence-transformers.

    uv run --group build python scripts/export_onnx.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from supportbot.config import EMBED_MODEL_HF, RERANK_MODEL_HF  # noqa: E402

OUT = ROOT / "artifacts" / "onnx"


def export(model_id: str, task: str, name: str) -> None:
    from optimum.exporters.onnx import main_export
    from onnxruntime.quantization import QuantType, quantize_dynamic

    tmp = OUT / f"_{name}"
    main_export(model_id, output=tmp, task=task, opset=17)
    dst = OUT / name
    dst.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(str(tmp / "model.onnx"), str(dst / "model.int8.onnx"), weight_type=QuantType.QInt8)
    shutil.copy(tmp / "model.onnx", dst / "model.onnx")
    shutil.copy(tmp / "tokenizer.json", dst / "tokenizer.json")
    shutil.rmtree(tmp)
    print(name, sorted(p.name for p in dst.iterdir()))


if __name__ == "__main__":
    export(EMBED_MODEL_HF, "feature-extraction", "embed")
    export(RERANK_MODEL_HF, "text-classification", "rerank")
