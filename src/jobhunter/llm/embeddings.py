"""Embeddings for semantic matching: Ollama (``nomic-embed-text``) -> OpenAI (unless ``local_only``)
-> hashed bag-of-words.

The last fallback needs no model at all, so scoring still works offline — just less semantically.
"""

from __future__ import annotations

import hashlib
import re

import httpx
import numpy as np

from jobhunter.config import Settings
from jobhunter.telemetry import get_logger

log = get_logger("llm.embed")
_TOKEN = re.compile(r"[a-ząćęłńóśźż0-9+#.]{2,}", re.I)


def _hashed_bow(texts: list[str], dim: int = 2048) -> np.ndarray:
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        for tok in _TOKEN.findall(t.lower()):
            out[i, int(hashlib.md5(tok.encode()).hexdigest(), 16) % dim] += 1.0
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
    sim: np.ndarray = a @ b.T
    return sim


class Embedder:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.backend = "unknown"

    async def embed(self, texts: list[str]) -> np.ndarray:
        cfg = self.settings.llm
        # nomic-embed-text expects task prefixes; "search_document" for both sides is fine for symmetric sim.
        prefixed = [f"search_document: {t[:8000]}" for t in texts]
        try:
            async with httpx.AsyncClient(base_url=cfg.ollama_base_url, timeout=120) as c:
                r = await c.post("/api/embed", json={"model": cfg.embedding_model, "input": prefixed})
                r.raise_for_status()
                self.backend = f"ollama:{cfg.embedding_model}"
                return np.asarray(r.json()["embeddings"], dtype=np.float32)
        except (httpx.HTTPError, KeyError, ValueError) as e:
            log.debug("ollama embeddings unavailable: {!r}", e)
        if self.settings.openai_api_key and not cfg.local_only:
            try:
                from openai import AsyncOpenAI

                client = AsyncOpenAI(api_key=self.settings.openai_api_key.get_secret_value())
                resp = await client.embeddings.create(
                    model="text-embedding-3-small", input=[t[:8000] for t in texts]
                )
                self.backend = "openai:text-embedding-3-small"
                return np.asarray([d.embedding for d in resp.data], dtype=np.float32)
            except Exception as e:
                log.warning("openai embeddings failed: {!r}", e)
        self.backend = "hashed-bow"
        log.warning("falling back to bag-of-words similarity (run: ollama pull {})", cfg.embedding_model)
        return _hashed_bow(texts)
