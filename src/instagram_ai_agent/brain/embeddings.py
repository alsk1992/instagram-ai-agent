"""Embedding provider chain.

One public function: :func:`embed(texts, model_hint)` returns a list of
unit-normalised float vectors plus the model identifier that produced them
(so the retriever can scope queries to the same backend).

Provider order (auto):

  1. ``gemini``  — Google text-embedding-004 via the public REST API.
                   Free tier: ~1500 requests/day, commercial-OK. Vector
                   dim = 768 by default.
  2. ``local``   — sentence-transformers (BAAI/bge-small-en-v1.5, MIT-ish,
                   384 dim, runs on CPU). Lazy-loaded singleton.
  3. ``none``    — no provider available; raises :class:`NoEmbeddingBackend`
                   so the caller can degrade gracefully.

All providers normalise output to unit length so cosine similarity is just
a dot product downstream.
"""
from __future__ import annotations

import asyncio
import math
import os
from collections.abc import Iterable
from dataclasses import dataclass

import httpx

from instagram_ai_agent.core.config import RAGConfig
from instagram_ai_agent.core.logging_setup import get_logger

log = get_logger(__name__)


class NoEmbeddingBackend(RuntimeError):
    """Raised when neither Gemini nor local sentence-transformers can run."""


@dataclass(frozen=True)
class EmbedResult:
    vectors: list[list[float]]
    model: str            # e.g. "gemini:text-embedding-004" or "local:bge-small"
    dim: int


# ─── Gemini provider ───
_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent"


async def _gemini_embed(texts: list[str], model: str) -> list[list[float]]:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise NoEmbeddingBackend("GEMINI_API_KEY missing")
    out: list[list[float]] = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(45.0)) as client:
        # Gemini's REST endpoint is one-text-at-a-time; batch by sequencing.
        # Simple parallelism with a small semaphore keeps the free tier happy.
        sem = asyncio.Semaphore(4)

        async def one(t: str) -> list[float]:
            async with sem:
                payload = {
                    "model": f"models/{model}",
                    "content": {"parts": [{"text": t}]},
                }
                r = await client.post(
                    _GEMINI_URL.format(model=model),
                    params={"key": key},
                    json=payload,
                )
                r.raise_for_status()
                data = r.json()
                values = data.get("embedding", {}).get("values") or []
                if not values:
                    raise RuntimeError(f"Gemini returned empty embedding for: {t[:60]!r}")
                return _normalise(values)

        out = await asyncio.gather(*[one(t) for t in texts])
    return out


# ─── Local provider (sentence-transformers) ───
_local_model: object | None = None


def _local_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        return True
    except Exception:
        return False


def _local_embed_sync(texts: list[str], model_name: str) -> list[list[float]]:
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        _local_model = SentenceTransformer(model_name, trust_remote_code=False)
    arr = _local_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [list(map(float, row)) for row in arr]


# ─── Helpers ───
def _normalise(v: Iterable[float]) -> list[float]:
    vec = list(v)
    norm = math.sqrt(sum(x * x for x in vec))
    if norm <= 0:
        return vec
    return [x / norm for x in vec]


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity for unit-normalised vectors == dot product."""
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b, strict=True))


def vec_to_blob(v: list[float]) -> bytes:
    import struct
    return struct.pack(f"{len(v)}f", *v)


def blob_to_vec(blob: bytes) -> list[float]:
    import struct
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


# ─── Public ───
async def embed(texts: list[str], cfg: RAGConfig) -> EmbedResult:
    if not texts:
        return EmbedResult(vectors=[], model="none:0", dim=0)

    provider = cfg.embedding_provider
    tried: list[str] = []

    candidates: list[str]
    if provider == "auto":
        candidates = ["gemini", "local"]
    else:
        candidates = [provider]

    last_err: Exception | None = None
    for which in candidates:
        try:
            if which == "gemini":
                vectors = await _gemini_embed(texts, cfg.gemini_model)
                return EmbedResult(
                    vectors=vectors,
                    model=f"gemini:{cfg.gemini_model}",
                    dim=len(vectors[0]) if vectors else 0,
                )
            if which == "local":
                if not _local_available():
                    raise NoEmbeddingBackend("sentence-transformers not installed")
                vectors = await asyncio.to_thread(_local_embed_sync, texts, cfg.local_model)
                return EmbedResult(
                    vectors=vectors,
                    model=f"local:{cfg.local_model}",
                    dim=len(vectors[0]) if vectors else 0,
                )
            if which == "none":
                raise NoEmbeddingBackend("provider explicitly set to none")
            raise NoEmbeddingBackend(f"unknown provider: {which}")
        except NoEmbeddingBackend as e:
            tried.append(f"{which}: {e}")
            last_err = e
            log.debug("embed: %s unavailable (%s)", which, e)
            continue
        except Exception as e:
            # Non-auth failures (rate limit, 5xx, network) deserve a louder
            # log so operators don't mistake them for "no backend configured".
            tried.append(f"{which}: {e}")
            last_err = e
            log.warning("embed: %s call failed (%s)", which, e)
            continue

    raise NoEmbeddingBackend(f"no embedding backend available — tried {tried}; last={last_err!r}")


async def embed_one(text: str, cfg: RAGConfig) -> EmbedResult:
    return await embed([text], cfg)
