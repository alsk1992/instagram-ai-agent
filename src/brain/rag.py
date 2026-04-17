"""Niche knowledge RAG.

Drop ``.txt`` / ``.md`` / ``.pdf`` into ``data/knowledge/`` and call
``index_dir()`` (or the CLI ``ig-agent index-knowledge``). Retrieval is
called from the pipeline via :func:`context_for(query, cfg, k)`, which
returns formatted snippets the caption + critic prompts can drop in.

Backed by SQLite + pure-Python cosine similarity. For single-agent scale
(hundreds-to-low-thousands of chunks) this is ample; the table schema is
chroma-compatible if anyone ever wants to swap it in.

Re-index detection is hash-based: each source file gets sha256'd; if the
hash is unchanged we skip re-embedding entirely.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.brain import embeddings as embed_mod
from src.core import db
from src.core.config import KNOWLEDGE_DIR, RAGConfig
from src.core.logging_setup import get_logger

log = get_logger(__name__)


# ─── Source readers ───
def _read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return p.read_text(encoding="latin-1", errors="replace")


def _read_pdf(p: Path) -> str:
    try:
        from pypdf import PdfReader
    except Exception:
        log.warning("pypdf not installed — skipping %s (install with `pip install pypdf`)", p.name)
        return ""
    try:
        r = PdfReader(str(p))
        return "\n\n".join((page.extract_text() or "").strip() for page in r.pages)
    except Exception as e:
        log.warning("pdf read failed %s: %s", p.name, e)
        return ""


def _read_source(p: Path) -> str:
    suffix = p.suffix.lower()
    if suffix in {".txt", ".md", ".markdown", ".rst", ".html"}:
        return _read_text(p)
    if suffix == ".pdf":
        return _read_pdf(p)
    if suffix in {".json", ".yaml", ".yml", ".csv"}:
        return _read_text(p)
    return ""


# ─── Chunker ───
_PARA_SPLIT = re.compile(r"\n\s*\n")


def chunk_text(text: str, *, max_chars: int, overlap: int) -> list[str]:
    """Paragraph-then-greedy-pack chunker. Preserves paragraph boundaries
    when possible; splits oversized paragraphs by sentence/word."""
    text = (text or "").strip()
    if not text:
        return []
    paragraphs = [p.strip() for p in _PARA_SPLIT.split(text) if p.strip()]
    chunks: list[str] = []
    cur = ""
    for para in paragraphs:
        if len(para) > max_chars:
            # Flush whatever we have
            if cur:
                chunks.append(cur.strip())
                cur = ""
            # Split the long paragraph by sentence boundaries
            for piece in _split_long(para, max_chars=max_chars):
                chunks.append(piece)
            continue
        if not cur:
            cur = para
        elif len(cur) + 2 + len(para) <= max_chars:
            cur = cur + "\n\n" + para
        else:
            chunks.append(cur.strip())
            cur = para
    if cur:
        chunks.append(cur.strip())

    # Apply overlap by prepending the tail of the previous chunk. We
    # still respect ``max_chars`` on the final output — otherwise a
    # configured cap is quietly exceeded by ~(overlap + 3) chars.
    if overlap > 0 and len(chunks) > 1:
        with_overlap: list[str] = [chunks[0]]
        bridge = " … "
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-overlap:]
            merged = (tail + bridge + chunks[i]).strip()
            if len(merged) > max_chars:
                # Trim the body from the left; keep the overlap tail visible
                keep_body = max_chars - len(tail) - len(bridge)
                if keep_body > 0:
                    merged = (tail + bridge + chunks[i][-keep_body:]).strip()
                else:
                    merged = chunks[i][:max_chars].strip()
            with_overlap.append(merged)
        chunks = with_overlap
    return chunks


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def _split_long(text: str, *, max_chars: int) -> list[str]:
    sentences = _SENT_SPLIT.split(text)
    out: list[str] = []
    cur = ""
    for s in sentences:
        if len(s) > max_chars:
            # Hard word-split as last resort
            words = s.split()
            buf: list[str] = []
            for w in words:
                if sum(len(x) + 1 for x in buf) + len(w) > max_chars:
                    out.append(" ".join(buf))
                    buf = [w]
                else:
                    buf.append(w)
            if buf:
                out.append(" ".join(buf))
            continue
        if not cur:
            cur = s
        elif len(cur) + 1 + len(s) <= max_chars:
            cur = cur + " " + s
        else:
            out.append(cur.strip())
            cur = s
    if cur:
        out.append(cur.strip())
    return out


# ─── Hashing ───
def _file_hash(p: Path) -> str:
    """Return sha256 of file bytes, or a path+mtime sentinel when unreadable.

    Returning an empty string here used to make every unreadable file dedup
    against every other unreadable file — we now stamp a unique sentinel so
    the re-index machinery can retry when the file becomes readable again.
    """
    h = hashlib.sha256()
    try:
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    except OSError:
        try:
            return f"ERR:{p.as_posix()}:{p.stat().st_mtime:.6f}"
        except OSError:
            return f"ERR:{p.as_posix()}"
    return h.hexdigest()


# ─── Indexing ───
@dataclass
class IndexStats:
    files_seen: int = 0
    files_indexed: int = 0
    files_skipped: int = 0
    chunks_added: int = 0
    embed_failures: int = 0
    last_error: str = ""


async def index_path(p: Path, cfg: RAGConfig) -> IndexStats:
    stats = IndexStats(files_seen=1)
    if not cfg.enabled:
        return stats
    raw = _read_source(p)
    if not raw.strip():
        return stats
    fhash = _file_hash(p)
    conn = db.get_conn()
    existing = conn.execute(
        "SELECT 1 FROM knowledge_chunks WHERE source=? AND source_hash=? LIMIT 1",
        (str(p), fhash),
    ).fetchone()
    if existing:
        stats.files_skipped = 1
        return stats

    # Drop any prior chunks for this source (file changed)
    conn.execute("DELETE FROM knowledge_chunks WHERE source=?", (str(p),))

    chunks = chunk_text(raw, max_chars=cfg.chunk_max_chars, overlap=cfg.chunk_overlap_chars)
    if not chunks:
        return stats

    try:
        result = await embed_mod.embed(chunks, cfg)
    except embed_mod.NoEmbeddingBackend as e:
        stats.embed_failures = 1
        stats.last_error = f"no embedding backend: {e}"
        return stats
    if not result.vectors:
        return stats

    for idx, (text, vec) in enumerate(zip(chunks, result.vectors, strict=True)):
        conn.execute(
            """
            INSERT INTO knowledge_chunks
              (source, source_hash, chunk_index, text, embedding, embedding_model)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (str(p), fhash, idx, text, embed_mod.vec_to_blob(vec), result.model),
        )
        stats.chunks_added += 1
    stats.files_indexed = 1
    return stats


_SUPPORTED_EXTS = {".txt", ".md", ".markdown", ".rst", ".html", ".pdf", ".json", ".yaml", ".yml", ".csv"}


async def index_dir(directory: Path | None = None, cfg: RAGConfig | None = None) -> IndexStats:
    """Walk the knowledge dir, embed any new/changed files."""
    target = directory or KNOWLEDGE_DIR
    if cfg is None:
        from src.core.config import load_niche
        cfg = load_niche().rag

    total = IndexStats()
    if not target.exists():
        return total
    if not cfg.enabled:
        return total

    files = [p for p in sorted(target.rglob("*")) if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTS]
    for p in files:
        try:
            s = await index_path(p, cfg)
        except Exception as e:
            log.warning("index failed for %s: %s", p, e)
            total.embed_failures += 1
            total.last_error = str(e)[:200]
            continue
        total.files_seen += s.files_seen
        total.files_indexed += s.files_indexed
        total.files_skipped += s.files_skipped
        total.chunks_added += s.chunks_added
        total.embed_failures += s.embed_failures
        if s.last_error and not total.last_error:
            total.last_error = s.last_error
    if total.chunks_added or total.files_indexed:
        log.info(
            "rag index: seen=%d indexed=%d skipped=%d new_chunks=%d",
            total.files_seen, total.files_indexed, total.files_skipped, total.chunks_added,
        )
    return total


def clear_index() -> int:
    cur = db.get_conn().execute("DELETE FROM knowledge_chunks")
    return cur.rowcount


# ─── Retrieval ───
@dataclass(frozen=True)
class Retrieval:
    text: str
    source: str
    score: float


async def retrieve(query: str, cfg: RAGConfig, *, k: int | None = None) -> list[Retrieval]:
    if not cfg.enabled:
        return []
    k = k or cfg.retrieve_k
    if not query.strip():
        return []

    try:
        q_result = await embed_mod.embed_one(query, cfg)
    except embed_mod.NoEmbeddingBackend:
        return []

    q_model = q_result.model
    q_vec = q_result.vectors[0]

    rows = db.get_conn().execute(
        "SELECT text, source, embedding FROM knowledge_chunks WHERE embedding_model=?",
        (q_model,),
    ).fetchall()
    if not rows:
        return []

    scored: list[Retrieval] = []
    for r in rows:
        v = embed_mod.blob_to_vec(r["embedding"])
        if len(v) != len(q_vec):
            continue
        s = embed_mod.cosine(q_vec, v)
        scored.append(Retrieval(text=r["text"], source=r["source"], score=s))
    scored.sort(key=lambda r: r.score, reverse=True)
    return scored[:k]


async def context_for(query: str, cfg: RAGConfig, *, k: int | None = None) -> str:
    """Format top-K chunks as a single string for prompt injection.
    Always returns "" when nothing retrieved or RAG disabled — caller can
    safely concatenate without conditional logic.
    """
    hits = await retrieve(query, cfg, k=k)
    if not hits:
        return ""
    pieces: list[str] = []
    used = 0
    for h in hits:
        rel = Path(h.source).name
        snippet = f"- [{rel}] {h.text}"
        if used + len(snippet) > cfg.max_inject_chars:
            break
        pieces.append(snippet)
        used += len(snippet)
    return "\n".join(pieces)


# ─── Stats (for dashboard) ───
def stats() -> dict:
    conn = db.get_conn()
    n_chunks = conn.execute("SELECT COUNT(*) c FROM knowledge_chunks").fetchone()["c"]
    n_sources = conn.execute("SELECT COUNT(DISTINCT source) c FROM knowledge_chunks").fetchone()["c"]
    by_model = [
        dict(r)
        for r in conn.execute(
            "SELECT embedding_model, COUNT(*) c FROM knowledge_chunks GROUP BY embedding_model"
        ).fetchall()
    ]
    return {"chunks": n_chunks, "sources": n_sources, "by_model": by_model}
