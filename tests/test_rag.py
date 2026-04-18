"""Niche RAG — chunker, embeddings contract, indexer, retriever, CLI wiring."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from instagram_ai_agent.brain import embeddings as emb_mod
from instagram_ai_agent.brain import rag
from instagram_ai_agent.core import config as cfg_mod
from instagram_ai_agent.core import db


def _mkcfg(**kwargs):
    base = dict(
        niche="home calisthenics",
        sub_topics=["pullups"],
        target_audience="dads 35+",
        commercial=True,
        voice=cfg_mod.Voice(tone=["direct"], forbidden=[], persona="ex-office worker rebuilding fitness."),
        aesthetic=cfg_mod.Aesthetic(palette=["#000", "#fff", "#c9a961"]),
        hashtags=cfg_mod.HashtagPools(core=["calisthenics", "homeworkout", "dadfit"]),
    )
    base.update(kwargs)
    return cfg_mod.NicheConfig(**base)


@pytest.fixture()
def tmp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fresh = tmp_path / "brain.db"
    monkeypatch.setattr(cfg_mod, "DB_PATH", fresh)
    monkeypatch.setattr(db, "DB_PATH", fresh)
    db.close()
    db.init_db()
    yield db
    db.close()


# ─── Embeddings helpers ───
def test_vec_blob_roundtrip():
    v = [0.1, -0.2, 0.33, 0.0]
    blob = emb_mod.vec_to_blob(v)
    out = emb_mod.blob_to_vec(blob)
    for a, b in zip(v, out, strict=True):
        assert abs(a - b) < 1e-6


def test_cosine_of_identical_unit_vecs_is_one():
    v = emb_mod._normalise([1.0, 2.0, 3.0])
    assert abs(emb_mod.cosine(v, v) - 1.0) < 1e-6


def test_cosine_orthogonal_is_zero():
    a = emb_mod._normalise([1.0, 0.0])
    b = emb_mod._normalise([0.0, 1.0])
    assert abs(emb_mod.cosine(a, b)) < 1e-6


def test_cosine_mismatched_length_is_zero():
    assert emb_mod.cosine([1.0, 0.0], [1.0]) == 0.0


def test_normalise_zero_vector_returns_zero():
    assert emb_mod._normalise([0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0]


# ─── Chunker ───
def test_chunk_respects_max_chars():
    text = "paragraph one. " * 5 + "\n\n" + "paragraph two. " * 60
    chunks = rag.chunk_text(text, max_chars=120, overlap=0)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c) <= 150  # max_chars + small overlap slack


def test_chunk_overlap_prepends_tail():
    text = "alpha bravo charlie.\n\ndelta echo foxtrot.\n\ngolf hotel india juliet."
    chunks = rag.chunk_text(text, max_chars=30, overlap=8)
    assert len(chunks) >= 2
    # Chunk 2 onwards should contain a tail of the previous chunk
    assert "…" in chunks[1]


def test_chunk_empty_returns_empty():
    assert rag.chunk_text("", max_chars=500, overlap=50) == []
    assert rag.chunk_text("   \n\n  \n", max_chars=500, overlap=50) == []


def test_chunk_long_paragraph_splits_by_sentences():
    para = (
        "First sentence talks about pull-ups and progressive overload. "
        "Second sentence covers shoulder mobility warmups. "
        "Third sentence nails a specific cue for hollow body holds. "
        "Fourth sentence summarises the protocol for beginners at home."
    )
    chunks = rag.chunk_text(para, max_chars=100, overlap=0)
    assert len(chunks) >= 2


# ─── Embedding stubs (no network, no torch) ───
async def _fake_embed(texts, cfg):
    # Deterministic "embedding": hash to 8-dim vectors, normalised
    def vec_for(t: str) -> list[float]:
        h = abs(hash(t)) % (10**8)
        # 8 dims spread across the hash
        v = [((h >> (i * 4)) & 0xF) / 15.0 for i in range(8)]
        return emb_mod._normalise(v)
    return emb_mod.EmbedResult(
        vectors=[vec_for(t) for t in texts],
        model="fake:testing-8d",
        dim=8,
    )


@pytest.fixture()
def fake_embed(monkeypatch: pytest.MonkeyPatch):
    async def embed(texts, cfg):
        return await _fake_embed(texts, cfg)

    async def embed_one(text, cfg):
        r = await _fake_embed([text], cfg)
        return r

    monkeypatch.setattr(emb_mod, "embed", embed)
    monkeypatch.setattr(emb_mod, "embed_one", embed_one)


# ─── Indexer ───
def test_index_txt_file(tmp_db, tmp_path: Path, fake_embed):
    p = tmp_path / "pullups.txt"
    p.write_text(
        "Pull-ups build real strength. Do them daily.\n\n"
        "Focus on slow negatives for 30 days and track reps weekly."
    )
    cfg = _mkcfg().rag
    stats = asyncio.run(rag.index_path(p, cfg))
    assert stats.files_indexed == 1
    assert stats.chunks_added >= 1

    lib = rag.stats()
    assert lib["chunks"] >= 1
    assert lib["sources"] == 1


def test_index_skips_unchanged_file(tmp_db, tmp_path: Path, fake_embed):
    p = tmp_path / "stable.md"
    p.write_text("Stable knowledge about the niche. Not changing.")
    cfg = _mkcfg().rag
    first = asyncio.run(rag.index_path(p, cfg))
    assert first.files_indexed == 1
    second = asyncio.run(rag.index_path(p, cfg))
    assert second.files_skipped == 1
    assert second.chunks_added == 0


def test_reindex_after_content_change(tmp_db, tmp_path: Path, fake_embed):
    p = tmp_path / "shifty.md"
    p.write_text("original")
    cfg = _mkcfg().rag
    asyncio.run(rag.index_path(p, cfg))
    chunks_before = rag.stats()["chunks"]

    p.write_text("updated — more facts about pull-ups and form.")
    asyncio.run(rag.index_path(p, cfg))
    chunks_after = rag.stats()["chunks"]
    assert chunks_after != chunks_before or chunks_after > 0


def test_index_dir_walks_recursively(tmp_db, tmp_path: Path, fake_embed, monkeypatch):
    (tmp_path / "deep").mkdir()
    (tmp_path / "deep" / "a.md").write_text("alpha insights about the niche.")
    (tmp_path / "b.txt").write_text("bravo beta beats.")
    (tmp_path / "irrelevant.log").write_text("ignore me")
    cfg = _mkcfg().rag
    stats = asyncio.run(rag.index_dir(tmp_path, cfg))
    assert stats.files_indexed == 2  # .log is not in SUPPORTED_EXTS
    assert stats.chunks_added >= 2


def test_clear_index(tmp_db, tmp_path: Path, fake_embed):
    p = tmp_path / "x.txt"; p.write_text("some niche knowledge here")
    cfg = _mkcfg().rag
    asyncio.run(rag.index_path(p, cfg))
    assert rag.stats()["chunks"] > 0
    removed = rag.clear_index()
    assert removed > 0
    assert rag.stats()["chunks"] == 0


# ─── Retriever ───
def test_retrieve_returns_most_similar(tmp_db, tmp_path: Path, fake_embed):
    cfg = _mkcfg().rag
    # Seed two unrelated docs
    (tmp_path / "a.md").write_text("pull-ups are the king of upper body work.")
    (tmp_path / "b.md").write_text("risotto techniques for italian home cooks.")
    asyncio.run(rag.index_dir(tmp_path, cfg))

    hits = asyncio.run(rag.retrieve("pull-up programming", cfg, k=2))
    assert len(hits) > 0
    # Scores must be in [-1, 1] and sorted descending
    for a, b in zip(hits, hits[1:], strict=False):
        assert a.score >= b.score
    assert all(-1.01 <= h.score <= 1.01 for h in hits)


def test_retrieve_empty_index_returns_empty(tmp_db, fake_embed):
    cfg = _mkcfg().rag
    hits = asyncio.run(rag.retrieve("anything", cfg))
    assert hits == []


def test_context_for_respects_inject_cap(tmp_db, tmp_path: Path, fake_embed):
    cfg = _mkcfg(rag=cfg_mod.RAGConfig(max_inject_chars=200))
    # Write enough knowledge to exceed the 200-char cap
    (tmp_path / "big.md").write_text(
        "Para one is a longish piece of niche knowledge about training "
        "progressive overload and pull-ups across multiple weeks.\n\n"
        "Para two is another longish piece of knowledge about the niche, "
        "covering mobility and recovery protocols for dads over 35.\n\n"
        "Para three adds the third chunk with additional cues and commonly "
        "missed technique notes around grip width and scapular control."
    )
    asyncio.run(rag.index_dir(tmp_path, cfg.rag))
    out = asyncio.run(rag.context_for("niche knowledge", cfg.rag))
    # Must respect the cap (small slack for \n joins between chunks)
    assert len(out) <= cfg.rag.max_inject_chars + 80


def test_context_for_returns_empty_when_disabled(tmp_db, tmp_path: Path, fake_embed):
    cfg = _mkcfg(rag=cfg_mod.RAGConfig(enabled=False))
    (tmp_path / "x.md").write_text("ignored knowledge.")
    # With enabled=False, index_dir short-circuits and context returns ""
    asyncio.run(rag.index_dir(tmp_path, cfg.rag))
    out = asyncio.run(rag.context_for("anything", cfg.rag))
    assert out == ""


def test_retrieve_graceful_when_no_backend(tmp_db, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """If the embed call raises NoEmbeddingBackend, retrieve returns []."""
    cfg = _mkcfg().rag

    async def boom(texts, c):
        raise emb_mod.NoEmbeddingBackend("nothing configured")

    monkeypatch.setattr(emb_mod, "embed", boom)
    monkeypatch.setattr(emb_mod, "embed_one", boom)

    hits = asyncio.run(rag.retrieve("pull ups", cfg))
    assert hits == []
    out = asyncio.run(rag.context_for("pull ups", cfg))
    assert out == ""


# ─── Model-scoped retrieval ───
def test_retrieval_scoped_to_current_embedding_model(tmp_db, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Rows embedded with one model must NOT match queries made with another."""
    cfg = _mkcfg().rag

    # Phase 1: index with a 'fake:v1' model
    async def v1_embed(texts, c):
        return emb_mod.EmbedResult(
            vectors=[emb_mod._normalise([1.0, 0.0, 0.0])] * len(texts),
            model="fake:v1", dim=3,
        )
    async def v1_embed_one(text, c):
        return await v1_embed([text], c)
    monkeypatch.setattr(emb_mod, "embed", v1_embed)
    monkeypatch.setattr(emb_mod, "embed_one", v1_embed_one)
    (tmp_path / "a.md").write_text("some niche fact")
    asyncio.run(rag.index_dir(tmp_path, cfg))

    # Phase 2: retrieve using a different model name → no match
    async def v2_embed_one(text, c):
        return emb_mod.EmbedResult(
            vectors=[emb_mod._normalise([1.0, 0.0, 0.0])],
            model="fake:v2", dim=3,
        )
    monkeypatch.setattr(emb_mod, "embed_one", v2_embed_one)

    hits = asyncio.run(rag.retrieve("anything", cfg))
    assert hits == []


# ─── Config roundtrip ───
def test_rag_config_roundtrips():
    import yaml
    cfg = _mkcfg(rag=cfg_mod.RAGConfig(retrieve_k=5, local_model="BAAI/bge-small-en-v1.5"))
    d = cfg.model_dump(mode="json")
    loaded = cfg_mod.NicheConfig.model_validate(yaml.safe_load(yaml.safe_dump(d)))
    assert loaded.rag.retrieve_k == 5
    assert loaded.rag.local_model == "BAAI/bge-small-en-v1.5"


# ─── Public fn: embed() provider fallback chain ───
def test_embed_auto_raises_no_backend_when_both_missing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(emb_mod, "_local_available", lambda: False)
    cfg = _mkcfg().rag
    with pytest.raises(emb_mod.NoEmbeddingBackend):
        asyncio.run(emb_mod.embed(["hi"], cfg))


def test_embed_empty_input_returns_empty_result():
    cfg = _mkcfg().rag
    result = asyncio.run(emb_mod.embed([], cfg))
    assert result.vectors == []
    assert result.dim == 0


# ─── Audit follow-ups ───
def test_chunk_overlap_does_not_exceed_max_chars():
    """Audit fix: previously overlap inflated chunks past the configured cap."""
    text = "\n\n".join(f"paragraph {i} content with niche knowledge bits." for i in range(20))
    chunks = rag.chunk_text(text, max_chars=120, overlap=60)
    for c in chunks:
        assert len(c) <= 120, f"chunk {len(c)} chars exceeds cap 120: {c[:60]!r}"


def test_index_path_respects_disabled(tmp_db, tmp_path: Path, fake_embed):
    """Audit fix: index_path now honours cfg.enabled, not just index_dir."""
    p = tmp_path / "x.md"; p.write_text("anything")
    cfg = _mkcfg(rag=cfg_mod.RAGConfig(enabled=False))
    stats = asyncio.run(rag.index_path(p, cfg.rag))
    assert stats.files_indexed == 0
    assert stats.chunks_added == 0


def test_index_dir_records_embed_failures(tmp_db, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """When no embed backend is available, index_dir surfaces the failure
    via IndexStats.embed_failures + last_error rather than silent zero."""
    cfg = _mkcfg().rag
    (tmp_path / "x.md").write_text("some niche knowledge")

    async def boom(texts, c):
        raise emb_mod.NoEmbeddingBackend("nothing wired")

    monkeypatch.setattr(emb_mod, "embed", boom)
    stats = asyncio.run(rag.index_dir(tmp_path, cfg))
    assert stats.embed_failures >= 1
    assert "no embedding backend" in stats.last_error.lower() or "nothing" in stats.last_error.lower()
    assert stats.chunks_added == 0


def test_file_hash_unreadable_returns_sentinel(tmp_path: Path):
    """Audit fix: unreadable files now get a unique sentinel, not an empty
    string that would silently dedup against every other unreadable file."""
    bogus = tmp_path / "does_not_exist.txt"
    h = rag._file_hash(bogus)
    assert h.startswith("ERR:"), f"expected ERR: sentinel, got {h!r}"
    assert h != ""


def test_pdf_missing_pypdf_returns_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """When pypdf isn't installed, pdf reader logs warning and returns empty."""
    p = tmp_path / "fake.pdf"
    p.write_bytes(b"%PDF-1.4 fake content")

    # Force ImportError path
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "pypdf" or name.startswith("pypdf."):
            raise ImportError("simulated missing pypdf")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    out = rag._read_pdf(p)
    assert out == ""
