"""Idea bank — seed ingest, picker, recency window, commercial filter."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from instagram_ai_agent.brain import idea_bank
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


# ─── Seed ingest ───
def test_seed_file_exists_and_parses():
    assert idea_bank.SEED_PATH.exists(), "data/ideas/seed.json must ship with the repo"
    data = json.loads(idea_bank.SEED_PATH.read_text())
    assert data.get("_license") == "CC0 (ig-agent curated)"
    assert len(data.get("ideas", [])) >= 40


def test_seed_from_file_populates_table(tmp_db):
    n = idea_bank.seed_from_file()
    assert n >= 40
    assert idea_bank.count() >= 40
    # Every row is commercial-safe
    for row in idea_bank.license_breakdown():
        lic = row["license"]
        assert lic.upper() not in {"CC-BY-NC", "CC-BY-NC-SA", "RESEARCH-ONLY"}


def test_seed_is_idempotent(tmp_db):
    idea_bank.seed_from_file()
    before = idea_bank.count()
    idea_bank.seed_from_file()   # second run: UNIQUE constraint means 0 new rows
    after = idea_bank.count()
    assert after == before


# ─── Picker ───
def test_pick_for_returns_an_idea(tmp_db):
    idea_bank.seed_from_file()
    cfg = _mkcfg()
    picked = idea_bank.pick_for(cfg, format_name="meme")
    assert picked is not None
    # format_hint must match meme OR be 'any'
    assert picked.format_hint in ("meme", "any")


def test_pick_for_unknown_format_still_returns_any(tmp_db):
    idea_bank.seed_from_file()
    cfg = _mkcfg()
    picked = idea_bank.pick_for(cfg, format_name="nonexistent_format")
    # All-`any` rows should still match
    assert picked is not None


def test_pick_respects_recent_window(tmp_db):
    idea_bank.seed_from_file()
    cfg = _mkcfg()
    # Count the eligible pool for `carousel` (has the most entries of any format)
    pool = db.get_conn().execute(
        "SELECT COUNT(*) c FROM ideas WHERE format_hint='carousel' OR format_hint='any'"
    ).fetchone()["c"]
    # Iterate up to min(pool-1, RECENT_WINDOW) — beyond that the picker must
    # fall back to repeats, which is expected behaviour.
    iterations = min(pool - 1, idea_bank.RECENT_WINDOW)
    assert iterations >= 5, f"seed has too few carousel/any ideas for this test (got {pool})"

    seen: list[int] = []
    for _ in range(iterations):
        idea = idea_bank.pick_for(cfg, format_name="carousel")
        assert idea is not None and idea.id is not None
        assert idea.id not in seen, "idea repeated inside recency window"
        seen.append(idea.id)
        idea_bank.mark_used(idea.id)

    # Recent list is pruned to RECENT_WINDOW
    assert len(idea_bank._recent_picks()) <= idea_bank.RECENT_WINDOW


def test_pick_falls_back_when_all_recent_used(tmp_db):
    """If every eligible idea is in the recent window, picker falls back to
    ignoring recency rather than returning None."""
    # Insert only one idea
    db.get_conn().execute(
        """
        INSERT INTO ideas (archetype, hook_formula, format_hint, license)
        VALUES ('lonely', 'only one here', 'meme', 'CC0')
        """
    )
    # Force it into the recent list
    only_id = db.get_conn().execute("SELECT id FROM ideas").fetchone()["id"]
    db.state_set_json(idea_bank._RECENT_PICK_KEY, [only_id])
    cfg = _mkcfg()
    picked = idea_bank.pick_for(cfg, format_name="meme")
    assert picked is not None
    assert picked.id == only_id


def test_commercial_only_excludes_non_commercial(tmp_db):
    conn = db.get_conn()
    conn.execute("INSERT INTO ideas (archetype, hook_formula, format_hint, license) VALUES ('safe', 'a', 'meme', 'CC0')")
    conn.execute("INSERT INTO ideas (archetype, hook_formula, format_hint, license) VALUES ('nc',   'b', 'meme', 'CC-BY-NC')")
    cfg = _mkcfg()

    cc0_count = 0
    other_count = 0
    # Drain the pool via marking; with commercial_only=True the NC row must never show
    for _ in range(10):
        idea = idea_bank.pick_for(cfg, format_name="meme", commercial_only=True)
        if idea is None:
            break
        if idea.license == "CC0":
            cc0_count += 1
        else:
            other_count += 1
        if idea.id is not None:
            idea_bank.mark_used(idea.id)
    assert cc0_count >= 1
    assert other_count == 0


def test_commercial_false_allows_everything(tmp_db):
    conn = db.get_conn()
    conn.execute("INSERT INTO ideas (archetype, hook_formula, format_hint, license) VALUES ('nc', 'x', 'meme', 'CC-BY-NC')")
    cfg = _mkcfg()
    idea = idea_bank.pick_for(cfg, format_name="meme", commercial_only=False)
    assert idea is not None


# ─── Score adjustment ───
def test_adjust_score_clamps(tmp_db):
    conn = db.get_conn()
    conn.execute("INSERT INTO ideas (archetype, hook_formula) VALUES ('x', 'y')")
    idea_id = conn.execute("SELECT id FROM ideas WHERE archetype='x'").fetchone()["id"]
    idea_bank.adjust_score(idea_id, +2.0)
    got = conn.execute("SELECT score FROM ideas WHERE id=?", (idea_id,)).fetchone()["score"]
    assert got <= 1.5
    idea_bank.adjust_score(idea_id, -10.0)
    got = conn.execute("SELECT score FROM ideas WHERE id=?", (idea_id,)).fetchone()["score"]
    assert got >= 0.0


# ─── Format breakdown ───
def test_format_breakdown_returns_mix(tmp_db):
    idea_bank.seed_from_file()
    rows = idea_bank.format_breakdown()
    formats = {r["format_hint"] for r in rows}
    # At minimum "any" and some specific formats are represented
    assert "any" in formats or len(formats) >= 2


# ─── Pipeline injection smoke ───
# ─── Audit follow-ups ───
def test_insert_many_returns_only_new_rows(tmp_db):
    """Second seed run reports 0 inserted (the counter no longer lies)."""
    first = idea_bank.seed_from_file()
    second = idea_bank.seed_from_file()
    assert first > 0
    assert second == 0


def test_mark_used_fires_only_after_successful_enqueue(tmp_db, monkeypatch):
    """A failed generation must NOT consume a recency slot."""
    import asyncio
    from instagram_ai_agent.content import pipeline as pipe

    idea_bank.seed_from_file()
    cfg = _mkcfg()

    # Stub _dispatch to raise — simulating a generator failure
    async def boom(format_name, cfg_, trend_context):
        raise RuntimeError("generator failed")

    monkeypatch.setattr(pipe, "_dispatch", boom)

    # Before: no recent picks
    assert idea_bank._recent_picks() == []

    # generate_one will try (cfg.safety.critic_max_regens + 1) times and
    # bail. With every dispatch raising, no enqueue happens.
    asyncio.run(pipe.generate_one(cfg, format_override="meme"))

    # After: still no recent picks recorded (mark_used fires post-enqueue only)
    assert idea_bank._recent_picks() == []
    # And the first archetype's use_count is unchanged
    row = db.get_conn().execute("SELECT SUM(use_count) s FROM ideas").fetchone()
    assert row["s"] == 0


def test_is_commercial_license_accepts_known_good():
    ok = (
        "CC0", "CC0 1.0", "CC-BY", "CC-BY-SA", "CC BY 4.0",
        "Apache-2.0", "MIT", "BSD-3-Clause", "Unlicense", "public-domain",
        "pixabay", "user-declared", "CC0 (ig-agent curated)",
    )
    for lic in ok:
        assert idea_bank.is_commercial_license(lic), f"{lic!r} should be commercial-safe"


def test_is_commercial_license_rejects_non_commercial():
    bad = (
        "CC-BY-NC", "CC BY-NC 4.0", "CC-BY-NC-SA", "CC-BY-NC-ND",
        "non-commercial", "research-only", "S-Lab License 1.0",
        "Coqui Public Model License", "Llama 2 Community License",
        "", None,
    )
    for lic in bad:
        assert not idea_bank.is_commercial_license(lic), f"{lic!r} should be rejected"


def test_seed_from_external_mocked_awesome_chatgpt(tmp_db, monkeypatch):
    """HTTP + CSV contract for the awesome-chatgpt source."""
    import httpx as _httpx

    csv_body = (
        "act,prompt\n"
        "\"Linux Terminal\",\"I want you to act as a linux terminal. Respond as the terminal would.\"\n"
        "\"Copywriter\",\"Act as a direct-response copywriter and write a hook.\"\n"
    )

    class FakeResp:
        text = csv_body
        def raise_for_status(self): pass

    class FakeClient:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return None
        def get(self, url):
            assert "awesome-chatgpt-prompts" in url
            return FakeResp()

    monkeypatch.setattr(_httpx, "Client", FakeClient)

    inserted = idea_bank.seed_from_external("awesome-chatgpt")
    assert inserted == 2
    rows = db.get_conn().execute(
        "SELECT archetype, license, source FROM ideas WHERE source='awesome-chatgpt'"
    ).fetchall()
    assert len(rows) == 2
    assert {r["archetype"] for r in rows} == {"linux_terminal", "copywriter"}
    # Every external row stamped CC0
    assert all(r["license"] == "CC0" for r in rows)


def test_seed_from_external_unknown_source_raises(tmp_db):
    with pytest.raises(ValueError, match="unknown external source"):
        idea_bank.seed_from_external("does-not-exist")


def test_all_formats_have_coverage(tmp_db):
    """Every FormatMix/StoryMix format should have ≥1 direct archetype or rely
    on ``any`` fallback; story/human/AI formats had zero direct coverage pre-fix.
    """
    idea_bank.seed_from_file()
    expected = {
        "meme", "quote_card", "carousel", "reel_stock", "reel_ai", "photo",
        "human_photo",
        "story_quote", "story_announcement", "story_photo", "story_video",
        "story_human",
    }
    rows = db.get_conn().execute(
        "SELECT DISTINCT format_hint FROM ideas"
    ).fetchall()
    present = {r["format_hint"] for r in rows}
    # 'any' provides universal fallback — don't require direct entries, but
    # assert it's present.
    assert "any" in present
    missing = expected - present
    # After the audit fix, no format should have zero direct archetypes
    assert missing == set(), f"formats with zero direct coverage: {missing}"


def test_pipeline_meta_records_archetype(tmp_db, monkeypatch: pytest.MonkeyPatch):
    """When generate_one enqueues content, the chosen archetype lands in meta."""
    import asyncio
    from instagram_ai_agent.content import pipeline as pipe
    from instagram_ai_agent.content.generators.base import GeneratedContent

    idea_bank.seed_from_file()
    cfg = _mkcfg()

    # Stub _dispatch to avoid network/ffmpeg
    async def fake_dispatch(format_name, cfg_, trend_context, *, contrarian=False):
        # Verify trend_context contains the archetype tag
        assert "[archetype:" in trend_context
        return GeneratedContent(
            format=format_name,
            media_paths=[str(tmp_db.DB_PATH)],  # bogus but exists
            visible_text="stub",
            caption_context="stub",
            generator="stub",
            meta={"stub": True},
        )

    monkeypatch.setattr(pipe, "_dispatch", fake_dispatch)

    # Stub caption candidates + critic to avoid LLM
    async def fake_caption_candidates(cfg, format_name, content, *, n, knowledge=None, **_kw):
        return [{
            "caption": "hook\n\n#a #b",
            "hashtags": ["a", "b"],
            "visible_text": "",
            "image_path": None,
        }]

    async def fake_rank(cfg, *, format_name, candidates, recent_captions, knowledge=None, **_kw):
        out = []
        for c in candidates:
            out.append({**c, "critique": {"overall": 0.9, "verdict": "approve", "reasons": "ok"}})
        return out

    monkeypatch.setattr(pipe, "_build_caption_candidates", fake_caption_candidates)
    from instagram_ai_agent.content import critic as critic_mod
    monkeypatch.setattr(critic_mod, "rank_candidates", fake_rank)

    # Avoid dedup IO by stubbing compute_phash
    from instagram_ai_agent.content import dedup
    monkeypatch.setattr(dedup, "compute_phash", lambda p: "0" * 32)
    monkeypatch.setattr(dedup, "is_duplicate", lambda h, thr, lookback=60: (False, None))

    cid = asyncio.run(pipe.generate_one(cfg, format_override="meme"))
    assert cid is not None
    row = db.content_get(cid)
    assert row is not None
    assert row["meta"].get("archetype"), "chosen archetype must be recorded in meta"
