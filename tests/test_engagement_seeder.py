"""Unit tests for the engagement seeder (deterministic, no network, no LLM)."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.brain import engagement_seeder
from src.core import config as cfg_mod
from src.core import db


def _mkcfg(**kwargs):
    base = dict(
        niche="home calisthenics",
        sub_topics=["pullups"],
        target_audience="dads 35+",
        commercial=True,
        voice=cfg_mod.Voice(tone=["direct"], forbidden=[], persona="ex-office worker rebuilding fitness."),
        aesthetic=cfg_mod.Aesthetic(palette=["#000", "#fff"]),
        hashtags=cfg_mod.HashtagPools(core=["calisthenics", "homeworkout", "dadfit"], growth=["fitover40"]),
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


def test_seed_competitors_respects_per_competitor_and_engaged_flag(tmp_db):
    cfg = _mkcfg(competitors=["a_user", "b_user"])
    # 3 posts per competitor, so per_competitor=2 leaves 1 unmarked each run
    for i in range(3):
        tmp_db.competitor_upsert(f"pk_a{i}", "a_user", f"post a{i}", 100 + i, 10, "2026-04-10T12:00:00Z")
        tmp_db.competitor_upsert(f"pk_b{i}", "b_user", f"post b{i}", 100 + i, 10, "2026-04-10T12:00:00Z")

    first = engagement_seeder.seed_from_competitors(cfg, per_competitor=2)
    assert first == 4   # 2 per competitor x 2 competitors

    # Already-engaged rows are skipped; only the 3rd per competitor remains
    second = engagement_seeder.seed_from_competitors(cfg, per_competitor=2)
    assert second == 2

    third = engagement_seeder.seed_from_competitors(cfg, per_competitor=2)
    assert third == 0  # nothing left with engaged=0


def test_seed_hashtags_caps_and_dedups_sameday(tmp_db):
    cfg = _mkcfg()
    # 5 posts in one tag, per_tag=3 → 3 go out first, 2 second, 0 third
    for i in range(5):
        tmp_db.hashtag_upsert("calisthenics", f"pk_h{i}", f"caption {i}", 500 + i, "2026-04-10T12:00:00Z")

    first = engagement_seeder.seed_from_hashtags(cfg, per_tag=3)
    assert first == 3

    second = engagement_seeder.seed_from_hashtags(cfg, per_tag=3)
    assert second == 2   # remaining 2 pulled (same-day dedup only stops *already-queued* pks)

    third = engagement_seeder.seed_from_hashtags(cfg, per_tag=3)
    assert third == 0    # all 5 now queued → nothing new


def test_seed_hashtags_skips_already_queued_pks(tmp_db):
    cfg = _mkcfg()
    tmp_db.hashtag_upsert("calisthenics", "pk_a", "a", 500, "2026-04-10T12:00:00Z")
    tmp_db.hashtag_upsert("calisthenics", "pk_b", "b", 400, "2026-04-10T12:00:00Z")
    # Pre-queue pk_a manually today
    tmp_db.engagement_enqueue("like", target_media="pk_a")
    # Now seed — pk_a should be skipped
    seeded = engagement_seeder.seed_from_hashtags(cfg, per_tag=5)
    assert seeded == 1
    # Verify the queue does not have pk_a twice
    conn = tmp_db.get_conn()
    count_a = conn.execute(
        "SELECT COUNT(*) c FROM engagement_queue WHERE target_media=?", ("pk_a",)
    ).fetchone()["c"]
    assert count_a == 1


def test_seed_no_competitors_is_noop(tmp_db):
    cfg = _mkcfg(competitors=[])
    assert engagement_seeder.seed_from_competitors(cfg) == 0


def test_clean_comment_strips_quotes_and_labels():
    assert engagement_seeder._clean_comment('"great hook"') == "great hook"
    assert engagement_seeder._clean_comment("Comment: hell yes") == "hell yes"
    assert engagement_seeder._clean_comment("line one\nline two") == "line one"
