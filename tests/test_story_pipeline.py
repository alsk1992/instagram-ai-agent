"""Tests for story config, budget separation, and format picker pool selection."""
from __future__ import annotations

from pathlib import Path

import pytest

from instagram_ai_agent.content.generators import format_picker
from instagram_ai_agent.core import config as cfg_mod
from instagram_ai_agent.core import db
from instagram_ai_agent.core.budget import allowed


def _mkcfg(**kwargs):
    base = dict(
        niche="home calisthenics",
        sub_topics=["pullups"],
        target_audience="dads 35+",
        commercial=True,
        voice=cfg_mod.Voice(tone=["direct"], forbidden=[], persona="ex-office worker rebuilding fitness."),
        aesthetic=cfg_mod.Aesthetic(palette=["#000", "#fff"]),
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


def test_story_mix_normalises():
    mix = cfg_mod.StoryMix(
        story_quote=2.0, story_announcement=1.0, story_photo=1.0, story_video=0.0,
    )
    norm = mix.normalized()
    assert abs(sum(norm.values()) - 1.0) < 1e-6
    assert norm["story_quote"] == pytest.approx(0.5)
    assert norm["story_video"] == 0.0


def test_niche_includes_stories_by_default():
    cfg = _mkcfg()
    dumped = cfg.model_dump(mode="json")
    assert "stories" in dumped
    assert "story_quote" in dumped["stories"]


def test_format_picker_story_pool(tmp_db):
    cfg = _mkcfg(
        stories=cfg_mod.StoryMix(
            story_quote=1.0, story_announcement=0.0,
            story_photo=0.0, story_video=0.0,
        )
    )
    # Pool forced → only story_quote has weight >0
    assert format_picker.pick_next(cfg, kind="story") == "story_quote"


def test_format_picker_feed_pool(tmp_db):
    cfg = _mkcfg(
        formats=cfg_mod.FormatMix(
            meme=1.0, quote_card=0.0, carousel=0.0,
            reel_stock=0.0, reel_ai=0.0, photo=0.0,
        )
    )
    assert format_picker.pick_next(cfg, kind="feed") == "meme"


def test_budget_story_post_separate_from_post(tmp_db):
    cfg = _mkcfg()
    # Default Schedule: posts_per_day=1, stories_per_day=3
    assert allowed("post", cfg) == (True, 0, 1)
    assert allowed("story_post", cfg) == (True, 0, 3)
    # Log a feed post — story budget untouched
    db.action_log("post", "pk1", "ok", 0)
    ok, used, cap = allowed("post", cfg)
    assert (ok, used, cap) == (False, 1, 1)
    ok, used, cap = allowed("story_post", cfg)
    assert (ok, used, cap) == (True, 0, 3)


def test_story_format_queue_routing(tmp_db):
    """Story items in content_queue are found by the story poster, not feed."""
    from instagram_ai_agent.workers import poster, story_poster

    cfg = _mkcfg()
    feed_id = db.content_enqueue(
        format="meme", caption="feed", hashtags=[], media_paths=["/tmp/m.jpg"],
        phash="aaaa", critic_score=0.8, critic_notes="", generator="meme", status="approved",
    )
    story_id = db.content_enqueue(
        format="story_quote", caption="story", hashtags=[], media_paths=["/tmp/s.jpg"],
        phash="bbbb", critic_score=0.8, critic_notes="", generator="story_quote", status="approved",
    )

    feed_item = poster._next_feed_item()
    story_item = story_poster._next_story()

    assert feed_item is not None and feed_item["id"] == feed_id
    assert story_item is not None and story_item["id"] == story_id


def test_story_scheduling_uses_spread_hours(tmp_db):
    from instagram_ai_agent.workers.poster import _story_hours

    # Given 3 peak hours, the story schedule should include intermediate hours
    hours = _story_hours([14, 18, 21])
    assert len(hours) > 3
    assert min(hours) >= 8 and max(hours) <= 23
    assert all(isinstance(h, int) for h in hours)


def test_story_scheduling_respects_per_day_cap(tmp_db):
    from instagram_ai_agent.workers.poster import _schedule_pool
    from instagram_ai_agent.core.config import STORY_FORMATS

    cfg = _mkcfg()
    # Queue 4 story items, per_day=2 — only 2 get today's slots, rest roll over
    for i in range(4):
        db.content_enqueue(
            format="story_quote",
            caption=f"s{i}",
            hashtags=[],
            media_paths=[f"/tmp/s{i}.jpg"],
            phash=f"hash{i}",
            critic_score=0.8,
            critic_notes="",
            generator="story_quote",
            status="approved",
        )
    scheduled = _schedule_pool(
        cfg,
        pool=STORY_FORMATS,
        per_day=2,
        best_hours=[10, 20],
        action_key="story_post",
    )
    assert scheduled == 4
    # 4 items at 2/day fit in 2 distinct days, but when the test runs after
    # the last best-hour of day-0, overflow pushes into a 3rd calendar date.
    rows = db.content_list(status="approved", limit=10)
    dates = sorted({r["scheduled_for"][:10] for r in rows if r.get("scheduled_for")})
    assert 2 <= len(dates) <= 3
    # Every date has at most `per_day=2` items
    from collections import Counter
    per_date = Counter(r["scheduled_for"][:10] for r in rows if r.get("scheduled_for"))
    assert all(v <= 2 for v in per_date.values()), per_date
