"""Tests for coverage rotator, watcher multi-target, hashtag discovery,
comment replier filters, follow-back triage plumbing, RSS parsing, critic v2."""
from __future__ import annotations

from pathlib import Path

import pytest

from instagram_ai_agent.core import config as cfg_mod
from instagram_ai_agent.core import db


def _mkcfg(**kwargs):
    base = dict(
        niche="home calisthenics",
        sub_topics=["pullups", "mobility", "dad strength"],
        target_audience="dads 35+",
        commercial=True,
        voice=cfg_mod.Voice(tone=["direct"], forbidden=["grind"],
                            persona="ex-office worker rebuilding fitness."),
        aesthetic=cfg_mod.Aesthetic(palette=["#000", "#fff", "#c9a961"]),
        hashtags=cfg_mod.HashtagPools(
            core=["calisthenics", "homeworkout", "dadfit"],
            growth=["fitover40", "fitmotivation"],
        ),
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


# ─── Coverage rotator ───
def test_coverage_prefers_uncovered(tmp_db):
    from instagram_ai_agent.brain import coverage
    cfg = _mkcfg()
    # Cover two of three topics
    coverage.record_coverage("pullups")
    coverage.record_coverage("mobility")
    picks = [coverage.pick_sub_topic(cfg) for _ in range(20)]
    # "dad strength" (uncovered) should appear often in first picks
    # (algorithm: ANY uncovered topic wins outright)
    assert "dad strength" in picks


def test_coverage_report_sort(tmp_db):
    from instagram_ai_agent.brain import coverage
    cfg = _mkcfg()
    coverage.record_coverage("pullups")
    report = coverage.coverage_report(cfg)
    assert isinstance(report, list)
    topics = [t for t, _ in report]
    assert set(topics) == set(cfg.sub_topics)
    # "never" comes first (sort-ascending puts "" before ISO date strings)
    assert report[0][1] == "never"


# ─── Multi-target watcher ───
def test_all_watch_targets_unifies_legacy_and_list():
    cfg = _mkcfg(
        watch_target="primary",
        watch_targets=["secondary", "primary", "@third"],
    )
    assert cfg.all_watch_targets() == ["secondary", "primary", "third"]


def test_all_watch_targets_empty_when_nothing_set():
    cfg = _mkcfg()
    assert cfg.all_watch_targets() == []


# ─── RSS parsing (no network) ───
def test_rss_parser_rss2():
    from instagram_ai_agent.brain.news_feed import _parse_items
    xml = """<?xml version="1.0"?><rss version="2.0"><channel>
    <item><title>Hit a new PR</title><link>https://e.com/1</link><description>Body</description></item>
    <item><title>Mobility tip</title><link>https://e.com/2</link><description>desc</description></item>
    </channel></rss>"""
    items = _parse_items(xml)
    assert len(items) == 2
    assert items[0]["title"] == "Hit a new PR"
    assert items[1]["link"] == "https://e.com/2"


def test_rss_parser_atom():
    from instagram_ai_agent.brain.news_feed import _parse_items
    xml = """<?xml version="1.0"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <title>Atom item</title>
        <link href="https://e.com/a1"/>
        <summary>summary text</summary>
      </entry>
    </feed>"""
    items = _parse_items(xml)
    assert len(items) == 1
    assert items[0]["title"] == "Atom item"
    assert items[0]["link"] == "https://e.com/a1"


def test_rss_parser_garbage_returns_empty():
    from instagram_ai_agent.brain.news_feed import _parse_items
    assert _parse_items("this isn't xml") == []


# ─── Hashtag discovery ───
def test_hashtag_mining(tmp_db):
    from instagram_ai_agent.brain import hashtag_discovery
    cfg = _mkcfg()
    # Insert competitor posts with hashtags
    tmp_db.competitor_upsert(
        "p1", "rival1",
        "Great workout #pullupprogression #shoulderhealth #calisthenics",
        500, 20, "2026-04-15T10:00:00Z",
    )
    tmp_db.competitor_upsert(
        "p2", "rival2",
        "Mobility reset #shoulderhealth #mobilityroutine",
        800, 25, "2026-04-14T10:00:00Z",
    )
    tmp_db.competitor_upsert(
        "p3", "rival1",
        "Pull-up tips #pullupprogression",
        400, 10, "2026-04-16T10:00:00Z",
    )
    suggestions = hashtag_discovery.mine_from_competitors(cfg)
    tags = {s["tag"] for s in suggestions}
    # Already in our pool → excluded
    assert "calisthenics" not in tags
    # New tags → present
    assert "shoulderhealth" in tags
    assert "pullupprogression" in tags
    # shoulderhealth should rank well (appears on 2 different users, 1300 likes total)
    sorted_tags = [s["tag"] for s in suggestions]
    assert sorted_tags[0] in {"shoulderhealth", "pullupprogression"}


def test_hashtag_approve_merges(tmp_db):
    from instagram_ai_agent.brain import hashtag_discovery
    cfg = _mkcfg()
    original_growth = list(cfg.hashtags.growth)
    # Fake niche.yaml save path so we don't accidentally overwrite real config
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tf:
        cfg_mod.NICHE_PATH = Path(tf.name)
    added = hashtag_discovery.approve_into_growth(cfg, ["shoulderhealth", "#newtag"])
    assert added == 2
    assert "shoulderhealth" in cfg.hashtags.growth
    assert "newtag" in cfg.hashtags.growth
    assert len(cfg.hashtags.growth) == len(original_growth) + 2


# ─── Comment replier filters ───
def test_looks_like_spam_detects_links():
    from instagram_ai_agent.workers.comment_replier import _looks_like_spam
    assert _looks_like_spam("check out example.com/deal")
    assert _looks_like_spam("DM me for crypto profit")
    assert _looks_like_spam("🔥🔥🔥🔥")
    assert _looks_like_spam("@user1 @user2")
    assert _looks_like_spam("")
    assert not _looks_like_spam("love this progression")
    assert not _looks_like_spam("what mobility routine do you recommend?")


def test_clean_strips_quotes_and_labels():
    from instagram_ai_agent.workers.comment_replier import _clean
    assert _clean('"nice progress mate"') == "nice progress mate"
    assert _clean("Reply: thanks for the tip") == "thanks for the tip"
    assert _clean("line one\nline two") == "line one"


# ─── Inbound comment + follower upserts ───
def test_inbound_comment_upsert_idempotent(tmp_db):
    new1 = tmp_db.inbound_comment_upsert(
        "c1", media_pk="m1", username="alice", user_id="42",
        text="nice", created_at="", is_own=False,
    )
    new2 = tmp_db.inbound_comment_upsert(
        "c1", media_pk="m1", username="alice", user_id="42",
        text="nice", created_at="", is_own=False,
    )
    assert new1 is True
    assert new2 is False


def test_comments_to_reply_excludes_own_and_ignored(tmp_db):
    tmp_db.inbound_comment_upsert("c1", media_pk="m1", username="me", user_id="1",
                                  text="self", created_at="", is_own=True)
    tmp_db.inbound_comment_upsert("c2", media_pk="m1", username="alice", user_id="2",
                                  text="nice", created_at="", is_own=False)
    tmp_db.inbound_comment_upsert("c3", media_pk="m1", username="bob", user_id="3",
                                  text="spam", created_at="", is_own=False)
    tmp_db.inbound_comment_ignore("c3")
    todo = tmp_db.inbound_comments_to_reply(limit=10)
    assert [r["comment_pk"] for r in todo] == ["c2"]


def test_follower_upsert_and_triage(tmp_db):
    assert tmp_db.follower_upsert("42", "alice", "Alice A") is True
    assert tmp_db.follower_upsert("42", "alice", "Alice A") is False
    pending = tmp_db.followers_pending()
    assert len(pending) == 1
    tmp_db.follower_triage("42", "followed_back", "on-niche")
    assert tmp_db.followers_pending() == []


# ─── Reciprocal engagement ───
def test_reciprocal_queues_story_views(tmp_db):
    from instagram_ai_agent.workers import follow_back
    tmp_db.inbound_comment_upsert("c1", media_pk="m1", username="alice", user_id="42",
                                  text="nice", created_at="", is_own=False)
    tmp_db.inbound_comment_upsert("c2", media_pk="m1", username="bob", user_id="43",
                                  text="cool", created_at="", is_own=False)
    n = follow_back.queue_reciprocal_from_recent_comments(batch=5)
    assert n == 2
    # Re-running on same day is a no-op (sameday dedup)
    again = follow_back.queue_reciprocal_from_recent_comments(batch=5)
    assert again == 0


# ─── Critic v2 additions ───
def test_critic_rubric_mentions_new_dimensions():
    from instagram_ai_agent.content.critic import _RUBRIC
    for dim in ("originality", "relevance_now", "competitor_edge", "weak_spots"):
        assert dim in _RUBRIC


def test_critic_persona_block_includes_forbidden():
    from instagram_ai_agent.content.critic import _persona_block
    cfg = _mkcfg()
    out = _persona_block(cfg)
    assert "grind" in out


# ─── Safety knobs ───
def test_safety_candidate_knobs_default():
    cfg = _mkcfg()
    assert cfg.safety.caption_candidates == 3
    assert cfg.safety.image_candidates == 1
    assert cfg.safety.vision_critic is True
