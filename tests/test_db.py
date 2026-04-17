"""Smoke tests against the SQLite layer using an in-tmpdir DB."""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture()
def tmp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point config.DB_PATH at a tmp file and reset the module-local connection."""
    from src.core import config, db

    fresh = tmp_path / "brain.db"
    monkeypatch.setattr(config, "DB_PATH", fresh)
    monkeypatch.setattr(db, "DB_PATH", fresh)
    # Reset thread-local connection
    db.close()
    db.init_db()
    yield db
    db.close()


def test_state_kv(tmp_db):
    tmp_db.state_set("foo", "bar")
    assert tmp_db.state_get("foo") == "bar"
    tmp_db.state_set_json("j", {"a": 1})
    assert tmp_db.state_get_json("j") == {"a": 1}


def test_content_queue_roundtrip(tmp_db):
    cid = tmp_db.content_enqueue(
        format="meme",
        caption="hello",
        hashtags=["a", "b"],
        media_paths=["/tmp/x.jpg"],
        phash="0000",
        critic_score=0.8,
        critic_notes="good",
        generator="meme",
        status="approved",
    )
    assert cid > 0
    row = tmp_db.content_get(cid)
    assert row["status"] == "approved"
    assert row["hashtags"] == ["a", "b"]
    assert row["media_paths"] == ["/tmp/x.jpg"]

    nxt = tmp_db.content_next_to_post()
    assert nxt and nxt["id"] == cid

    tmp_db.content_mark_posted(cid, "ig_pk_123")
    row = tmp_db.content_get(cid)
    assert row["status"] == "posted"
    assert row["ig_media_pk"] == "ig_pk_123"


def test_action_log_and_budget(tmp_db):
    tmp_db.action_log("like", None, "ok", 100)
    tmp_db.action_log("like", None, "ok", 90)
    tmp_db.action_log("like", None, "failed", 5)
    assert tmp_db.action_count_today("like") == 2  # only ok


def test_context_feed_priority(tmp_db):
    tmp_db.push_context("a", "low", priority=1)
    tmp_db.push_context("b", "hi", priority=5)
    tmp_db.push_context("c", "mid", priority=3)
    out = tmp_db.pop_context(limit=3)
    assert [r["source"] for r in out] == ["b", "c", "a"]
