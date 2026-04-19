"""Voice fingerprint — picker ranking + block builder + empty-history fallback."""
from __future__ import annotations

from pathlib import Path

import pytest

from instagram_ai_agent.content import voice_fingerprint
from instagram_ai_agent.core import config as cfg_mod
from instagram_ai_agent.core import db


@pytest.fixture()
def tmp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fresh = tmp_path / "brain.db"
    monkeypatch.setattr(cfg_mod, "DB_PATH", fresh)
    monkeypatch.setattr(db, "DB_PATH", fresh)
    db.close()
    db.init_db()
    yield db
    db.close()


def _enqueue(caption: str, *, status: str, score: float, media_paths=None):
    return db.content_enqueue(
        format="meme",
        caption=caption,
        hashtags=[],
        media_paths=media_paths or [],
        phash="",
        critic_score=score,
        critic_notes=None,
        generator="test",
        status=status,
    )


# ─── Picker ───
def test_empty_history_returns_empty_list(tmp_db):
    assert voice_fingerprint.pick_voice_examples() == []


def test_picker_prefers_posted_over_approved(tmp_db):
    posted_long = "I ran the 4-minute CNS primer for 6 weeks — first pullup arrived at week 5. Not a routine. A reset."
    approved_long = "Three mobility fixes that unlocked my shoulders after years of office work. Took 10 minutes a day."
    _enqueue(posted_long, status="posted", score=0.9)
    _enqueue(approved_long, status="approved", score=0.95)  # higher score BUT approved

    examples = voice_fingerprint.pick_voice_examples(n=5)
    assert len(examples) == 2
    # posted must come first in the picker order, before approved fallback
    assert examples[0] == posted_long
    assert examples[1] == approved_long


def test_picker_ranks_by_critic_score(tmp_db):
    low = "The 2-week block periodisation fix that kept my deadlift progressing past the stall plateau at 140kg."
    mid = "My shoulder mobility routine shifted when I swapped band pull-aparts for scapular pullups at week 4."
    high = "Stopped eating in a deficit Tuesdays. Strength numbers on Thursdays jumped 8% in three weeks running."
    _enqueue(low, status="posted", score=0.5)
    _enqueue(mid, status="posted", score=0.75)
    _enqueue(high, status="posted", score=0.9)

    examples = voice_fingerprint.pick_voice_examples(n=3)
    assert examples == [high, mid, low]


def test_picker_filters_too_short(tmp_db):
    _enqueue("too short", status="posted", score=0.9)
    long = "Three mobility fixes that unlocked my shoulders after years of office work. Took 10 minutes a day."
    _enqueue(long, status="posted", score=0.6)

    examples = voice_fingerprint.pick_voice_examples()
    assert examples == [long]


def test_picker_strips_hashtag_block(tmp_db):
    caption = (
        "Stopped eating in a deficit Tuesdays. Strength numbers on Thursdays jumped 8% in three weeks running.\n\n"
        "#calisthenics #homeworkout #dadfit"
    )
    _enqueue(caption, status="posted", score=0.9)

    [got] = voice_fingerprint.pick_voice_examples()
    assert "#calisthenics" not in got
    assert "Stopped eating" in got


# ─── Block builder ───
def test_block_empty_on_no_examples():
    assert voice_fingerprint.build_voice_block([]) == ""


def test_block_empty_on_single_example():
    # 1 example is noise, not a fingerprint
    assert voice_fingerprint.build_voice_block(["only one"]) == ""


def test_block_formats_examples():
    a = "Stopped eating in a deficit Tuesdays. Strength jumped 8% on Thursdays."
    b = "Three mobility fixes that unlocked my shoulders after years of office work."
    block = voice_fingerprint.build_voice_block([a, b])
    assert "VOICE FINGERPRINT" in block
    assert "example 1:" in block
    assert "example 2:" in block
    assert a in block
    assert b in block
    # Must tell the model to borrow cadence not topics
    assert "topics" in block.lower()


def test_block_limits_to_requested_count(tmp_db):
    for i in range(10):
        cap = (
            f"Caption number {i} with enough body text to satisfy the minimum "
            f"length filter for voice examples — entry {i}."
        )
        _enqueue(cap, status="posted", score=0.5 + i * 0.01)

    examples = voice_fingerprint.pick_voice_examples(n=3)
    assert len(examples) == 3
