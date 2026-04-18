"""Behavioural anti-detection layer — pre-post scroll, post cooldown,
typing delay, caption entropy check, aspect-ratio pre-flight,
client rotation, first-comment hashtags."""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.core import config as cfg_mod
from src.core import db
from src.plugins import human_mimic


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
def tmp_db(tmp_path: Path, monkeypatch):
    fresh = tmp_path / "brain.db"
    monkeypatch.setattr(cfg_mod, "DB_PATH", fresh)
    monkeypatch.setattr(db, "DB_PATH", fresh)
    db.close()
    db.init_db()
    yield db
    db.close()


# ─── HumanMimicConfig defaults ───
def test_human_mimic_config_defaults_all_on():
    cfg = _mkcfg()
    hm = cfg.human_mimic
    assert hm.pre_post_scroll is True
    assert hm.post_cooldown is True
    assert hm.comment_reply_delay is True
    assert hm.typing_delays is True
    assert hm.caption_entropy_check is True
    assert hm.aspect_ratio_check is True
    assert hm.rotate_client is True
    # First-comment hashtags is opt-in (most users don't want the split)
    assert hm.first_comment_hashtags is False


# ─── Post cooldown ───
def test_post_cooldown_zero_when_never_posted(tmp_db):
    assert human_mimic.post_cooldown_remaining_s() == 0.0
    assert human_mimic.post_cooldown_ok() is True


def test_post_cooldown_nonzero_after_stamp(tmp_db):
    human_mimic.stamp_post()
    remaining = human_mimic.post_cooldown_remaining_s()
    # Must be in the 30-90min window (1800-5400s)
    assert 1800 <= remaining <= 5400
    assert human_mimic.post_cooldown_ok() is False


def test_post_cooldown_decays_over_time(tmp_db, monkeypatch):
    # Stamp a post 2h ago — cooldown must be clear
    past = time.time() - 2 * 3600
    human_mimic.stamp_post(now_s=past)
    assert human_mimic.post_cooldown_remaining_s() == 0.0
    assert human_mimic.post_cooldown_ok() is True


def test_post_cooldown_threshold_is_deterministic_within_window(tmp_db):
    """Two calls in the same cooldown window return the SAME required
    silence — no random re-roll that could let a second check sneak
    through."""
    human_mimic.stamp_post()
    first = human_mimic.post_cooldown_remaining_s()
    time.sleep(0.01)
    second = human_mimic.post_cooldown_remaining_s()
    # Threshold is fixed; only elapsed changes (by ~10ms)
    assert abs((second - first) - (-0.01)) < 0.5


# ─── Typing delay ───
def test_typing_delay_is_length_proportional():
    short = human_mimic.typing_delay_s("hi")
    medium = human_mimic.typing_delay_s("This is a medium comment with some real words")
    long_text = "Lorem ipsum " * 20
    long_delay = human_mimic.typing_delay_s(long_text)
    # Relative ordering holds (with some randomness tolerance)
    assert short < medium < long_delay


def test_typing_delay_floor_and_ceiling():
    # Tiny input → floor
    assert human_mimic.typing_delay_s("") >= 0.5
    # Huge input → ceiling (18s)
    assert human_mimic.typing_delay_s("x" * 10_000) <= 18.0


# ─── Caption entropy ───
def test_caption_entropy_passes_unique_caption():
    recent = [
        "Your morning routine is killing you. Here's the fix.",
        "Three pullup mistakes that stall every beginner.",
    ]
    new = "Why your deadlift plateau is actually a mobility issue."
    assert human_mimic.captions_too_similar(new, recent) is False


def test_caption_entropy_catches_near_duplicate():
    recent = ["Your morning routine is killing you. Here's the fix."]
    new = "Your morning routine is killing you. Here is the fix."
    assert human_mimic.captions_too_similar(new, recent) is True


def test_caption_entropy_ignores_hashtag_block():
    """Hashtag block is stripped before similarity — a repeated tag
    set isn't the same as a repeated caption."""
    recent = ["Cold start killing results.\n\n#fitness #dadfit #calisthenics"]
    # Different body, same hashtags → must NOT flag
    new = "Recovery beats volume every time.\n\n#fitness #dadfit #calisthenics"
    assert human_mimic.captions_too_similar(new, recent) is False


def test_caption_entropy_tolerates_empty_inputs():
    assert human_mimic.captions_too_similar("", []) is False
    assert human_mimic.captions_too_similar("short", []) is False


def test_caption_entropy_skips_under_20_chars():
    recent = ["hello world"]
    assert human_mimic.captions_too_similar("hello world", recent) is False


# ─── Aspect ratio ───
def test_aspect_ratio_accepts_portrait_feed(tmp_path: Path):
    from PIL import Image
    p = tmp_path / "portrait.jpg"
    Image.new("RGB", (1080, 1350), (0, 0, 0)).save(p, "JPEG")
    assert human_mimic.validate_aspect_ratio(p, kind="feed") is True


def test_aspect_ratio_accepts_square_feed(tmp_path: Path):
    from PIL import Image
    p = tmp_path / "square.jpg"
    Image.new("RGB", (1080, 1080), (0, 0, 0)).save(p, "JPEG")
    assert human_mimic.validate_aspect_ratio(p, kind="feed") is True


def test_aspect_ratio_rejects_oddshape_for_feed(tmp_path: Path):
    from PIL import Image
    p = tmp_path / "wide.jpg"
    # 3:1 banner shape → way off
    Image.new("RGB", (1800, 600), (0, 0, 0)).save(p, "JPEG")
    assert human_mimic.validate_aspect_ratio(p, kind="feed") is False


def test_aspect_ratio_accepts_9_16_for_reel(tmp_path: Path):
    from PIL import Image
    p = tmp_path / "reel.jpg"
    Image.new("RGB", (1080, 1920), (0, 0, 0)).save(p, "JPEG")
    assert human_mimic.validate_aspect_ratio(p, kind="reel") is True


def test_aspect_ratio_rejects_square_for_reel(tmp_path: Path):
    from PIL import Image
    p = tmp_path / "sq.jpg"
    Image.new("RGB", (1080, 1080), (0, 0, 0)).save(p, "JPEG")
    assert human_mimic.validate_aspect_ratio(p, kind="reel") is False


def test_aspect_ratio_noop_on_missing_file(tmp_path: Path):
    """Non-fatal: missing file → True (don't block upload)."""
    assert human_mimic.validate_aspect_ratio(tmp_path / "ghost.jpg") is True


# ─── Client rotation ───
def test_should_rotate_client_false_when_fresh():
    assert human_mimic.should_rotate_client(client_age_s=60) is False
    assert human_mimic.should_rotate_client(client_age_s=3000) is False   # 50min
    # Over 2h could be True depending on the random threshold for this
    # process's seed, so we test the lower bound only.


def test_should_rotate_client_deterministic_for_same_seed():
    """Same seed_ts → same threshold → same decision. Prevents
    reload flicker on adjacent checks."""
    seed = 100.0
    a = human_mimic.should_rotate_client(client_age_s=10_000, seed_ts=seed)
    b = human_mimic.should_rotate_client(client_age_s=10_000, seed_ts=seed)
    assert a == b


def test_should_rotate_client_true_beyond_ceiling():
    """A client older than the max 4h threshold ALWAYS rotates, regardless of seed."""
    assert human_mimic.should_rotate_client(client_age_s=5 * 3600, seed_ts=1.0) is True
    assert human_mimic.should_rotate_client(client_age_s=5 * 3600, seed_ts=999.9) is True


# ─── Pre-post scroll (stubbed Client) ───
def test_pre_post_scroll_silent_on_missing_methods():
    """A Client without the expected helpers → scroll no-ops silently."""
    cl = MagicMock(spec=[])   # empty spec — no methods
    n = human_mimic.pre_post_scroll(cl)
    assert n == 0


def test_pre_post_scroll_calls_get_timeline_and_media_seen(monkeypatch):
    """Happy path: feed returned → media_seen called per touched item."""
    monkeypatch.setattr("time.sleep", lambda *_: None)  # no real sleeps

    fake_items = [MagicMock(pk=f"pk_{i}") for i in range(5)]
    cl = MagicMock()
    cl.get_timeline_feed.return_value = fake_items
    cl.media_seen.return_value = True

    n = human_mimic.pre_post_scroll(cl, min_items=3, max_items=5)
    assert n >= 3
    assert cl.media_seen.call_count >= 3


def test_pre_post_scroll_survives_feed_exception(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    cl = MagicMock()
    cl.get_timeline_feed.side_effect = RuntimeError("API flaked")
    # Must NOT propagate — scroll is optional
    assert human_mimic.pre_post_scroll(cl) == 0
