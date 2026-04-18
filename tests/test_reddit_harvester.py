"""Reddit question harvester — PRAW contract, filters, dedup, safety."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from instagram_ai_agent.brain import reddit_harvester as rh
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


# ─── Config defaults ───
def test_reddit_defaults_sane():
    cfg = _mkcfg()
    assert cfg.reddit_enabled is True
    assert cfg.reddit_subs == []
    assert 1 <= cfg.reddit_posts_per_sub <= 50
    assert cfg.reddit_lookback_hours >= 1


def test_reddit_config_roundtrips():
    import yaml
    cfg = _mkcfg(reddit_subs=["bodyweightfitness", "AskMen"], reddit_min_score=10)
    d = cfg.model_dump(mode="json")
    loaded = cfg_mod.NicheConfig.model_validate(yaml.safe_load(yaml.safe_dump(d)))
    assert loaded.reddit_subs == ["bodyweightfitness", "AskMen"]
    assert loaded.reddit_min_score == 10


# ─── Question-title heuristics ───
@pytest.mark.parametrize("title,expected", [
    ("How do I do a muscle-up?", True),
    ("What's the best pull-up progression", True),   # implicit question via "what"
    ("Why my pull-ups plateau at 10", True),
    ("Should I train to failure?", True),
    ("Just hit my first one-arm pull-up!", False),
    ("Mobility reset routine for tight shoulders", False),
    ("", False),
    ("   ", False),
    ("Is this form correct?", True),
    ("\"Can I start calisthenics at 40?\"", True),
    ("Best post-workout meal", False),
])
def test_is_question_title(title, expected):
    assert rh.is_question_title(title) is expected


# ─── Sensitive-sub filter ───
def test_is_sensitive_blocks_nsfw_flags():
    assert rh.is_sensitive("fitness", over_18=True, nsfw_post=False) is True
    assert rh.is_sensitive("fitness", over_18=False, nsfw_post=True) is True
    assert rh.is_sensitive("gonewild", over_18=False, nsfw_post=False) is True
    assert rh.is_sensitive("GoneWild", over_18=False, nsfw_post=False) is True
    assert rh.is_sensitive("calisthenics", over_18=False, nsfw_post=False) is False


# ─── Availability / creds ───
def test_praw_available_false_without_install(monkeypatch: pytest.MonkeyPatch):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "praw":
            raise ImportError("not installed")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert rh._praw_available() is False


def test_creds_configured(monkeypatch: pytest.MonkeyPatch):
    for k in ("REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USER_AGENT"):
        monkeypatch.delenv(k, raising=False)
    assert rh._creds_configured() is False

    monkeypatch.setenv("REDDIT_CLIENT_ID", "x")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "y")
    monkeypatch.setenv("REDDIT_USER_AGENT", "ig-agent test")
    assert rh._creds_configured() is True


def test_get_reddit_returns_none_when_disabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)
    assert rh._get_reddit() is None


# ─── Fetch flow with a fake PRAW shape ───
def _fake_post(*, pid: str, title: str, score: int = 10, comments: int = 3,
               over_18: bool = False, created_offset_hours: float = 1.0,
               permalink: str = "/r/test/comments/x/") -> SimpleNamespace:
    ts = (datetime.now(timezone.utc) - timedelta(hours=created_offset_hours)).timestamp()
    return SimpleNamespace(
        id=pid,
        title=title,
        score=score,
        num_comments=comments,
        over_18=over_18,
        created_utc=ts,
        permalink=permalink,
        author=SimpleNamespace(name="fakeuser"),
    )


def _fake_subreddit(name: str, *, over18: bool = False, posts: list | None = None) -> SimpleNamespace:
    posts = posts or []

    def top(time_filter="day", limit=10):
        for p in posts[:limit]:
            yield p

    return SimpleNamespace(over18=over18, top=top, display_name=name)


class _FakeReddit:
    def __init__(self, subs: dict[str, SimpleNamespace]):
        self._subs = subs

    def subreddit(self, name: str) -> SimpleNamespace:
        return self._subs[name]


def test_fetch_filters_to_questions_above_min_score(monkeypatch: pytest.MonkeyPatch):
    posts = [
        _fake_post(pid="a", title="How do I do a muscle-up?", score=50),
        _fake_post(pid="b", title="Just hit 10 reps today!", score=200),   # not a question
        _fake_post(pid="c", title="What grip for pull-ups?", score=3),      # below min_score
        _fake_post(pid="d", title="Can I train twice a day?", score=25),
    ]
    fake = _FakeReddit({"bodyweightfitness": _fake_subreddit("bodyweightfitness", posts=posts)})
    monkeypatch.setattr(rh, "_get_reddit", lambda: fake)

    cfg = _mkcfg(reddit_subs=["bodyweightfitness"], reddit_min_score=5, reddit_posts_per_sub=10)
    out = rh.fetch_questions(cfg)
    ids = [q.post_id for q in out]
    assert ids == ["a", "d"] or ids == ["d", "a"]   # sorted by score desc
    assert out[0].score >= out[-1].score


def test_fetch_skips_over_18_sub(monkeypatch: pytest.MonkeyPatch):
    nsfw = _fake_subreddit("adult_sub", over18=True, posts=[
        _fake_post(pid="x", title="How do I ...?", score=999),
    ])
    safe = _fake_subreddit("fitness", posts=[
        _fake_post(pid="y", title="How to improve form?", score=40),
    ])
    fake = _FakeReddit({"adult_sub": nsfw, "fitness": safe})
    monkeypatch.setattr(rh, "_get_reddit", lambda: fake)

    cfg = _mkcfg(reddit_subs=["adult_sub", "fitness"])
    out = rh.fetch_questions(cfg)
    assert all(q.subreddit != "adult_sub" for q in out)
    assert any(q.subreddit == "fitness" for q in out)


def test_fetch_skips_over_18_posts(monkeypatch: pytest.MonkeyPatch):
    sub = _fake_subreddit("mixed", posts=[
        _fake_post(pid="ok", title="How to deload?", score=30),
        _fake_post(pid="nsfw", title="Should I ...?", score=30, over_18=True),
    ])
    fake = _FakeReddit({"mixed": sub})
    monkeypatch.setattr(rh, "_get_reddit", lambda: fake)

    out = rh.fetch_questions(_mkcfg(reddit_subs=["mixed"]))
    assert [q.post_id for q in out] == ["ok"]


def test_fetch_respects_lookback_hours(monkeypatch: pytest.MonkeyPatch):
    recent = _fake_post(pid="r", title="How to deload?", score=30, created_offset_hours=2)
    stale = _fake_post(pid="s", title="What about protein?", score=30, created_offset_hours=48)
    sub = _fake_subreddit("fitness", posts=[recent, stale])
    fake = _FakeReddit({"fitness": sub})
    monkeypatch.setattr(rh, "_get_reddit", lambda: fake)

    out = rh.fetch_questions(_mkcfg(reddit_subs=["fitness"], reddit_lookback_hours=24))
    assert [q.post_id for q in out] == ["r"]


def test_fetch_handles_normalised_sub_names(monkeypatch: pytest.MonkeyPatch):
    """Users may write `r/fitness`, `/r/fitness`, or just `fitness`."""
    sub = _fake_subreddit("fitness", posts=[_fake_post(pid="a", title="How?", score=20)])
    calls = []

    class Recording(_FakeReddit):
        def subreddit(self, name: str):
            calls.append(name)
            return sub

    monkeypatch.setattr(rh, "_get_reddit", lambda: Recording({"fitness": sub}))

    rh.fetch_questions(_mkcfg(reddit_subs=["r/fitness", "/r/fitness", "fitness"]))
    assert calls == ["fitness", "fitness", "fitness"]


def test_fetch_returns_empty_when_no_praw(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(rh, "_get_reddit", lambda: None)
    assert rh.fetch_questions(_mkcfg(reddit_subs=["any"])) == []


def test_fetch_skips_blocked_subs_even_if_user_adds_them(monkeypatch: pytest.MonkeyPatch):
    """Even if user adds gonewild to their config, we never harvest it."""
    sub = _fake_subreddit("gonewild", posts=[_fake_post(pid="x", title="How?", score=999)])
    fake = _FakeReddit({"gonewild": sub})
    monkeypatch.setattr(rh, "_get_reddit", lambda: fake)
    out = rh.fetch_questions(_mkcfg(reddit_subs=["gonewild"]))
    assert out == []


# ─── Dedup + run_once ───
def test_run_once_pushes_once_then_dedupes(tmp_db, monkeypatch: pytest.MonkeyPatch):
    posts = [
        _fake_post(pid="a", title="How do I train?", score=50),
        _fake_post(pid="b", title="What helps recovery?", score=30),
    ]
    fake = _FakeReddit({"fitness": _fake_subreddit("fitness", posts=posts)})
    monkeypatch.setattr(rh, "_get_reddit", lambda: fake)

    cfg = _mkcfg(reddit_subs=["fitness"])
    first = asyncio.run(rh.run_once(cfg))
    assert first == 2
    second = asyncio.run(rh.run_once(cfg))
    assert second == 0

    # Context feed holds both entries, priority 3
    rows = tmp_db.get_conn().execute(
        "SELECT text, priority FROM context_feed WHERE source LIKE 'reddit.%'"
    ).fetchall()
    assert len(rows) == 2
    assert all(r["priority"] == 3 for r in rows)


def test_run_once_no_op_when_disabled(tmp_db, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(rh, "_get_reddit", lambda: object())
    cfg = _mkcfg(reddit_enabled=False, reddit_subs=["fitness"])
    assert asyncio.run(rh.run_once(cfg)) == 0


def test_run_once_no_op_when_empty_subs(tmp_db):
    cfg = _mkcfg(reddit_subs=[])
    assert asyncio.run(rh.run_once(cfg)) == 0


def test_run_once_survives_per_sub_failure(tmp_db, monkeypatch: pytest.MonkeyPatch):
    good = _fake_subreddit("fitness", posts=[_fake_post(pid="a", title="How?", score=10)])

    class Broken(_FakeReddit):
        def subreddit(self, name):
            if name == "broken":
                class Bomb:
                    @property
                    def over18(self): raise RuntimeError("boom")
                return Bomb()
            return good

    monkeypatch.setattr(rh, "_get_reddit", lambda: Broken({"fitness": good}))

    pushed = asyncio.run(rh.run_once(_mkcfg(reddit_subs=["broken", "fitness"])))
    assert pushed == 1


def test_seen_store_prunes_old_entries(tmp_db):
    """Entries older than 2× lookback_hours are dropped on write."""
    now = datetime.now(timezone.utc).timestamp()
    stale_ts = now - 300 * 3600   # 300 hours ago
    db.state_set_json("reddit_questions_seen", {"fitness:old": stale_ts})
    rh._record_seen([
        rh.RedditQuestion(post_id="new", subreddit="fitness", title="How?",
                          score=10, num_comments=1, url="", created_utc=now, author=""),
    ], lookback_hours=24)
    store = db.state_get_json("reddit_questions_seen") or {}
    assert "fitness:old" not in store
    assert "fitness:new" in store