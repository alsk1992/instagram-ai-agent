"""Round-5 keyless feeds: HackerNews, Dev.to, Wikipedia OTD, Openverse."""
from __future__ import annotations

from pathlib import Path

import pytest

from instagram_ai_agent.brain import devto, hackernews, wiki_otd
from instagram_ai_agent.core import config as cfg_mod
from instagram_ai_agent.core import db
from instagram_ai_agent.plugins import openverse


def _mkcfg(**overrides):
    base = dict(
        niche="home calisthenics",
        sub_topics=["pullups"],
        target_audience="dads 35+",
        commercial=True,
        voice=cfg_mod.Voice(tone=["direct"], forbidden=[], persona="ex-office worker"),
        aesthetic=cfg_mod.Aesthetic(palette=["#000", "#fff", "#c9a961"]),
        hashtags=cfg_mod.HashtagPools(core=["calisthenics", "homeworkout", "dadfit"]),
    )
    base.update(overrides)
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


# ─── HackerNews ───
@pytest.mark.asyncio
async def test_hackernews_fetches_and_filters(monkeypatch, tmp_db):
    async def fake_client_get(self, url, params=None):
        class _R:
            status_code = 200
            def raise_for_status(self): return None
            def json(self_inner):
                return {"hits": [
                    {"title": "pullup form is a scam — you should try this",
                     "url": "https://hn.example/1",
                     "points": 400, "num_comments": 120, "objectID": "1"},
                    {"title": "unrelated js framework release",
                     "url": "https://hn.example/2",
                     "points": 10, "num_comments": 3, "objectID": "2"},
                ]}
        return _R()

    async def fake_generate_json(task, prompt, *, system, max_tokens):
        return {"picks": [{"i": 0, "angle": "riff on the pullup-form debate"}]}

    import httpx
    monkeypatch.setattr(httpx.AsyncClient, "__aenter__", lambda self: self.__class__.__aenter__.__wrapped__(self) if hasattr(self.__class__.__aenter__, "__wrapped__") else None)
    # Simpler: patch the module's httpx.AsyncClient entirely
    class FakeClient:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url, params=None):
            class _R:
                status_code = 200
                def raise_for_status(self): return None
                def json(self_inner):
                    return {"hits": [
                        {"title": "niche-relevant story",
                         "url": "https://hn.example/1",
                         "points": 400, "num_comments": 120, "objectID": "1"},
                    ]}
            return _R()
    monkeypatch.setattr(hackernews.httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(hackernews, "generate_json", fake_generate_json)

    n = await hackernews.run_once(_mkcfg())
    assert n == 1
    # Verify the pick landed on context_feed
    ctx = db.pop_context(limit=5)
    assert any("niche-relevant story" in r["text"] for r in ctx)


@pytest.mark.asyncio
async def test_hackernews_handles_fetch_failure(monkeypatch, tmp_db):
    class FakeClient:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url, params=None):
            raise RuntimeError("network down")
    monkeypatch.setattr(hackernews.httpx, "AsyncClient", FakeClient)

    n = await hackernews.run_once(_mkcfg())
    assert n == 0


# ─── Dev.to ───
@pytest.mark.asyncio
async def test_devto_disabled_without_tags(monkeypatch, tmp_db):
    calls = {"n": 0}
    class FakeClient:
        def __init__(self, *a, **k): calls["n"] += 1
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
    monkeypatch.setattr(devto.httpx, "AsyncClient", FakeClient)

    n = await devto.run_once(_mkcfg())  # devto_tags empty
    assert n == 0
    assert calls["n"] == 0


@pytest.mark.asyncio
async def test_devto_fetches_and_filters(monkeypatch, tmp_db):
    class FakeClient:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url, params=None):
            class _R:
                status_code = 200
                def raise_for_status(self): return None
                def json(self_inner):
                    return [{"title": "10 fitness habits for engineers",
                             "url": "https://dev.to/x/1",
                             "description": "dev productivity + fitness",
                             "public_reactions_count": 250,
                             "comments_count": 30}]
            return _R()

    async def fake_generate_json(task, prompt, *, system, max_tokens):
        return {"picks": [{"i": 0, "angle": "fitness-for-devs carousel"}]}

    monkeypatch.setattr(devto.httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(devto, "generate_json", fake_generate_json)

    n = await devto.run_once(_mkcfg(devto_tags=["productivity", "fitness"]))
    assert n == 1


# ─── Wikipedia On This Day ───
@pytest.mark.asyncio
async def test_wiki_otd_disabled_by_default(monkeypatch, tmp_db):
    calls = {"n": 0}
    class FakeClient:
        def __init__(self, *a, **k): calls["n"] += 1
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
    monkeypatch.setattr(wiki_otd.httpx, "AsyncClient", FakeClient)

    n = await wiki_otd.run_once(_mkcfg())
    assert n == 0
    assert calls["n"] == 0


@pytest.mark.asyncio
async def test_wiki_otd_pulls_events_when_enabled(monkeypatch, tmp_db):
    class FakeClient:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url):
            class _R:
                status_code = 200
                def raise_for_status(self): return None
                def json(self_inner):
                    return {"events": [
                        {"year": 1969, "text": "Apollo 11 landed",
                         "pages": [{"content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Apollo_11"}}}]},
                        {"year": 1844, "text": "gym-related historical event"},
                    ]}
            return _R()

    async def fake_generate_json(task, prompt, *, system, max_tokens):
        return {"picks": [{"i": 0, "angle": "anniversary post"}]}

    monkeypatch.setattr(wiki_otd.httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(wiki_otd, "generate_json", fake_generate_json)

    n = await wiki_otd.run_once(_mkcfg(wiki_otd_enabled=True))
    assert n == 1


# ─── Openverse ───
@pytest.mark.asyncio
async def test_openverse_search_returns_commercial_safe(monkeypatch):
    class FakeClient:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url, params=None):
            class _R:
                status_code = 200
                def raise_for_status(self): return None
                def json(self_inner):
                    return {"results": [
                        {"url": "https://cdn.example/a.jpg",
                         "thumbnail": "https://cdn.example/a.jpg",
                         "license": "by-nc",  # excluded — non-commercial
                         "license_url": "",
                         "title": "NC image"},
                        {"url": "https://cdn.example/b.jpg",
                         "thumbnail": "https://cdn.example/b.jpg",
                         "license": "cc0",
                         "license_url": "",
                         "title": "Public domain pullup"},
                    ]}
            return _R()
    monkeypatch.setattr(openverse.httpx, "AsyncClient", FakeClient)

    img = await openverse.search("pullup", commercial_only=True)
    assert img is not None
    assert img["license"] == "cc0"
    assert img["title"] == "Public domain pullup"


@pytest.mark.asyncio
async def test_openverse_no_results_returns_none(monkeypatch):
    class FakeClient:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url, params=None):
            class _R:
                status_code = 200
                def raise_for_status(self): return None
                def json(self_inner):
                    return {"results": []}
            return _R()
    monkeypatch.setattr(openverse.httpx, "AsyncClient", FakeClient)
    img = await openverse.search("xyz")
    assert img is None


def test_openverse_attribution_cc0_empty():
    assert openverse.attribution_line({"license": "cc0", "title": "x", "creator": "y"}) == ""


def test_openverse_attribution_cc_by_formats():
    line = openverse.attribution_line({
        "license": "by",
        "title": "Pullup photo",
        "creator": "jane",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
    })
    assert "Pullup photo" in line
    assert "jane" in line
    assert "CC BY" in line


# ─── Niche config new flags ───
def test_niche_config_defaults_disable_new_feeds():
    cfg = _mkcfg()
    assert cfg.hackernews_keywords == []
    assert cfg.devto_tags == []
    assert cfg.wiki_otd_enabled is False
