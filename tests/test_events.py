"""Event calendar — Nager.Date parse, user events, window, dedup."""
from __future__ import annotations

import asyncio
from datetime import date, timedelta
from pathlib import Path

import pytest

from src.brain import events as events_mod
from src.core import config as cfg_mod
from src.core import db


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
def test_event_calendar_defaults_sane():
    cfg = _mkcfg()
    assert cfg.holidays_enabled is True
    assert cfg.holiday_country == "US"
    assert 1 <= cfg.events_lookahead_days <= 60
    assert cfg.events_calendar == []


def test_events_roundtrip_user_calendar():
    import yaml
    cfg = _mkcfg(events_calendar=[
        {"date": "2026-05-01", "label": "Launch", "note": "big drop"},
        {"date": "2026-05-20", "label": "Anniversary"},
    ])
    d = cfg.model_dump(mode="json")
    loaded = cfg_mod.NicheConfig.model_validate(yaml.safe_load(yaml.safe_dump(d)))
    assert len(loaded.events_calendar) == 2


# ─── Event dataclass + event_id ───
def test_event_id_is_deterministic():
    e1 = events_mod.Event(date=date(2026, 7, 4), label="Independence Day", source="holiday")
    e2 = events_mod.Event(date=date(2026, 7, 4), label="Independence Day", source="holiday")
    assert e1.event_id == e2.event_id


def test_event_id_differs_by_source_or_date():
    e1 = events_mod.Event(date=date(2026, 7, 4), label="Independence Day", source="holiday")
    e2 = events_mod.Event(date=date(2026, 7, 4), label="Independence Day", source="user")
    e3 = events_mod.Event(date=date(2026, 7, 5), label="Independence Day", source="holiday")
    assert len({e1.event_id, e2.event_id, e3.event_id}) == 3


def test_event_id_handles_non_alnum():
    e = events_mod.Event(date=date(2026, 5, 1), label="Labour Day / May Day!!!", source="holiday")
    # No non-alnum chars in the slug portion
    slug = e.event_id.split(":", 2)[2]
    assert all(c.isalnum() or c == "-" for c in slug)


# ─── Nager parse ───
def test_fetch_holidays_parses_valid_payload(monkeypatch: pytest.MonkeyPatch):
    payload = [
        {"date": "2026-01-01", "localName": "New Year's Day", "name": "New Year's Day"},
        {"date": "2026-07-04", "localName": "Independence Day", "name": "Independence Day"},
        {"date": "not-a-date", "localName": "Junk"},        # skipped
        {"localName": "No-date-entry"},                     # skipped
    ]

    class FakeResp:
        def raise_for_status(self): pass
        def json(self): return payload

    class FakeClient:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return None
        async def get(self, url):
            assert "nager.at" in url
            assert "/2026/US" in url
            return FakeResp()

    monkeypatch.setattr(events_mod.httpx, "AsyncClient", lambda *a, **k: FakeClient())

    out = asyncio.run(events_mod.fetch_holidays("us", 2026))
    assert len(out) == 2
    assert out[0].label == "New Year's Day"
    assert out[0].source == "holiday"
    assert out[1].date == date(2026, 7, 4)


def test_fetch_holidays_network_failure_returns_empty(monkeypatch: pytest.MonkeyPatch):
    class FakeClient:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return None
        async def get(self, url):
            raise RuntimeError("network down")

    monkeypatch.setattr(events_mod.httpx, "AsyncClient", lambda *a, **k: FakeClient())
    assert asyncio.run(events_mod.fetch_holidays("US", 2026)) == []


def test_fetch_holidays_empty_country():
    assert asyncio.run(events_mod.fetch_holidays("", 2026)) == []


# ─── User events ───
def test_user_events_parses_valid_entries():
    cfg = _mkcfg(events_calendar=[
        {"date": "2026-05-01", "label": "Launch", "note": "product drop"},
        {"date": "2026-05-20", "label": "Anniversary"},
        {"date": "bogus", "label": "skip"},
        {"date": "2026-06-01"},  # missing label
        {"label": "Hover"},       # missing date
    ])
    events = events_mod.user_events(cfg)
    assert len(events) == 2
    assert events[0].label == "Launch"
    assert events[0].note == "product drop"
    assert events[1].note == ""
    assert all(e.source == "user" for e in events)


# ─── Window filter ───
def test_in_window_sorted_ascending():
    today = date(2026, 5, 1)
    # Window is [today, today+days) — half-open. 14 days → inclusive up to 5/14
    events = [
        events_mod.Event(date(2026, 5, 10), "mid", "holiday"),
        events_mod.Event(date(2026, 5, 2), "near", "user"),
        events_mod.Event(date(2026, 4, 30), "past", "holiday"),        # excluded
        events_mod.Event(date(2026, 5, 30), "out", "holiday"),          # 30d > 14d
    ]
    out = events_mod.in_window(events, today=today, days=14)
    assert [e.label for e in out] == ["near", "mid"]


def test_in_window_includes_today_excludes_end():
    today = date(2026, 5, 1)
    events = [
        events_mod.Event(today, "today", "holiday"),
        events_mod.Event(today + timedelta(days=14), "end-day", "holiday"),  # half-open
    ]
    out = events_mod.in_window(events, today=today, days=14)
    assert [e.label for e in out] == ["today"]


# ─── Dedup ───
def test_run_once_pushes_fresh_events_only(tmp_db, monkeypatch: pytest.MonkeyPatch):
    cfg = _mkcfg(
        holidays_enabled=False,       # skip network path
        events_calendar=[
            {"date": _iso_today(), "label": "Today thing"},
            {"date": _iso_today(days=1), "label": "Tomorrow thing"},
        ],
    )

    pushed_once = asyncio.run(events_mod.run_once(cfg))
    assert pushed_once == 2

    # Second run same day → 0 new pushes (debounced)
    pushed_again = asyncio.run(events_mod.run_once(cfg))
    assert pushed_again == 0

    # Verify context_feed contains exactly the 2 entries from the first run
    rows = tmp_db.get_conn().execute(
        "SELECT text FROM context_feed WHERE source LIKE 'event.%'"
    ).fetchall()
    assert len(rows) == 2
    texts = [r["text"] for r in rows]
    assert any("Today thing" in t for t in texts)
    assert any("Tomorrow thing" in t for t in texts)


def test_upcoming_combines_holidays_and_user(tmp_db, monkeypatch: pytest.MonkeyPatch):
    # Stub Nager
    async def fake_holidays(country, year):
        return [events_mod.Event(date.today() + timedelta(days=3), "Fake Holiday", "holiday")]

    monkeypatch.setattr(events_mod, "fetch_holidays", fake_holidays)
    cfg = _mkcfg(
        holiday_country="US",
        events_calendar=[{"date": _iso_today(days=5), "label": "Launch day"}],
    )
    up = asyncio.run(events_mod.upcoming(cfg))
    sources = [e.source for e in up]
    assert "holiday" in sources
    assert "user" in sources


def test_upcoming_respects_disabled_holidays(tmp_db, monkeypatch: pytest.MonkeyPatch):
    async def fake_holidays(country, year):
        raise AssertionError("should not be called when holidays_enabled=False")

    monkeypatch.setattr(events_mod, "fetch_holidays", fake_holidays)
    cfg = _mkcfg(
        holidays_enabled=False,
        events_calendar=[{"date": _iso_today(days=1), "label": "User only"}],
    )
    up = asyncio.run(events_mod.upcoming(cfg))
    assert len(up) == 1
    assert up[0].label == "User only"


def test_run_once_prunes_by_event_date_not_push_date(tmp_db):
    """Audit fix: prune must hinge on the EVENT date, not when we pushed it.

    Previously an event pushed 10 days ago whose date is still 5 days away
    got pruned — and then re-pushed on the next cycle.
    """
    # Event pushed 10 days ago, but its event_date is 5 days IN THE FUTURE
    # → must NOT be pruned (event hasn't happened yet).
    future_id = f"{(date.today() + timedelta(days=5)).isoformat()}:user:future"
    db.state_set_json("events_seen", {
        future_id: {
            "event_date": (date.today() + timedelta(days=5)).isoformat(),
            "pushes": [(date.today() - timedelta(days=10)).isoformat()],
        },
    })

    events_mod._record_seen([])   # trigger prune via a no-op call
    store = db.state_get_json("events_seen") or {}
    assert future_id in store, "future event must survive prune"

    # Separately: a recent push on an event 4 days PAST must be pruned
    past_id = f"{(date.today() - timedelta(days=4)).isoformat()}:holiday:past"
    store[past_id] = {
        "event_date": (date.today() - timedelta(days=4)).isoformat(),
        "pushes": [date.today().isoformat()],
    }
    db.state_set_json("events_seen", store)
    events_mod._record_seen([])
    store = db.state_get_json("events_seen") or {}
    assert past_id not in store, "past-event record must be pruned"


def test_run_once_does_not_spam_over_window(tmp_db):
    """Audit fix: a single event should NOT push every day it sits in the window.

    With REPUSH_LEAD_DAYS = {0, 1}, a 14-day-out event pushes once on first
    sight, is silent for days 13..2 remaining, then pushes again on T-1 and T-0.
    """
    cfg = _mkcfg(
        holidays_enabled=False,
        events_calendar=[{"date": _iso_today(days=10), "label": "Far off"}],
    )

    # Day 0 — first sighting, should push
    first = asyncio.run(events_mod.run_once(cfg))
    assert first == 1

    # Same-day second run — no re-push
    again = asyncio.run(events_mod.run_once(cfg))
    assert again == 0


def test_run_once_repushes_on_T_minus_one(tmp_db, monkeypatch: pytest.MonkeyPatch):
    """Policy: an event re-pushes on the day before it fires."""
    # Event is 5 days out in real calendar time
    target = date.today() + timedelta(days=5)
    cfg = _mkcfg(
        holidays_enabled=False,
        events_calendar=[{"date": target.isoformat(), "label": "Launch"}],
    )

    # First push "today"
    assert asyncio.run(events_mod.run_once(cfg)) == 1

    # Fast-forward the agent's perception of today → 4 days later = T-1
    monkeypatch.setattr(events_mod, "_today_utc", lambda: target - timedelta(days=1))
    assert asyncio.run(events_mod.run_once(cfg)) == 1

    # Same T-1 same day — no double-push
    assert asyncio.run(events_mod.run_once(cfg)) == 0

    # Now T-0
    monkeypatch.setattr(events_mod, "_today_utc", lambda: target)
    assert asyncio.run(events_mod.run_once(cfg)) == 1


def test_seen_store_migrates_old_schema(tmp_db):
    """Old schema was {id: iso_date}; new is {id: {event_date, pushes}}.
    Loading must not crash or drop data."""
    old_id = f"{_iso_today(days=3)}:user:legacy"
    db.state_set_json("events_seen", {old_id: date.today().isoformat()})
    migrated = events_mod._seen()
    assert old_id in migrated
    assert "event_date" in migrated[old_id]
    assert "pushes" in migrated[old_id]


# ─── helpers ───
def _iso_today(*, days: int = 0) -> str:
    return (date.today() + timedelta(days=days)).isoformat()