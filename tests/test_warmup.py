"""Warmup phase scaling + budget gating."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest

from src.core import config as cfg_mod
from src.core import db, warmup
from src.core.budget import allowed


def _mkcfg():
    return cfg_mod.NicheConfig(
        niche="home calisthenics",
        sub_topics=["pullups"],
        target_audience="dads 35+",
        commercial=True,
        voice=cfg_mod.Voice(tone=["direct"], forbidden=[], persona="ex-office worker rebuilding fitness."),
        aesthetic=cfg_mod.Aesthetic(palette=["#000", "#fff"]),
        hashtags=cfg_mod.HashtagPools(core=["calisthenics", "homeworkout", "dadfit"]),
    )


@pytest.fixture()
def tmp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fresh = tmp_path / "brain.db"
    monkeypatch.setattr(cfg_mod, "DB_PATH", fresh)
    monkeypatch.setattr(db, "DB_PATH", fresh)
    db.close()
    db.init_db()
    yield db
    db.close()


def _set_warmup_start(days_ago: int) -> None:
    start_date = date.today() - timedelta(days=days_ago)
    db.state_set("warmup_start", start_date.isoformat())


def test_warmup_not_started_returns_full_caps(tmp_db):
    cfg = _mkcfg()
    budget = warmup.effective_caps(cfg)
    assert budget.multiplier == 1.0
    assert budget.allow_posts is True
    assert budget.caps["post"] == cfg.schedule.posts_per_day


def test_warmup_day_1_silent(tmp_db):
    _set_warmup_start(0)  # day 1
    cfg = _mkcfg()
    b = warmup.effective_caps(cfg)
    assert b.day == 1
    assert b.phase_label == "silent"
    assert b.allow_posts is False
    assert b.allow_dms is False
    assert b.caps["post"] == 0
    assert b.caps["story_post"] == 0
    assert b.caps["dm"] == 0
    assert b.caps["like"] == int(cfg.budget.likes * 0.20)


def test_warmup_day_8_posts_unlocked(tmp_db):
    _set_warmup_start(7)  # day 8
    cfg = _mkcfg()
    b = warmup.effective_caps(cfg)
    assert b.phase_label == "post-safe"
    assert b.allow_posts is True
    assert b.allow_dms is False
    assert b.caps["post"] == int(cfg.schedule.posts_per_day * 0.6)


def test_warmup_day_15_full_caps(tmp_db):
    _set_warmup_start(14)  # day 15
    cfg = _mkcfg()
    b = warmup.effective_caps(cfg)
    assert b.phase_label == "complete"
    assert b.multiplier == 1.0
    assert b.caps["post"] == cfg.schedule.posts_per_day


def test_budget_allowed_respects_warmup(tmp_db):
    _set_warmup_start(0)  # day 1 — posts forbidden
    cfg = _mkcfg()
    ok, used, cap = allowed("post", cfg)
    assert ok is False
    assert cap == 0
    ok, used, cap = allowed("like", cfg)
    assert cap == int(cfg.budget.likes * 0.20)


def test_ensure_started_idempotent(tmp_db):
    warmup.ensure_started()
    first = db.state_get("warmup_start")
    warmup.ensure_started()  # calling again must not reset
    second = db.state_get("warmup_start")
    assert first == second
    assert first is not None
