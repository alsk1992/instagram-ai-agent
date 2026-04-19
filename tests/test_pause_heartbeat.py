"""Round-4 ops-tier changes: pause/resume, heartbeat, status pulse."""
from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from instagram_ai_agent import cli
from instagram_ai_agent.core import config as cfg_mod
from instagram_ai_agent.core import db


@pytest.fixture()
def tmp_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    niche = tmp_path / "niche.yaml"
    env = tmp_path / ".env"
    dbf = tmp_path / "brain.db"
    monkeypatch.setattr(cfg_mod, "NICHE_PATH", niche)
    monkeypatch.setattr(cfg_mod, "ENV_PATH", env)
    monkeypatch.setattr(cfg_mod, "DB_PATH", dbf)
    monkeypatch.setattr(cli, "NICHE_PATH", niche)
    monkeypatch.setattr(cli, "ENV_PATH", env)
    monkeypatch.setattr(db, "DB_PATH", dbf)
    monkeypatch.setattr(cfg_mod, "ROOT", tmp_path)
    monkeypatch.setattr(cfg_mod, "DATA_DIR", tmp_path / "data")
    db.close()
    db.init_db()

    # Seed a minimal niche.yaml so _require_niche() passes on status call
    import yaml
    cfg = cfg_mod.NicheConfig(
        niche="home calisthenics",
        sub_topics=["pullups"],
        target_audience="dads 35+",
        commercial=True,
        voice=cfg_mod.Voice(tone=["direct"], forbidden=[], persona="ex-office worker"),
        aesthetic=cfg_mod.Aesthetic(palette=["#000", "#fff", "#c9a961"]),
        hashtags=cfg_mod.HashtagPools(core=["calisthenics", "homeworkout", "dadfit"]),
    )
    niche.write_text(
        yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    yield tmp_path
    db.close()


# ─── pause / resume ───
def test_pause_sets_state_flag(tmp_state):
    runner = CliRunner()
    r = runner.invoke(cli.app, ["pause"])
    assert r.exit_code == 0
    assert "paused" in r.output.lower()
    assert (db.state_get("paused") or "").lower() in ("1", "true", "yes")


def test_resume_clears_flag(tmp_state):
    db.state_set("paused", "1")
    runner = CliRunner()
    r = runner.invoke(cli.app, ["resume"])
    assert r.exit_code == 0
    assert "resumed" in r.output.lower()
    assert (db.state_get("paused") or "").lower() == "0"


def test_resume_idempotent_when_not_paused(tmp_state):
    runner = CliRunner()
    r = runner.invoke(cli.app, ["resume"])
    assert r.exit_code == 0
    assert "already running" in r.output.lower() or "no change" in r.output.lower()


# ─── status reflects pause + heartbeat ───
def test_status_shows_paused(tmp_state):
    db.state_set("paused", "1")
    runner = CliRunner()
    r = runner.invoke(cli.app, ["status"])
    assert r.exit_code == 0
    assert "PAUSED" in r.output


def test_status_shows_heartbeat_when_recent(tmp_state):
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    db.state_set("last_heartbeat", now)
    runner = CliRunner()
    r = runner.invoke(cli.app, ["status"])
    assert r.exit_code == 0
    assert "heartbeat" in r.output.lower()
    # Recent heartbeat renders as green "X.X min ago"
    assert "min ago" in r.output or "h ago" in r.output


def test_status_flags_stale_heartbeat_as_red(tmp_state):
    # Old heartbeat > 60 min → red "orchestrator may be down"
    db.state_set("last_heartbeat", "2020-01-01T00:00:00Z")
    runner = CliRunner()
    r = runner.invoke(cli.app, ["status"])
    assert r.exit_code == 0
    assert "may be down" in r.output.lower()


def test_status_shows_upcoming_posts(tmp_state):
    # Seed an approved item with a scheduled_for so status surfaces it
    cid = db.content_enqueue(
        format="carousel", caption="first line / second / third",
        hashtags=[], media_paths=[], phash="abc",
        critic_score=0.8, critic_notes="", generator="carousel",
        status="approved",
    )
    db.get_conn().execute(
        "UPDATE content_queue SET scheduled_for=? WHERE id=?",
        ("2099-01-01T12:00:00Z", cid),
    )
    runner = CliRunner()
    r = runner.invoke(cli.app, ["status"])
    assert r.exit_code == 0
    assert "next 3 scheduled" in r.output.lower()
    assert "carousel" in r.output


# ─── orchestrator heartbeat logic (unit-test the helper) ───
def test_paused_helper_reflects_state(tmp_state):
    from instagram_ai_agent import orchestrator

    assert orchestrator._paused() is False
    db.state_set("paused", "1")
    assert orchestrator._paused() is True
    db.state_set("paused", "0")
    assert orchestrator._paused() is False
    # Accepts truthy variants
    for truthy in ("1", "true", "True", "YES", "yes"):
        db.state_set("paused", truthy)
        assert orchestrator._paused() is True, f"failed for {truthy!r}"
