"""DM funnel state machine tests — no LLM/IG calls."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

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


def test_dm_upsert_idempotent(tmp_db):
    tmp_db.dm_upsert_contact("alice", source="hashtag:fit", priority=1)
    tmp_db.dm_upsert_contact("alice", source="competitor:x", priority=3)
    rows = tmp_db.get_conn().execute("SELECT * FROM dm_contacts WHERE username='alice'").fetchall()
    assert len(rows) == 1
    # Priority is monotonic-up, source sticks to first non-null
    assert rows[0]["priority"] == 3


def test_dm_stage_advance(tmp_db):
    tmp_db.dm_upsert_contact("bob", source="hashtag:cal")
    tmp_db.dm_advance("bob", "targeted")
    due = tmp_db.dm_contacts_due("targeted")
    assert len(due) == 1 and due[0]["username"] == "bob"


def test_dm_due_respects_next_action_at(tmp_db):
    tmp_db.dm_upsert_contact("carol", source="h:a")
    future = (datetime.now(timezone.utc) + timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%SZ")
    tmp_db.dm_advance("carol", "targeted", next_after_iso=future)
    # Due should be empty because next_action_at is in the future
    due = tmp_db.dm_contacts_due("targeted")
    assert due == []


def test_dm_record_and_step_count(tmp_db):
    tmp_db.dm_upsert_contact("dave", source="h:a")
    tmp_db.dm_record_message("dave", "out", "hi", step=0)
    tmp_db.dm_record_message("dave", "out", "follow", step=1)
    tmp_db.dm_record_message("dave", "in", "hello back")
    assert tmp_db.dm_step_count("dave", "out") == 2
    assert tmp_db.dm_step_count("dave", "in") == 1
    last = tmp_db.dm_last_out("dave")
    assert last and last["body"] == "follow"


def test_interleave_helper():
    from instagram_ai_agent.workers.dm_worker import _interleave
    a = [{"u": 1}, {"u": 2}, {"u": 3}]
    b = [{"u": "a"}, {"u": "b"}]
    out = _interleave(a, b)
    assert [x["u"] for x in out] == [1, "a", 2, "b", 3]


def test_cooldown_gate():
    from instagram_ai_agent.workers.dm_worker import _cooldown_ok, COOLDOWN_HOURS

    # Never messaged → OK to send
    assert _cooldown_ok({"last_action_at": None}) is True
    # Messaged just now → blocked
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert _cooldown_ok({"last_action_at": now_iso}) is False
    # Messaged long ago → OK
    past = (datetime.now(timezone.utc) - timedelta(hours=COOLDOWN_HOURS)).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert _cooldown_ok({"last_action_at": past}) is True


def test_dm_sanitise_strips_quotes_and_labels():
    from instagram_ai_agent.workers.dm_worker import _sanitise
    assert _sanitise('"Hi Hey there"') == "Hi Hey there"
    assert _sanitise("Message: real body") == "real body"
    assert _sanitise("line one\nline two") == "line one"
