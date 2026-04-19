"""Dashboard route smoke tests via FastAPI TestClient."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from instagram_ai_agent.core import config as cfg_mod
from instagram_ai_agent.core import db


@pytest.fixture()
def prepped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Isolated DB and niche.yaml
    monkeypatch.setattr(cfg_mod, "DB_PATH", tmp_path / "brain.db")
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "brain.db")
    db.close()
    db.init_db()

    niche_path = tmp_path / "niche.yaml"
    cfg = cfg_mod.NicheConfig(
        niche="home calisthenics",
        sub_topics=["pullups"],
        target_audience="dads 35+",
        commercial=True,
        voice=cfg_mod.Voice(tone=["direct"], forbidden=[], persona="ex-office worker rebuilding fitness."),
        aesthetic=cfg_mod.Aesthetic(palette=["#000", "#fff", "#c9a961"]),
        hashtags=cfg_mod.HashtagPools(core=["calisthenics", "homeworkout", "dadfit"]),
    )
    niche_path.write_text(
        yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    monkeypatch.setattr(cfg_mod, "NICHE_PATH", niche_path)

    # Pre-populate some state the dashboard will render
    db.content_enqueue(
        format="meme", caption="test caption", hashtags=["a", "b"],
        media_paths=[], phash="abcd1234", critic_score=0.82, critic_notes="ok",
        generator="meme", status="approved",
    )
    db.action_log("post", "fake_pk", "ok", 100)
    db.health_record(100, 20, 3, 0.04, False)

    from instagram_ai_agent.dashboard import create_app
    app = create_app()
    yield TestClient(app)
    db.close()


def test_index_renders(prepped: TestClient):
    r = prepped.get("/")
    assert r.status_code == 200
    body = r.text
    assert "home calisthenics" in body
    assert "queue" in body.lower()
    assert "warmup" in body.lower()


def test_api_state(prepped: TestClient):
    r = prepped.get("/api/state")
    assert r.status_code == 200
    data = r.json()
    assert data["niche"] == "home calisthenics"
    assert "queue_by_status" in data
    assert data["queue_by_status"].get("approved") == 1


def test_api_queue_filter(prepped: TestClient):
    r = prepped.get("/api/queue?status_filter=approved")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 1
    assert data[0]["format"] == "meme"


def test_media_path_traversal_blocked(prepped: TestClient):
    r = prepped.get("/media/../../../../etc/passwd")
    assert r.status_code in (403, 404)


def test_auth_enforced_when_env_set(prepped: TestClient, monkeypatch: pytest.MonkeyPatch):
    # Rebuild with auth env set
    monkeypatch.setenv("DASH_USER", "admin")
    monkeypatch.setenv("DASH_PASS", "secret")
    from instagram_ai_agent.dashboard import create_app
    app = create_app()
    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 401
    r2 = client.get("/", auth=("admin", "secret"))
    assert r2.status_code == 200
    r3 = client.get("/", auth=("admin", "wrong"))
    assert r3.status_code == 401


# ─── approve / reject actions ───
def test_review_page_empty_state(prepped: TestClient):
    """No pending items → page renders with 'Nothing to review' empty state."""
    r = prepped.get("/review")
    assert r.status_code == 200
    assert "Nothing to review" in r.text


def test_review_approve_endpoint_flips_status(prepped: TestClient):
    cid = db.content_enqueue(
        format="carousel", caption="pending item", hashtags=[], media_paths=[],
        phash="ffff0000", critic_score=0.7, critic_notes="", generator="carousel",
        status="pending_review",
    )
    r = prepped.post(f"/api/queue/{cid}/approve")
    assert r.status_code == 200
    data = r.json()
    assert data["id"] == cid
    assert data["status"] == "approved"

    rows = db.content_list(status="approved")
    assert any(row["id"] == cid for row in rows)


def test_review_reject_endpoint_flips_status(prepped: TestClient):
    cid = db.content_enqueue(
        format="meme", caption="bad draft", hashtags=[], media_paths=[],
        phash="0000abcd", critic_score=0.3, critic_notes="weak", generator="meme",
        status="pending_review",
    )
    r = prepped.post(f"/api/queue/{cid}/reject")
    assert r.status_code == 200
    assert r.json()["status"] == "rejected"


def test_review_page_lists_pending(prepped: TestClient):
    """Pending item with caption must be visible on /review."""
    db.content_enqueue(
        format="quote_card", caption="visible pending caption here for human review",
        hashtags=[], media_paths=[], phash="beadbeef",
        critic_score=0.6, critic_notes="", generator="quote_card",
        status="pending_review",
    )
    r = prepped.get("/review")
    assert r.status_code == 200
    assert "visible pending caption" in r.text
    assert 'data-action="approve"' in r.text
    assert 'data-action="reject"' in r.text
