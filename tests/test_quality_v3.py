"""Tests for image ranker, retro learning, story viewer, ComfyUI plumbing."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.core import config as cfg_mod
from src.core import db


def _mkcfg(**kwargs):
    base = dict(
        niche="home calisthenics",
        sub_topics=["pullups", "mobility"],
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


# ─── Image ranker ───
@pytest.mark.asyncio
async def test_image_rank_single_shortcut():
    from src.content import image_rank
    cfg = _mkcfg()
    out = await image_rank.rank(cfg, ["/tmp/only.jpg"])
    assert out == [{"path": "/tmp/only.jpg", "score": 1.0, "reason": "only candidate"}]


@pytest.mark.asyncio
async def test_image_rank_no_vision_no_local_preserves_order(monkeypatch: pytest.MonkeyPatch):
    """With local_aesthetic disabled AND no vision provider, rank preserves
    input order — first candidate takes the top slot."""
    from src.content import image_rank

    monkeypatch.setattr(image_rank, "_vision_ready", lambda: False)
    cfg = _mkcfg(safety=cfg_mod.Safety(local_aesthetic=False, vision_critic=True))
    ranked = await image_rank.rank(cfg, ["/tmp/a.jpg", "/tmp/b.jpg", "/tmp/c.jpg"])
    assert [r["path"] for r in ranked] == ["/tmp/a.jpg", "/tmp/b.jpg", "/tmp/c.jpg"]
    assert ranked[0]["score"] == 1.0
    assert ranked[1]["score"] == 0.0


@pytest.mark.asyncio
async def test_image_rank_sorts_by_score(monkeypatch: pytest.MonkeyPatch):
    from src.content import image_rank

    monkeypatch.setattr(image_rank, "_vision_ready", lambda: True)

    # Fake data_url so we don't need real files
    monkeypatch.setattr(image_rank, "_data_url", lambda p: f"data:fake:{p}")

    scores = {"/tmp/a.jpg": 0.4, "/tmp/b.jpg": 0.9, "/tmp/c.jpg": 0.7}

    async def fake_desc(url, question=""):
        path = url.replace("data:fake:", "")
        return f"score: {scores[path]}\nreason: fake"

    monkeypatch.setattr(image_rank, "describe_image", fake_desc)

    cfg = _mkcfg()
    ranked = await image_rank.rank(cfg, list(scores))
    assert ranked[0]["path"] == "/tmp/b.jpg"
    assert ranked[-1]["path"] == "/tmp/a.jpg"
    assert ranked[0]["score"] == pytest.approx(0.9)


# ─── Retro learning ───
def test_top_posts_order_by_engagement(tmp_db):
    from src.brain import retro

    now = db.now_iso()
    # Pretend 3 posts with different engagement
    db.post_record("p1", None, "meme", "hook A")
    db.post_record("p2", None, "carousel", "hook B")
    db.post_record("p3", None, "reel_stock", "hook C")
    db.post_update_metrics("p1", likes=10, comments=1)
    db.post_update_metrics("p2", likes=40, comments=5)
    db.post_update_metrics("p3", likes=25, comments=8)

    top = retro.top_posts(limit=3)
    # Score = likes + comments * 3  → p2: 55, p3: 49, p1: 13
    assert [r["ig_media_pk"] for r in top] == ["p2", "p3", "p1"]


def test_performance_by_format(tmp_db):
    from src.brain import retro

    db.post_record("p1", None, "meme", "m1")
    db.post_record("p2", None, "meme", "m2")
    db.post_record("p3", None, "carousel", "c1")
    db.post_update_metrics("p1", likes=10, comments=1)
    db.post_update_metrics("p2", likes=30, comments=2)
    db.post_update_metrics("p3", likes=60, comments=5)

    perf = retro.performance_by_format()
    assert perf["meme"]["n"] == 2
    assert perf["meme"]["avg_likes"] == pytest.approx(20.0)
    assert perf["carousel"]["avg_likes"] == pytest.approx(60.0)


def test_push_retro_context_writes_signal(tmp_db):
    from src.brain import retro
    db.post_record("p1", None, "meme", "funny hook line\ntail")
    db.post_update_metrics("p1", likes=99, comments=12)
    pushed = retro.push_retro_context(limit=1)
    assert pushed == 1

    ctx = db.pop_context(limit=5)
    assert any("retro" in r["source"] for r in ctx)


# ─── Story viewer scoring ───
def test_story_viewer_already_viewed_today(tmp_db):
    from src.workers import story_viewer

    db.action_log("story_view", "uid-123", "ok", 0)
    assert story_viewer._already_viewed_today("uid-123") is True
    assert story_viewer._already_viewed_today("uid-999") is False


def test_story_viewer_queued_today_idempotent(tmp_db):
    from src.workers import story_viewer

    db.engagement_enqueue("story_view", target_user="uid-1")
    assert story_viewer._already_queued_today("uid-1") is True
    assert story_viewer._already_queued_today("uid-2") is False


# ─── ComfyUI client ───
def test_comfyui_unconfigured_is_false(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("COMFYUI_URL", raising=False)
    from src.plugins import comfyui
    assert comfyui.configured() is False


def test_comfyui_load_default_workflow(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("COMFYUI_URL", "http://localhost:8188")
    from src.plugins import comfyui

    # Point WORKFLOW_DIR at empty tmp
    monkeypatch.setattr(comfyui, "WORKFLOW_DIR", tmp_path / "workflows")
    wf = comfyui._load_workflow()
    assert "3" in wf and "4" in wf
    assert wf["3"]["class_type"] == "KSampler"


def test_comfyui_apply_params_fills_prompt_and_seed(monkeypatch: pytest.MonkeyPatch):
    from src.plugins import comfyui

    wf = comfyui._load_workflow()
    out = comfyui._apply_params(
        wf, prompt="dad mid-pullup, 35mm, natural light",
        negative="bad anatomy", width=1080, height=1350, seed=42424242,
    )
    assert out["3"]["inputs"]["seed"] == 42424242
    assert out["5"]["inputs"]["width"] == 1080
    assert out["5"]["inputs"]["height"] == 1350
    # Positive prompt should land on the first CLIPTextEncode
    assert "dad mid-pullup" in out["6"]["inputs"]["text"]
    assert "bad anatomy" in out["7"]["inputs"]["text"]


def test_comfyui_collect_image_refs():
    from src.plugins.comfyui import _collect_image_refs
    outputs = {
        "9": {"images": [{"filename": "a.png", "subfolder": "", "type": "output"}]},
        "10": {"images": [{"filename": "b.png", "subfolder": "", "type": "output"},
                          {"filename": "c.png", "subfolder": "", "type": "output"}]},
        "11": {"gifs": []},
    }
    refs = _collect_image_refs(outputs)
    assert [r["filename"] for r in refs] == ["a.png", "b.png", "c.png"]


# ─── Config still roundtrips ───
def test_safety_image_candidates_knob():
    from src.core.config import NicheConfig, Safety
    cfg = _mkcfg(safety=Safety(image_candidates=3))
    dumped = cfg.model_dump(mode="json")
    loaded = NicheConfig.model_validate(dumped)
    assert loaded.safety.image_candidates == 3
