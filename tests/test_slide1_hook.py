"""Slide 1 hook optimiser — winner selection + lock-override in carousel outline."""
from __future__ import annotations

import pytest

from instagram_ai_agent.content import slide1_hook
from instagram_ai_agent.content.generators import carousel as carousel_gen
from instagram_ai_agent.core import config as cfg_mod


def _mkcfg():
    return cfg_mod.NicheConfig(
        niche="home calisthenics",
        sub_topics=["pullups"],
        target_audience="dads 35+",
        commercial=True,
        voice=cfg_mod.Voice(tone=["direct"], forbidden=[], persona="ex-office worker"),
        aesthetic=cfg_mod.Aesthetic(palette=["#000", "#fff", "#c9a961"]),
        hashtags=cfg_mod.HashtagPools(core=["calisthenics", "homeworkout", "dadfit"]),
    )


_VALID = {
    "candidates": [
        {
            "title": "most guys quit pullups one rep early",
            "body": "the 3 CNS-level fixes I used at week 5",
            "pattern": "contrarian",
            "scroll_stop": 9, "specificity": 8, "visual_dominance": 7, "total": 24,
        },
    ],
    "winner": {
        "title": "most guys quit pullups one rep early",
        "body": "the 3 CNS-level fixes I used at week 5",
        "why": "specific number + lived-experience + contrarian claim",
    },
}


# ─── Winner selection ───
@pytest.mark.asyncio
async def test_returns_slide1_winner(monkeypatch):
    async def fake_json(route, prompt, *, system=None, max_tokens=None, temperature=None):
        assert route == "bulk"
        return _VALID

    monkeypatch.setattr(slide1_hook, "generate_json", fake_json)

    out = await slide1_hook.best_slide1_hook(
        _mkcfg(), trend_context="reddit: first pullup plateau",
    )
    assert out is not None
    assert out.title == _VALID["winner"]["title"]
    assert "week 5" in out.body
    block = out.as_constraint_block()
    assert "SLIDE 1 HARD CONSTRAINT" in block
    assert out.title in block


@pytest.mark.asyncio
async def test_banned_phrase_rejected(monkeypatch):
    async def fake_json(*a, **k):
        return {
            "candidates": [],
            "winner": {
                "title": "unlock the secret pullup hack",
                "body": "the 3 fixes",
                "why": "x",
            },
        }

    monkeypatch.setattr(slide1_hook, "generate_json", fake_json)

    out = await slide1_hook.best_slide1_hook(_mkcfg(), trend_context="")
    assert out is None


@pytest.mark.asyncio
async def test_too_long_title_rejected(monkeypatch):
    async def fake_json(*a, **k):
        return {
            "candidates": [],
            "winner": {
                "title": " ".join(["word"] * 15),  # 15 words, over 12-word ceiling
                "body": "short body",
                "why": "x",
            },
        }

    monkeypatch.setattr(slide1_hook, "generate_json", fake_json)
    out = await slide1_hook.best_slide1_hook(_mkcfg(), trend_context="")
    assert out is None


@pytest.mark.asyncio
async def test_llm_exception_returns_none(monkeypatch):
    async def fake_json(*a, **k):
        raise RuntimeError("all down")

    monkeypatch.setattr(slide1_hook, "generate_json", fake_json)
    out = await slide1_hook.best_slide1_hook(_mkcfg(), trend_context="")
    assert out is None


@pytest.mark.asyncio
async def test_missing_winner_returns_none(monkeypatch):
    async def fake_json(*a, **k):
        return {"candidates": _VALID["candidates"]}

    monkeypatch.setattr(slide1_hook, "generate_json", fake_json)
    out = await slide1_hook.best_slide1_hook(_mkcfg(), trend_context="")
    assert out is None


@pytest.mark.asyncio
async def test_empty_title_returns_none(monkeypatch):
    async def fake_json(*a, **k):
        return {"candidates": [], "winner": {"title": "", "body": "x", "why": "y"}}

    monkeypatch.setattr(slide1_hook, "generate_json", fake_json)
    out = await slide1_hook.best_slide1_hook(_mkcfg(), trend_context="")
    assert out is None


# ─── Integration: carousel outline locks slide 1 ───
@pytest.mark.asyncio
async def test_outline_locks_slide1_from_winner(monkeypatch):
    """Even if the script model returns a different slide 1, we must
    override with the upstream winner — the lock is not negotiable."""
    async def fake_json(route, prompt, *, system=None, max_tokens=None, temperature=None):
        # Simulate model that "drifted" from the lock directive
        return {"slides": [
            {"kind": "hook", "title": "DIFFERENT HOOK", "body": "model ignored lock"},
            {"kind": "content", "title": "b", "body": "b1"},
            {"kind": "content", "title": "c", "body": "c1"},
            {"kind": "cta", "title": "save", "body": "save it"},
        ]}

    monkeypatch.setattr(carousel_gen, "generate_json", fake_json)

    winner = slide1_hook.Slide1Hook(
        title="locked title",
        body="locked body",
        why="x",
    )
    outline = await carousel_gen._llm_outline(
        _mkcfg(), trend_context="ctx", slides=4, slide1=winner,
    )
    assert outline[0]["title"] == "locked title"
    assert outline[0]["body"] == "locked body"
    # Slides 2..N must come from the model, untouched
    assert outline[1]["title"] == "b"


@pytest.mark.asyncio
async def test_outline_passes_through_without_winner(monkeypatch):
    """Backwards compatibility: without slide1, LLM output is respected."""
    async def fake_json(*a, **k):
        return {"slides": [
            {"kind": "hook", "title": "model hook", "body": "model body"},
            {"kind": "content", "title": "b", "body": "b1"},
            {"kind": "cta", "title": "save", "body": "save it"},
        ]}

    monkeypatch.setattr(carousel_gen, "generate_json", fake_json)

    outline = await carousel_gen._llm_outline(
        _mkcfg(), trend_context="", slides=3, slide1=None,
    )
    assert outline[0]["title"] == "model hook"
