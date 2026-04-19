"""Specificity rewrite pass — short-circuit + guardrails + voice preserve."""
from __future__ import annotations

import pytest

from instagram_ai_agent.content import specificity_pass
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


# ─── has_generic_filler ───
def test_filler_detect_positive():
    assert specificity_pass.has_generic_filler("pro tips for better pullups")
    assert specificity_pass.has_generic_filler("this is a GAME-CHANGER")
    assert specificity_pass.has_generic_filler("unlock your next level")


def test_filler_detect_negative():
    assert not specificity_pass.has_generic_filler(
        "I did 3 shoulder fixes for 6 weeks — first pullup arrived at week 5."
    )
    assert not specificity_pass.has_generic_filler("")


# ─── concretize ───
@pytest.mark.asyncio
async def test_clean_draft_short_circuits_no_llm(monkeypatch):
    """Already-specific drafts skip the LLM entirely."""
    calls = {"n": 0}

    async def fake_gen(*a, **k):
        calls["n"] += 1
        return "should not be called"

    monkeypatch.setattr(specificity_pass, "generate", fake_gen)

    clean_draft = "did 3 shoulder-position fixes for 6 weeks — first pullup arrived at week 5."
    out = await specificity_pass.concretize(_mkcfg(), clean_draft)
    assert out == clean_draft
    assert calls["n"] == 0


@pytest.mark.asyncio
async def test_rewrites_generic_draft(monkeypatch):
    """Generic filler → LLM produces a cleaner rewrite that wins the guardrails."""
    async def fake_gen(task, prompt, *, system, max_tokens, temperature):
        assert task == "caption"
        return "the 3 shoulder fixes I used for 6 weeks — first pullup came at week 5."

    monkeypatch.setattr(specificity_pass, "generate", fake_gen)

    out = await specificity_pass.concretize(
        _mkcfg(),
        "pro tips for better pullups",
        context="reddit: how do I get my first pullup",
    )
    assert "pro tips" not in out.lower()
    assert "week 5" in out


@pytest.mark.asyncio
async def test_empty_draft_passes_through(monkeypatch):
    calls = {"n": 0}

    async def fake_gen(*a, **k):
        calls["n"] += 1
        return "x"

    monkeypatch.setattr(specificity_pass, "generate", fake_gen)

    out = await specificity_pass.concretize(_mkcfg(), "")
    assert out == ""
    assert calls["n"] == 0


@pytest.mark.asyncio
async def test_llm_exception_returns_original(monkeypatch):
    async def fake_gen(*a, **k):
        raise RuntimeError("all providers down")

    monkeypatch.setattr(specificity_pass, "generate", fake_gen)

    original = "pro tips for pullups"
    out = await specificity_pass.concretize(_mkcfg(), original)
    assert out == original


@pytest.mark.asyncio
async def test_rewrite_still_generic_keeps_original(monkeypatch):
    """LLM failed its job (rewrite still has banned phrase) → keep original."""
    async def fake_gen(*a, **k):
        return "even better pro tips for pullups"  # still has "pro tips"

    monkeypatch.setattr(specificity_pass, "generate", fake_gen)

    original = "pro tips for pullups"
    out = await specificity_pass.concretize(_mkcfg(), original)
    assert out == original


@pytest.mark.asyncio
async def test_rewrite_too_long_keeps_original(monkeypatch):
    """A rewrite that's 2× as long means the model rambled. Refuse."""
    async def fake_gen(*a, **k):
        return "a much longer rewrite " * 50

    monkeypatch.setattr(specificity_pass, "generate", fake_gen)

    original = "pro tips for pullups"
    out = await specificity_pass.concretize(_mkcfg(), original)
    assert out == original


@pytest.mark.asyncio
async def test_rewrite_too_short_keeps_original(monkeypatch):
    """A rewrite that collapses to 2 words means the model lost content. Refuse."""
    async def fake_gen(*a, **k):
        return "okay"

    monkeypatch.setattr(specificity_pass, "generate", fake_gen)

    original = "pro tips for pullups week 5 is where it starts working"
    out = await specificity_pass.concretize(_mkcfg(), original)
    assert out == original


@pytest.mark.asyncio
async def test_rewrite_strips_label_prefixes(monkeypatch):
    async def fake_gen(*a, **k):
        return 'Rewrite: "the 3 CNS-level fixes I used for 6 weeks — first pullup at week 5"'

    monkeypatch.setattr(specificity_pass, "generate", fake_gen)

    out = await specificity_pass.concretize(
        _mkcfg(), "pro tips for pullups",
    )
    assert not out.startswith("Rewrite:")
    assert not out.startswith('"')
    assert "week 5" in out


@pytest.mark.asyncio
async def test_identical_rewrite_returns_original(monkeypatch):
    async def fake_gen(*a, **k):
        return "pro tips for pullups"  # same string

    monkeypatch.setattr(specificity_pass, "generate", fake_gen)

    original = "pro tips for pullups"
    out = await specificity_pass.concretize(_mkcfg(), original)
    assert out is original
