"""Pre-generation angle + hook brainstorm — winner selection, failure modes."""
from __future__ import annotations

import pytest

from instagram_ai_agent.content import angle_brainstorm
from instagram_ai_agent.core import config as cfg_mod


def _mkcfg():
    return cfg_mod.NicheConfig(
        niche="home calisthenics",
        sub_topics=["pullups"],
        target_audience="dads 35+",
        commercial=True,
        voice=cfg_mod.Voice(tone=["direct"], forbidden=["hustle"], persona="ex-office worker"),
        aesthetic=cfg_mod.Aesthetic(palette=["#000", "#fff", "#c9a961"]),
        hashtags=cfg_mod.HashtagPools(core=["calisthenics", "homeworkout", "dadfit"]),
    )


_VALID_RESPONSE = {
    "angles": [
        {
            "angle": "most guys quit pullups one rep before CNS adapts",
            "hook": "most guys quit pullups one rep early",
            "source_signal": "reddit /r/bodyweightfitness top-rated thread",
            "specificity": 8,
            "hook_strength": 9,
            "evidence_anchor": 7,
            "total": 24,
        },
    ],
    "winner": {
        "angle": "most guys quit pullups one rep before CNS adapts",
        "hook": "most guys quit pullups one rep early",
        "why": "anchors a widely-felt failure to a specific physiological claim",
    },
}


@pytest.mark.asyncio
async def test_returns_winning_angle(monkeypatch):
    async def fake_json(route, prompt, *, system=None, max_tokens=None, temperature=None):
        assert route == "bulk"
        assert "pullups" in prompt
        assert "CONTRARIAN" not in prompt  # default contrarian=False
        return _VALID_RESPONSE

    monkeypatch.setattr(angle_brainstorm, "generate_json", fake_json)

    out = await angle_brainstorm.brainstorm_angle(
        _mkcfg(),
        format_name="carousel",
        sub_topic="pullups",
        research_context="[reddit] how do I get my first pullup?",
    )
    assert out is not None
    assert out.hook == "most guys quit pullups one rep early"
    assert "CNS" in out.angle
    # Context block carries the hard directive downstream
    block = out.as_context_block()
    assert "WINNING_ANGLE" in block
    assert out.hook in block
    assert "no clichés" in block.lower() or "clichés" in block.lower()


@pytest.mark.asyncio
async def test_contrarian_flag_injects_directive(monkeypatch):
    captured: dict[str, str] = {}

    async def fake_json(route, prompt, *, system=None, max_tokens=None, temperature=None):
        captured["prompt"] = prompt
        return _VALID_RESPONSE

    monkeypatch.setattr(angle_brainstorm, "generate_json", fake_json)

    await angle_brainstorm.brainstorm_angle(
        _mkcfg(),
        format_name="meme",
        sub_topic="mobility",
        research_context="",
        contrarian=True,
    )
    assert "CONTRARIAN MODE" in captured["prompt"]


@pytest.mark.asyncio
async def test_archetype_hook_appears_in_prompt(monkeypatch):
    captured: dict[str, str] = {}

    async def fake_json(route, prompt, *, system=None, max_tokens=None, temperature=None):
        captured["prompt"] = prompt
        return _VALID_RESPONSE

    monkeypatch.setattr(angle_brainstorm, "generate_json", fake_json)

    await angle_brainstorm.brainstorm_angle(
        _mkcfg(),
        format_name="carousel",
        sub_topic="pullups",
        research_context="",
        archetype_hook="unpopular-opinion: X is actually Y",
    )
    assert "unpopular-opinion: X is actually Y" in captured["prompt"]


@pytest.mark.asyncio
async def test_llm_exception_returns_none(monkeypatch):
    async def fake_json(*a, **k):
        raise RuntimeError("all providers down")

    monkeypatch.setattr(angle_brainstorm, "generate_json", fake_json)

    out = await angle_brainstorm.brainstorm_angle(
        _mkcfg(), format_name="meme", sub_topic=None, research_context="",
    )
    assert out is None


@pytest.mark.asyncio
async def test_missing_winner_returns_none(monkeypatch):
    async def fake_json(*a, **k):
        return {"angles": _VALID_RESPONSE["angles"]}  # no "winner"

    monkeypatch.setattr(angle_brainstorm, "generate_json", fake_json)

    out = await angle_brainstorm.brainstorm_angle(
        _mkcfg(), format_name="meme", sub_topic=None, research_context="",
    )
    assert out is None


@pytest.mark.asyncio
async def test_empty_winner_fields_return_none(monkeypatch):
    async def fake_json(*a, **k):
        return {"angles": [], "winner": {"angle": "", "hook": "", "why": ""}}

    monkeypatch.setattr(angle_brainstorm, "generate_json", fake_json)

    out = await angle_brainstorm.brainstorm_angle(
        _mkcfg(), format_name="meme", sub_topic=None, research_context="",
    )
    assert out is None


@pytest.mark.asyncio
async def test_forbidden_hook_cliche_rejected(monkeypatch):
    """Model smuggled a cliché through — we reject rather than ship it."""
    async def fake_json(*a, **k):
        return {
            "angles": [],
            "winner": {
                "angle": "anything",
                "hook": "unlock the secret to better pullups",
                "why": "test",
            },
        }

    monkeypatch.setattr(angle_brainstorm, "generate_json", fake_json)

    out = await angle_brainstorm.brainstorm_angle(
        _mkcfg(), format_name="meme", sub_topic=None, research_context="",
    )
    assert out is None


@pytest.mark.asyncio
async def test_non_dict_response_returns_none(monkeypatch):
    async def fake_json(*a, **k):
        return ["not", "a", "dict"]

    monkeypatch.setattr(angle_brainstorm, "generate_json", fake_json)

    out = await angle_brainstorm.brainstorm_angle(
        _mkcfg(), format_name="meme", sub_topic=None, research_context="",
    )
    assert out is None
