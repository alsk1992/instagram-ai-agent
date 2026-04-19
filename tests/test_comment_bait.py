"""Comment-bait CTA engineer — pattern selection + rewrite guardrails."""
from __future__ import annotations

import pytest

from instagram_ai_agent.content import comment_bait
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


_GENERIC_CTA_CAPTION = (
    "I trained pullups for 8 weeks.\n"
    "At week 5 the first rep arrived — CNS adaptation, not strength.\n"
    "save this for later."
)


# ─── Pattern selection ───
def test_pick_pattern_contrarian_wins():
    assert comment_bait.pick_pattern("carousel", contrarian=True) == "unpopular_roast"
    assert comment_bait.pick_pattern("reel_stock", contrarian=True) == "unpopular_roast"


def test_pick_pattern_carousel_with_numbers():
    assert comment_bait.pick_pattern("carousel", has_numbers=True) == "number_drop"


def test_pick_pattern_carousel_without_numbers():
    assert comment_bait.pick_pattern("carousel") == "story_invite"


def test_pick_pattern_quote_emoji_react():
    assert comment_bait.pick_pattern("quote_card") == "emoji_react"


def test_pick_pattern_reel_binary():
    assert comment_bait.pick_pattern("reel_stock") == "binary_pick"
    assert comment_bait.pick_pattern("reel_ai") == "binary_pick"
    assert comment_bait.pick_pattern("photo") == "binary_pick"


def test_pick_pattern_fallback():
    assert comment_bait.pick_pattern("some_unknown_format") == "fill_in_blank"


# ─── Format targeting ───
@pytest.mark.asyncio
async def test_non_comment_format_skipped(monkeypatch):
    """Memes are share-optimised — don't rewrite them."""
    calls = {"n": 0}

    async def fake_gen(*a, **k):
        calls["n"] += 1
        return "should not be called"

    monkeypatch.setattr(comment_bait, "generate", fake_gen)

    out = await comment_bait.engineer(
        _mkcfg(), _GENERIC_CTA_CAPTION, format_name="meme",
    )
    assert out == _GENERIC_CTA_CAPTION
    assert calls["n"] == 0


@pytest.mark.asyncio
async def test_story_format_skipped(monkeypatch):
    """story_* captions don't get comment-bait (stickers do the work)."""
    calls = {"n": 0}

    async def fake_gen(*a, **k):
        calls["n"] += 1
        return "x"

    monkeypatch.setattr(comment_bait, "generate", fake_gen)

    for fmt in ("story_quote", "story_announcement", "story_video"):
        out = await comment_bait.engineer(
            _mkcfg(), _GENERIC_CTA_CAPTION, format_name=fmt,
        )
        assert out == _GENERIC_CTA_CAPTION
    assert calls["n"] == 0


# ─── Already-engineered short-circuit ───
@pytest.mark.asyncio
async def test_caption_ending_in_question_short_circuits(monkeypatch):
    calls = {"n": 0}

    async def fake_gen(*a, **k):
        calls["n"] += 1
        return "x"

    monkeypatch.setattr(comment_bait, "generate", fake_gen)

    caption = "I trained pullups for 8 weeks.\nAt week 5 the first rep clicked.\nwhich week was yours?"
    out = await comment_bait.engineer(_mkcfg(), caption, format_name="carousel")
    assert out == caption
    assert calls["n"] == 0


# ─── Successful rewrite ───
@pytest.mark.asyncio
async def test_rewrites_generic_cta_to_comment_bait(monkeypatch):
    async def fake_gen(task, prompt, *, system, max_tokens, temperature):
        assert task == "caption"
        return (
            "I trained pullups for 8 weeks.\n"
            "At week 5 the first rep arrived — CNS adaptation, not strength.\n"
            "what week did yours click? drop it below 👇"
        )

    monkeypatch.setattr(comment_bait, "generate", fake_gen)

    out = await comment_bait.engineer(
        _mkcfg(), _GENERIC_CTA_CAPTION, format_name="carousel",
    )
    assert "save this for later" not in out
    assert "what week" in out
    # CTA must be question/emoji-terminated (engineered signal)
    assert out.rstrip().endswith(("?", "👇"))


# ─── Guardrails ───
@pytest.mark.asyncio
async def test_llm_failure_returns_original(monkeypatch):
    async def fake_gen(*a, **k):
        raise RuntimeError("providers down")

    monkeypatch.setattr(comment_bait, "generate", fake_gen)

    out = await comment_bait.engineer(
        _mkcfg(), _GENERIC_CTA_CAPTION, format_name="carousel",
    )
    assert out == _GENERIC_CTA_CAPTION


@pytest.mark.asyncio
async def test_rewrite_without_cta_signal_rejected(monkeypatch):
    """LLM returned a caption whose last line isn't ? or emoji — reject."""
    async def fake_gen(*a, **k):
        return (
            "I trained pullups for 8 weeks.\n"
            "At week 5 the first rep arrived.\n"
            "just a flat statement no trigger"
        )

    monkeypatch.setattr(comment_bait, "generate", fake_gen)

    out = await comment_bait.engineer(
        _mkcfg(), _GENERIC_CTA_CAPTION, format_name="carousel",
    )
    assert out == _GENERIC_CTA_CAPTION


@pytest.mark.asyncio
async def test_rewrite_length_drift_rejected(monkeypatch):
    async def fake_gen(*a, **k):
        return "short?"

    monkeypatch.setattr(comment_bait, "generate", fake_gen)

    out = await comment_bait.engineer(
        _mkcfg(), _GENERIC_CTA_CAPTION, format_name="carousel",
    )
    assert out == _GENERIC_CTA_CAPTION


@pytest.mark.asyncio
async def test_empty_caption_returns_unchanged(monkeypatch):
    async def fake_gen(*a, **k):
        return "shouldn't be called"

    monkeypatch.setattr(comment_bait, "generate", fake_gen)

    out = await comment_bait.engineer(_mkcfg(), "", format_name="carousel")
    assert out == ""
