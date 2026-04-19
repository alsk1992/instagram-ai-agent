"""Story-arc validator — prescription detection + lived-experience rewrite."""
from __future__ import annotations

import pytest

from instagram_ai_agent.content import story_arc
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


# ─── Detection ───
def test_prescription_heavy_detected():
    text = "You should stop doing pullups the wrong way. Here's how to fix it. Follow these steps."
    assert story_arc.is_heavy_prescriptive(text)


def test_lived_experience_not_flagged():
    text = (
        "I did 3 shoulder fixes for 6 weeks. At week 5 my first pullup arrived. "
        "Turns out CNS adaptation takes longer than the muscle side."
    )
    assert not story_arc.is_heavy_prescriptive(text)


def test_mixed_leans_toward_lower_score():
    text = (
        "You should try pullups. I did them for 8 weeks and found the real issue. "
        "After 2 months the fix clicked."
    )
    # Lived-experience signals cancel prescription roughly 1:1 — result should be below threshold
    assert not story_arc.is_heavy_prescriptive(text)


def test_empty_text_scores_zero():
    assert story_arc.score_prescription("") == 0.0
    assert story_arc.score_prescription("   ") == 0.0


def test_score_is_length_invariant():
    """A heavy-prescriptive 2-line draft should flag higher than a single
    prescription in a 10-sentence draft."""
    short = "You should stop doing X. Here's what to do instead."
    long = "I trained pullups every morning for 6 weeks. The CNS adaptation kicked in at week 3. My shoulder mobility was the limiter. I added band pull-aparts. Then scapular pullups. You should try scapular pullups. Mine unlocked at week 5. Strict form always. No kipping."
    assert story_arc.score_prescription(short) > story_arc.score_prescription(long)


# ─── Rewrite ───
@pytest.mark.asyncio
async def test_clean_draft_short_circuits_no_llm(monkeypatch):
    calls = {"n": 0}

    async def fake_gen(*a, **k):
        calls["n"] += 1
        return "should not be called"

    monkeypatch.setattr(story_arc, "generate", fake_gen)

    clean = "I trained pullups for 6 weeks. At week 5 the first rep arrived. Turns out CNS was the limiter."
    out = await story_arc.convert_to_story(_mkcfg(), clean)
    assert out == clean
    assert calls["n"] == 0


@pytest.mark.asyncio
async def test_rewrites_prescriptive_draft(monkeypatch):
    async def fake_gen(task, prompt, *, system, max_tokens, temperature):
        assert task == "caption"
        return (
            "I spent 6 weeks trying the 3 fixes most pullup videos skip. "
            "At week 5 my first rep arrived — turns out CNS adaptation, not strength, was the block."
        )

    monkeypatch.setattr(story_arc, "generate", fake_gen)

    original = "You should stop doing pullups the wrong way. Here's how to fix it."
    out = await story_arc.convert_to_story(_mkcfg(), original)
    assert "I spent" in out
    assert "week 5" in out
    assert "you should" not in out.lower()


@pytest.mark.asyncio
async def test_llm_failure_returns_original(monkeypatch):
    async def fake_gen(*a, **k):
        raise RuntimeError("providers down")

    monkeypatch.setattr(story_arc, "generate", fake_gen)

    original = "You should stop doing pullups the wrong way."
    out = await story_arc.convert_to_story(_mkcfg(), original)
    assert out == original


@pytest.mark.asyncio
async def test_still_prescriptive_rewrite_rejected(monkeypatch):
    async def fake_gen(*a, **k):
        # LLM returned another prescriptive draft — reject and keep original
        return "You should also try these drills. Here's what to do. Follow these steps and stop doing the wrong thing."

    monkeypatch.setattr(story_arc, "generate", fake_gen)

    original = "You should stop doing pullups the wrong way. Here's how to fix it."
    out = await story_arc.convert_to_story(_mkcfg(), original)
    assert out == original


@pytest.mark.asyncio
async def test_rewrite_length_drift_rejected(monkeypatch):
    async def fake_gen(*a, **k):
        return "I " + " ".join(["word"] * 200)

    monkeypatch.setattr(story_arc, "generate", fake_gen)

    original = "You should stop doing pullups the wrong way. Here's how to fix it."
    out = await story_arc.convert_to_story(_mkcfg(), original)
    assert out == original


@pytest.mark.asyncio
async def test_strips_label_prefixes(monkeypatch):
    async def fake_gen(*a, **k):
        return 'Rewrite: "I trained pullups for 8 weeks — at week 6 the first rep arrived."'

    monkeypatch.setattr(story_arc, "generate", fake_gen)

    original = "You should do pullups the right way. Stop doing them wrong."
    out = await story_arc.convert_to_story(_mkcfg(), original)
    assert not out.startswith("Rewrite:")
    assert not out.startswith('"')
    assert "week 6" in out


# ─── STORY_ARC_FORMATS ───
def test_story_arc_formats_includes_carousel_and_reels():
    assert "carousel" in story_arc.STORY_ARC_FORMATS
    assert "reel_stock" in story_arc.STORY_ARC_FORMATS
    assert "reel_ai" in story_arc.STORY_ARC_FORMATS
    # Memes + quotes + stories are deliberately left out — prescription fits there
    assert "meme" not in story_arc.STORY_ARC_FORMATS
    assert "quote_card" not in story_arc.STORY_ARC_FORMATS
    assert "story_quote" not in story_arc.STORY_ARC_FORMATS
