"""Smoke tests for config serialization / hashtag pool logic."""
from __future__ import annotations

import pytest

from instagram_ai_agent.content.hashtags import build_hashtags, format_hashtags
from instagram_ai_agent.core.config import (
    Aesthetic,
    Budget,
    FormatMix,
    HashtagPools,
    NicheConfig,
    Safety,
    Schedule,
    Voice,
)


def _demo_cfg() -> NicheConfig:
    return NicheConfig(
        niche="home calisthenics",
        sub_topics=["pullups", "mobility"],
        target_audience="dads over 35 rebuilding fitness at home",
        commercial=True,
        voice=Voice(
            tone=["direct", "dry"],
            forbidden=["grind"],
            persona="ex-office worker, 40, rebuilt body at home — talks like a mate.",
        ),
        aesthetic=Aesthetic(palette=["#0a0a0a", "#f5f5f0", "#c9a961"]),
        hashtags=HashtagPools(
            core=["calisthenics", "homeworkout", "bodyweighttraining"],
            growth=["fitmotivation", "dadfit", "fitover40"],
            long_tail=["pullupprogression"],
            per_post=10,
        ),
        formats=FormatMix(),
        schedule=Schedule(),
        budget=Budget(),
        safety=Safety(),
    )


def test_niche_roundtrip():
    cfg = _demo_cfg()
    dumped = cfg.model_dump(mode="json")
    cfg2 = NicheConfig.model_validate(dumped)
    assert cfg2.niche == cfg.niche
    assert cfg2.formats.normalized() == cfg.formats.normalized()


def test_format_mix_normalises():
    mix = FormatMix(meme=2.0, quote_card=1.0, carousel=1.0, reel_stock=0.0, reel_ai=0.0, photo=0.0)
    norm = mix.normalized()
    assert abs(sum(norm.values()) - 1.0) < 1e-6
    assert norm["meme"] == pytest.approx(0.5)


def test_hashtag_count():
    cfg = _demo_cfg()
    tags = build_hashtags(cfg, seed=42)
    # Can never return more unique tags than exist in the pools
    total_pool = len({*cfg.hashtags.core, *cfg.hashtags.growth, *cfg.hashtags.long_tail})
    assert len(tags) == min(cfg.hashtags.per_post, total_pool)
    assert all(not t.startswith("#") for t in tags)
    rendered = format_hashtags(tags)
    assert rendered.startswith("#")


def test_hashtag_count_with_large_pool():
    cfg = _demo_cfg()
    cfg.hashtags.core = [f"core{i}" for i in range(8)]
    cfg.hashtags.growth = [f"growth{i}" for i in range(8)]
    cfg.hashtags.long_tail = [f"long{i}" for i in range(8)]
    cfg.hashtags.per_post = 15
    tags = build_hashtags(cfg, seed=42)
    assert len(tags) == 15
    # Should draw from every pool, not just one
    assert any(t.startswith("core") for t in tags)
    assert any(t.startswith("growth") for t in tags)


def test_palette_validator():
    with pytest.raises(ValueError):
        Aesthetic(palette=["not-a-hex", "#ffffff"])


def test_best_hours_sorted_unique():
    s = Schedule(best_hours_utc=[21, 9, 9, 15])
    assert s.best_hours_utc == [9, 15, 21]
