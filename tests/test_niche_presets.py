"""Noob-onboarding presets — every preset must produce a NicheConfig
that validates cleanly."""
from __future__ import annotations

import pytest

from src.core import config as cfg_mod
from src.niche_presets import PRESETS, by_key, to_niche_config_fields


def test_at_least_eight_presets_shipped():
    """8 starter niches cover the most common use cases + give noobs
    choice without overwhelm."""
    assert len(PRESETS) >= 8


def test_preset_keys_are_unique_and_lowercase():
    keys = [p.key for p in PRESETS]
    assert len(set(keys)) == len(keys), "duplicate preset keys"
    for k in keys:
        assert k.islower() and k.replace("_", "").isalnum(), f"bad key: {k}"


def test_every_preset_has_minimum_viable_fields():
    for p in PRESETS:
        assert len(p.niche) >= 3, f"{p.key}: niche too short"
        assert len(p.sub_topics) >= 1, f"{p.key}: no sub_topics"
        assert len(p.target_audience) >= 5, f"{p.key}: audience too short"
        assert len(p.persona) >= 10, f"{p.key}: persona too short"
        assert len(p.voice_tone) >= 1, f"{p.key}: no voice_tone"
        assert len(p.palette) >= 3, f"{p.key}: palette must have ≥3 colours"
        # Hex validation — each palette entry starts with # and is 7 chars
        for c in p.palette:
            assert c.startswith("#") and len(c) == 7, f"{p.key}: bad hex {c}"
        assert len(p.core_hashtags) >= 3, f"{p.key}: need ≥3 core hashtags"
        assert len(p.best_hours_utc) >= 1, f"{p.key}: need at least 1 best hour"
        for h in p.best_hours_utc:
            assert 0 <= h <= 23, f"{p.key}: hour {h} out of range"


@pytest.mark.parametrize("key", [p.key for p in PRESETS])
def test_every_preset_builds_a_valid_niche_config(key):
    """Preset must produce a NicheConfig that passes pydantic validation."""
    p = by_key(key)
    assert p is not None
    fields = to_niche_config_fields(p)
    cfg = cfg_mod.NicheConfig(
        niche=fields["niche"],
        sub_topics=fields["sub_topics"],
        target_audience=fields["target_audience"],
        commercial=True,
        voice=cfg_mod.Voice(
            tone=fields["voice_tone"],
            forbidden=fields["voice_forbidden"],
            persona=fields["persona"],
        ),
        aesthetic=cfg_mod.Aesthetic(palette=fields["palette"]),
        hashtags=cfg_mod.HashtagPools(core=fields["core_hashtags"]),
    )
    assert cfg.niche == fields["niche"]


def test_by_key_returns_none_on_unknown():
    assert by_key("nonexistent") is None
    assert by_key("") is None


def test_preset_forbidden_phrases_dont_clash_with_tone():
    """Every preset's forbidden list must NOT include its own tone words —
    would cause the voice validator to fight itself."""
    for p in PRESETS:
        tone_set = {t.lower() for t in p.voice_tone}
        forbidden_set = {f.lower() for f in p.voice_forbidden}
        overlap = tone_set & forbidden_set
        assert not overlap, f"{p.key}: tone/forbidden overlap {overlap}"
