"""Human-photo generator config + prompt-builder tests (no network)."""
from __future__ import annotations

from pathlib import Path

import pytest

from instagram_ai_agent.content.generators import human_photo
from instagram_ai_agent.core import config as cfg_mod


def _base_cfg(**overrides):
    base = dict(
        niche="home calisthenics",
        sub_topics=["pullups"],
        target_audience="dads 35+",
        commercial=True,
        voice=cfg_mod.Voice(tone=["direct"], forbidden=[], persona="ex-office worker rebuilding fitness."),
        aesthetic=cfg_mod.Aesthetic(palette=["#0a0a0a", "#f5f5f0", "#c9a961"]),
        hashtags=cfg_mod.HashtagPools(core=["calisthenics", "homeworkout", "dadfit"]),
    )
    base.update(overrides)
    return cfg_mod.NicheConfig(**base)


def test_human_photo_defaults_disabled():
    cfg = _base_cfg()
    assert cfg.human_photo.enabled is False
    assert cfg.human_photo.model == "flux-realism"
    assert "flux" in cfg.human_photo.model_fallbacks
    assert cfg.human_photo.character.enabled is False


def test_format_mix_human_default_zero():
    cfg = _base_cfg()
    norm = cfg.formats.normalized()
    assert norm.get("human_photo", 0.0) == 0.0


def test_persona_unique_mode_returns_from_pool():
    cfg = _base_cfg(human_photo=cfg_mod.HumanPhoto(enabled=True))
    persona, seed = human_photo._persona_for_gen(cfg)
    assert persona in cfg.human_photo.diversity_pool
    assert seed > 0


def test_persona_brand_mode_uses_character_seed():
    ch = cfg_mod.BrandCharacter(
        enabled=True,
        age_range="30s",
        gender="woman",
        ethnicity="unspecified",
        hair="short dark hair",
        build="athletic",
        wardrobe_style="black hoodie",
        vibe="focused",
        seed=424242,
    )
    cfg = _base_cfg(human_photo=cfg_mod.HumanPhoto(enabled=True, character=ch))
    persona, seed = human_photo._persona_for_gen(cfg)
    assert seed == 424242
    for token in ("30s", "woman", "short dark hair", "athletic", "black hoodie", "focused"):
        assert token in persona


def test_brand_persona_omits_unspecified_ethnicity():
    ch = cfg_mod.BrandCharacter(enabled=True, ethnicity="unspecified", hair="bald", build="stocky")
    out = human_photo._brand_persona(ch)
    assert "unspecified" not in out
    assert "bald" in out


def test_build_prompt_includes_palette_hint():
    cfg = _base_cfg()
    p = human_photo._build_prompt("dad mid-pullup in home gym", "40s man, grey stubble", cfg)
    assert "dad mid-pullup" in p
    assert "40s man" in p
    assert "photorealistic" in p
    assert "#0a0a0a" in p or "0a0a0a" in p  # palette hint present


def test_negative_prompt_merges_user_terms():
    ch = cfg_mod.BrandCharacter(enabled=True, negative="logos, text signs")
    out = human_photo._negative(ch)
    assert "logos" in out
    assert "anime" in out


def test_generate_raises_when_disabled():
    import asyncio

    cfg = _base_cfg()  # human_photo.enabled == False
    with pytest.raises(RuntimeError, match="disabled"):
        asyncio.run(human_photo.generate(cfg))


def test_format_mix_human_weight_normalises():
    mix = cfg_mod.FormatMix(
        meme=1.0, quote_card=0.0, carousel=0.0, reel_stock=0.0, reel_ai=0.0, photo=0.0,
        human_photo=1.0,
    )
    norm = mix.normalized()
    assert abs(sum(norm.values()) - 1.0) < 1e-6
    assert norm["human_photo"] == pytest.approx(0.5)
    assert norm["meme"] == pytest.approx(0.5)


def test_story_mix_human_weight_normalises():
    mix = cfg_mod.StoryMix(
        story_quote=1.0, story_announcement=0.0, story_photo=0.0, story_video=0.0,
        story_human=1.0,
    )
    norm = mix.normalized()
    assert abs(sum(norm.values()) - 1.0) < 1e-6
    assert norm["story_human"] == pytest.approx(0.5)


def test_config_roundtrip_with_human_block(tmp_path: Path):
    import yaml

    ch = cfg_mod.BrandCharacter(
        enabled=True,
        age_range="40s",
        gender="man",
        hair="grey stubble",
        build="stocky",
        wardrobe_style="black gym tee",
        vibe="no-nonsense",
        seed=1234567,
    )
    hp = cfg_mod.HumanPhoto(enabled=True, character=ch)
    cfg = _base_cfg(human_photo=hp)
    yaml_path = tmp_path / "niche.yaml"
    yaml_path.write_text(yaml.safe_dump(cfg.model_dump(mode="json")), encoding="utf-8")
    loaded = cfg_mod.NicheConfig.model_validate(yaml.safe_load(yaml_path.read_text()))
    assert loaded.human_photo.enabled is True
    assert loaded.human_photo.character.seed == 1234567
    assert loaded.human_photo.character.hair == "grey stubble"


def test_pipeline_routes_human_formats():
    """Verify dispatch doesn't fall through to ValueError for human formats."""
    import inspect
    from instagram_ai_agent.content import pipeline

    src = inspect.getsource(pipeline._dispatch)
    assert 'format_name == "human_photo"' in src
    assert 'format_name == "story_human"' in src


def test_cli_helpers_set_human_weight():
    from instagram_ai_agent.cli import _apply_human_weight

    base = cfg_mod.FormatMix()  # default weights
    out = _apply_human_weight(base, 0.25)
    norm = out.normalized()
    assert abs(sum(norm.values()) - 1.0) < 1e-6
    assert norm["human_photo"] == pytest.approx(0.25)
    # All other buckets preserved proportionally
    assert norm["meme"] > norm["reel_ai"]  # default ordering respected
