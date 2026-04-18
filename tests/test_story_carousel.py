"""Character-consistent narrative carousel — config, persona building,
seed-lock consistency, pipeline dispatch."""
from __future__ import annotations

from pathlib import Path

import pytest

from instagram_ai_agent.core import config as cfg_mod
from instagram_ai_agent.content.generators import story_carousel as sc_gen


def _mkcfg(**kwargs):
    base = dict(
        niche="home calisthenics",
        sub_topics=["pullups"],
        target_audience="dads 35+",
        commercial=True,
        voice=cfg_mod.Voice(tone=["direct"], forbidden=[], persona="ex-office worker rebuilding fitness."),
        aesthetic=cfg_mod.Aesthetic(palette=["#000", "#fff", "#c9a961"]),
        hashtags=cfg_mod.HashtagPools(core=["calisthenics", "homeworkout", "dadfit"]),
    )
    base.update(kwargs)
    return cfg_mod.NicheConfig(**base)


# ─── Config ───
def test_story_carousel_config_defaults():
    cfg = _mkcfg()
    assert cfg.story_carousel.slides == 6
    assert cfg.story_carousel.template_variant == "photo_caption"
    assert cfg.story_carousel.seed is None


def test_story_carousel_config_bounds():
    with pytest.raises(Exception):
        cfg_mod.StoryCarouselConfig(slides=2)
    with pytest.raises(Exception):
        cfg_mod.StoryCarouselConfig(slides=15)


def test_story_carousel_roundtrips():
    import yaml
    cfg = _mkcfg(story_carousel=cfg_mod.StoryCarouselConfig(slides=5, seed=42))
    d = cfg.model_dump(mode="json")
    loaded = cfg_mod.NicheConfig.model_validate(yaml.safe_load(yaml.safe_dump(d)))
    assert loaded.story_carousel.slides == 5
    assert loaded.story_carousel.seed == 42


# ─── FormatMix extension ───
def test_formatmix_includes_story_carousel_weight():
    fm = cfg_mod.FormatMix()
    assert hasattr(fm, "story_carousel")
    assert fm.story_carousel == 0.0


def test_formatmix_normalizes_story_carousel_into_sum():
    fm = cfg_mod.FormatMix(
        meme=0.0, quote_card=0.0, carousel=0.0, reel_stock=0.0,
        reel_ai=0.0, photo=0.0, human_photo=0.0, story_carousel=1.0,
    )
    n = fm.normalized()
    assert n["story_carousel"] == pytest.approx(1.0)


def test_formatmix_rejects_all_zero_including_story():
    with pytest.raises(Exception):
        cfg_mod.FormatMix(
            meme=0.0, quote_card=0.0, carousel=0.0, reel_stock=0.0,
            reel_ai=0.0, photo=0.0, human_photo=0.0, story_carousel=0.0,
        ).normalized()


# ─── Persona extraction ───
def test_default_persona_for_niche_mentions_niche():
    out = sc_gen._default_persona_for_niche("home calisthenics")
    assert "home calisthenics" in out


def test_brand_persona_one_liner_skips_unspecified():
    ch = cfg_mod.BrandCharacter(
        enabled=True,
        age_range="30s",
        gender="androgynous",    # skipped
        ethnicity="unspecified",  # skipped
        hair="short brown hair",
        build="lean athletic",
        wardrobe_style="",
        vibe="tired but determined",
    )
    out = sc_gen._brand_persona_one_liner(ch)
    assert "androgynous" not in out
    assert "unspecified" not in out
    assert "30s" in out
    assert "lean athletic" in out
    assert "tired but determined" in out


def test_persona_description_prefers_brand_character():
    ch = cfg_mod.BrandCharacter(
        enabled=True, age_range="40s", gender="man",
        hair="grey stubble", build="dad bod",
        vibe="wry half-smile", seed=12345,
    )
    cfg = _mkcfg(human_photo=cfg_mod.HumanPhoto(enabled=True, character=ch))
    out = sc_gen._persona_description(cfg)
    assert "40s" in out
    assert "dad bod" in out


def test_persona_description_falls_back_to_niche_default():
    cfg = _mkcfg()  # no brand character
    out = sc_gen._persona_description(cfg)
    assert cfg.niche in out


# ─── Prompt composition ───
def test_full_prompt_puts_persona_before_scene():
    out = sc_gen._full_prompt("dad bod lean athletic", "at the bar mid-pullup", "calisthenics")
    assert out.index("dad bod") < out.index("at the bar")
    # Quality descriptors appended at the end
    assert "photorealistic" in out


# ─── Scene outline ───
@pytest.mark.asyncio
async def test_outline_scenes_enforces_hook_and_cta(monkeypatch):
    async def fake_json(route, prompt, *, system, max_tokens=1200, temperature=0.85):
        return {"slides": [
            {"kind": "content", "title": "A", "body": "a", "scene_prompt": "s1"},
            {"kind": "content", "title": "B", "body": "b", "scene_prompt": "s2"},
            {"kind": "content", "title": "C", "body": "c", "scene_prompt": "s3"},
        ]}

    import instagram_ai_agent.content.generators.story_carousel as mod
    monkeypatch.setattr(mod, "generate_json", fake_json)

    cfg = _mkcfg()
    out = await sc_gen._outline_scenes(cfg, "persona", "trend", n_slides=3)
    assert len(out) == 3
    assert out[0]["kind"] == "hook"
    assert out[-1]["kind"] == "cta"
    # Middle stays content
    assert out[1]["kind"] == "content"


@pytest.mark.asyncio
async def test_outline_scenes_raises_on_short_response(monkeypatch):
    async def fake_json(*a, **k):
        return {"slides": [{"kind": "hook", "title": "a", "body": "b", "scene_prompt": "s"}]}

    import instagram_ai_agent.content.generators.story_carousel as mod
    monkeypatch.setattr(mod, "generate_json", fake_json)

    cfg = _mkcfg()
    with pytest.raises(ValueError):
        await sc_gen._outline_scenes(cfg, "p", "t", n_slides=5)


# ─── Seed-locked image generation behaviour ───
@pytest.mark.asyncio
async def test_seed_is_locked_across_slides(monkeypatch, tmp_path):
    """Every slide must invoke comfyui.generate() with the SAME seed
    so Character Consistency By Seed-Lock actually holds."""
    cfg = _mkcfg(story_carousel=cfg_mod.StoryCarouselConfig(slides=4, seed=777))

    # Stub scene outline
    async def fake_outline(cfg_, persona, ctx, *, n_slides):
        return [
            {"kind": "hook", "title": "h", "body": "", "scene_prompt": "s1", "index": 1},
            {"kind": "content", "title": "t2", "body": "b2", "scene_prompt": "s2", "index": 2},
            {"kind": "content", "title": "t3", "body": "b3", "scene_prompt": "s3", "index": 3},
            {"kind": "cta", "title": "t4", "body": "b4", "scene_prompt": "s4", "index": 4},
        ]

    seeds_seen: list[int] = []

    async def fake_gen_image(prompt, seed, cfg_):
        seeds_seen.append(seed)
        p = tmp_path / f"img_{len(seeds_seen)}.jpg"
        p.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 32)
        return p

    async def fake_render(html, out, **kw):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"\xff\xd8jpeg")
        return out

    import instagram_ai_agent.content.generators.story_carousel as mod
    monkeypatch.setattr(mod, "_outline_scenes", fake_outline)
    monkeypatch.setattr(mod, "_generate_slide_image", fake_gen_image)
    monkeypatch.setattr(mod, "render_html_to_png", fake_render)
    monkeypatch.setattr(mod, "apply_lut_image", lambda p, c: p)

    content = await sc_gen.generate(cfg)
    assert content.format == "story_carousel"
    # All 4 slides used the same locked seed
    assert len(seeds_seen) == 4
    assert len(set(seeds_seen)) == 1
    assert seeds_seen[0] == 777


@pytest.mark.asyncio
async def test_seed_is_random_when_unset(monkeypatch, tmp_path):
    """When seed=None, the generator picks ONE random seed and locks
    across all slides — NOT a different seed per slide."""
    cfg = _mkcfg(story_carousel=cfg_mod.StoryCarouselConfig(slides=3))

    async def fake_outline(cfg_, persona, ctx, *, n_slides):
        return [
            {"kind": "hook", "title": "a", "body": "", "scene_prompt": "s1", "index": 1},
            {"kind": "content", "title": "b", "body": "b", "scene_prompt": "s2", "index": 2},
            {"kind": "cta", "title": "c", "body": "c", "scene_prompt": "s3", "index": 3},
        ]

    seeds_seen: list[int] = []

    async def fake_gen_image(prompt, seed, cfg_):
        seeds_seen.append(seed)
        p = tmp_path / f"i_{len(seeds_seen)}.jpg"
        p.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 32)
        return p

    async def fake_render(html, out, **kw):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"\xff\xd8jpeg")
        return out

    import instagram_ai_agent.content.generators.story_carousel as mod
    monkeypatch.setattr(mod, "_outline_scenes", fake_outline)
    monkeypatch.setattr(mod, "_generate_slide_image", fake_gen_image)
    monkeypatch.setattr(mod, "render_html_to_png", fake_render)
    monkeypatch.setattr(mod, "apply_lut_image", lambda p, c: p)

    await sc_gen.generate(cfg)
    assert len(seeds_seen) == 3
    # One random seed, reused for every slide
    assert len(set(seeds_seen)) == 1


@pytest.mark.asyncio
async def test_generate_returns_meta_with_consistency_path(monkeypatch, tmp_path):
    cfg = _mkcfg(story_carousel=cfg_mod.StoryCarouselConfig(slides=3, seed=1))

    async def fake_outline(cfg_, persona, ctx, *, n_slides):
        return [
            {"kind": "hook", "title": "a", "body": "", "scene_prompt": "s", "index": 1},
            {"kind": "content", "title": "b", "body": "b", "scene_prompt": "s", "index": 2},
            {"kind": "cta", "title": "c", "body": "c", "scene_prompt": "s", "index": 3},
        ]

    async def fake_gen_image(prompt, seed, cfg_):
        p = tmp_path / f"i_{seed}.jpg"
        p.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 32)
        return p

    async def fake_render(html, out, **kw):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"\xff\xd8jpeg")
        return out

    import instagram_ai_agent.content.generators.story_carousel as mod
    monkeypatch.setattr(mod, "_outline_scenes", fake_outline)
    monkeypatch.setattr(mod, "_generate_slide_image", fake_gen_image)
    monkeypatch.setattr(mod, "render_html_to_png", fake_render)
    monkeypatch.setattr(mod, "apply_lut_image", lambda p, c: p)

    content = await sc_gen.generate(cfg)
    assert content.meta["consistency_path"] == "seed-lock"
    assert content.meta["slide_count"] == 3
    assert content.meta["seed"] == 1
    assert content.meta["persona"]


@pytest.mark.asyncio
async def test_generate_rejects_template_without_background_image(monkeypatch, tmp_path):
    """Audit-inspired: if the template variant doesn't declare
    $background_image, abort rather than silently dropping the photo."""
    cfg = _mkcfg(story_carousel=cfg_mod.StoryCarouselConfig(
        slides=3, seed=1, template_variant="default",  # default has no $background_image
    ))

    async def fake_outline(cfg_, persona, ctx, *, n_slides):
        return [
            {"kind": "hook", "title": "a", "body": "", "scene_prompt": "s", "index": 1},
            {"kind": "content", "title": "b", "body": "b", "scene_prompt": "s", "index": 2},
            {"kind": "cta", "title": "c", "body": "c", "scene_prompt": "s", "index": 3},
        ]

    async def fake_gen_image(prompt, seed, cfg_):
        p = tmp_path / f"i_{seed}.jpg"
        p.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 32)
        return p

    import instagram_ai_agent.content.generators.story_carousel as mod
    monkeypatch.setattr(mod, "_outline_scenes", fake_outline)
    monkeypatch.setattr(mod, "_generate_slide_image", fake_gen_image)

    with pytest.raises(RuntimeError) as exc:
        await sc_gen.generate(cfg)
    assert "background_image" in str(exc.value)


# ─── Pipeline dispatch ───
@pytest.mark.asyncio
async def test_pipeline_dispatches_story_carousel(monkeypatch):
    from instagram_ai_agent.content import pipeline as pipe
    captured = {}

    async def fake_gen(cfg_, trend_ctx):
        captured["called"] = True
        from instagram_ai_agent.content.generators.base import GeneratedContent
        return GeneratedContent(format="story_carousel", media_paths=["/x.jpg"])

    import instagram_ai_agent.content.generators.story_carousel as mod
    monkeypatch.setattr(mod, "generate", fake_gen)
    cfg = _mkcfg()
    out = await pipe._dispatch("story_carousel", cfg, "some trend")
    assert captured.get("called") is True
    assert out.format == "story_carousel"


# ─── LUT failure handling (audit-fix regression guard) ───
def test_apply_lut_image_falls_back_on_ffmpeg_error(tmp_path: Path, monkeypatch):
    """Audit fix: if ffmpeg fails on a bad LUT, return the raw image
    instead of crashing the pipeline."""
    from instagram_ai_agent.content import style

    img = tmp_path / "in.jpg"
    img.write_bytes(b"\xff\xd8\xff\xe0fake")
    lut = tmp_path / "fake.cube"
    lut.write_text("not a real cube")

    cfg = _mkcfg(aesthetic=cfg_mod.Aesthetic(
        palette=["#000", "#fff", "#c9a961"], lut=str(lut),
    ))

    monkeypatch.setattr(style, "_resolve_lut", lambda _name: lut)
    import subprocess as _sub

    def fake_run(cmd, **kw):
        raise _sub.CalledProcessError(returncode=1, cmd=cmd, stderr=b"bad LUT")

    monkeypatch.setattr(style.subprocess, "run", fake_run)
    out = style.apply_lut_image(img, cfg)
    # Must return raw image, not raise
    assert out == img
    assert out.exists()
