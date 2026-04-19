"""Film emulation + character lock + new carousel templates."""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from instagram_ai_agent.core import config as cfg_mod
from instagram_ai_agent.core import db
from instagram_ai_agent.plugins import film_emulation


def _mkcfg(**overrides):
    base = dict(
        niche="home calisthenics",
        sub_topics=["pullups"],
        target_audience="dads 35+",
        commercial=True,
        voice=cfg_mod.Voice(tone=["direct"], forbidden=[], persona="ex-office worker"),
        aesthetic=cfg_mod.Aesthetic(palette=["#000", "#fff", "#c9a961"]),
        hashtags=cfg_mod.HashtagPools(core=["calisthenics", "homeworkout", "dadfit"]),
    )
    base.update(overrides)
    return cfg_mod.NicheConfig(**base)


# ─── film_emulation module ───
def test_apply_film_look_off_returns_original(tmp_path):
    p = tmp_path / "in.jpg"
    Image.new("RGB", (64, 64), (120, 120, 120)).save(p, "JPEG")
    original_size = p.stat().st_size
    out = film_emulation.apply_film_look(p, strength="off")
    assert out == p
    assert p.stat().st_size == original_size  # untouched


def test_apply_film_look_actually_modifies_the_file(tmp_path):
    p = tmp_path / "in.jpg"
    Image.new("RGB", (128, 128), (120, 120, 120)).save(p, "JPEG", quality=95)
    before = p.read_bytes()
    out = film_emulation.apply_film_look(p, strength="medium", seed=42)
    assert out == p
    # Bytes must differ — the image was actually processed
    assert p.read_bytes() != before


def test_apply_film_look_deterministic_with_seed(tmp_path):
    p1 = tmp_path / "a.jpg"
    p2 = tmp_path / "b.jpg"
    Image.new("RGB", (80, 80), (130, 130, 130)).save(p1, "JPEG", quality=95)
    Image.new("RGB", (80, 80), (130, 130, 130)).save(p2, "JPEG", quality=95)
    film_emulation.apply_film_look(p1, strength="medium", seed=123)
    film_emulation.apply_film_look(p2, strength="medium", seed=123)
    # Same input + same seed → same output bytes
    assert p1.read_bytes() == p2.read_bytes()


def test_apply_film_look_handles_missing_file(tmp_path):
    p = tmp_path / "does_not_exist.jpg"
    out = film_emulation.apply_film_look(p, strength="medium")
    assert out == p  # returns original path, doesn't crash


def test_strength_presets_defined():
    for s in ("off", "subtle", "medium", "strong"):
        assert s in film_emulation.STRENGTH_PRESETS


# ─── Aesthetic config ───
def test_aesthetic_default_film_strength_is_medium():
    cfg = _mkcfg()
    assert cfg.aesthetic.film_strength == "medium"


def test_aesthetic_rejects_invalid_film_strength():
    import pytest
    with pytest.raises(Exception):
        cfg_mod.Aesthetic(
            palette=["#000", "#fff", "#c9a961"],
            film_strength="extreme",
        )


# ─── style.apply_film_look wrapper ───
def test_style_apply_film_look_respects_off(tmp_path):
    from instagram_ai_agent.content import style

    cfg = _mkcfg(aesthetic=cfg_mod.Aesthetic(
        palette=["#000", "#fff", "#c9a961"], film_strength="off",
    ))
    p = tmp_path / "in.jpg"
    Image.new("RGB", (64, 64), (120, 120, 120)).save(p, "JPEG")
    original_size = p.stat().st_size
    out = style.apply_film_look(p, cfg)
    assert out == p
    assert p.stat().st_size == original_size


# ─── Brand character seed lock ───
@pytest.fixture()
def tmp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fresh = tmp_path / "brain.db"
    monkeypatch.setattr(cfg_mod, "DB_PATH", fresh)
    monkeypatch.setattr(db, "DB_PATH", fresh)
    db.close()
    db.init_db()
    yield db
    db.close()


def test_brand_character_seed_locked_across_calls(tmp_db, monkeypatch):
    from instagram_ai_agent.content.generators import human_photo

    cfg = _mkcfg(human_photo=cfg_mod.HumanPhoto(
        enabled=True,
        character=cfg_mod.BrandCharacter(
            enabled=True,
            age_range="30s",
            gender="male",
            hair="short dark",
            build="athletic",
            wardrobe_style="streetwear",
            vibe="approachable",
            # seed left None → first-call generation should cache
        ),
    ))

    _, seed1 = human_photo._persona_for_gen(cfg)
    _, seed2 = human_photo._persona_for_gen(cfg)
    _, seed3 = human_photo._persona_for_gen(cfg)

    assert seed1 == seed2 == seed3
    # And it's persisted in db.state
    assert db.state_get(human_photo._BRAND_SEED_STATE_KEY) == str(seed1)


def test_brand_character_explicit_seed_wins_over_cache(tmp_db):
    from instagram_ai_agent.content.generators import human_photo

    # Pre-populate a different cached seed
    db.state_set(human_photo._BRAND_SEED_STATE_KEY, "999999")

    cfg = _mkcfg(human_photo=cfg_mod.HumanPhoto(
        enabled=True,
        character=cfg_mod.BrandCharacter(
            enabled=True, seed=12345, age_range="30s",
        ),
    ))
    _, seed = human_photo._persona_for_gen(cfg)
    assert seed == 12345  # explicit config wins


# ─── Carousel template variants ship ───
def test_carousel_templates_present():
    tpl_dir = Path(__file__).resolve().parent.parent / "src" / "instagram_ai_agent" / "content" / "templates" / "carousels"
    existing = {p.stem for p in tpl_dir.glob("*.html")}
    # Originals still there
    assert {"default", "data", "magazine", "photo_caption"}.issubset(existing)
    # New variants shipped
    assert {"flush_left", "bottom_anchor", "oversized_number"}.issubset(existing)
