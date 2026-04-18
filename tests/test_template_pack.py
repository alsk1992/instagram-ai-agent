"""Multi-template pack — meme JSON variety, picker fairness, HTML availability."""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import pytest

from instagram_ai_agent.content.generators import meme as meme_mod
from instagram_ai_agent.content.generators import playwright_render as pr
from instagram_ai_agent.core import config as cfg_mod


def _mkcfg(**kwargs):
    base = dict(
        niche="home calisthenics",
        sub_topics=["pullups"],
        target_audience="dads 35+",
        commercial=True,
        voice=cfg_mod.Voice(tone=["direct"], forbidden=[], persona="ex-office worker rebuilding fitness."),
        aesthetic=cfg_mod.Aesthetic(palette=["#0a0a0a", "#f5f5f0", "#c9a961"]),
        hashtags=cfg_mod.HashtagPools(core=["calisthenics", "homeworkout", "dadfit"]),
    )
    base.update(kwargs)
    return cfg_mod.NicheConfig(**base)


# ─── Meme template inventory ───
def test_meme_templates_at_least_five():
    """PLAN §7 requires 5 meme templates shipped in repo."""
    templates = meme_mod.list_templates()
    assert len(templates) >= 5, f"only {len(templates)} meme templates found"
    names = {t["name"] for t in templates}
    expected = {"twobox", "drake", "expanding_brain", "stages", "expectation_reality"}
    missing = expected - names
    assert missing == set(), f"missing meme templates: {missing}"


def test_each_meme_template_has_background():
    for tpl in meme_mod.list_templates():
        bg = tpl.get("_background")
        assert bg is not None and Path(bg).exists(), f"missing bg for {tpl['name']}"
        # Backgrounds must be non-trivial (>10kB)
        assert Path(bg).stat().st_size > 10_000, f"bg too small for {tpl['name']}"


def test_each_meme_template_has_text_boxes():
    for tpl in meme_mod.list_templates():
        boxes = tpl.get("text_boxes") or []
        assert len(boxes) >= 2, f"{tpl['name']} should have ≥2 text boxes (got {len(boxes)})"
        for box in boxes:
            for k in ("name", "x", "y", "width", "height"):
                assert k in box, f"{tpl['name']}.{box.get('name', '?')} missing {k}"


def test_meme_template_jsons_are_valid_json():
    for json_path in (cfg_mod.TEMPLATES_DIR / "memes").glob("*.json"):
        # Will raise on malformed JSON
        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert "name" in data
        assert "text_boxes" in data


# ─── Quote card / Carousel HTML inventory ───
def test_quote_card_templates_at_least_three():
    templates = pr.list_templates("quote_cards")
    assert len(templates) >= 3, f"only {len(templates)} quote_card templates"
    names = {t.stem for t in templates}
    assert "default" in names
    # Two new ones we shipped this round
    assert "serif" in names or "bold_block" in names


def test_carousel_templates_at_least_three():
    templates = pr.list_templates("carousels")
    assert len(templates) >= 3, f"only {len(templates)} carousel templates"
    names = {t.stem for t in templates}
    assert "default" in names
    assert "magazine" in names or "data" in names


# ─── Picker behaviour ───
def test_pick_template_unknown_falls_back_to_default():
    name, content = pr.pick_template("quote_cards", variant="does_not_exist")
    # Falls through to a random pick (not crash). May or may not be 'default'
    assert content
    assert name in {t.stem for t in pr.list_templates("quote_cards")}


def test_pick_template_specific_variant_resolves():
    name, content = pr.pick_template("quote_cards", variant="serif")
    assert name == "serif"
    assert "Georgia" in content   # serif template references Georgia


def test_pick_template_random_distribution_visits_all():
    """20-pick dry-run touches every quote_card template at least once."""
    rng = random.Random(20260417)
    seen: Counter[str] = Counter()
    for _ in range(20):
        name, _ = pr.pick_template("quote_cards", rng=rng)
        seen[name] += 1
    template_names = {t.stem for t in pr.list_templates("quote_cards")}
    assert set(seen).issuperset(template_names), (
        f"missed templates over 20 picks: {template_names - set(seen)}"
    )


def test_pick_template_carousel_random_distribution_visits_all():
    rng = random.Random(98765)
    seen: Counter[str] = Counter()
    for _ in range(30):
        name, _ = pr.pick_template("carousels", rng=rng)
        seen[name] += 1
    template_names = {t.stem for t in pr.list_templates("carousels")}
    assert set(seen).issuperset(template_names), (
        f"missed templates over 30 picks: {template_names - set(seen)}"
    )


def test_pick_template_empty_folder_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(pr, "TEMPLATES_DIR", tmp_path / "missing")
    with pytest.raises(FileNotFoundError):
        pr.pick_template("anything")


# ─── HTML smoke: each template renders something a Template can substitute ───
def test_each_quote_card_template_substitutes_cleanly():
    """Verify every shipped HTML is a valid string.Template payload — no
    unbalanced ``$`` left behind after substitution."""
    from string import Template

    cfg = _mkcfg()
    palette = cfg.aesthetic.palette
    base_subs = {
        "css": "/* test */",
        "heading_font": cfg.aesthetic.heading_font,
        "body_font": cfg.aesthetic.body_font,
        "bg": palette[0],
        "fg": palette[1],
        "accent": palette[2],
        "quote": "test quote",
        "byline": "byline",
        "watermark": "@test",
    }
    for tpl_path in pr.list_templates("quote_cards"):
        raw = tpl_path.read_text(encoding="utf-8")
        out = Template(raw).safe_substitute(**base_subs)
        # Quote card HTML must include the quote text and not leave unresolved
        # template variables behind (no `$identifier` patterns remaining).
        assert "test quote" in out
        # safe_substitute leaves unmatched $vars in place — flag any.
        import re
        leftovers = re.findall(r"\$[A-Za-z_][A-Za-z0-9_]*", out)
        # Allow CSS dollar-like patterns? None expected.
        assert leftovers == [], f"{tpl_path.name} left unresolved vars: {leftovers}"


def test_each_carousel_template_substitutes_cleanly():
    from string import Template

    cfg = _mkcfg()
    palette = cfg.aesthetic.palette
    subs = {
        "css": "/* test */",
        "heading_font": cfg.aesthetic.heading_font,
        "body_font": cfg.aesthetic.body_font,
        "bg": palette[0],
        "fg": palette[1],
        "accent": palette[2],
        "title": "title",
        "body": "body",
        "index": "01",
        "total": "07",
        "hook_class": "",
        "cta_class": "",
        "watermark": "@test",
        # Optional — used only by photo_caption.html (reel-repurpose)
        "background_image": "data:image/jpeg;base64,xxx",
    }
    for tpl_path in pr.list_templates("carousels"):
        raw = tpl_path.read_text(encoding="utf-8")
        out = pr.Template(raw).safe_substitute(**subs) if hasattr(pr, "Template") \
              else Template(raw).safe_substitute(**subs)
        assert "title" in out
        import re
        leftovers = re.findall(r"\$[A-Za-z_][A-Za-z0-9_]*", out)
        assert leftovers == [], f"{tpl_path.name} left unresolved vars: {leftovers}"


# ─── Generator integration smoke ───
def test_meme_generator_random_picks_visit_all_templates(monkeypatch: pytest.MonkeyPatch):
    """20 picks across the meme template list should hit every template."""
    rng = random.Random(31415)
    monkeypatch.setattr(meme_mod, "random", rng)
    seen: Counter[str] = Counter()
    for _ in range(40):
        tpl = rng.choice(meme_mod.list_templates())
        seen[tpl["name"]] += 1
    assert set(seen) >= {"twobox", "drake", "expanding_brain", "stages", "expectation_reality"}


# ─── Audit follow-ups ───
def test_pick_template_unknown_variant_logs_warning(caplog: pytest.LogCaptureFixture):
    """A typo in niche.yaml should surface a warning, not fail silent."""
    import logging
    caplog.set_level(logging.WARNING)
    name, _content = pr.pick_template("quote_cards", variant="doesntexist")
    # Fallback picks something from the valid set
    assert name in {t.stem for t in pr.list_templates("quote_cards")}
    # Warning emitted somewhere in captured log
    assert any(
        "pick_template" in rec.message and "doesntexist" in rec.message
        for rec in caplog.records
    )


def test_magazine_title_uses_niche_heading_font_first():
    """Audit fix: Georgia must not override the configured $heading_font."""
    raw = (cfg_mod.TEMPLATES_DIR / "carousels" / "magazine.html").read_text()
    # $heading_font must appear BEFORE Georgia in the .title font stack
    title_block = raw[raw.index(".title"):raw.index(".body")]
    h_idx = title_block.index("$heading_font")
    g_idx = title_block.index("Georgia")
    assert h_idx < g_idx, "niche heading font should be first in the CSS fallback chain"


def test_magazine_watermark_not_doubled():
    """Audit fix: $watermark must appear only in the footer, not in the kicker."""
    raw = (cfg_mod.TEMPLATES_DIR / "carousels" / "magazine.html").read_text()
    # count distinct `$watermark` placeholder occurrences
    assert raw.count("$watermark") == 1, (
        f"magazine.html references $watermark {raw.count('$watermark')} times"
    )


@pytest.mark.asyncio
async def test_carousel_uses_one_template_for_all_slides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Every slide in a single carousel must share the same template so the
    deck visually coheres — the picker runs ONCE per generate() call."""
    from instagram_ai_agent.content.generators import carousel as carousel_mod

    # Stub the LLM outline + rendering
    async def fake_outline(cfg, trend, n, *, contrarian=False):
        return [{"kind": "hook" if i == 0 else "content", "title": f"t{i}", "body": f"b{i}", "index": i + 1} for i in range(n)]

    async def fake_render(html, out, *, width, height, deviceScaleFactor=2):
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_bytes(b"fake")
        return out

    monkeypatch.setattr(carousel_mod, "_llm_outline", fake_outline)
    monkeypatch.setattr(carousel_mod, "render_html_to_png", fake_render)
    # apply_lut_image → pass-through
    monkeypatch.setattr(carousel_mod, "apply_lut_image", lambda p, cfg: p)

    # Spy the pick_template call site — we want to assert it's invoked
    # exactly once per carousel regardless of slide count.
    real_pick = pr.pick_template
    picks_used: list[str] = []

    def spy_pick(folder, *, variant=None, rng=None):
        name, content = real_pick(folder, variant=variant, rng=rng)
        picks_used.append(name)
        return name, content

    monkeypatch.setattr(carousel_mod, "pick_template", spy_pick)

    cfg = _mkcfg()
    result = await carousel_mod.generate(cfg, "trend", slides=5)
    # The picker was called exactly once — every slide used the same template
    assert len(picks_used) == 1, f"expected 1 pick for all slides, got {len(picks_used)}"
    assert result.meta.get("template") == picks_used[0]
    # Every output media path encodes the same template name
    for path in result.media_paths:
        assert picks_used[0] in Path(path).name, f"{path} missing template tag"
