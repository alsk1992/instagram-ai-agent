"""Quote-card generator — real curated quote → HTML/CSS → Playwright screenshot.

The v1 of this generator asked an LLM to invent both the quote AND the
attribution. Result: plausible-sounding fake quotes with invented bylines
("Your desk chair isn't a throne" — nobody said that). User feedback on
a real post was direct: it's AI slop pretending to be profound.

v2 (this version) loads a curated library of REAL quotes with verified
attributions (Marcus Aurelius, Seneca, Churchill, Jocko, James Clear,
Muhammad Ali, etc.) — bundled as ``library.json`` in this directory. We
filter by tags that match the niche config (via simple keyword overlap
with sub_topics + niche), pick one at random from the filtered pool,
and use it unchanged. LLM invention is a last-resort fallback when the
library has no matches (shouldn't happen for fitness/discipline niches
which are well-represented; matters for e.g. finance or pets where we
ship fewer library entries).
"""
from __future__ import annotations

import json
import random
from html import escape
from pathlib import Path
from string import Template

from instagram_ai_agent.content.generators.base import GeneratedContent, staging_path
from instagram_ai_agent.content.generators.playwright_render import (
    base_css,
    pick_template,
    render_html_to_png,
)
from instagram_ai_agent.content.style import apply_lut_image, apply_watermark
from instagram_ai_agent.core.config import NicheConfig
from instagram_ai_agent.core.llm import generate_json
from instagram_ai_agent.core.logging_setup import get_logger

log = get_logger(__name__)


# Lazy-loaded cache of the quote library.
_LIBRARY_CACHE: list[dict] | None = None


def _load_library() -> list[dict]:
    global _LIBRARY_CACHE
    if _LIBRARY_CACHE is not None:
        return _LIBRARY_CACHE
    lib_path = Path(__file__).parent.parent / "templates" / "quote_cards" / "library.json"
    try:
        data = json.loads(lib_path.read_text(encoding="utf-8"))
        _LIBRARY_CACHE = list(data.get("quotes") or [])
    except Exception as e:
        log.warning("quote_card: library load failed — %s (LLM fallback only)", e)
        _LIBRARY_CACHE = []
    return _LIBRARY_CACHE


def _niche_keywords(cfg: NicheConfig) -> set[str]:
    """Keywords to match against quote tags — derived from niche + sub-topics."""
    words = set()
    for src in (cfg.niche, cfg.target_audience, *cfg.sub_topics):
        for tok in src.lower().replace("-", " ").split():
            if len(tok) >= 3:
                words.add(tok)
    # Fitness/calisthenics common themes map well to our library tags:
    fitness_markers = {"fitness", "calisthenics", "gym", "workout", "training",
                       "bodyweight", "pullup", "pullups", "mobility", "strength"}
    if words & fitness_markers:
        words.update({"fitness", "discipline", "consistency", "persistence",
                      "effort", "habits", "growth", "mindset", "training"})
    # Generic fallback — always-valid tags that fit any aspirational niche
    words.update({"discipline", "action", "mindset", "effort", "persistence"})
    return words


def _pick_from_library(cfg: NicheConfig) -> dict | None:
    """Pick a curated quote whose tags overlap with the niche. Returns
    None when library is empty or no matches."""
    library = _load_library()
    if not library:
        return None
    kws = _niche_keywords(cfg)
    matches = [
        q for q in library
        if any(tag in kws for tag in (q.get("tags") or []))
    ]
    pool = matches or library  # fall back to full library if no tag overlap
    choice = random.choice(pool)
    return {
        "quote": str(choice.get("quote") or "").strip(),
        "byline": str(choice.get("byline") or "").strip(),
    }


async def _llm_quote(
    cfg: NicheConfig, trend_context: str, *, contrarian: bool = False,
) -> dict:
    """Pick a real curated quote matching the niche. Falls back to LLM
    invention only if the library has zero entries (shouldn't happen in
    practice — library ships with 80+ quotes covering common themes)."""
    # Primary path: curated library (real attributions, no hallucinations).
    if not contrarian:
        picked = _pick_from_library(cfg)
        if picked and picked["quote"]:
            return picked

    # Fallback: LLM invention. Only used for contrarian mode (library has
    # no contrarian takes by design) or if library load failed entirely.
    log.debug("quote_card: falling back to LLM quote invention (contrarian=%s)", contrarian)
    system = (
        f"You write short, original, niche-specific quotes for Instagram quote cards.\n"
        f"Niche: {cfg.niche}. Audience: {cfg.target_audience}.\n"
        f"Voice: {cfg.voice.persona}. Tone: {', '.join(cfg.voice.tone)}.\n"
        f"Avoid: clichés, {', '.join(cfg.voice.forbidden) or 'n/a'}.\n"
        "Constraints: ≤18 words, punchy, specific, no quotation marks, no hashtags."
    )
    if contrarian:
        system += (
            "\n\nCONTRARIAN QUOTE — the quote must challenge a widely-held "
            "niche assumption. Memorable because it contradicts consensus.\n"
            f"Avoid takes on: {', '.join(cfg.contrarian.avoid_topics) or 'none'}.\n"
            "No medical, conspiracy, self-harm, or political-group claims."
        )
    # When inventing, don't fake an attribution. Empty byline means the
    # template renders the quote clean, no "— Some Fake Person".
    prompt = (
        f"Context/trend to respond to:\n{trend_context or '(general niche wisdom)'}\n\n"
        "Return JSON: {\"quote\": str, \"byline\": \"\"}  "
        "— byline MUST be empty. Do NOT invent an attribution."
    )
    data = await generate_json("caption", prompt, system=system, max_tokens=300)
    return {
        "quote": str(data.get("quote") or "").strip().strip('"'),
        "byline": "",  # never fake an attribution
    }


def _render_html(cfg: NicheConfig, quote: str, byline: str, *, variant: str | None = None) -> tuple[str, str]:
    template_name, tpl = pick_template("quote_cards", variant=variant)
    palette = cfg.aesthetic.palette
    bg = palette[0]
    fg = palette[1] if len(palette) > 1 else "#ffffff"
    accent = palette[2] if len(palette) > 2 else fg

    css = base_css(
        width=1080,
        height=1350,
        bg=bg,
        fg=fg,
        body_font=cfg.aesthetic.body_font,
        heading_font=cfg.aesthetic.heading_font,
    )
    heading_font = cfg.aesthetic.heading_font
    watermark = cfg.aesthetic.watermark or ""

    doc = Template(tpl).safe_substitute(
        css=css,
        heading_font=heading_font,
        body_font=cfg.aesthetic.body_font,
        bg=bg,
        accent=accent,
        fg=fg,
        quote=escape(quote),
        byline=escape(byline),
        watermark=escape(watermark),
    )
    return template_name, doc


async def generate(
    cfg: NicheConfig,
    trend_context: str = "",
    *,
    variant: str | None = None,
    contrarian: bool = False,
) -> GeneratedContent:
    data = await _llm_quote(cfg, trend_context, contrarian=contrarian)
    quote = data["quote"] or "the grind is the gift"
    byline = data["byline"]
    template_name, html = _render_html(cfg, quote, byline, variant=variant)
    out = staging_path(f"quote_{template_name}", ".jpg")
    await render_html_to_png(html, out, width=1080, height=1350)

    # LUT + watermark (watermark might already be inside the HTML, but
    # apply_watermark is a no-op if cfg.aesthetic.watermark is None/empty)
    styled = apply_lut_image(out, cfg)
    final = apply_watermark(styled, cfg) if not cfg.aesthetic.watermark else styled

    return GeneratedContent(
        format="quote_card",
        media_paths=[str(final)],
        visible_text=f"{quote} — {byline}".strip(" —"),
        caption_context=f"Quote card showing: '{quote}'. Caption should expand, not repeat.",
        generator=f"quote_card:{template_name}",
        meta={"quote": quote, "byline": byline, "template": template_name},
    )
