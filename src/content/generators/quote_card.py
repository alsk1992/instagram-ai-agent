"""Quote-card generator — LLM quote → HTML/CSS → Playwright screenshot."""
from __future__ import annotations

from html import escape
from string import Template

from src.content.generators.base import GeneratedContent, staging_path
from src.content.generators.playwright_render import base_css, pick_template, render_html_to_png
from src.content.style import apply_lut_image, apply_watermark
from src.core.config import NicheConfig
from src.core.llm import generate_json


async def _llm_quote(
    cfg: NicheConfig, trend_context: str, *, contrarian: bool = False,
) -> dict:
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
    prompt = (
        f"Context/trend to respond to:\n{trend_context or '(general niche wisdom)'}\n\n"
        "Return JSON: {\"quote\": str, \"byline\": str (optional, ≤30 chars)}"
    )
    data = await generate_json("caption", prompt, system=system, max_tokens=300)
    return {
        "quote": str(data.get("quote") or "").strip().strip('"'),
        "byline": str(data.get("byline") or "").strip().strip('"'),
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
