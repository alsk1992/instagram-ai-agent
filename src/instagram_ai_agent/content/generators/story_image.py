"""Story image generator — 1080x1920 vertical static stories.

Three variants share one HTML template with conditional styling:

  - ``story_quote``       A short niche quote + byline.
  - ``story_announcement``A bold one-liner (new drop, incoming post, CTA).
  - ``story_photo``       A photo (AI-gen via Pollinations) with minimal overlay.

Each returns a ``GeneratedContent`` with optional sticker data in ``meta``
so the IG uploader can attach mentions / hashtags / links at post time.
"""
from __future__ import annotations

import os
import random
import time
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from string import Template
from urllib.parse import quote_plus

import httpx

from instagram_ai_agent.content.generators.base import GeneratedContent, staging_path
from instagram_ai_agent.content.generators.playwright_render import base_css, load_template, render_html_to_png
from instagram_ai_agent.content.style import apply_lut_image, apply_watermark
from instagram_ai_agent.core.config import NicheConfig
from instagram_ai_agent.core.llm import generate_json
from instagram_ai_agent.core.logging_setup import get_logger

log = get_logger(__name__)

STORY_W, STORY_H = 1080, 1920

POLLINATIONS_IMAGE = (
    "https://image.pollinations.ai/prompt/{prompt}"
    "?width={w}&height={h}&model={model}&nologo=true&enhance=true&seed={seed}"
)


# ───────── LLM ideation per variant ─────────
async def _ideate_quote(cfg: NicheConfig, trend_context: str) -> dict:
    system = (
        f"You write short niche-specific quotes for a page about {cfg.niche}.\n"
        f"Audience: {cfg.target_audience}.\n"
        f"Voice: {cfg.voice.persona}. Tone: {', '.join(cfg.voice.tone)}.\n"
        f"Never: {', '.join(cfg.voice.forbidden) or 'n/a'}.\n"
        "Rules: ≤14 words hook. No quotation marks. No emojis."
    )
    prompt = (
        f"Trend/context:\n{trend_context or '(niche wisdom)'}\n\n"
        "Return JSON: {\"hook\": str, \"byline\": str (≤30 chars)}"
    )
    data = await generate_json("caption", prompt, system=system, max_tokens=250)
    return {"hook": str(data.get("hook") or "").strip().strip('"'),
            "byline": str(data.get("byline") or "").strip().strip('"')}


async def _ideate_announcement(cfg: NicheConfig, trend_context: str) -> dict:
    system = (
        f"You write bold Instagram story announcements for a page about {cfg.niche}.\n"
        f"Voice: {cfg.voice.persona}. Tone: {', '.join(cfg.voice.tone)}.\n"
        f"Never: {', '.join(cfg.voice.forbidden) or 'n/a'}.\n"
        "Rules: a scroll-stopping HOOK (≤8 words) + a DETAIL (≤20 words) + a CTA (2–4 words)."
    )
    prompt = (
        f"Context / reason for the announcement:\n{trend_context or '(new post imminent)'}\n\n"
        "Return JSON: {\"hook\": str, \"detail\": str, \"cta\": str}"
    )
    data = await generate_json("caption", prompt, system=system, max_tokens=300)
    return {
        "hook": str(data.get("hook") or "").strip().strip('"'),
        "detail": str(data.get("detail") or "").strip().strip('"'),
        "cta": str(data.get("cta") or "").strip().strip('"'),
    }


async def _ideate_photo_prompt(cfg: NicheConfig, trend_context: str) -> str:
    system = (
        f"You craft photography prompts for an Instagram story background on a page about {cfg.niche}.\n"
        f"Palette hint: {', '.join(cfg.aesthetic.palette)}.\n"
        "Rules: subject + mood + lighting + camera style, 20–40 words, one sentence, photorealistic. "
        "No on-image text."
    )
    prompt = f"Trend/context:\n{trend_context or '(ambient niche scene)'}\n\nReturn ONLY the prompt."
    from instagram_ai_agent.core.llm import generate as _g
    out = await _g("bulk", prompt, system=system, max_tokens=200, temperature=0.8)
    return out.strip().strip('"')


# ───────── Rendering ─────────
def _render_html(
    cfg: NicheConfig,
    *,
    kind: str,
    hook: str,
    detail: str = "",
    cta: str = "",
    background_image: str | None = None,
) -> str:
    tpl = load_template("stories")
    palette = cfg.aesthetic.palette
    bg = palette[0]
    fg = palette[1] if len(palette) > 1 else "#ffffff"
    accent = palette[2] if len(palette) > 2 else fg

    css = base_css(
        width=STORY_W,
        height=STORY_H,
        bg=bg,
        fg=fg,
        body_font=cfg.aesthetic.body_font,
        heading_font=cfg.aesthetic.heading_font,
    )
    # Background image override (for story_photo variant)
    if background_image:
        css += (
            "\nbody {"
            f"  background-image: linear-gradient(to bottom, rgba(0,0,0,0.35), rgba(0,0,0,0.75)), "
            f"url('file://{background_image}');"
            "  background-size: cover;"
            "  background-position: center;"
            "}\n"
        )

    kind_class = {
        "story_quote": "quote",
        "story_announcement": "announcement",
        "story_photo": "photo",
    }.get(kind, "")
    kind_label = {
        "story_quote": "note",
        "story_announcement": "drop",
        "story_photo": "caught this",
    }.get(kind, "story")

    cta_block = (
        f'<div class="cta">{escape(cta)}</div>' if cta else ""
    )
    swipe_hint = {
        "story_announcement": "tap up →",
        "story_quote": "save this",
        "story_photo": "new post",
    }.get(kind, "")

    return Template(tpl).safe_substitute(
        css=css,
        heading_font=cfg.aesthetic.heading_font,
        body_font=cfg.aesthetic.body_font,
        accent=accent,
        fg=fg,
        bg=bg,
        kind_class=kind_class,
        kind_label=kind_label,
        hook=escape(hook),
        detail=escape(detail),
        cta_block=cta_block,
        timestamp=datetime.now(timezone.utc).strftime("%b %d").lower(),
        watermark=escape(cfg.aesthetic.watermark or ""),
        swipe_hint=swipe_hint,
    )


async def _fetch_photo_bg(cfg: NicheConfig, text_prompt: str) -> str:
    dest = staging_path("storybg", ".jpg")
    model = os.environ.get("POLLINATIONS_IMAGE_MODEL", "flux")
    url = POLLINATIONS_IMAGE.format(
        prompt=quote_plus(text_prompt),
        w=STORY_W,
        h=STORY_H,
        model=model,
        seed=random.randint(1, 10_000_000),
    )
    headers = {}
    token = os.environ.get("POLLINATIONS_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        async with client.stream("GET", url, headers=headers) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                async for chunk in r.aiter_bytes():
                    f.write(chunk)
    return str(dest.resolve())


# ───────── Stickers ─────────
def _default_stickers(cfg: NicheConfig) -> dict:
    """Build sticker metadata for the uploader.

    Keys are sticker types understood by src/plugins/ig.py#upload_story_image.
    """
    stickers: dict = {}
    if cfg.aesthetic.watermark:
        mention = cfg.aesthetic.watermark.lstrip("@")
        if mention:
            stickers["mention"] = mention
    # First core hashtag becomes a story sticker too (clickable)
    if cfg.hashtags.core:
        stickers["hashtag"] = cfg.hashtags.core[0].lstrip("#")
    return stickers


# ───────── Public entry points ─────────
async def generate_quote(cfg: NicheConfig, trend_context: str = "") -> GeneratedContent:
    data = await _ideate_quote(cfg, trend_context)
    hook = data["hook"] or "the grind is the gift"
    detail = data["byline"]
    html = _render_html(cfg, kind="story_quote", hook=hook, detail=detail)
    out = staging_path("story_quote", ".jpg")
    await render_html_to_png(html, out, width=STORY_W, height=STORY_H)
    styled = apply_lut_image(out, cfg)
    final = apply_watermark(styled, cfg) if cfg.aesthetic.watermark else styled
    return GeneratedContent(
        format="story_quote",
        media_paths=[str(final)],
        visible_text=f"{hook} — {detail}".strip(" —"),
        caption_context=f"Story: quote '{hook}'",
        generator="story_quote",
        meta={"stickers": _default_stickers(cfg)},
    )


async def generate_announcement(cfg: NicheConfig, trend_context: str = "") -> GeneratedContent:
    data = await _ideate_announcement(cfg, trend_context)
    hook = data["hook"] or "new drop"
    detail = data["detail"]
    cta = data["cta"] or "link in bio"
    html = _render_html(cfg, kind="story_announcement", hook=hook, detail=detail, cta=cta)
    out = staging_path("story_announce", ".jpg")
    await render_html_to_png(html, out, width=STORY_W, height=STORY_H)
    styled = apply_lut_image(out, cfg)
    final = apply_watermark(styled, cfg) if cfg.aesthetic.watermark else styled
    return GeneratedContent(
        format="story_announcement",
        media_paths=[str(final)],
        visible_text=f"{hook}. {detail}. {cta}",
        caption_context=f"Announcement story: '{hook}'. Detail: '{detail}'.",
        generator="story_announcement",
        meta={"stickers": _default_stickers(cfg)},
    )


async def generate_photo(cfg: NicheConfig, trend_context: str = "") -> GeneratedContent:
    text_prompt = await _ideate_photo_prompt(cfg, trend_context)
    bg_path = await _fetch_photo_bg(cfg, text_prompt)

    # Intentionally no finish-pass here. The BG is rendered as a CSS
    # ``background-size: cover`` under a 1080×1920 viewport and Playwright
    # owns the final resolution via ``deviceScaleFactor=2``. Upscaling the
    # BG first would be immediately discarded by the browser downsample.

    data = await _ideate_quote(cfg, trend_context)
    hook = data["hook"] or cfg.niche
    html = _render_html(
        cfg,
        kind="story_photo",
        hook=hook,
        detail="",
        background_image=bg_path,
    )
    out = staging_path("story_photo", ".jpg")
    await render_html_to_png(html, out, width=STORY_W, height=STORY_H)
    final = apply_lut_image(out, cfg)
    return GeneratedContent(
        format="story_photo",
        media_paths=[str(final)],
        visible_text=hook,
        caption_context=f"Photo story. Visual: {text_prompt}. Overlay: '{hook}'.",
        generator="story_photo",
        meta={"stickers": _default_stickers(cfg), "prompt": text_prompt},
    )


STORY_IMAGE_DISPATCH = {
    "story_quote": generate_quote,
    "story_announcement": generate_announcement,
    "story_photo": generate_photo,
}


async def generate(cfg: NicheConfig, trend_context: str = "", *, variant: str | None = None) -> GeneratedContent:
    """Pick a variant at random (or use `variant`)."""
    if variant and variant in STORY_IMAGE_DISPATCH:
        return await STORY_IMAGE_DISPATCH[variant](cfg, trend_context)
    fn = random.choice(list(STORY_IMAGE_DISPATCH.values()))
    return await fn(cfg, trend_context)
