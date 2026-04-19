"""Carousel generator — LLM outline → HTML slides → Playwright screenshots."""
from __future__ import annotations

from html import escape
from string import Template

from instagram_ai_agent.content import slide1_hook as slide1_mod
from instagram_ai_agent.content.generators.base import GeneratedContent, staging_path
from instagram_ai_agent.content.generators.playwright_render import (
    base_css,
    pick_template,
    render_html_to_png,
)
from instagram_ai_agent.content.style import apply_lut_image
from instagram_ai_agent.core.config import NicheConfig
from instagram_ai_agent.core.llm import generate_json


async def _llm_outline(
    cfg: NicheConfig,
    trend_context: str,
    slides: int,
    *,
    contrarian: bool = False,
    slide1: slide1_mod.Slide1Hook | None = None,
) -> list[dict]:
    system = (
        f"You design Instagram carousels for a page about {cfg.niche}.\n"
        f"Audience: {cfg.target_audience}.\n"
        f"Voice: {cfg.voice.persona}. Tone: {', '.join(cfg.voice.tone)}.\n"
        f"Never: {', '.join(cfg.voice.forbidden) or 'n/a'}.\n"
        "Rules:\n"
        "- Slide 1 is the HOOK: a bold scroll-stopping statement, ≤9 words.\n"
        "- Middle slides deliver 1 concrete idea each, ≤24 words of body.\n"
        "- Final slide is a CTA tied to the niche (save, follow, share, tag).\n"
        "- Each slide has a short title (≤6 words) AND body (≤24 words).\n"
        "- Zero fluff. No emojis. No hashtags."
    )
    if contrarian:
        system += (
            "\n\nCONTRARIAN MODE — frame the whole carousel as a hot take:\n"
            "- Slide 1 names a widely-held niche belief the audience holds.\n"
            "- Middle slides build the counter-case with ONE specific reason,\n"
            "  number, or example per slide (no vague outrage).\n"
            "- Final slide delivers the re-framed truth + CTA.\n"
            f"- Never make contrarian claims about: {', '.join(cfg.contrarian.avoid_topics) or 'none'}.\n"
            "- Never touch: medical advice, vaccines, cancer cures, extreme\n"
            "  diets, self-harm, political candidates, ethnic generalisations."
        )
    # When the upstream slide1_hook stage produced a winner, slide 1 is
    # locked and the LLM's job is only to build slides 2..N around it.
    if slide1 is not None:
        slide1_directive = (
            "\n\nSLIDE 1 IS LOCKED by an upstream scroll-stop optimiser. "
            f"Use EXACTLY — title: {slide1.title!r} body: {slide1.body!r}. "
            "Do not paraphrase. Slides 2..N must deliver on its specific promise."
        )
        system += slide1_directive
    prompt = (
        f"Trend/context to riff on:\n{trend_context or '(general niche value-post)'}\n\n"
        f"Produce exactly {slides} slides.\n"
        "Return JSON: {\"slides\":[{\"kind\":\"hook|content|cta\",\"title\":str,\"body\":str}, ...]}"
    )
    data = await generate_json("script", prompt, system=system, max_tokens=1200)
    out = data.get("slides") or []
    if not isinstance(out, list) or len(out) < slides:
        raise ValueError(f"LLM returned {len(out)} slides, expected {slides}")

    parsed = [
        {
            "kind": str(s.get("kind") or "content"),
            "title": str(s.get("title") or "").strip(),
            "body": str(s.get("body") or "").strip(),
            "index": i + 1,
        }
        for i, s in enumerate(out[:slides])
    ]
    # Hard override — if the upstream winner exists, slide 1 is locked no
    # matter what the model produced. This protects against models that
    # "reinterpret" the lock directive.
    if slide1 is not None and parsed:
        parsed[0] = {
            "kind": "hook",
            "title": slide1.title,
            "body": slide1.body,
            "index": 1,
        }
    return parsed


def _render_slide_html(cfg: NicheConfig, slide: dict, total: int, *, tpl: str) -> str:
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
    is_hook = slide["kind"] == "hook"
    is_cta = slide["kind"] == "cta"
    return Template(tpl).safe_substitute(
        css=css,
        heading_font=cfg.aesthetic.heading_font,
        body_font=cfg.aesthetic.body_font,
        bg=bg,
        accent=accent,
        fg=fg,
        title=escape(slide["title"]),
        body=escape(slide["body"]),
        index=f"{slide['index']:02d}",
        total=f"{total:02d}",
        hook_class="hook" if is_hook else "",
        cta_class="cta" if is_cta else "",
        watermark=escape(cfg.aesthetic.watermark or ""),
    )


async def generate(
    cfg: NicheConfig,
    trend_context: str = "",
    *,
    slides: int = 7,
    variant: str | None = None,
    contrarian: bool = False,
) -> GeneratedContent:
    slides = max(3, min(10, slides))
    # Upstream scroll-stop optimiser: pick the best-of-8 slide 1 hook
    # before the outline stage locks the rest of the carousel around it.
    slide1 = await slide1_mod.best_slide1_hook(
        cfg,
        trend_context=trend_context,
        contrarian=contrarian,
    )
    outline = await _llm_outline(
        cfg, trend_context, slides, contrarian=contrarian, slide1=slide1,
    )
    # Pick ONE template for the whole carousel — every slide stays consistent
    template_name, tpl = pick_template("carousels", variant=variant)
    paths: list[str] = []
    for slide in outline:
        html = _render_slide_html(cfg, slide, total=len(outline), tpl=tpl)
        out = staging_path(f"slide{slide['index']:02d}_{template_name}", ".jpg")
        await render_html_to_png(html, out, width=1080, height=1350)
        out = apply_lut_image(out, cfg)
        paths.append(str(out))
    hook = outline[0]["title"]
    body_preview = " / ".join(s["body"] for s in outline[:3])
    return GeneratedContent(
        format="carousel",
        media_paths=paths,
        visible_text=f"Hook: {hook}. Slides: {body_preview}",
        caption_context=(
            f"A {len(outline)}-slide carousel. Hook: '{hook}'. "
            "Caption must tease the value without giving it away."
        ),
        generator=f"carousel:{template_name}",
        meta={
            "slides": outline,
            "template": template_name,
            "slide1_optimised": slide1 is not None,
            "slide1_why": slide1.why if slide1 else None,
        },
    )
