"""Character-consistent narrative carousel.

Produces a multi-slide carousel where the SAME persona appears across
every slide — "A day in the life of Jake", "Three mistakes I made at
the gym", etc. Reuses the photo_caption.html overlay template from
the reel-repurpose feature.

Character consistency — **seed-lock**: every slide is generated with
the same seed + same persona prompt prefix + a per-slide scene tail.
Gives ~90% character coherence with zero extra dependencies (always
commercial-safe).

Composition:
  * Works with or without a trained brand LoRA (#13). LoRA gives
    "perfect" character identity; seed-lock gives "good" pose/face
    coherence between slides. Stack both for best results.
  * Works with or without ControlNet pose conditioning (#15). When both
    are active, the LoRA chains into the ControlNet-wrapped
    conditioning — no collisions.
"""
from __future__ import annotations

import base64
import random
from html import escape
from pathlib import Path
from string import Template

import httpx

from instagram_ai_agent.content.generators.base import GeneratedContent, staging_path
from instagram_ai_agent.content.generators.playwright_render import (
    base_css,
    pick_template,
    render_html_to_png,
)
from instagram_ai_agent.content.style import apply_lut_image
from instagram_ai_agent.core.config import BrandCharacter, NicheConfig
from instagram_ai_agent.core.llm import generate_json
from instagram_ai_agent.core.logging_setup import get_logger

log = get_logger(__name__)

TARGET_W = 1080
TARGET_H = 1350


# ─── Persona extraction ─────────
def _persona_description(cfg: NicheConfig) -> str:
    """Build the character description string that gets prefixed onto
    every slide prompt. Reuses the brand character persona when the
    user's configured one; falls back to a niche-themed default.

    (Kept module-local rather than importing from human_photo.py to
    avoid coupling story_carousel to that generator's internals.)"""
    hp = getattr(cfg, "human_photo", None)
    if hp is not None and hp.character.enabled:
        return _brand_persona_one_liner(hp.character)
    return _default_persona_for_niche(cfg.niche)


def _brand_persona_one_liner(ch: BrandCharacter) -> str:
    parts: list[str] = []
    if ch.age_range:
        parts.append(ch.age_range)
    if ch.gender and ch.gender != "androgynous":
        parts.append(ch.gender)
    if ch.ethnicity and ch.ethnicity.lower() not in ("unspecified", "any"):
        parts.append(ch.ethnicity)
    for extra in (ch.hair, ch.build, ch.wardrobe_style, ch.vibe):
        if extra:
            parts.append(extra)
    return ", ".join(parts) or "ordinary person"


def _default_persona_for_niche(niche: str) -> str:
    """When no brand character is configured, generate a plausible
    niche-aligned persona. The same string is used on every slide so
    the seed-lock still anchors the face/build."""
    return f"ordinary person on-niche for {niche}, natural expression, candid moment"


# ─── Scene ideation ─────────
async def _outline_scenes(
    cfg: NicheConfig,
    persona: str,
    trend_context: str,
    *,
    n_slides: int,
) -> list[dict]:
    """Ask the LLM for a sequence of N coherent scenes + per-slide
    caption copy. Returns a list of {scene_prompt, title, body, kind}.
    Kind follows the carousel convention: first=hook, last=cta."""
    system = (
        f"You direct story-driven Instagram carousels for a page about {cfg.niche}.\n"
        f"Audience: {cfg.target_audience}. Voice: {cfg.voice.persona}.\n"
        f"Tone: {', '.join(cfg.voice.tone)}.\n"
        f"Never: {', '.join(cfg.voice.forbidden) or 'n/a'}.\n"
        "Rules:\n"
        f"- Produce exactly {n_slides} slides that tell ONE narrative arc.\n"
        "- Slide 1 is the HOOK, last slide is the CTA, middle slides escalate.\n"
        "- The SAME PERSONA appears in every slide — describe their current\n"
        "  activity, setting, and emotional state, NOT their appearance\n"
        "  (their appearance stays constant across slides).\n"
        "- title ≤6 words, body ≤24 words, scene_prompt 20–45 words.\n"
        "- scene_prompt is a photorealistic brief: activity + setting + lighting\n"
        "  + camera/lens hint. NO on-image text. Candid, not posed."
    )
    prompt = (
        f"Persona (constant across all slides): {persona}.\n"
        f"Trend/context:\n{trend_context or '(a story worth telling for this niche)'}\n\n"
        f"Return JSON: {{\"slides\": [\n"
        "  {\"kind\": \"hook|content|cta\",\n"
        "   \"title\": str,\n"
        "   \"body\": str,\n"
        "   \"scene_prompt\": str},\n"
        f"  ... x{n_slides}\n"
        "]}"
    )
    data = await generate_json("script", prompt, system=system, max_tokens=1800, temperature=0.85)
    raw = data.get("slides") or []
    if not isinstance(raw, list) or len(raw) < n_slides:
        raise ValueError(f"LLM returned {len(raw)} slides, expected {n_slides}")
    out: list[dict] = []
    for i, s in enumerate(raw[:n_slides]):
        out.append({
            "kind": str(s.get("kind") or "content"),
            "title": str(s.get("title") or "").strip(),
            "body": str(s.get("body") or "").strip(),
            "scene_prompt": str(s.get("scene_prompt") or "").strip(),
            "index": i + 1,
        })
    # Enforce shape invariants — same safety net as carousel_repurpose.
    if out:
        out[0]["kind"] = "hook"
    if len(out) > 1:
        out[-1]["kind"] = "cta"
    return out


# ─── Image generation (seed-locked + character-consistent) ─────────
def _full_prompt(persona: str, scene: str, niche: str) -> str:
    """Compose the full photo prompt: persona FIRST (so the LoRA /
    seed-lock anchors on it), scene SECOND."""
    quality = (
        "photorealistic, 35mm film look, natural skin texture, "
        "candid, soft natural lighting, color grade matches palette"
    )
    return f"{persona}, {scene}, {quality}, niche context: {niche}"


_NEGATIVE = (
    "bad anatomy, lowres, watermark, text, overlay, logo, artifacts, "
    "deformed hands, extra fingers, disfigured, noisy, oversharpened, "
    "duplicate face, cartoon, 3d render, cgi"
)


async def _generate_slide_image(
    prompt: str,
    seed: int,
    cfg: NicheConfig,
) -> Path:
    """One slide's image. Routes through ComfyUI when configured (LoRA +
    ControlNet + StoryDiffusion injections fire here), falls back to
    Pollinations Flux-realism when not."""
    from instagram_ai_agent.plugins import comfyui
    if comfyui.configured():
        return await comfyui.generate(
            prompt,
            negative=_NEGATIVE,
            width=TARGET_W,
            height=TARGET_H,
            seed=seed,
            cfg=cfg,
        )

    # Pollinations fallback — commercial-safe, no key needed. Uses the
    # flux-realism model for niche content.
    import os
    from urllib.parse import quote_plus
    url = (
        f"https://image.pollinations.ai/prompt/{quote_plus(prompt)}"
        f"?width={TARGET_W}&height={TARGET_H}&model=flux-realism"
        f"&seed={seed}&nologo=true&enhance=true"
    )
    headers = {}
    token = os.environ.get("POLLINATIONS_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    dest = staging_path(f"story_{seed}", ".jpg")
    async with httpx.AsyncClient(timeout=httpx.Timeout(180.0)) as client:
        async with client.stream("GET", url, headers=headers) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as f:
                async for chunk in resp.aiter_bytes():
                    f.write(chunk)
    if dest.stat().st_size < 10_000:
        raise RuntimeError(f"Pollinations returned a tiny response for seed={seed}")
    return dest


# ─── Slide render (overlay text on generated image) ─────────
def _image_data_url(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}.get(
        suffix, "image/jpeg",
    )
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def _render_slide_html(
    cfg: NicheConfig,
    slide: dict,
    *,
    background: Path,
    total: int,
    tpl: str,
) -> str:
    palette = cfg.aesthetic.palette
    bg = palette[0]
    fg = palette[1] if len(palette) > 1 else "#ffffff"
    accent = palette[2] if len(palette) > 2 else fg

    css = base_css(
        width=TARGET_W,
        height=TARGET_H,
        bg=bg,
        fg=fg,
        body_font=cfg.aesthetic.body_font,
        heading_font=cfg.aesthetic.heading_font,
    )
    is_hook = slide.get("kind") == "hook"
    is_cta = slide.get("kind") == "cta"
    return Template(tpl).safe_substitute(
        css=css,
        heading_font=cfg.aesthetic.heading_font,
        body_font=cfg.aesthetic.body_font,
        bg=bg,
        accent=accent,
        fg=fg,
        title=escape(slide.get("title") or ""),
        body=escape(slide.get("body") or ""),
        index=f"{slide['index']:02d}",
        total=f"{total:02d}",
        hook_class="hook" if is_hook else "",
        cta_class="cta" if is_cta else "",
        watermark=escape(cfg.aesthetic.watermark or ""),
        background_image=_image_data_url(background),
    )


# ─── Public entry point ─────────
async def generate(cfg: NicheConfig, trend_context: str = "") -> GeneratedContent:
    sc = cfg.story_carousel
    n_slides = sc.slides
    persona = _persona_description(cfg)

    # Lock the seed across every slide so the character stays consistent
    # (cheap-but-effective alternative to true Consistent Self-Attention).
    seed = sc.seed if sc.seed is not None else random.randint(1, 10**7)

    outline = await _outline_scenes(cfg, persona, trend_context, n_slides=n_slides)

    # Render images: sequential so ComfyUI doesn't get swamped, but the
    # fallback path could in theory parallelise. Sequential is also
    # easier to observe in the log stream when debugging character drift.
    image_paths: list[Path] = []
    for slide in outline:
        full = _full_prompt(persona, slide["scene_prompt"], cfg.niche)
        try:
            img = await _generate_slide_image(full, seed, cfg)
        except Exception as e:
            log.warning("story_carousel: slide %d image-gen failed: %s", slide["index"], e)
            # Re-raise — a missing slide breaks the narrative; better
            # to fail loud so the pipeline's retry logic kicks in.
            raise
        image_paths.append(img)

    # Slide template — same family used by reel-repurpose
    template_name, tpl = pick_template("carousels", variant=sc.template_variant)
    if "$background_image" not in tpl:
        raise RuntimeError(
            f"story_carousel: template {template_name!r} doesn't declare "
            "$background_image — pick a photo-capable variant."
        )

    # Render each slide as an HTML overlay screenshot
    slide_jpgs: list[str] = []
    for slide, bg in zip(outline, image_paths, strict=True):
        html = _render_slide_html(
            cfg, slide,
            background=bg, total=n_slides, tpl=tpl,
        )
        out = staging_path(f"story_slide{slide['index']:02d}_{template_name}", ".jpg")
        await render_html_to_png(html, out, width=TARGET_W, height=TARGET_H)
        out = apply_lut_image(out, cfg)
        slide_jpgs.append(str(out))

    hook = outline[0]["title"]
    body_preview = " / ".join(s["body"] for s in outline[:3])

    return GeneratedContent(
        format="story_carousel",
        media_paths=slide_jpgs,
        visible_text=f"Hook: {hook}. Story: {body_preview}",
        caption_context=(
            f"A {n_slides}-slide story carousel featuring a single persona "
            f"across a narrative arc. Hook: '{hook}'. Caption must tease the "
            "story without revealing the payoff."
        ),
        generator=f"story_carousel:{template_name}:seed-lock",
        meta={
            "slides": outline,
            "template": template_name,
            "persona": persona,
            "seed": seed,
            "consistency_path": "seed-lock",
            "slide_count": n_slides,
        },
    )
