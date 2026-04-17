"""Photorealistic human-subject generator via Pollinations Flux-Realism.

Runs on a free public endpoint and produces niche-relevant people in niche
contexts (e.g. "calisthenics dad mid-pullup"). Two modes:

  1. ``unique``  — each post draws a random persona from the diversity pool
                   so the feed feels like real community, not one character.
  2. ``brand``   — when ``HumanPhoto.character.enabled`` is true, the same
                   persona + seed anchors every generation for a consistent
                   "face" across the feed (brand character pattern).

Output lands in the feed queue as ``format="human_photo"`` or as a story
via ``story_human``. The underlying model is a free-tier Flux variant; we
route through a short fallback chain on errors.
"""
from __future__ import annotations

import asyncio
import os
import random
from pathlib import Path
from urllib.parse import quote_plus

import httpx

from src.content import image_rank
from src.content.generators.base import GeneratedContent, staging_path
from src.content.style import apply_lut_image, apply_watermark
from src.core.config import BrandCharacter, HumanPhoto, NicheConfig
from src.core.llm import generate
from src.core.logging_setup import get_logger

log = get_logger(__name__)

POLLINATIONS_IMAGE = (
    "https://image.pollinations.ai/prompt/{prompt}"
    "?width={w}&height={h}&model={model}&nologo=true&enhance=true&seed={seed}"
)


# ───── Persona helpers ─────
def _persona_for_gen(cfg: NicheConfig) -> tuple[str, int]:
    """Return (persona_description, seed_to_use)."""
    hp = cfg.human_photo
    if hp.character.enabled:
        return _brand_persona(hp.character), hp.character.seed or random.randint(1, 10**7)
    if hp.diversity_pool:
        pick = random.choice(hp.diversity_pool)
    else:
        pick = "ordinary person, niche-appropriate look"
    return pick, random.randint(1, 10**7)


def _brand_persona(ch: BrandCharacter) -> str:
    """Compact one-line description from a BrandCharacter."""
    parts: list[str] = []
    if ch.age_range:
        parts.append(ch.age_range)
    if ch.gender and ch.gender != "androgynous":
        parts.append(ch.gender)
    if ch.ethnicity and ch.ethnicity.lower() not in ("unspecified", "any"):
        parts.append(ch.ethnicity)
    for f in (ch.hair, ch.build, ch.wardrobe_style, ch.vibe):
        if f:
            parts.append(f)
    return ", ".join(parts) or "thoughtful person"


# ───── Scene ideation ─────
async def _scene_idea(cfg: NicheConfig, persona: str, trend_context: str, *, vertical: bool) -> dict:
    """LLM decides a concrete on-niche situation featuring the persona."""
    orient = "portrait/vertical" if vertical else "square"
    system = (
        f"You direct niche Instagram photography for a page about {cfg.niche}.\n"
        f"Audience: {cfg.target_audience}.\n"
        f"Voice: {cfg.voice.persona}.\n"
        f"Aesthetic palette hint: {', '.join(cfg.aesthetic.palette)}.\n"
        "You write photography prompts for a Flux-realism image model.\n"
        f"Orientation: {orient}.\n"
        "Rules for the prompt:\n"
        "- ONE specific scene, shot, and activity on-niche.\n"
        "- Subject is a real human matching the given persona.\n"
        "- Include lens/focal length hint (e.g. 35mm, 50mm, 85mm), lighting "
        "(soft window, golden hour, dim gym, overcast), and environment.\n"
        "- Photorealistic. Candid, not posed. No on-image text.\n"
        "- 30–55 words. One sentence."
    )
    prompt = (
        f"Persona: {persona}.\n"
        f"Trend/context:\n{trend_context or '(ordinary on-niche moment)'}\n\n"
        "Return JSON: {\"scene\": str, \"caption_context\": str (≤40 words)}"
    )
    from src.core.llm import generate_json
    return await generate_json("caption", prompt, system=system, max_tokens=500, temperature=0.85)


_QUALITY_TAGS = (
    "photorealistic",
    "high detail",
    "natural skin texture",
    "unretouched",
    "shot on Fujifilm X-T5",
    "soft natural lighting",
    "color grade matches palette",
    "35mm film look",
)


def _negative(ch: BrandCharacter) -> str:
    base = (
        "anime, illustration, 3d render, cartoon, plastic skin, deformed face, "
        "extra fingers, bad anatomy, watermark, text overlay, over-saturated, "
        "stock photo look, oversharpened"
    )
    return f"{base}, {ch.negative}" if ch.negative else base


def _build_prompt(scene: str, persona: str, cfg: NicheConfig) -> str:
    # Flux interprets natural language + tags. We append quality tags + palette hint.
    palette_hint = f"dominant tones: {' '.join(cfg.aesthetic.palette[:3])}"
    tags = ", ".join(_QUALITY_TAGS)
    return f"{scene} | subject: {persona} | {tags} | {palette_hint}"


# ───── Image fetch w/ model fallback ─────
async def _fetch(
    client: httpx.AsyncClient,
    prompt: str,
    seed: int,
    *,
    width: int,
    height: int,
    model_chain: list[str],
    negative: str,
    cfg: NicheConfig | None = None,
) -> Path:
    # Try local ComfyUI first when configured (usually better humans on a good GPU).
    from src.plugins import comfyui
    if comfyui.configured():
        try:
            return await comfyui.generate(
                prompt,
                width=width,
                height=height,
                seed=seed,
                negative=negative,
                cfg=cfg,
            )
        except Exception as e:
            log.warning("human_photo: ComfyUI failed, falling back to Pollinations: %s", e)

    headers = {}
    token = os.environ.get("POLLINATIONS_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    last_err: Exception | None = None
    for model in model_chain:
        url = POLLINATIONS_IMAGE.format(
            prompt=quote_plus(f"{prompt} | negative: {negative}"),
            w=width,
            h=height,
            model=model,
            seed=seed,
        )
        try:
            dest = staging_path(f"human_{model}", ".jpg")
            async with client.stream("GET", url, headers=headers) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    async for chunk in r.aiter_bytes():
                        f.write(chunk)
            if dest.stat().st_size < 12_000:
                # Some providers serve a placeholder PNG on failure
                raise RuntimeError(f"Tiny response from {model}")
            log.debug("human_photo: got %s via %s", dest.name, model)
            return dest
        except Exception as e:
            last_err = e
            log.warning("human_photo: model %s failed — %s", model, e)
            continue

    raise RuntimeError(f"All Pollinations models failed: {last_err!r}")


# ───── Entry points ─────
async def generate(
    cfg: NicheConfig,
    trend_context: str = "",
    *,
    width: int = 1080,
    height: int = 1350,
    format_name: str = "human_photo",
) -> GeneratedContent:
    if not cfg.human_photo.enabled:
        raise RuntimeError(
            "human_photo generation is disabled in niche.yaml "
            "(set human_photo.enabled: true to turn it on)."
        )

    persona, seed = _persona_for_gen(cfg)
    vertical = height > width
    scene_data = await _scene_idea(cfg, persona, trend_context, vertical=vertical)
    scene = str(scene_data.get("scene") or "").strip().strip('"')
    caption_ctx = str(scene_data.get("caption_context") or "").strip()

    if not scene:
        raise RuntimeError("LLM did not return a usable scene prompt")

    prompt = _build_prompt(scene, persona, cfg)
    negative = _negative(cfg.human_photo.character)
    models = [cfg.human_photo.model] + list(cfg.human_photo.model_fallbacks)

    # Best-of-N: generate multiple variants with different seeds, vision-rank, pick best.
    n_candidates = max(1, cfg.safety.image_candidates)
    seeds = [seed if i == 0 else random.randint(1, 10**7) for i in range(n_candidates)]

    async with httpx.AsyncClient(timeout=httpx.Timeout(180.0)) as client:
        fetch_tasks = [
            _fetch(
                client, prompt, s,
                width=width, height=height,
                model_chain=models, negative=negative,
                cfg=cfg,
            )
            for s in seeds
        ]
        results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

    candidates = [r for r in results if isinstance(r, Path) and r.exists()]
    if not candidates:
        # Re-raise the first real error so the pipeline's retry logic sees it
        for r in results:
            if isinstance(r, Exception):
                raise r
        raise RuntimeError("No image candidates produced")

    ranked_meta: dict = {}
    if len(candidates) > 1:
        best_path, ranked_meta = await image_rank.pick_best(
            cfg,
            candidates,
            subject_is_human=True,
        )
        raw = Path(best_path)
    else:
        raw = candidates[0]

    # Finish pass — THE reason this pipeline exists. Upscale + face restore.
    from src.plugins import finish_pass
    finish = finish_pass.enhance(raw, cfg.finish, subject_is_human=True)
    if finish.notes:
        log.debug("human_photo finish: %s", finish.notes)

    styled = apply_lut_image(finish.path, cfg)
    final = apply_watermark(styled, cfg) if cfg.aesthetic.watermark else styled

    return GeneratedContent(
        format=format_name,
        media_paths=[str(final)],
        visible_text="",
        caption_context=(caption_ctx or f"Human-subject photo. Scene: {scene}. Persona: {persona}."),
        generator="human_photo",
        meta={
            "scene": scene,
            "persona": persona,
            "seed": seed,
            "candidates": len(candidates),
            "ranking": ranked_meta.get("ranked", []),
            "model_used": models[0],
            "width": width,
            "height": height,
            "brand_locked": cfg.human_photo.character.enabled,
            "finish_backend": finish.backend,
            "upscaled": finish.upscaled,
            "face_restored": finish.face_restored,
        },
    )


async def generate_story(cfg: NicheConfig, trend_context: str = "") -> GeneratedContent:
    """Vertical human story — 1080x1920 output, caption context tuned for stories."""
    from src.content.generators.story_image import _default_stickers

    content = await generate(
        cfg,
        trend_context,
        width=1080,
        height=1920,
        format_name="story_human",
    )
    content.meta.setdefault("stickers", _default_stickers(cfg))
    return content
