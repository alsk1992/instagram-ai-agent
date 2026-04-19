"""Photo generator — text-to-image via Pollinations with best-of-N vision ranking."""
from __future__ import annotations

import asyncio
import os
import random
from pathlib import Path
from urllib.parse import quote_plus

import httpx

from instagram_ai_agent.content import image_rank
from instagram_ai_agent.content.generators.base import GeneratedContent, staging_path
from instagram_ai_agent.content.style import apply_film_look, apply_lut_image, apply_watermark
from instagram_ai_agent.core.config import NicheConfig
from instagram_ai_agent.core.llm import generate
from instagram_ai_agent.core.logging_setup import get_logger

log = get_logger(__name__)

POLLINATIONS_URL = (
    "https://image.pollinations.ai/prompt/{prompt}"
    "?width={w}&height={h}&model={model}&nologo=true&enhance=true&seed={seed}"
)


async def _ideate_prompt(cfg: NicheConfig, trend_context: str) -> str:
    # Rotate composition hints so consecutive posts don't all centre-frame
    # their subject — the #1 "AI-looking" visual pattern. Each hint is a
    # real photographer's framing rule; picking one randomly gives a
    # varied grid without the LLM having to invent compositions on its own.
    composition = random.choice([
        "subject positioned on the left third, negative space to the right, 35mm lens wide aperture",
        "low-angle shot with leading lines drawing toward the subject, 50mm lens",
        "subject off-centre in the upper third, long horizontal composition, 85mm lens",
        "tight close-up at f/1.8, shallow depth of field, subject filling bottom-right quadrant",
        "rule-of-thirds: subject on a grid intersection, not centre-framed, 35mm",
        "over-the-shoulder POV, foreground blur, subject at mid-depth, cinematic 50mm",
        "three-quarter profile, hand/prop in foreground, soft background bokeh, 85mm",
    ])
    system = (
        f"You craft vivid photography prompts for an Instagram page about {cfg.niche}.\n"
        f"Audience: {cfg.target_audience}. Voice: {cfg.voice.persona}.\n"
        f"Aesthetic palette hint: {', '.join(cfg.aesthetic.palette)}.\n"
        f"Composition directive (MUST be reflected): {composition}.\n"
        "Rules: subject + mood + lighting + camera style + composition. No text-on-image. "
        "Photorealistic. Candid, not posed. Slight imperfection welcome "
        "(uneven light, mild motion, natural skin texture). 25–45 words. "
        "One sentence. No lists."
    )
    prompt = f"Trend/idea to riff on:\n{trend_context or '(general niche scene)'}\n\nReturn ONLY the prompt sentence."
    return (await generate("bulk", prompt, system=system, max_tokens=240)).strip().strip('"')


async def _fetch_single(
    client: httpx.AsyncClient,
    prompt: str,
    seed: int,
    *,
    width: int,
    height: int,
    model: str,
    cfg: NicheConfig | None = None,
) -> Path:
    # Prefer local ComfyUI when available — better quality on decent GPU.
    from instagram_ai_agent.plugins import comfyui
    if comfyui.configured():
        try:
            return await comfyui.generate(
                prompt,
                width=width,
                height=height,
                seed=seed,
                negative="bad anatomy, lowres, watermark, text, artifacts",
                cfg=cfg,
            )
        except Exception as e:
            log.warning("ComfyUI local gen failed (falling back to Pollinations): %s", e)

    token = os.environ.get("POLLINATIONS_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    dest = staging_path(f"photo_{seed}", ".jpg")
    url = POLLINATIONS_URL.format(
        prompt=quote_plus(prompt),
        w=width,
        h=height,
        model=model,
        seed=seed,
    )
    async with client.stream("GET", url, headers=headers) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as f:
            async for chunk in resp.aiter_bytes():
                f.write(chunk)
    if dest.stat().st_size < 10_000:
        raise RuntimeError(f"tiny response for seed={seed}")
    return dest


async def generate_image(
    cfg: NicheConfig,
    trend_context: str = "",
    *,
    width: int = 1080,
    height: int = 1350,
    model: str = "flux",
) -> GeneratedContent:
    text_prompt = await _ideate_prompt(cfg, trend_context)

    n = max(1, cfg.safety.image_candidates)
    seeds = [random.randint(1, 10**7) for _ in range(n)]

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        tasks = [
            _fetch_single(client, text_prompt, s, width=width, height=height, model=model, cfg=cfg)
            for s in seeds
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    candidates = [r for r in results if isinstance(r, Path) and r.exists()]
    if not candidates:
        for r in results:
            if isinstance(r, Exception):
                raise r
        raise RuntimeError("No image candidates produced")

    ranked_meta: dict = {}
    if len(candidates) > 1:
        best_path, ranked_meta = await image_rank.pick_best(cfg, candidates)
        raw = Path(best_path)
    else:
        raw = candidates[0]

    # Finish pass: upscale (+ face restore when human). Silent no-op if disabled.
    from instagram_ai_agent.plugins import finish_pass
    finish = finish_pass.enhance(raw, cfg.finish, subject_is_human=False)
    if finish.notes:
        log.debug("photo finish: %s", finish.notes)

    styled = apply_lut_image(finish.path, cfg)
    # Film emulation — grain + vignette + colour cast → kills the sterile
    # AI look. Runs on the LUT-graded result so the grain sits in the
    # final colour space.
    filmic = apply_film_look(styled, cfg)
    final = apply_watermark(filmic, cfg)

    return GeneratedContent(
        format="photo",
        media_paths=[str(final)],
        visible_text="",
        caption_context=f"Photo post. Visual: {text_prompt}",
        generator="photo",
        meta={
            "prompt": text_prompt,
            "model": model,
            "candidates": len(candidates),
            "ranking": ranked_meta.get("ranked", []),
            "finish_backend": finish.backend,
            "upscaled": finish.upscaled,
            "face_restored": finish.face_restored,
        },
    )


# Backwards-compatible alias used by pipeline dispatch
generate = generate_image  # type: ignore[assignment]
