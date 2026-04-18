"""Story video generator — 8–12s vertical clip with voiceover + captions.

Lightweight vs reels:
  - single scene (one AI image OR one stock clip)
  - one-sentence voiceover
  - ken-burns motion if image, direct normalise if stock
  - captions in a big/punchy story-friendly style

Output goes to the content queue as format=``story_video`` and is posted
via IGClient.upload_story_video.
"""
from __future__ import annotations

import asyncio
import os
import random
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import quote_plus

import httpx

from instagram_ai_agent.content.generators import reel_stock
from instagram_ai_agent.content.generators.base import GeneratedContent, staging_path
from instagram_ai_agent.content.generators.story_image import _default_stickers
from instagram_ai_agent.content.style import apply_lut_video
from instagram_ai_agent.core.config import NicheConfig
from instagram_ai_agent.core.llm import generate_json
from instagram_ai_agent.core.logging_setup import get_logger

log = get_logger(__name__)

STORY_W, STORY_H = 1080, 1920


POLLINATIONS_IMAGE = (
    "https://image.pollinations.ai/prompt/{prompt}"
    "?width={w}&height={h}&model={model}&nologo=true&enhance=true&seed={seed}"
)


async def _script(cfg: NicheConfig, trend_context: str, *, mode: str) -> dict:
    if mode == "ai":
        visual_hint = "A vivid photorealistic VISUAL prompt for an image model."
    else:
        visual_hint = "A 2–4 keyword QUERY for a stock-footage search (portrait orientation)."

    system = (
        f"You write a 10-second Instagram STORY script for a page about {cfg.niche}.\n"
        f"Audience: {cfg.target_audience}.\n"
        f"Voice: {cfg.voice.persona}. Tone: {', '.join(cfg.voice.tone)}.\n"
        f"Never: {', '.join(cfg.voice.forbidden) or 'n/a'}.\n"
        "Rules:\n"
        f"- One scene, one LINE of voiceover (≤18 words).\n"
        f"- {visual_hint}\n"
        "- No on-screen text; captions are auto-burned."
    )
    prompt = (
        f"Trend/context:\n{trend_context or '(niche value moment)'}\n\n"
        "Return JSON: {\"visual\": str, \"line\": str}"
    )
    data = await generate_json("caption", prompt, system=system, max_tokens=400)
    visual = str(data.get("visual") or "").strip()
    line = str(data.get("line") or "").strip()
    if not (visual and line):
        raise ValueError("Story video script missing visual or line")
    return {"visual": visual, "line": line}


# ───────── Source helpers ─────────
async def _gen_ai_image(visual: str, dest: Path) -> Path:
    model = os.environ.get("POLLINATIONS_IMAGE_MODEL", "flux")
    url = POLLINATIONS_IMAGE.format(
        prompt=quote_plus(visual),
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
    if dest.stat().st_size < 10_000:
        raise RuntimeError("Pollinations returned too-small image")
    return dest


def _ken_burns_vertical(image: Path, dur: float, out: Path) -> Path:
    fps = 30
    frames = max(30, int(dur * fps))
    z = "min(zoom+0.0012,1.22)"
    x = "iw/2-(iw/zoom/2)"
    y = "ih/2-(ih/zoom/2)"
    filters = (
        f"scale=-1:{STORY_H * 2},"
        f"zoompan=z='{z}':x='{x}':y='{y}':d={frames}:s={STORY_W}x{STORY_H}:fps={fps},"
        f"crop={STORY_W}:{STORY_H},format=yuv420p"
    )
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-loop", "1",
            "-t", f"{dur:.3f}",
            "-i", str(image),
            "-vf", filters,
            "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
            str(out),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    return out


def _normalize_story_clip(src: Path, dur: float, out: Path) -> Path:
    """Trim to `dur`, fit to 1080x1920, strip audio."""
    filters = (
        f"scale={STORY_W}:{STORY_H}:force_original_aspect_ratio=increase,"
        f"crop={STORY_W}:{STORY_H},fps=30,format=yuv420p"
    )
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-t", f"{dur:.3f}",
            "-i", str(src),
            "-vf", filters,
            "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
            str(out),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    return out


# ───────── Entry point ─────────
async def generate(
    cfg: NicheConfig,
    trend_context: str = "",
    *,
    mode: str | None = None,  # "ai" | "stock"; None = auto
) -> GeneratedContent:
    # Prefer stock when keys are set (higher success rate, lower load on Pollinations)
    if mode is None:
        mode = "stock" if (os.environ.get("PEXELS_API_KEY") or os.environ.get("PIXABAY_API_KEY")) else "ai"

    script = await _script(cfg, trend_context, mode=mode)
    visual, line = script["visual"], script["line"]

    with tempfile.TemporaryDirectory(prefix="story_") as _wd:
        work = Path(_wd)

        # 1. Voice
        voice = os.environ.get("TTS_VOICE", "en-US-GuyNeural")
        vo_path = work / "voice.mp3"
        await reel_stock.tts_to_file(line, vo_path, voice=voice)
        dur = max(7.0, min(14.0, reel_stock.probe_duration(vo_path) + 0.5))

        # 2. Source video
        base_clip = work / "base.mp4"
        if mode == "stock":
            stock_src = await reel_stock._fetch_stock_clip(visual, work / "stock.mp4")
            if stock_src is None:
                log.warning("Stock fetch failed for story — falling back to AI image")
                mode = "ai"
        if mode == "ai" or not base_clip.exists():
            img = await _gen_ai_image(visual, work / "img.jpg")
            _ken_burns_vertical(img, dur, base_clip)
        else:
            _normalize_story_clip(stock_src, dur, base_clip)  # type: ignore[arg-type]

        # 3. Captions (kinetic ASS for stories by default)
        cap_path, force_style = reel_stock.render_captions(
            vo_path, work, cfg,
            is_story=True, video_w=STORY_W, video_h=STORY_H,
        )

        # 4. Optional music bed on story videos (lighter volume — stories are 10s)
        from instagram_ai_agent.plugins import audio_mix, music as music_plugin
        final_audio = vo_path
        if cfg.music.enabled:
            try:
                bed = await music_plugin.find_music(cfg)
                if bed is not None:
                    mixed = work / "mixed.m4a"
                    audio_mix.mix_vo_and_music(
                        vo_path, bed.path, mixed,
                        duration_s=dur, music_cfg=cfg.music,
                    )
                    final_audio = mixed
            except Exception as e:
                log.warning("story music bed failed: %s", e)

        # 5. Mux + caption burn
        out = staging_path("story_video", ".mp4")
        reel_stock.mux_audio_captions(
            base_clip, final_audio, cap_path, out, subtitle_style=force_style,
        )

    styled = apply_lut_video(out, cfg)
    # Hook overlay on story videos too — same top-third fade as reels.
    from instagram_ai_agent.plugins import video_overlay
    hook_text = line  # single-scene stories → line IS the hook
    with_hook = video_overlay.add_hook_overlay(styled, hook_text, cfg)
    final = reel_stock.overlay_watermark_video(with_hook, cfg) if cfg.aesthetic.watermark else with_hook

    return GeneratedContent(
        format="story_video",
        media_paths=[str(final)],
        visible_text=line,
        caption_context=f"Story video. VO: '{line}'. Caption hooks swipers.",
        generator="story_video",
        meta={
            "script": script, "mode": mode,
            "stickers": _default_stickers(cfg),
            "hook_text": hook_text,
        },
    )


