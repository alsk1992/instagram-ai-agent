"""AI reel generator — Pollinations Flux images + ken-burns motion + TTS + captions.

No GPU required; no paid API. Quality depends on Pollinations' free queue, so
this format should sit at a small weight (5–10%) in the format mix.

Pipeline:
  1. LLM script (scenes w/ visual_prompt + voice line).
  2. Pollinations image per scene.
  3. ffmpeg zoompan → 2–4s vertical clip per scene.
  4. Concat + edge-tts voiceover + whisper SRT + subtitle burn-in.
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import quote_plus

import httpx

from instagram_ai_agent.content.generators import reel_stock
from instagram_ai_agent.content.generators.base import GeneratedContent, staging_path
from instagram_ai_agent.content.style import apply_lut_video
from instagram_ai_agent.core.config import NicheConfig
from instagram_ai_agent.core.llm import generate_json
from instagram_ai_agent.core.logging_setup import get_logger

log = get_logger(__name__)

TARGET_W, TARGET_H = reel_stock.TARGET_W, reel_stock.TARGET_H
# Pollinations image generation is the only permanent-free endpoint they ship
POLLINATIONS_IMAGE = (
    "https://image.pollinations.ai/prompt/{prompt}"
    "?width={w}&height={h}&model={model}&nologo=true&enhance=true&seed={seed}"
)


# ───────── Script ─────────
async def _script(cfg: NicheConfig, trend_context: str) -> list[dict]:
    system = (
        f"You write a 25-second Instagram reel script for a page about {cfg.niche}.\n"
        f"Audience: {cfg.target_audience}.\n"
        f"Voice: {cfg.voice.persona}. Tone: {', '.join(cfg.voice.tone)}.\n"
        f"Never: {', '.join(cfg.voice.forbidden) or 'n/a'}.\n"
        f"Aesthetic palette hint: {', '.join(cfg.aesthetic.palette)}.\n"
        "Rules:\n"
        "- 4–6 scenes. First is the HOOK, last is the CTA.\n"
        "- Each scene: VISUAL (vivid, photorealistic, one sentence prompt for an image model)\n"
        "  + LINE (voiceover, ≤14 words, conversational).\n"
        "- Do not mention on-screen text; images are text-free."
    )
    prompt = (
        f"Trend/context to riff on:\n{trend_context or '(niche value post)'}\n\n"
        "Return JSON: {\"scenes\":[{\"visual\":str, \"line\":str}, ...]}"
    )
    data = await generate_json("script", prompt, system=system, max_tokens=1000)
    raw_scenes = data.get("scenes") or []
    cleaned: list[dict] = []
    for s in raw_scenes[:6]:
        visual = str(s.get("visual") or "").strip()
        line = str(s.get("line") or "").strip()
        if visual and line:
            cleaned.append({"visual": visual, "line": line})
    if len(cleaned) < 3:
        raise ValueError(f"AI reel script too short after cleaning ({len(cleaned)} scenes)")
    return cleaned


# ───────── Image generation ─────────
async def _gen_image(client: httpx.AsyncClient, visual: str, dest: Path, idx: int) -> Path:
    model = os.environ.get("POLLINATIONS_IMAGE_MODEL", "flux")
    url = POLLINATIONS_IMAGE.format(
        prompt=quote_plus(visual),
        w=TARGET_W,
        h=TARGET_H,
        model=model,
        seed=1000 + idx,
    )
    headers = {}
    token = os.environ.get("POLLINATIONS_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    async with client.stream("GET", url, headers=headers) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            async for chunk in r.aiter_bytes():
                f.write(chunk)
    if dest.stat().st_size < 10_000:
        raise RuntimeError(f"Pollinations returned too-small image for scene {idx}")
    return dest


# ───────── Ken-Burns clip ─────────
def _ken_burns(image: Path, dur: float, out: Path, *, direction: str = "in") -> Path:
    """Pan/zoom a still image into a vertical clip via ffmpeg zoompan."""
    fps = 30
    frames = max(30, int(dur * fps))
    # zoompan expects a d-value (frames) and a z-expression
    if direction == "in":
        z = "min(zoom+0.0015,1.25)"
        x = "iw/2-(iw/zoom/2)"
        y = "ih/2-(ih/zoom/2)"
    elif direction == "out":
        z = "if(eq(on,1),1.25,max(zoom-0.0015,1.0))"
        x = "iw/2-(iw/zoom/2)"
        y = "ih/2-(ih/zoom/2)"
    elif direction == "left":
        z = "1.15"
        x = "iw - (iw/zoom) - on*3"
        y = "ih/2-(ih/zoom/2)"
    else:  # "right"
        z = "1.15"
        x = "on*3"
        y = "ih/2-(ih/zoom/2)"

    filters = (
        f"scale=-1:{TARGET_H * 2},"
        f"zoompan=z='{z}':x='{x}':y='{y}':d={frames}:s={TARGET_W}x{TARGET_H}:fps={fps},"
        f"crop={TARGET_W}:{TARGET_H},format=yuv420p"
    )
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1",
        "-t", f"{dur:.3f}",
        "-i", str(image),
        "-vf", filters,
        "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
        str(out),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return out


# ───────── Entry point ─────────
async def generate(cfg: NicheConfig, trend_context: str = "") -> GeneratedContent:
    scenes = await _script(cfg, trend_context)

    with tempfile.TemporaryDirectory(prefix="reelai_") as _wd:
        work = Path(_wd)

        # 1. Voiceover
        voice = os.environ.get("TTS_VOICE", "en-US-GuyNeural")
        full_text = " ".join(s["line"] for s in scenes)
        vo_path = work / "voice.mp3"
        await reel_stock.tts_to_file(full_text, vo_path, voice=voice)
        vo_dur = reel_stock.probe_duration(vo_path)
        if vo_dur <= 0:
            raise RuntimeError("Voiceover duration probe failed")

        # 2. Proportional scene durations
        total_words = sum(len(s["line"].split()) for s in scenes) or 1
        scene_durs = [
            max(1.8, vo_dur * (len(s["line"].split()) / total_words))
            for s in scenes
        ]
        total = sum(scene_durs)
        scene_durs = [d * (vo_dur / total) for d in scene_durs]

        # 2b. Fetch music + beat-sync scene boundaries before rendering clips.
        from instagram_ai_agent.plugins import audio_mix, beat_sync, music as music_plugin

        bed = None
        if cfg.music.enabled:
            try:
                bed = await music_plugin.find_music(cfg)
            except Exception as e:
                log.warning("reel_ai music lookup failed: %s", e)

        beat_synced = False
        if bed is not None and cfg.music.beat_window_s > 0:
            try:
                # reel_ai uses slower pacing than reel_stock — take the max
                # of the niche-configured floor and this generator's 1.8s
                # aspiration so beat snap can never undercut either.
                min_scene_s = max(cfg.music.beat_min_scene_s, 1.8)
                scene_durs, beat_synced = beat_sync.snap_scene_durs(
                    scene_durs, bed.path,
                    vo_duration_s=vo_dur,
                    window_s=cfg.music.beat_window_s,
                    min_scene_s=min_scene_s,
                )
            except Exception as e:
                log.warning("reel_ai beat sync failed: %s", e)

        # 3. Image generation (parallel, bounded concurrency)
        sem = asyncio.Semaphore(3)
        async with httpx.AsyncClient(timeout=httpx.Timeout(180.0)) as client:
            async def _gen(idx: int, visual: str) -> Path:
                async with sem:
                    return await _gen_image(client, visual, work / f"img_{idx}.jpg", idx)

            images = await asyncio.gather(
                *[_gen(i, s["visual"]) for i, s in enumerate(scenes)],
                return_exceptions=True,
            )

        # Handle partial failures by regenerating with a simpler prompt
        async with httpx.AsyncClient(timeout=httpx.Timeout(180.0)) as client:
            for i, res in enumerate(images):
                if isinstance(res, Exception) or res is None or not Path(res).exists():
                    log.warning("Retrying image %d with simpler prompt", i)
                    simpler = " ".join(scenes[i]["visual"].split()[:6])
                    images[i] = await _gen_image(client, simpler, work / f"img_retry_{i}.jpg", i + 500)

        # 4. Ken-burns per scene
        directions = ["in", "left", "out", "right", "in", "out"]
        clips: list[Path] = []
        for i, (img, dur) in enumerate(zip(images, scene_durs, strict=False)):
            direction = directions[i % len(directions)]
            clip_out = work / f"kb_{i}.mp4"
            _ken_burns(Path(img), dur, clip_out, direction=direction)
            clips.append(clip_out)

        # 5. Concat + captions + mux (+ optional music bed)
        concat_path = work / "concat.mp4"
        reel_stock.concat_clips(clips, concat_path, work)
        cap_path, force_style = reel_stock.render_captions(
            vo_path, work, cfg,
            is_story=False, video_w=TARGET_W, video_h=TARGET_H,
        )

        final_audio = vo_path
        music_meta: dict = {}
        if bed is not None:
            try:
                mixed = work / "mixed.m4a"
                audio_mix.mix_vo_and_music(
                    vo_path, bed.path, mixed,
                    duration_s=vo_dur, music_cfg=cfg.music,
                )
                final_audio = mixed
                music_meta = {
                    "music_source": bed.source,
                    "music_title": bed.title,
                    "music_license": bed.license,
                    "beat_synced": beat_synced,
                }
            except Exception as e:
                log.warning("reel_ai music mix failed: %s", e)

        out = staging_path("reelai", ".mp4")
        reel_stock.mux_audio_captions(
            concat_path, final_audio, cap_path, out, subtitle_style=force_style,
        )

    # 6. Style pass + hook overlay + watermark
    styled = apply_lut_video(out, cfg)
    from instagram_ai_agent.plugins import video_overlay
    hook_text = video_overlay.pick_hook_text(scenes)
    with_hook = video_overlay.add_hook_overlay(styled, hook_text, cfg)
    final = reel_stock.overlay_watermark_video(with_hook, cfg) if cfg.aesthetic.watermark else with_hook

    visible = " / ".join(s["line"] for s in scenes)
    return GeneratedContent(
        format="reel_ai",
        media_paths=[str(final)],
        visible_text=visible,
        caption_context=f"AI-visual reel with voiceover: '{visible}'. Hook-focused caption.",
        generator="reel_ai",
        meta={"scenes": scenes, "hook_text": hook_text, "beat_synced": beat_synced, **music_meta},
    )
