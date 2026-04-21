"""HF-Spaces AI video reel generator.

Unlike `reel_ai` (Pollinations-image + Ken Burns pan/zoom — cheap fake
motion), this format produces TRUE text-to-video via HuggingFace Spaces
running open-weight models (LTX-2, Wan2.2, SVD). Zero API keys required;
authenticated HF token gets ZeroGPU priority if set.

Pipeline:
  1. LLM writes a one-hook reel script (single scene, 5-8s).
  2. hf_video.generate_t2v() calls the rotating Spaces pool until one
     returns a video.
  3. edge-tts synthesises voiceover over the video length.
  4. Whisper word-timed captions burned in via libass.
  5. Music bed + LUT + watermark — same stack as reel_ai.

Opt-in: register as a format in niche.yaml's `formats` weights to enable.
Degrades gracefully via reel_ai fallback when the HF pool is cold.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

from instagram_ai_agent.content.generators import reel_ai, reel_stock
from instagram_ai_agent.content.generators.base import GeneratedContent, staging_path
from instagram_ai_agent.content.style import apply_lut_video
from instagram_ai_agent.core.config import NicheConfig
from instagram_ai_agent.core.llm import generate_json_model
from instagram_ai_agent.core.logging_setup import get_logger
from instagram_ai_agent.plugins import hf_video

log = get_logger(__name__)

TARGET_W, TARGET_H = 1080, 1920


class _HFReelScript(BaseModel):
    visual_prompt: str = Field(..., description="One-sentence vivid visual prompt for text-to-video. Concrete subject + action + setting + style cues. Think: 'close-up of a man finishing a one-arm pullup at golden hour in a home gym, cinematic, 9:16 vertical'.")
    voiceover: str = Field(..., description="2-3 sentence voiceover script matching the visual. Direct voice. ≤50 words total.")
    hook: str = Field(..., description="3-8 word on-screen hook overlay text.")
    cta: str = Field("save for later", description="Short end-of-caption CTA.")


async def _script(cfg: NicheConfig, trend_context: str) -> _HFReelScript:
    system = (
        f"You write single-scene Instagram reel scripts for a page about {cfg.niche}.\n"
        f"Audience: {cfg.target_audience}. Voice: {cfg.voice.persona}. "
        f"Tone: {', '.join(cfg.voice.tone)}.\n"
        f"Never: {', '.join(cfg.voice.forbidden) or 'n/a'}.\n"
        "Output will feed a text-to-video model (LTX-2 / Wan2.2), so the "
        "visual_prompt must be ONE photorealistic scene — clear subject, "
        "concrete action, lighting/setting. Vertical 9:16 framing.\n"
        "Voiceover is 2-3 direct sentences, max 50 words total. The hook "
        "overlay is 3-8 words that burn on-screen in slide-1-style typography."
    )
    prompt = (
        f"Niche trend/context:\n{trend_context or '(general niche wisdom)'}\n\n"
        "Write the single-scene reel spec."
    )
    return await generate_json_model(
        "script", prompt, _HFReelScript, system=system, max_tokens=500,
    )


async def generate(cfg: NicheConfig, trend_context: str = "") -> GeneratedContent:
    """Single-scene HF-Spaces T2V reel. On any HF pool failure, raises so
    the pipeline falls through to another format."""
    script = await _script(cfg, trend_context)

    # 1. Generate the video via the HF pool
    video_raw = await hf_video.generate_t2v(
        script.visual_prompt, duration_s=6.0,
    )
    if video_raw is None:
        raise RuntimeError("hf_video pool returned nothing — all Spaces cooling / failed")

    with tempfile.TemporaryDirectory(prefix="reelhf_") as _wd:
        work = Path(_wd)

        # Normalise the raw video to 9:16 1080×1920 (models emit various aspect
        # ratios — LTX emits 9:16 natively; Wan variants sometimes emit 16:9
        # or square). Scale-pad-crop keeps the subject centred.
        normalised = work / "norm.mp4"
        vf = (
            f"scale={TARGET_W}:-2:force_original_aspect_ratio=decrease,"
            f"pad={TARGET_W}:{TARGET_H}:(ow-iw)/2:(oh-ih)/2:black,"
            f"scale={TARGET_W}:{TARGET_H}"
        )
        subprocess.run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(video_raw),
            "-vf", vf,
            "-c:v", "libx264", "-crf", "22", "-preset", "medium",
            "-pix_fmt", "yuv420p",
            "-an",  # drop source audio — we replace with our own voiceover
            str(normalised),
        ], check=True)

        video_dur = reel_stock.probe_duration(normalised)
        if video_dur <= 0:
            raise RuntimeError("normalised HF video has zero duration")

        # 2. Voiceover — synthesise to video length
        vo_path = work / "voice.mp3"
        await reel_stock.tts_to_file(script.voiceover, vo_path)
        vo_dur = reel_stock.probe_duration(vo_path)

        # If VO is longer than video, loop the video to cover VO. If shorter,
        # trim to VO (users watch for the spoken content, not silent tail).
        target_dur = max(vo_dur, 5.0)
        looped = work / "looped.mp4"
        if video_dur < target_dur:
            # Loop the video to reach target_dur
            subprocess.run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-stream_loop", "-1",
                "-i", str(normalised),
                "-t", f"{target_dur:.2f}",
                "-c", "copy",
                str(looped),
            ], check=True)
        else:
            subprocess.run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(normalised),
                "-t", f"{target_dur:.2f}",
                "-c", "copy",
                str(looped),
            ], check=True)

        # 3. Captions from voiceover
        try:
            cap_path, force_style = reel_stock.render_captions(
                vo_path, work, cfg, is_story=False,
                video_w=TARGET_W, video_h=TARGET_H,
            )
        except Exception as e:
            log.warning("reel_hf: caption render failed (non-fatal) — %s", e)
            cap_path, force_style = None, None

        # 4. Mux video + VO + captions
        out = staging_path("reelhf", ".mp4")
        reel_stock.mux_audio_captions(
            looped, vo_path, cap_path, out, subtitle_style=force_style,
        )

    # 5. Style + hook overlay + watermark
    styled = apply_lut_video(out, cfg)
    from instagram_ai_agent.plugins import video_overlay
    with_hook = video_overlay.add_hook_overlay(styled, script.hook, cfg)
    final = reel_stock.overlay_watermark_video(with_hook, cfg) if cfg.aesthetic.watermark else with_hook

    return GeneratedContent(
        format="reel_ai",   # reuse reel_ai format slot for poster compatibility
        media_paths=[str(final)],
        visible_text=f"{script.hook}. {script.voiceover}",
        caption_context=(
            f"AI-generated reel (HF Spaces T2V). Visual: {script.visual_prompt}. "
            f"Voice: '{script.voiceover}'. Caption should lead with the hook "
            f"and end with '{script.cta}'."
        ),
        generator="reel_hf",
        meta={
            "visual_prompt": script.visual_prompt,
            "hook_text": script.hook,
            "backend": "hf_spaces",
        },
    )
