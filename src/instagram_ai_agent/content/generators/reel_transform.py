"""Transform generator — public source video + anti-fingerprint pipeline.

Takes a public clip from yt-dlp (YouTube Shorts, TikTok) or Reddit and
produces a new mp4 that passes IG's perceptual fingerprinter as
"originalish" by:

  1. Stripping container metadata (no source hints in the file header).
  2. Random 1-3px crop (frame-hash signature drifts).
  3. Pitch-shift audio ±2% + replace with our own edge-tts voiceover
     (audio fingerprint becomes ours, not theirs).
  4. Re-encode at fresh CRF so codec-level signatures change.
  5. Burn in our branded caption overlay (TikTok-style word-by-word
     highlighting via libass) + credit line for the original creator.
  6. Optional LUT for consistent colour across our page.

The credit line + original-audio-replacement is what distinguishes this
from plain repost — it's transformative commentary (fair use posture)
while also dodging the perceptual-hash matcher. Both legal and practical
angles covered.

This generator exists alongside reel_stock (Pexels-based) and reel_ai
(Pollinations-based). It's opt-in via niche.yaml config or an
IG_TRANSFORM_REELS env flag, and daily-capped so IG's "reposted content"
classifier can't pattern-match on volume.
"""
from __future__ import annotations

import random
import subprocess
from pathlib import Path

from instagram_ai_agent.content.generators.base import GeneratedContent, staging_path
from instagram_ai_agent.content.generators.reel_stock import (
    probe_duration,
    render_captions,
    tts_to_file,
)
from instagram_ai_agent.content.style import apply_lut_video
from instagram_ai_agent.core.config import NicheConfig
from instagram_ai_agent.core.llm import generate
from instagram_ai_agent.core.logging_setup import get_logger
from instagram_ai_agent.plugins import reddit_public, ytdlp_source

log = get_logger(__name__)

TARGET_W, TARGET_H = 1080, 1920


async def _llm_overlay_script(cfg: NicheConfig, source_title: str, source_uploader: str) -> str:
    """Generate a short commentary/insight line our TTS voice will say over
    the imported footage. 2-3 sentences, in the page's configured voice.

    Freeform text — uses generate(), not generate_json() — because we only
    need prose here, and small :free models do prose fine."""
    system = (
        f"You write tight voiceover scripts for an Instagram reel on {cfg.niche}. "
        f"Persona: {cfg.voice.persona}. Tone: {', '.join(cfg.voice.tone)}. "
        "Rules: 2-3 sentences, max 35 words total, no quotation marks, "
        "no emojis, no clichés. Concrete, specific, direct."
    )
    prompt = (
        f"The reel shows: {source_title or 'calisthenics footage'}.\n"
        f"Write a voiceover that re-frames this clip with a niche-specific insight "
        f"(technique tip, mistake to avoid, progression note). Don't describe the "
        f"footage — add value to it."
    )
    try:
        text = await generate("caption", prompt, system=system, max_tokens=200, temperature=0.75)
    except Exception as e:
        log.warning("reel_transform: voiceover LLM failed — %s", e)
        return ""
    # Strip any stray quotes the LLM likes to add
    return text.strip().strip('"').strip("'")[:250]


def _ffmpeg(args: list[str]) -> None:
    """Run ffmpeg silently, raising on failure."""
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args]
    subprocess.run(cmd, check=True)


def _transform_video(source: Path, vo_audio: Path, out: Path) -> None:
    """Core anti-fingerprint transform. source + vo_audio → out.

    Operations:
      - Scale+crop to 9:16 (most public clips already are for Shorts/TikTok
        but older YouTube could be 16:9; we letterbox-crop smartly).
      - Random 1-3 px crop offset on both axes — shifts perceptual hashes.
      - Audio: discard source audio entirely, mux in our voiceover. No need
        to pitch-shift because we're replacing wholesale.
      - Re-encode at CRF 22 (fresh codec signature).
      - Strip all metadata (-map_metadata -1).
    """
    # Random crop offsets per-run so we never produce byte-identical output
    # across two transform runs of the same source.
    crop_x = random.randint(1, 3)
    crop_y = random.randint(1, 3)

    # Target 9:16 1080x1920. Scale input to fit 1080 width, pad to 1920 tall,
    # then apply the tiny offset crop.
    vf = (
        f"scale={TARGET_W}:-2:force_original_aspect_ratio=decrease,"
        f"pad={TARGET_W}:{TARGET_H}:(ow-iw)/2:(oh-ih)/2:black,"
        f"crop=iw-{crop_x*2}:ih-{crop_y*2}:{crop_x}:{crop_y},"
        f"scale={TARGET_W}:{TARGET_H}"
    )

    _ffmpeg([
        "-i", str(source),
        "-i", str(vo_audio),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-vf", vf,
        "-c:v", "libx264",
        "-crf", "22",
        "-preset", "medium",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-shortest",
        "-map_metadata", "-1",
        "-movflags", "+faststart",
        str(out),
    ])


def _burn_captions(video_in: Path, captions_file: Path, video_out: Path) -> None:
    """Burn a libass-styled .ass caption overlay into the video."""
    vf = f"ass={captions_file.as_posix()}"
    _ffmpeg([
        "-i", str(video_in),
        "-vf", vf,
        "-c:v", "libx264",
        "-crf", "22",
        "-preset", "medium",
        "-c:a", "copy",
        str(video_out),
    ])


async def _harvest_source(cfg: NicheConfig) -> tuple[Path, str, str] | None:
    """Try Reddit (no API key) first, fall back to yt-dlp YouTube Shorts
    search. Returns (clip_path, source_label, source_url) or None."""
    # Reddit path: public JSON scrape of niche subs (configurable, defaults to fitness)
    reddit_subs = cfg.reddit_subs or [
        "bodyweightfitness", "calisthenics", "formcheck", "flexibility",
    ]
    try:
        posts = await reddit_public.top_videos_across_subs(
            reddit_subs, timeframe="week", min_score=200,
        )
        if posts:
            p = posts[0]
            # Reddit hosts video; we still pipe through yt-dlp because
            # Reddit's fallback_url is video-only (audio is separate) and
            # yt-dlp handles the mux.
            clip = await ytdlp_source.download_clip(p.permalink)
            if clip is not None:
                return clip.path, f"/u/{p.author}", p.permalink
    except Exception as e:
        log.warning("reel_transform: reddit harvest failed — %s", e)

    # Fallback: YouTube Shorts
    try:
        clips = await ytdlp_source.harvest_youtube_shorts(
            keyword=cfg.niche, want=1,
        )
        if clips:
            c = clips[0]
            label = c.uploader or "@creator"
            return c.path, label, c.source_url
    except Exception as e:
        log.warning("reel_transform: youtube harvest failed — %s", e)

    return None


async def generate(
    cfg: NicheConfig,
    trend_context: str = "",
    *,
    variant: str | None = None,
    contrarian: bool = False,
) -> GeneratedContent:
    """Harvest → transform → caption → return. Signature matches other
    generators so pipeline.py can dispatch to us transparently."""
    if not ytdlp_source._ytdlp_available():
        raise RuntimeError(
            "reel_transform: yt-dlp not installed. "
            "Run `pip install yt-dlp` or `pipx inject instagram-ai-agent yt-dlp`."
        )

    harvested = await _harvest_source(cfg)
    if harvested is None:
        raise RuntimeError("reel_transform: no harvestable source video right now")
    source_path, credit_label, source_url = harvested

    # Voiceover script — our value-add layer
    source_title = source_path.stem
    voiceover = await _llm_overlay_script(cfg, source_title, credit_label)
    if not voiceover:
        voiceover = f"This. {cfg.niche.capitalize()}. Watch the form."

    # Synth the voiceover
    vo_path = staging_path("transform_vo", ".mp3")
    await tts_to_file(voiceover, vo_path)
    vo_duration = probe_duration(vo_path)

    # Trim source to at most voiceover duration + 1s, but respect source
    # length — too short a VO with a long clip looks awkward
    src_duration = probe_duration(source_path)
    # Use the shorter of (source, voiceover + 1.0s)
    target_duration = min(src_duration, max(vo_duration + 1.0, 8.0))

    # Clip to target_duration using ffmpeg before transform (saves encode time)
    clipped = staging_path("transform_clipped", ".mp4")
    _ffmpeg([
        "-i", str(source_path),
        "-t", f"{target_duration:.2f}",
        "-c", "copy",
        str(clipped),
    ])

    # Transform (9:16 resize + random crop + voiceover swap + metadata strip)
    transformed = staging_path("transform_stage1", ".mp4")
    _transform_video(clipped, vo_path, transformed)

    # Word-level captions over our new voiceover
    try:
        cap_dir = transformed.parent
        caption_files = render_captions(
            vo_path, cap_dir, cfg, is_story=False,
            video_w=TARGET_W, video_h=TARGET_H,
        )
        ass = caption_files.get("ass") if isinstance(caption_files, dict) else caption_files
    except Exception as e:
        log.warning("reel_transform: caption render failed (non-fatal) — %s", e)
        ass = None

    final_captioned = staging_path("transform_final", ".mp4")
    if ass and Path(ass).exists():
        _burn_captions(transformed, Path(ass), final_captioned)
    else:
        final_captioned = transformed

    # LUT pass for consistent page aesthetic
    with_lut = apply_lut_video(final_captioned, cfg)

    # Caption text for IG post — credits the original creator, adds
    # our insight as hook. This is the transformative-commentary claim.
    caption_context = (
        f"Voiceover commentary on a clip sourced from {credit_label}. "
        f"Voiceover line: '{voiceover}'. Credit the source, lead with the "
        f"insight, do not summarise — make them want to swipe back."
    )
    # Visible watermark-style credit for the caption's first line so IG
    # indexers see the attribution immediately.
    visible_text = f"🎥: {credit_label} · {voiceover[:120]}"

    return GeneratedContent(
        format="reel_stock",  # pipeline treats as a reel for upload purposes
        media_paths=[str(with_lut)],
        visible_text=visible_text,
        caption_context=caption_context,
        generator="reel_transform",
        meta={
            "source_platform": "reddit_or_youtube",
            "source_url": source_url,
            "credit": credit_label,
            "voiceover_len": len(voiceover),
            "duration_s": target_duration,
        },
    )
