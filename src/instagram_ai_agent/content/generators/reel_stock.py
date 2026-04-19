"""Reel generator: stock footage + edge-tts voiceover + whisper word-level captions.

Pipeline:
  1. LLM produces a script: { scenes:[{query, line}], hook, cta }.
  2. For each scene, fetch a matching vertical clip from Pexels (fallback Pixabay).
  3. edge-tts synthesises the full voiceover.
  4. faster-whisper transcribes → word timestamps → SRT chunks.
  5. ffmpeg composes: stack clips → fit 9:16 → overlay captions → mix audio.
  6. Optional LUT and watermark applied by style layer afterwards.

Everything is commercial-safe on Pexels/Pixabay + edge-tts (Microsoft public
endpoint) + faster-whisper (MIT).
"""
from __future__ import annotations

import asyncio
import os
import random
import re
import subprocess
import tempfile
from pathlib import Path

import edge_tts
import httpx

from instagram_ai_agent.content.generators.base import GeneratedContent, staging_path
from instagram_ai_agent.content.style import apply_lut_video
from instagram_ai_agent.core.config import NicheConfig
from instagram_ai_agent.core.llm import generate_json
from instagram_ai_agent.core.logging_setup import get_logger

log = get_logger(__name__)

TARGET_W, TARGET_H = 1080, 1920  # 9:16

PEXELS_SEARCH = "https://api.pexels.com/videos/search"
PIXABAY_SEARCH = "https://pixabay.com/api/videos/"


# ───────── Script ─────────
REEL_VARIANTS = {
    "tips": (
        "- 4–6 scenes of 1-idea-each niche tips.\n"
        "- Scene 1 = HOOK (promise a payoff); last = CTA.\n"
        "- Each scene: `query` = stock search, `line` ≤ 15 words."
    ),
    "listicle": (
        "- Exactly 5 scenes: an intro hook, then 3 numbered items, then a CTA.\n"
        "- Numbered lines start with the count (e.g. \"1.\", \"2.\", \"3.\").\n"
        "- `query` searches for stock footage that visually supports each item."
    ),
    "talking_head": (
        "- 3–4 scenes, each a direct-address line from the persona to the viewer.\n"
        "- `query` should return portrait shots of PEOPLE in the niche context "
        "(trainers, workers, enthusiasts — matching the niche).\n"
        "- Every line sounds like someone talking AT camera."
    ),
    "myth_bust": (
        "- 4 scenes: \"They say X\" (myth) → \"But actually Y\" (counter) → "
        "the real tip → a CTA.\n"
        "- Tight, contrarian voice. Stock queries should match each beat visually."
    ),
    "before_after": (
        "- 4–5 scenes alternating BEFORE/AFTER beats with a hook and CTA.\n"
        "- Stock queries should reflect the transformation arc of the niche."
    ),
}


async def _script(
    cfg: NicheConfig, trend_context: str, *,
    variant: str | None = None, contrarian: bool = False,
) -> dict:
    if variant is None or variant not in REEL_VARIANTS:
        variant = random.choice(list(REEL_VARIANTS))
    rules = REEL_VARIANTS[variant]
    system = (
        f"You write 30-second Instagram reel scripts for a page about {cfg.niche}.\n"
        f"Audience: {cfg.target_audience}.\n"
        f"Voice: {cfg.voice.persona}. Tone: {', '.join(cfg.voice.tone)}.\n"
        f"Never: {', '.join(cfg.voice.forbidden) or 'n/a'}.\n"
        f"Reel style: {variant}.\n"
        f"Rules for this style:\n{rules}\n"
        "Common rules: total voiceover ≤ 90 words. Queries are 2–4 keywords, portrait orientation."
    )
    if contrarian:
        system += (
            "\n\nCONTRARIAN FRAMING — structure the voiceover as 'they say X, "
            "but actually Y':\n"
            "- Scene 1 names the mainstream belief (hook promises to flip it).\n"
            "- Middle scenes build the counter-case with specifics.\n"
            "- Final scene delivers the re-framed truth.\n"
            f"Avoid takes on: {', '.join(cfg.contrarian.avoid_topics) or 'none'}.\n"
            "Never touch medical, conspiracy, self-harm, or political-group claims."
        )
    prompt = (
        f"Trend/context to riff on:\n{trend_context or '(general niche value post)'}\n\n"
        "Return JSON: {\n"
        "  \"variant\": \"" + variant + "\",\n"
        "  \"scenes\": [ {\"query\":str, \"line\":str}, ... ]\n"
        "}"
    )
    data = await generate_json("script", prompt, system=system, max_tokens=1000)
    data["variant"] = variant
    scenes = data.get("scenes") or []
    if not isinstance(scenes, list) or len(scenes) < 3:
        raise ValueError("Script must have at least 3 scenes")
    cleaned = []
    for s in scenes[:6]:
        q = str(s.get("query") or "").strip()
        line = str(s.get("line") or "").strip()
        if q and line:
            cleaned.append({"query": q, "line": line})
    if len(cleaned) < 3:
        raise ValueError("Fewer than 3 usable scenes after cleaning")
    return {"scenes": cleaned}


# ───────── Stock footage ─────────
async def _fetch_stock_clip(query: str, dest: Path) -> Path | None:
    """Try Pexels then Pixabay. Returns path of downloaded mp4 or None."""
    pexels_key = os.environ.get("PEXELS_API_KEY")
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        if pexels_key:
            try:
                r = await client.get(
                    PEXELS_SEARCH,
                    params={
                        "query": query,
                        "per_page": 10,
                        "orientation": "portrait",
                        "size": "medium",
                    },
                    headers={"Authorization": pexels_key},
                )
                r.raise_for_status()
                videos = r.json().get("videos") or []
                if videos:
                    video = random.choice(videos[: min(5, len(videos))])
                    files = video.get("video_files") or []
                    # Prefer mp4 in HD portrait
                    portrait_files = [
                        f for f in files
                        if f.get("file_type") == "video/mp4"
                        and f.get("height", 0) >= f.get("width", 0)
                    ]
                    pick = (portrait_files or files)[0] if files else None
                    if pick and pick.get("link"):
                        return await _download(client, pick["link"], dest)
            except Exception as e:
                log.warning("Pexels fetch failed for %r: %s", query, e)

        pixabay_key = os.environ.get("PIXABAY_API_KEY")
        if pixabay_key:
            try:
                r = await client.get(
                    PIXABAY_SEARCH,
                    params={
                        "key": pixabay_key,
                        "q": query,
                        "per_page": 10,
                        "video_type": "film",
                    },
                )
                r.raise_for_status()
                hits = r.json().get("hits") or []
                if hits:
                    hit = random.choice(hits[: min(5, len(hits))])
                    videos = hit.get("videos") or {}
                    # Pixabay returns several rendition keys
                    for key in ("medium", "small", "large", "tiny"):
                        v = videos.get(key) or {}
                        if v.get("url"):
                            return await _download(client, v["url"], dest)
            except Exception as e:
                log.warning("Pixabay fetch failed for %r: %s", query, e)

    return None


async def _download(client: httpx.AsyncClient, url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    async with client.stream("GET", url) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            async for chunk in r.aiter_bytes():
                f.write(chunk)
    return dest


# ───────── Voiceover ─────────
async def tts_to_file(text: str, out_path: Path, voice: str = "en-US-GuyNeural") -> Path:
    """Render text via edge-tts to an mp3 file (async)."""
    comm = edge_tts.Communicate(text=text, voice=voice, rate="+6%", volume="+0%")
    await comm.save(str(out_path))
    return out_path


# Backwards-compat alias used within this module
_tts = tts_to_file


def probe_duration(path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, check=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


# Backwards-compat alias retained for in-module callers
_probe_duration = probe_duration


# ───────── Transcription + captions (SRT or kinetic ASS) ─────────
def render_captions(
    audio: Path,
    out_dir: Path,
    cfg: NicheConfig,
    *,
    is_story: bool = False,
    video_w: int = TARGET_W,
    video_h: int = TARGET_H,
) -> tuple[Path, str]:
    """Transcribe and render captions for the configured style.

    Returns ``(caption_file_path, force_style_for_ffmpeg)``. For ASS
    karaoke, ``force_style`` is an empty string because the ASS itself
    carries the styling.
    """
    from instagram_ai_agent.content import captions_render as cr
    from instagram_ai_agent.content.transcribe import transcribe_words

    words = transcribe_words(audio, prefer_whisperx=cfg.captions.prefer_whisperx)
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.captions.style == "karaoke":
        out = out_dir / "captions.ass"
        cr.render_ass_karaoke(
            words, out, cfg=cfg, captions=cfg.captions,
            video_w=video_w, video_h=video_h, is_story=is_story,
        )
        return out, ""  # ASS has baked-in styling

    out = out_dir / "captions.srt"
    cr.render_srt(words, out, chunk_size=cfg.captions.chunk_size)
    return out, subtitle_style(cfg) if not is_story else _story_srt_style(cfg)


# Backwards-compat: existing callers that pass an explicit srt path.
def transcribe_to_srt(audio: Path, srt_out: Path) -> Path:
    from instagram_ai_agent.content.captions_render import render_srt
    from instagram_ai_agent.content.transcribe import transcribe_words

    words = transcribe_words(audio)
    return render_srt(words, srt_out, chunk_size=4)


_transcribe_to_srt = transcribe_to_srt


def _story_srt_style(cfg: NicheConfig) -> str:
    """Story-specific SRT force_style (higher margin, bigger font)."""
    base = subtitle_style(cfg)
    return f"{base},FontSize=20,MarginV={cfg.captions.margin_v_story}"


# ───────── ffmpeg composition ─────────
def normalize_clip(src: Path, dur: float, out: Path) -> Path:
    """Trim to `dur` seconds, scale-pad to 9:16, 30fps, strip audio."""
    filters = (
        f"scale={TARGET_W}:{TARGET_H}:force_original_aspect_ratio=increase,"
        f"crop={TARGET_W}:{TARGET_H},"
        "fps=30,format=yuv420p"
    )
    cmd = [
        "ffmpeg", "-y",
        "-t", f"{dur:.3f}",
        "-i", str(src),
        "-vf", filters,
        "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
        str(out),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return out


def concat_clips(clips: list[Path], out: Path, work: Path) -> Path:
    list_file = work / "concat.txt"
    with open(list_file, "w", encoding="utf-8") as f:
        for c in clips:
            f.write(f"file '{c.as_posix()}'\n")
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(out),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return out


def mux_audio_captions(
    video: Path, audio: Path, srt: Path, out: Path, *, subtitle_style: str
) -> Path:
    # Escape the caption path for ffmpeg's subtitles filter (: and , are special)
    path_esc = str(srt).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    # ASS carries its own styling; only apply force_style for SRT
    if subtitle_style:
        vf = f"subtitles='{path_esc}':force_style='{subtitle_style}'"
    else:
        vf = f"subtitles='{path_esc}'"
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video),
        "-i", str(audio),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        str(out),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return out


def subtitle_style(cfg: NicheConfig) -> str:
    # ASS-style overrides supported by ffmpeg libass. Use palette for accent.
    heading = cfg.aesthetic.heading_font.replace("'", "") or "Arial"
    primary_bgr = _hex_to_bgr(cfg.aesthetic.palette[1] if len(cfg.aesthetic.palette) > 1 else "#FFFFFF")
    outline_bgr = _hex_to_bgr("#000000")
    back_bgr = _hex_to_bgr(cfg.aesthetic.palette[0])
    # Alignment 2 = bottom center. Use bold, large outline + shadow for reel readability.
    return (
        f"FontName={heading},FontSize=16,Bold=1,Outline=3,Shadow=0,"
        f"PrimaryColour=&H00{primary_bgr}&,OutlineColour=&H00{outline_bgr}&,"
        f"BackColour=&H80{back_bgr}&,BorderStyle=1,Alignment=2,MarginV=160"
    )


_HEX_RE = re.compile(r"^#?([0-9a-fA-F]{6})$")


def _hex_to_bgr(hex_color: str) -> str:
    m = _HEX_RE.match(hex_color)
    if not m:
        return "FFFFFF"
    rgb = m.group(1)
    r, g, b = rgb[0:2], rgb[2:4], rgb[4:6]
    return (b + g + r).upper()


# ───────── Entry point ─────────
async def generate(
    cfg: NicheConfig,
    trend_context: str = "",
    *,
    variant: str | None = None,
    contrarian: bool = False,
) -> GeneratedContent:
    script_data = await _script(cfg, trend_context, variant=variant, contrarian=contrarian)
    scenes = script_data["scenes"]
    variant = script_data.get("variant", "tips")

    with tempfile.TemporaryDirectory(prefix="reel_") as _wd:
        work = Path(_wd)

        # 1. Synthesise voiceover first (per scene, for precise scene timing)
        full_text = " ".join(s["line"] for s in scenes)
        voice = os.environ.get("TTS_VOICE", "en-US-GuyNeural")
        vo_path = work / "voice.mp3"
        await _tts(full_text, vo_path, voice=voice)

        # 2. Measure per-scene target durations, proportional to line word count
        vo_dur = probe_duration(vo_path)
        if vo_dur <= 0:
            raise RuntimeError("Voiceover duration probe failed")
        total_words = sum(len(s["line"].split()) for s in scenes) or 1
        scene_durs = [
            max(1.5, vo_dur * (len(s["line"].split()) / total_words))
            for s in scenes
        ]
        # Rescale to match voiceover exactly
        total = sum(scene_durs)
        scene_durs = [d * (vo_dur / total) for d in scene_durs]

        # 3. Music bed picked EARLY so we can beat-sync scene durations
        #    before normalising the stock clips to those durations.
        from instagram_ai_agent.plugins import audio_mix, beat_sync
        from instagram_ai_agent.plugins import music as music_plugin

        bed = None
        if cfg.music.enabled:
            try:
                bed = await music_plugin.find_music(cfg)
            except Exception as e:
                log.warning("music lookup failed (continuing without): %s", e)

        beat_synced = False
        if bed is not None and cfg.music.beat_window_s > 0:
            try:
                scene_durs, beat_synced = beat_sync.snap_scene_durs(
                    scene_durs, bed.path,
                    vo_duration_s=vo_dur,
                    window_s=cfg.music.beat_window_s,
                    min_scene_s=cfg.music.beat_min_scene_s,
                )
                if beat_synced:
                    log.info("reel beat-synced against %s", bed.path.name)
            except Exception as e:
                log.warning("beat sync failed (continuing with original cuts): %s", e)

        # 4. Fetch + normalise clips to (beat-snapped) scene_durs
        stock_clips: list[Path] = []
        fetch_tasks = []
        for i, s in enumerate(scenes):
            dest = work / f"stock_{i}.mp4"
            fetch_tasks.append(_fetch_stock_clip(s["query"], dest))
        fetched = await asyncio.gather(*fetch_tasks, return_exceptions=True)

        for i, result in enumerate(fetched):
            if isinstance(result, Exception) or result is None:
                simpler = " ".join(scenes[i]["query"].split()[:1]) or cfg.niche
                result = await _fetch_stock_clip(simpler, work / f"stock_fb_{i}.mp4")
                if result is None:
                    raise RuntimeError(
                        f"No stock footage for scene {i} ('{scenes[i]['query']}'). "
                        "Set PEXELS_API_KEY or PIXABAY_API_KEY."
                    )
            normalized = normalize_clip(result, scene_durs[i], work / f"norm_{i}.mp4")
            stock_clips.append(normalized)

        # 5. Concatenate
        concat_path = work / "concat.mp4"
        concat_clips(stock_clips, concat_path, work)

        # 6. Transcribe voice → captions (SRT or kinetic ASS by niche.yaml)
        cap_path, force_style = render_captions(
            vo_path, work, cfg,
            is_story=False, video_w=TARGET_W, video_h=TARGET_H,
        )

        # 7. Mix VO with the music bed we already fetched
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
                log.warning("music mix failed (continuing with VO only): %s", e)

        # 8. Mux + captions
        out = staging_path("reel", ".mp4")
        mux_audio_captions(concat_path, final_audio, cap_path, out, subtitle_style=force_style)

    # LUT + hook overlay + watermark
    styled = apply_lut_video(out, cfg)
    # Hook overlay goes BEFORE watermark so the watermark stays on top.
    from instagram_ai_agent.plugins import video_overlay
    hook_text = video_overlay.pick_hook_text(scenes)
    with_hook = video_overlay.add_hook_overlay(styled, hook_text, cfg)
    final = overlay_watermark_video(with_hook, cfg) if cfg.aesthetic.watermark else with_hook

    visible = " / ".join(s["line"] for s in scenes)
    return GeneratedContent(
        format="reel_stock",
        media_paths=[str(final)],
        visible_text=visible,
        caption_context=(
            f"Reel (style={variant}) voiceover: '{visible}'. "
            "Caption hooks viewer to watch the full clip."
        ),
        generator=f"reel_stock:{variant}",
        meta={
            "scenes": scenes,
            "variant": variant,
            "hook_text": hook_text,
            "beat_synced": beat_synced,
            **music_meta,
        },
    )


def overlay_watermark_video(video_path: Path, cfg: NicheConfig) -> Path:
    text = cfg.aesthetic.watermark or ""
    if not text:
        return video_path
    out = video_path.with_name(video_path.stem + "_wm" + video_path.suffix)
    # Escape single quotes + colons for drawtext
    safe = text.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:")
    accent = cfg.aesthetic.palette[2] if len(cfg.aesthetic.palette) > 2 else "#FFFFFF"
    vf = (
        f"drawtext=text='{safe}':"
        "fontsize=36:"
        f"fontcolor={accent}:"
        "x=w-tw-40:"
        "y=h-th-60:"
        "box=1:"
        "boxcolor=black@0.35:"
        "boxborderw=12:"
        "borderw=2:"
        "bordercolor=black@0.5"
    )
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
        "-c:a", "copy",
        str(out),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return out
