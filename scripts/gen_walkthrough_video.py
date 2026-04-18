#!/usr/bin/env python
"""Generate a narrated walkthrough video (README hero / onboarding).

Produces ``docs/media/walkthrough.mp4`` — a ~90-second narrated video
showing: install → init wizard → doctor → first post. Every visual is
a rasterised SVG we already generate + a synthesised AI voice-over.

Stack (all commercial-safe):
  * edge-tts (MIT wrapper on free Azure neural voices) — default
  * Piper (MIT) — optional, fully offline
  * Kokoro (Apache-2.0) — optional, flagship 2025 quality
  * ffmpeg — compositing

Run:
    python scripts/gen_walkthrough_video.py

Voice override:
    python scripts/gen_walkthrough_video.py --voice en-US-GuyNeural
    python scripts/gen_walkthrough_video.py --backend piper
    python scripts/gen_walkthrough_video.py --backend kokoro --voice af_bella

Output:
  docs/media/walkthrough.mp4  (~5MB, 1920x1080, H.264)
"""
from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _ffmpeg_binary() -> str:
    """Prefer system ffmpeg; fall back to imageio-ffmpeg's bundled static
    binary (pip install imageio-ffmpeg) so users without a system install
    still get a working pipeline."""
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return ""


def _ffprobe_binary() -> str:
    if shutil.which("ffprobe"):
        return "ffprobe"
    # imageio-ffmpeg doesn't ship ffprobe — fall back to parsing via
    # ffmpeg's stderr, but for simplicity just return "ffprobe" and
    # let the caller handle absence with a shorter fallback.
    return "ffprobe"


FFMPEG = ""   # resolved in main()

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "docs" / "media"
OUT = MEDIA / "walkthrough.mp4"

# ─── Script ────────────────────────────────────────────────────
# Each panel is (SVG-or-PNG path relative to docs/media, narration text,
# seconds to hold on screen beyond the narration itself).
PANELS: list[tuple[str, str, float]] = [
    (
        "title",
        "Meet instagram-ai-agent. An autonomous AI agent that runs your "
        "Instagram. Describe your niche once, give it an account, and it runs.",
        1.0,
    ),
    (
        "presets.svg",
        "Step one: pick a starter preset. Fitness, food, travel, finance — "
        "eight curated niches come pre-tuned so you don't invent defaults.",
        0.5,
    ),
    (
        "wizard-preview.svg",
        "The init wizard fills every field from the preset. You just edit "
        "whatever's specific to your page. Takes about sixty seconds.",
        0.5,
    ),
    (
        "doctor.svg",
        "Run doctor any time to self-check every dependency, API key, and "
        "configuration step. Eighty percent of issues solve themselves here.",
        0.5,
    ),
    (
        "warmup-status.svg",
        "Fresh accounts get a fourteen-day warmup ramp. Day one through seven, "
        "no posts — just lurking. This is the single biggest ban-avoidance lever.",
        0.5,
    ),
    (
        "review.svg",
        "Every generated post lands in a review queue. Approve, reject, or "
        "tweak before it goes live. Or flip auto-approve once you trust it.",
        0.5,
    ),
    (
        "status.svg",
        "Queue depth, engagement stats, shadowban probes — all visible in status. "
        "Or open the local dashboard for the same view in a browser.",
        0.5,
    ),
    (
        "outro",
        "That's it. Pipx install, configure once, walk away. "
        "Links to the repo in the description.",
        1.5,
    ),
]


# ─── Main orchestration ────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=["edge-tts", "piper", "kokoro"],
        default="edge-tts",
        help="TTS backend (default: edge-tts; already a dep)",
    )
    parser.add_argument(
        "--voice",
        default="en-US-GuyNeural",
        help="Voice ID — backend-specific. edge-tts: en-US-GuyNeural | "
        "en-US-AriaNeural | en-GB-RyanNeural. "
        "piper: en_US-ryan-medium | en_GB-alan-medium. "
        "kokoro: af_bella | am_michael.",
    )
    parser.add_argument("--keep-work", action="store_true",
                        help="Don't delete the intermediate workdir.")
    args = parser.parse_args()

    global FFMPEG
    FFMPEG = _ffmpeg_binary()
    if not FFMPEG:
        print(
            "✗ ffmpeg required. Options:\n"
            "    pip install imageio-ffmpeg   (bundled static binary, zero system deps)\n"
            "    macOS: brew install ffmpeg\n"
            "    Ubuntu: sudo apt install ffmpeg"
        )
        return 1

    work = Path(tempfile.mkdtemp(prefix="ig_walkthrough_"))
    print(f"▶ Working dir: {work}")
    try:
        print("▶ Rendering visual panels")
        panel_pngs = _render_panels(work)

        print(f"▶ Synthesising narration via {args.backend}")
        narrations = asyncio.run(
            _synth_narration(work, args.backend, args.voice)
        )

        print("▶ Composing video")
        _compose_video(work, panel_pngs, narrations)

        MEDIA.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(work / "walkthrough.mp4", OUT)
        print(f"✓ Written: {OUT.relative_to(ROOT)}  ({OUT.stat().st_size / 1024 / 1024:.1f} MB)")

        # Additional artefacts for the README — GIF plays inline on GitHub
        # where <video> doesn't, poster is the fallback thumbnail.
        gif_path = MEDIA / "walkthrough.gif"
        poster_path = MEDIA / "walkthrough-poster.jpg"
        print("▶ Generating GIF preview + poster thumbnail")
        _derive_gif_and_poster(OUT, gif_path, poster_path, work)
        print(f"  → {gif_path.relative_to(ROOT)}  ({gif_path.stat().st_size / 1024 / 1024:.1f} MB)")
        print(f"  → {poster_path.relative_to(ROOT)}  ({poster_path.stat().st_size / 1024:.0f} KB)")
        return 0
    finally:
        if not args.keep_work:
            shutil.rmtree(work, ignore_errors=True)


# ─── Visual panels ─────────────────────────────────────────────
def _render_panels(work: Path) -> list[Path]:
    """Rasterise every SVG panel to a 1920x1080 PNG with centred content
    on a dark backdrop. Synthesises the title + outro cards as PIL images."""
    from PIL import Image, ImageDraw, ImageFont

    W, H = 1920, 1080
    BG = (14, 14, 18)          # deep slate
    FG = (232, 230, 220)
    ACCENT = (201, 169, 97)

    # Font resolution — reuse what the installer shipped
    fonts_dir = ROOT / "data" / "fonts"
    def _font(size: int) -> ImageFont.FreeTypeFont:
        for candidate in (fonts_dir / "Archivo Black.ttf",
                          fonts_dir / "Inter.ttf",
                          Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")):
            if candidate.exists():
                return ImageFont.truetype(str(candidate), size)
        return ImageFont.load_default()

    out_paths: list[Path] = []
    for idx, (name, narration, _hold) in enumerate(PANELS):
        out = work / f"panel_{idx:02d}.png"
        if name in ("title", "outro"):
            img = Image.new("RGB", (W, H), BG)
            draw = ImageDraw.Draw(img)
            if name == "title":
                draw.text((W // 2, H // 2 - 80), "instagram-ai-agent",
                          font=_font(120), fill=FG, anchor="mm")
                draw.text((W // 2, H // 2 + 40),
                          "autonomous AI content agent for Instagram",
                          font=_font(44), fill=ACCENT, anchor="mm")
                draw.text((W // 2, H - 120),
                          "github.com/alsk1992/instagram-ai-agent",
                          font=_font(28), fill=(150, 150, 160), anchor="mm")
            else:
                draw.text((W // 2, H // 2 - 40),
                          "pipx install git+...",
                          font=_font(72), fill=ACCENT, anchor="mm")
                draw.text((W // 2, H // 2 + 40),
                          "ig-agent init && ig-agent run",
                          font=_font(56), fill=FG, anchor="mm")
            img.save(out, "PNG", quality=95)
        else:
            # Rasterise SVG → PNG, then composite centred on the dark bg
            svg = MEDIA / name
            tmp_png = work / f"raw_{idx:02d}.png"
            _svg_to_png(svg, tmp_png, width=1600)
            raw = Image.open(tmp_png).convert("RGBA")
            canvas = Image.new("RGBA", (W, H), BG + (255,))
            # Scale to fit height with 100px margin
            max_h = H - 200
            if raw.height > max_h:
                ratio = max_h / raw.height
                raw = raw.resize(
                    (int(raw.width * ratio), max_h),
                    Image.LANCZOS,
                )
            x = (W - raw.width) // 2
            y = (H - raw.height) // 2
            canvas.paste(raw, (x, y), raw if raw.mode == "RGBA" else None)
            canvas.convert("RGB").save(out, "PNG", quality=95)
        out_paths.append(out)
        print(f"  → panel_{idx:02d}.png ({name})")
    return out_paths


def _svg_to_png(svg: Path, png: Path, *, width: int = 1600) -> None:
    """Best-effort SVG → PNG. Tries resvg → cairosvg → rsvg-convert."""
    if shutil.which("resvg"):
        subprocess.run(
            ["resvg", "--width", str(width), str(svg), str(png)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        return
    try:
        import cairosvg
        cairosvg.svg2png(
            url=str(svg), write_to=str(png),
            output_width=width, background_color="white",
        )
        return
    except Exception:
        pass
    if shutil.which("rsvg-convert"):
        subprocess.run(
            ["rsvg-convert", "-w", str(width), str(svg), "-o", str(png)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        return
    raise RuntimeError(
        "SVG-to-PNG failed — install one of: "
        "pip install cairosvg  OR  brew install resvg  OR  apt install librsvg2-bin"
    )


# ─── TTS narration ─────────────────────────────────────────────
async def _synth_narration(
    work: Path, backend: str, voice: str,
) -> list[tuple[Path, float]]:
    """Render each panel's narration to an mp3/wav. Returns list of
    (path, measured_duration_seconds)."""
    out: list[tuple[Path, float]] = []
    for idx, (_, text, _) in enumerate(PANELS):
        dest = work / f"voice_{idx:02d}.mp3"
        if backend == "edge-tts":
            await _edge_tts(text, dest, voice=voice)
        elif backend == "piper":
            _piper_tts(text, dest, voice=voice)
        elif backend == "kokoro":
            _kokoro_tts(text, dest, voice=voice)
        dur = _probe_duration(dest)
        out.append((dest, dur))
        print(f"  → voice_{idx:02d}  {dur:.2f}s")
    return out


async def _edge_tts(text: str, dest: Path, *, voice: str) -> None:
    """Synthesise via Microsoft Edge's neural voices (MIT wrapper, free endpoint)."""
    import edge_tts
    comm = edge_tts.Communicate(text=text, voice=voice, rate="+2%", volume="+0%")
    await comm.save(str(dest))


def _piper_tts(text: str, dest: Path, *, voice: str) -> None:
    """Offline MIT-licensed TTS. Requires `pip install piper-tts` +
    model download. Voice format: en_US-ryan-medium."""
    try:
        from piper import PiperVoice
    except ImportError:
        raise RuntimeError("piper not installed. `pip install piper-tts`")
    # Piper models live in ~/.local/share/piper or user-supplied path.
    import wave
    wav_path = dest.with_suffix(".wav")
    # This is a minimal inline invocation — for real production, cache
    # the model between calls.
    v = PiperVoice.load(voice)  # expects the model JSON + onnx in CWD
    with wave.open(str(wav_path), "wb") as wf:
        v.synthesize(text, wf)
    subprocess.run(
        [FFMPEG, "-y", "-i", str(wav_path), "-c:a", "libmp3lame",
         "-q:a", "2", str(dest)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )


def _kokoro_tts(text: str, dest: Path, *, voice: str) -> None:
    """Kokoro 2025 — 82M-param flagship OSS TTS, Apache-2.0."""
    try:
        from kokoro import KPipeline
    except ImportError:
        raise RuntimeError("kokoro not installed. `pip install kokoro soundfile`")
    import soundfile as sf
    pipeline = KPipeline(lang_code="a")   # auto
    audio_chunks = []
    for _, _, audio in pipeline(text, voice=voice):
        audio_chunks.append(audio)
    import numpy as np
    combined = np.concatenate(audio_chunks) if len(audio_chunks) > 1 else audio_chunks[0]
    wav_path = dest.with_suffix(".wav")
    sf.write(str(wav_path), combined, samplerate=24000)
    subprocess.run(
        [FFMPEG, "-y", "-i", str(wav_path), "-c:a", "libmp3lame",
         "-q:a", "2", str(dest)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )


def _probe_duration(path: Path) -> float:
    """MP3/WAV duration in seconds. Prefers ffprobe if present; falls
    back to mutagen (pure-Python, works without a system ffprobe)."""
    if shutil.which("ffprobe"):
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, check=True,
        )
        return float(r.stdout.strip() or 0.0)
    try:
        from mutagen import File as MutagenFile
        m = MutagenFile(str(path))
        if m is not None and getattr(m.info, "length", None):
            return float(m.info.length)
    except Exception:
        pass
    # Last-resort: use ffmpeg itself to probe (parses stderr).
    r = subprocess.run(
        [FFMPEG, "-i", str(path)], capture_output=True, text=True,
    )
    import re as _re
    m2 = _re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", r.stderr)
    if m2:
        h, mn, s = m2.groups()
        return int(h) * 3600 + int(mn) * 60 + float(s)
    return 0.0


# ─── Composition ───────────────────────────────────────────────
def _compose_video(
    work: Path,
    panel_pngs: list[Path],
    narrations: list[tuple[Path, float]],
) -> None:
    """For each panel: generate a video segment of (narration_duration +
    hold) seconds holding the panel PNG with the voice-over audio.
    Concat all segments into walkthrough.mp4."""
    segments: list[Path] = []
    for idx, ((panel_path, narration_pair, (_, _, hold_s))) in enumerate(
        zip(panel_pngs, narrations, PANELS, strict=True)
    ):
        _ = panel_path   # unpack helper (kept explicit to avoid confusion)

    # Simpler loop now that the above is clearer
    segments = []
    for idx in range(len(PANELS)):
        png = panel_pngs[idx]
        voice_mp3, voice_dur = narrations[idx]
        _, _, hold = PANELS[idx]
        seg_duration = voice_dur + hold
        seg = work / f"segment_{idx:02d}.mp4"
        # Single still image for N seconds + audio track padded with silence.
        cmd = [
            FFMPEG, "-y",
            "-loop", "1", "-t", f"{seg_duration:.3f}",
            "-i", str(png),
            "-i", str(voice_mp3),
            "-af", f"apad=whole_dur={seg_duration:.3f}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-r", "30",
            "-shortest",
            str(seg),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        segments.append(seg)

    # Concat list
    concat_list = work / "concat.txt"
    concat_list.write_text(
        "\n".join(f"file '{s.as_posix()}'" for s in segments) + "\n",
        encoding="utf-8",
    )
    subprocess.run(
        [FFMPEG, "-y",
         "-f", "concat", "-safe", "0", "-i", str(concat_list),
         "-c", "copy",
         str(work / "walkthrough.mp4")],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )


def _derive_gif_and_poster(
    mp4: Path, gif: Path, poster: Path, work: Path,
) -> None:
    """Produce two extras from the final mp4:
      * walkthrough.gif  (720-wide, 10fps, palette-dithered) — plays
        inline on GitHub README where <video> elements don't.
      * walkthrough-poster.jpg — single-frame thumbnail for the
        <video> tag's poster attribute.
    """
    # Poster — frame at t=1s (title card, already settled)
    subprocess.run(
        [FFMPEG, "-y", "-ss", "1.0", "-i", str(mp4),
         "-frames:v", "1", "-q:v", "2",
         "-vf", "scale=1280:-1", str(poster)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )

    # GIF — two-pass palette for sharp text. 720w keeps it <2MB for
    # an 80s video while still readable on desktop.
    palette = work / "palette.png"
    subprocess.run(
        [FFMPEG, "-y", "-i", str(mp4),
         "-vf", "fps=10,scale=720:-1:flags=lanczos,palettegen=max_colors=128",
         str(palette)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    subprocess.run(
        [FFMPEG, "-y", "-i", str(mp4), "-i", str(palette),
         "-filter_complex",
         "fps=10,scale=720:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=4",
         str(gif)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )


if __name__ == "__main__":
    sys.exit(main())
