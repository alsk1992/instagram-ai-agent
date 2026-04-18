#!/usr/bin/env python
"""Stitch the real recordings + Kokoro narration into one walkthrough.mp4.

Takes docs/media/real-cli-*.mp4 + docs/media/real-dashboard.mp4,
scales each to 1280x720, adds a spoken narration track via Kokoro,
and writes the final docs/media/walkthrough.mp4 + a still poster.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "docs" / "media"
CANVAS_W, CANVAS_H = 1280, 720
FPS = 25

# (source_mp4, narration_text, hold_seconds_after)
CLIPS = [
    (
        None,  # title card
        "ig-agent. an autonomous Instagram content agent. "
        "here's what it actually looks like running.",
        0.0,
    ),
    (
        MEDIA / "real-cli-doctor.mp4",
        "first — doctor. it verifies Python, ffmpeg, Playwright, "
        "and your environment so setup problems fail loud, not silent.",
        0.5,
    ),
    (
        MEDIA / "real-cli-status.mp4",
        "status — your niche, content format mix, providers, and queue counts, "
        "all in one place.",
        0.5,
    ),
    (
        MEDIA / "real-cli-warmup.mp4",
        "warmup-status — the daily ramp. likes, follows, comments, "
        "each capped by day of the account's age.",
        0.5,
    ),
    (
        MEDIA / "real-dashboard.mp4",
        "the local dashboard. review generated posts, approve or reject them, "
        "and watch what the agent published. all on your machine.",
        1.0,
    ),
    (
        None,  # outro
        "pipx install instagram-ai-agent. give it a niche. it does the rest.",
        0.0,
    ),
]


def ffmpeg_bin() -> str:
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def narrate(text: str, out_mp3: Path) -> float:
    """Synthesise narration via Kokoro CPU. Returns duration in seconds."""
    import numpy as np  # noqa: F401
    import soundfile as sf
    from kokoro import KPipeline
    import torch

    global _PIPE
    try:
        _PIPE
    except NameError:
        _PIPE = KPipeline(lang_code="a", device=torch.device("cpu"))

    audio_chunks = []
    for _, _, audio in _PIPE(text, voice="af_bella"):
        audio_chunks.append(audio.numpy())

    import numpy as np
    full = np.concatenate(audio_chunks) if audio_chunks else np.zeros(24000)
    wav_tmp = out_mp3.with_suffix(".wav")
    sf.write(str(wav_tmp), full, 24000)

    subprocess.run(
        [ffmpeg_bin(), "-y", "-i", str(wav_tmp),
         "-codec:a", "libmp3lame", "-qscale:a", "2",
         str(out_mp3)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    wav_tmp.unlink(missing_ok=True)

    from mutagen.mp3 import MP3
    return float(MP3(str(out_mp3)).info.length)


def render_title_card(text: str, duration: float, out: Path) -> None:
    """Rasterise a centered text card via PIL, then encode as a still video."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (CANVAS_W, CANVAS_H), color=(10, 10, 10))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 52
    )

    # Wrap text to fit 80% canvas width
    max_w = int(CANVAS_W * 0.8)
    words = text.split()
    lines: list[str] = []
    cur = ""
    for w in words:
        candidate = f"{cur} {w}".strip()
        if draw.textlength(candidate, font=font) <= max_w:
            cur = candidate
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)

    line_h = 66
    total_h = len(lines) * line_h
    y = (CANVAS_H - total_h) // 2
    for line in lines:
        tw = draw.textlength(line, font=font)
        draw.text(((CANVAS_W - tw) // 2, y), line, fill=(245, 245, 240), font=font)
        y += line_h

    # Gold accent bar near the bottom
    bar_y = int(CANVAS_H * 0.85)
    draw.rectangle([(0, bar_y), (CANVAS_W, bar_y + 6)], fill=(201, 169, 97))

    png = out.with_suffix(".png")
    img.save(png)

    ffmpeg = ffmpeg_bin()
    subprocess.run(
        [ffmpeg, "-y",
         "-loop", "1", "-t", f"{duration:.3f}",
         "-i", str(png),
         "-vf", f"fps={FPS},format=yuv420p",
         "-c:v", "libx264", "-preset", "medium", "-crf", "20",
         str(out)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    png.unlink(missing_ok=True)


def scale_clip(src: Path, audio_mp3: Path, duration: float, out: Path) -> None:
    """Scale source to 1280x720 (letterbox) and mux in the narration mp3,
    stretching/truncating video to match narration duration."""
    ffmpeg = ffmpeg_bin()
    vf = (
        f"scale={CANVAS_W}:{CANVAS_H}:force_original_aspect_ratio=decrease,"
        f"pad={CANVAS_W}:{CANVAS_H}:(ow-iw)/2:(oh-ih)/2:color=0x0a0a0a,"
        f"fps={FPS},"
        f"setpts=PTS*{duration}/(PTS_END)"  # will be overwritten by -t
    )
    # Simpler approach: loop/trim the video to exactly match narration duration
    subprocess.run(
        [ffmpeg, "-y",
         "-stream_loop", "-1", "-i", str(src),
         "-i", str(audio_mp3),
         "-t", f"{duration:.3f}",
         "-vf", (
             f"scale={CANVAS_W}:{CANVAS_H}:force_original_aspect_ratio=decrease,"
             f"pad={CANVAS_W}:{CANVAS_H}:(ow-iw)/2:(oh-ih)/2:color=0x0a0a0a,"
             f"fps={FPS}"
         ),
         "-c:v", "libx264", "-preset", "medium", "-crf", "22",
         "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "128k",
         "-shortest",
         str(out)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )


def combine_title(card: Path, audio_mp3: Path, duration: float, out: Path) -> None:
    ffmpeg = ffmpeg_bin()
    subprocess.run(
        [ffmpeg, "-y",
         "-i", str(card), "-i", str(audio_mp3),
         "-t", f"{duration:.3f}",
         "-c:v", "libx264", "-preset", "medium", "-crf", "20",
         "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "128k",
         "-shortest",
         str(out)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )


def concat_clips(clips: list[Path], out: Path) -> None:
    ffmpeg = ffmpeg_bin()
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for c in clips:
            f.write(f"file '{c.resolve()}'\n")
        concat_list = f.name
    try:
        subprocess.run(
            [ffmpeg, "-y", "-f", "concat", "-safe", "0",
             "-i", concat_list,
             "-c:v", "libx264", "-preset", "medium", "-crf", "22",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart",
             "-c:a", "aac", "-b:a", "128k",
             str(out)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
    finally:
        Path(concat_list).unlink(missing_ok=True)


def poster_from_mp4(mp4: Path, out_jpg: Path) -> None:
    """Grab a poster frame at 25% into the video."""
    ffmpeg = ffmpeg_bin()
    subprocess.run(
        [ffmpeg, "-y", "-ss", "6", "-i", str(mp4),
         "-frames:v", "1", "-q:v", "3", str(out_jpg)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="ig_compose_"))
    try:
        per_clip: list[Path] = []
        for i, (src, narration, hold) in enumerate(CLIPS):
            audio = tmp / f"nar_{i:02d}.mp3"
            dur = narrate(narration, audio) + hold
            print(f"  [{i}] narration {dur:.2f}s — {narration[:60]}…")

            out = tmp / f"clip_{i:02d}.mp4"
            if src is None:
                card = tmp / f"card_{i:02d}.mp4"
                render_title_card(narration.split('.')[0] + '.', dur, card)
                combine_title(card, audio, dur, out)
            else:
                scale_clip(src, audio, dur, out)
            per_clip.append(out)

        final = MEDIA / "walkthrough.mp4"
        print("  Concatenating…")
        concat_clips(per_clip, final)
        print(f"  → {final.relative_to(ROOT)}  ({final.stat().st_size / 1024 / 1024:.1f} MB)")

        print("  Deriving poster…")
        poster_from_mp4(final, MEDIA / "walkthrough-poster.jpg")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
