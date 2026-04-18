"""ffmpeg audio mixer — voiceover + looping-ducked music bed.

One function: ``mix_vo_and_music`` — takes the raw voiceover, a music path,
and an output destination, writes a stereo 48k AAC-in-MP4 audio track of
the configured length. Music is looped to cover the whole clip, faded in
and out, and attenuated to sit under the VO.

Mix strategy (simple + predictable):
  [music] → aloop -1 → volume(duck_gain) → afade in → afade out → [bed]
  [vo]    → volume(vo_gain)                                      → [vo_g]
  amix [bed][vo_g]                                                → [mixed]

We stay away from sidechaincompress because free-tier ffmpeg builds have
historically shipped inconsistent compressor coefficients.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from instagram_ai_agent.core.config import MusicConfig
from instagram_ai_agent.core.logging_setup import get_logger

log = get_logger(__name__)


def mix_vo_and_music(
    voiceover: Path,
    music: Path,
    out_path: Path,
    *,
    duration_s: float,
    music_cfg: MusicConfig,
) -> Path:
    """Render VO + music to ``out_path`` (m4a/aac). Caller supplies the
    total duration so we can pad/loop music and fade it out cleanly."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fade_in = max(0.0, music_cfg.fade_in_s)
    fade_out = max(0.0, music_cfg.fade_out_s)
    duck_gain = float(music_cfg.duck_gain)
    vo_gain = float(music_cfg.vo_gain)
    total = max(1.0, duration_s)

    fade_out_start = max(0.0, total - fade_out)

    filter_complex = (
        # Music: loop, gain, fade-in, fade-out
        f"[1:a]aloop=loop=-1:size=2e9,"
        f"volume={duck_gain:.3f},"
        f"afade=t=in:st=0:d={fade_in:.2f},"
        f"afade=t=out:st={fade_out_start:.2f}:d={fade_out:.2f}[bed];"
        # Voiceover gain
        f"[0:a]volume={vo_gain:.3f}[vog];"
        # Mix — use 'duration=first' so music obeys VO length
        f"[vog][bed]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[out]"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", str(voiceover),
        "-stream_loop", "-1",
        "-i", str(music),
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-t", f"{total:.3f}",
        "-ar", "48000",
        "-ac", "2",
        "-c:a", "aac",
        "-b:a", "192k",
        str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        log.error("audio mix failed: %s", e.stderr.decode(errors="ignore")[-800:])
        raise
    return out_path


def mux_video_with_audio(
    video: Path,
    audio: Path,
    srt: Path,
    out: Path,
    *,
    subtitle_style: str,
) -> Path:
    """Drop-in replacement for reel_stock.mux_audio_captions but with an
    arbitrary pre-mixed audio track (music + VO). Re-used by reel_stock,
    reel_ai and story_video when a music bed is present.
    """
    # Escape SRT path for subtitles filter
    srt_escaped = str(srt).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    vf = f"subtitles='{srt_escaped}':force_style='{subtitle_style}'"
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
