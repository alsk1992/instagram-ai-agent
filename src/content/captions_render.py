"""Caption file renderers.

Two outputs, same word-level input:

  * :func:`render_srt` — 3–4 word chunks, static subtitles (backward-compat
    with the old ``faster_whisper`` → SRT path).
  * :func:`render_ass_karaoke` — one-word-at-a-time kinetic ASS with fade-in,
    scale-bounce and accent-colour highlight. The dominant IG-reel style in
    2026. Burned via the same ffmpeg ``subtitles`` filter.
"""
from __future__ import annotations

import re
from pathlib import Path

from src.content.transcribe import Word
from src.core.config import CaptionsConfig, NicheConfig


# ───── SRT (chunked) ─────
def render_srt(words: list[Word], out: Path, *, chunk_size: int = 4) -> Path:
    """Group words into ``chunk_size``-word entries; punctuation triggers an
    early chunk break so we don't orphan a fragment."""
    entries: list[tuple[float, float, str]] = []
    buf: list[Word] = []
    for w in words:
        buf.append(w)
        # Only true sentence terminators force an early break; commas stay
        # inside the chunk so we don't orphan 1-2 word fragments.
        ends_sentence = w.text.rstrip().endswith((".", "!", "?"))
        if len(buf) >= chunk_size or (len(buf) >= 3 and ends_sentence):
            entries.append((buf[0].start, buf[-1].end, " ".join(x.text for x in buf)))
            buf = []
    if buf:
        entries.append((buf[0].start, buf[-1].end, " ".join(x.text for x in buf)))

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for i, (start, end, text) in enumerate(entries, 1):
            f.write(f"{i}\n{_fmt_srt_time(start)} --> {_fmt_srt_time(end)}\n{text}\n\n")
    return out


def _fmt_srt_time(t: float) -> str:
    t = max(0.0, t)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int((t - int(t)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


# ───── ASS karaoke ─────
#
# ASS override tags we lean on:
#   {\fad(in,out)}        fade-in/out in ms
#   {\t(t1,t2,\fscx...)}  time-based transform → scale bounce
#   {\fscx N\fscy N}      base scale %
#   {\an5}                alignment: 5 = centre
#   {\pos(x,y)}           absolute position
#   {\c&HBBGGRR&}         primary colour override per-line (not needed — style handles it)
#
# One event per word; caller's aesthetic palette drives font + colour.

_HEX_RE = re.compile(r"^#?([0-9a-fA-F]{6})$")


def _hex_to_bgr(hex_colour: str) -> str:
    m = _HEX_RE.match(hex_colour or "")
    if not m:
        return "FFFFFF"
    r, g, b = m.group(1)[0:2], m.group(1)[2:4], m.group(1)[4:6]
    return (b + g + r).upper()


def _ass_time(t: float) -> str:
    """ASS uses H:MM:SS.cs (centiseconds)."""
    t = max(0.0, t)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    cs = int(round((t - int(t)) * 100))
    if cs == 100:
        cs = 99
    return f"{h}:{m:02}:{s:02}.{cs:02}"


def _ass_escape(text: str) -> str:
    # ASS treats {} as override-tag delimiters; escape them and backslashes.
    return text.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")


def render_ass_karaoke(
    words: list[Word],
    out: Path,
    *,
    cfg: NicheConfig,
    captions: CaptionsConfig,
    video_w: int = 1080,
    video_h: int = 1920,
    is_story: bool = False,
) -> Path:
    """Write an ASS file with one-word-at-a-time kinetic captions.

    Each word is its own ``Dialogue`` event timed to ``[word.start, word.end]``
    (with a ~40ms pad so adjacent words don't crack audibly). Every event
    carries a fade + a scale bounce so the word "pops in" on hit.
    """
    if not words:
        out.write_text("", encoding="utf-8")
        return out

    palette = cfg.aesthetic.palette
    fg = palette[1] if len(palette) > 1 else "#ffffff"
    accent = captions.highlight_colour or (palette[2] if len(palette) > 2 else fg)
    outline = "#000000"

    # Font sizing — ASS PlayResY scales font_size relative to the rendered video.
    # We use PlayResY=video_h for a 1:1 mapping.
    base_font_px = int(video_h * (0.045 if is_story else 0.055))
    margin_v = captions.margin_v_story if is_story else captions.margin_v_feed

    heading_font = (cfg.aesthetic.heading_font or "Arial").replace("'", "")

    # ASS styles: primary text colour set via style; per-event colour changes are
    # achieved by writing the accent colour as the PrimaryColour in the hook style.
    header = _ass_header(
        play_res_x=video_w,
        play_res_y=video_h,
        font_name=heading_font,
        font_size=base_font_px,
        primary_bgr=_hex_to_bgr(accent),
        outline_bgr=_hex_to_bgr(outline),
        back_bgr=_hex_to_bgr("#000000"),
        margin_v=margin_v,
    )

    peak = captions.font_scale_peak
    base_scale = 100
    peak_scale = int(round(peak * 100))
    events: list[str] = []
    n = len(words)
    for i, w in enumerate(words):
        start = max(0.0, w.start)
        # Hold slightly past the spoken end so the word doesn't flash off,
        # but cap hold at the NEXT word's start so we don't stack two events
        # on the same layer (avoids libass double-draw / flicker).
        held = max(w.end + 0.08, start + 0.12)
        if i + 1 < n:
            held = min(held, words[i + 1].start)
            # Guarantee a visible minimum duration
            held = max(held, start + 0.12)
        end = held
        duration_ms = max(120, int((end - start) * 1000))
        bounce_ms = min(220, duration_ms // 2)
        fade_in_ms = min(120, duration_ms // 3)
        fade_out_ms = min(160, duration_ms // 3)

        # Build the override block. Braces are real ASS syntax; every {{ in an
        # f-string emits one literal `{`, every }} emits one literal `}`.
        x = video_w // 2
        y = video_h - margin_v
        tags = (
            f"{{\\an5\\pos({x},{y})"
            f"\\fad({fade_in_ms},{fade_out_ms})"
            f"\\fscx{peak_scale}\\fscy{peak_scale}"
            f"\\t(0,{bounce_ms},\\fscx{base_scale}\\fscy{base_scale})}}"
        )
        text = _ass_escape(w.text)
        events.append(
            f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Karaoke,,0,0,0,,"
            f"{tags}{text}"
        )

    body = "\n".join(events) + "\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(header + body)
    return out


def _ass_header(
    *,
    play_res_x: int,
    play_res_y: int,
    font_name: str,
    font_size: int,
    primary_bgr: str,
    outline_bgr: str,
    back_bgr: str,
    margin_v: int,
) -> str:
    """Minimal ASS header with a single Karaoke style."""
    # BorderStyle=1 with OutlineColour handled via per-style; we keep it simple.
    return f"""[Script Info]
ScriptType: v4.00+
WrapStyle: 2
ScaledBorderAndShadow: yes
YCbCr Matrix: None
PlayResX: {play_res_x}
PlayResY: {play_res_y}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Karaoke,{font_name},{font_size},&H00{primary_bgr}&,&H00FFFFFF&,&H00{outline_bgr}&,&H80{back_bgr}&,1,0,0,0,100,100,0,0,1,5,0,5,60,60,{margin_v},0

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
