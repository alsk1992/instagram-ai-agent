"""Video overlay primitives — burned-in hook text, watermark, etc.

This module concentrates every ``ffmpeg drawtext`` incantation so the
generators don't each reinvent escaping, font resolution and easing.

``add_hook_overlay(video, hook_text, cfg)`` is the primary surface. Runs a
single drawtext pass that fades the hook in, holds it, fades it out — all
inside the first N seconds configured by ``HookOverlay.duration_s``. The
rest of the clip is untouched.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable

from src.content.generators.base import staging_path
from src.core.config import FONTS_DIR, HookOverlay, NicheConfig
from src.core.logging_setup import get_logger

log = get_logger(__name__)


# ─── Text wrapping ───
def wrap_hook(text: str, *, max_words: int, max_chars_per_line: int, max_lines: int) -> list[str]:
    """Greedy word-wrap. Trims to ``max_words`` then packs up to ``max_lines``
    lines of ``max_chars_per_line``.

    When words are dropped (either by the word cap or the line cap) we append
    a single ellipsis character to the last line, so readers can see the hook
    was truncated rather than reading a sentence fragment that looks final.
    """
    original_words = (text or "").split()
    if not original_words:
        return []

    words = original_words[:max_words]
    words_dropped = len(original_words) > max_words

    lines: list[str] = []
    cur: list[str] = []
    cur_len = 0
    used = 0
    for w in words:
        w_len = len(w)
        if not cur:
            cur = [w]
            cur_len = w_len
            used += 1
            continue
        if cur_len + 1 + w_len <= max_chars_per_line:
            cur.append(w)
            cur_len += 1 + w_len
            used += 1
        else:
            lines.append(" ".join(cur))
            cur = [w]
            cur_len = w_len
            used += 1
            if len(lines) >= max_lines - 1:
                break
    if cur and len(lines) < max_lines:
        lines.append(" ".join(cur))

    # Any remaining words from ``words`` (the max_words-capped list) that we
    # didn't consume PLUS anything trimmed by ``max_words`` → signal with an
    # ellipsis on the final line.
    words_dropped = words_dropped or (used < len(words))
    if words_dropped and lines:
        last = lines[-1]
        if not last.endswith(("…", "...")):
            # Respect the per-line char cap — if the line is already full,
            # chop the last word to make room for the ellipsis.
            suffix = "…"
            if len(last) + len(suffix) <= max_chars_per_line:
                lines[-1] = last + suffix
            else:
                tokens = last.split()
                tokens = tokens[:-1] if len(tokens) > 1 else tokens
                lines[-1] = " ".join(tokens) + suffix
    return lines


# ─── Font resolution ───
def _resolve_font_path(family: str) -> Path | None:
    if not FONTS_DIR.exists():
        return None
    # Normalise: "Archivo Black" → "archivoblack"
    key = "".join(ch for ch in family.lower() if ch.isalnum())
    candidates: list[Path] = []
    for p in FONTS_DIR.iterdir():
        if p.suffix.lower() not in {".ttf", ".otf"}:
            continue
        stem = "".join(ch for ch in p.stem.lower() if ch.isalnum())
        if key and (key in stem or stem.startswith(key)):
            candidates.append(p)
    if not candidates:
        return None
    preferred = [c for c in candidates if "regular" in c.stem.lower() or c.stem.lower().endswith(key)]
    return (preferred or candidates)[0]


def _fallback_system_font() -> Path | None:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        p = Path(candidate)
        if p.exists():
            return p
    return None


# ─── ffmpeg probe helpers ───
def _probe_video_size(video: Path) -> tuple[int, int]:
    try:
        r = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=p=0:s=,",
                str(video),
            ],
            capture_output=True, text=True, check=True,
        )
        w_str, h_str = r.stdout.strip().split(",")
        return int(w_str), int(h_str)
    except Exception:
        return 1080, 1920


def _escape_drawtext_param(s: str) -> str:
    """Escape a value for ffmpeg filter-graph parameters.

    Filter args are parsed by libavfilter; ``:`` and ``'`` and ``\\`` are the
    hot items. Backslash first, single quote second.
    """
    return (
        s.replace("\\", r"\\")
         .replace(":", r"\:")
         .replace("'", r"\'")
    )


# ─── Colour helpers (reuses aesthetic.palette) ───
def _hex_to_ff(hex_color: str, alpha: float = 1.0) -> str:
    """`#RRGGBB` → `0xRRGGBB@A` for drawtext.``{font,box,border}color``."""
    import re as _re
    m = _re.match(r"^#?([0-9a-fA-F]{6})$", hex_color or "")
    rgb = m.group(1) if m else "FFFFFF"
    alpha = max(0.0, min(1.0, alpha))
    return f"0x{rgb.upper()}@{alpha:.2f}"


# ─── Public: hook overlay ───
def add_hook_overlay(
    video: Path,
    hook_text: str,
    cfg: NicheConfig,
    *,
    out: Path | None = None,
) -> Path:
    """Burn a fading hook overlay over the first ``cfg.hook_overlay.duration_s``.

    Graceful no-op: if text is empty, feature disabled, font can't be found,
    or ffmpeg errors, returns ``video`` unchanged. Callers never need to
    guard this.
    """
    ho: HookOverlay = cfg.hook_overlay
    if not ho.enabled:
        return video
    text = (hook_text or "").strip()
    if not text:
        return video

    wrapped = wrap_hook(
        text,
        max_words=ho.max_words,
        max_chars_per_line=ho.max_chars_per_line,
        max_lines=ho.max_lines,
    )
    if not wrapped:
        return video

    # Font path (niche heading > body > system fallback)
    font_path = (
        _resolve_font_path(cfg.aesthetic.heading_font)
        or _resolve_font_path(cfg.aesthetic.body_font)
        or _fallback_system_font()
    )
    if font_path is None:
        log.info("hook overlay: no font resolved — skipping")
        return video

    w, h = _probe_video_size(video)
    fontsize = max(18, int(h * ho.font_scale))
    y_abs = int(h * ho.y_ratio)

    palette = cfg.aesthetic.palette
    fg = palette[1] if len(palette) > 1 else "#FFFFFF"
    bg = palette[0] if len(palette) > 0 else "#000000"

    # Alpha envelope: fade in → hold → fade out
    d = float(ho.duration_s)
    fi = max(0.0, ho.fade_in_s)
    fo = max(0.0, ho.fade_out_s)
    hold_start = fi
    hold_end = max(hold_start, d - fo)
    # ffmpeg expressions: alpha=0 outside the window, ramp in/out inside
    alpha_expr = (
        f"if(lt(t,{fi:.3f}),t/{max(fi,0.001):.3f},"
        f"if(lt(t,{hold_end:.3f}),1,"
        f"if(lt(t,{d:.3f}),(({d:.3f}-t)/{max(fo,0.001):.3f}),0)))"
    )

    # Textfile — keeps any special char out of the filter syntax entirely
    textfile = staging_path("hook", ".txt")
    textfile.write_text("\n".join(wrapped), encoding="utf-8")

    # Font path needs escaping for the filter-graph
    font_esc = _escape_drawtext_param(str(font_path.resolve()))
    tf_esc = _escape_drawtext_param(str(textfile.resolve()))

    drawtext_parts = [
        f"textfile='{tf_esc}'",
        f"fontfile='{font_esc}'",
        f"fontcolor={_hex_to_ff(fg, 1.0)}",
        f"fontsize={fontsize}",
        f"line_spacing={max(6, fontsize // 8)}",
        # centre horizontally, fixed y
        "x=(w-text_w)/2",
        f"y={y_abs}",
        # Fade window: enable only for the first `d` seconds, alpha ramps
        f"enable='between(t,0,{d:.3f})'",
        f"alpha='{alpha_expr}'",
        # Bold outline for readability
        "borderw=4",
        f"bordercolor={_hex_to_ff(bg, 0.9)}",
    ]
    if ho.box:
        drawtext_parts += [
            "box=1",
            f"boxcolor={_hex_to_ff(bg, ho.box_alpha)}",
            f"boxborderw={ho.box_borderw}",
        ]

    vf = "drawtext=" + ":".join(drawtext_parts)

    dst = out or staging_path(video.stem + "_hook", video.suffix)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(dst),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except FileNotFoundError:
        # ffmpeg binary missing — strip-down env (fresh CI, minimal container).
        # The whole feature is optional; do not crash the reel pipeline.
        log.warning("hook overlay: ffmpeg not found on PATH — returning original")
        return video
    except subprocess.CalledProcessError as e:
        log.warning(
            "hook overlay ffmpeg failed — returning original: %s",
            e.stderr.decode(errors="ignore")[-400:],
        )
        return video
    finally:
        try:
            textfile.unlink(missing_ok=True)
        except OSError:
            pass

    return dst


# ─── Convenience for generators ───
def pick_hook_text(scenes: Iterable[dict]) -> str:
    """Return the best hook candidate from a reel script's scene list.

    We prefer scenes explicitly marked ``hook: true`` or ``kind=="hook"``.
    Fallback to the first scene's ``line``.
    """
    first_line = ""
    for i, s in enumerate(scenes):
        if i == 0:
            first_line = str(s.get("line") or s.get("text") or "").strip()
        if s.get("hook") is True or s.get("kind") == "hook":
            return str(s.get("line") or s.get("text") or "").strip() or first_line
    return first_line
