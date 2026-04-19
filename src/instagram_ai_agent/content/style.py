"""Style applicator — palette/font/LUT/watermark/film-look across generators."""
from __future__ import annotations

import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from instagram_ai_agent.core.config import FONTS_DIR, LUTS_DIR, NicheConfig
from instagram_ai_agent.plugins import film_emulation


def apply_watermark(image_path: str | Path, cfg: NicheConfig) -> Path:
    """Burn the niche watermark onto the bottom-right with safe padding."""
    if not cfg.aesthetic.watermark:
        return Path(image_path)

    img_p = Path(image_path)
    with Image.open(img_p).convert("RGBA") as img:
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)

        # Size the watermark proportionally to the image
        size = max(14, img.height // 48)
        font = _load_font(cfg.aesthetic.body_font, size)
        text = cfg.aesthetic.watermark
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        pad = size // 2
        x = img.width - tw - pad * 2
        y = img.height - th - pad * 2

        # Semi-opaque pill background for legibility
        draw.rounded_rectangle(
            [x - pad, y - pad, x + tw + pad, y + th + pad],
            radius=size // 2,
            fill=(0, 0, 0, 140),
        )
        draw.text((x, y), text, font=font, fill=(255, 255, 255, 230))

        composed = Image.alpha_composite(img, layer).convert("RGB")
        out = img_p.with_name(img_p.stem + "_wm" + img_p.suffix)
        composed.save(out, quality=94)
        return out


def apply_film_look(image_path: str | Path, cfg: NicheConfig) -> Path:
    """Apply grain + vignette + subtle colour cast so AI outputs look photographed.

    Strength follows ``cfg.aesthetic.film_strength`` (default: medium). This
    is the single biggest visual-realness lever — raw Flux/SDXL output is
    sterile and sterile = reads as AI. Called by every generator that
    ships AI-rendered media (photo, human_photo, AI reel frames).
    """
    strength = cfg.aesthetic.film_strength or "medium"
    if strength == "off":
        return Path(image_path)
    return film_emulation.apply_film_look(image_path, strength=strength)


def apply_lut_image(image_path: str | Path, cfg: NicheConfig) -> Path:
    """Apply a .cube LUT to a still image via ffmpeg's lut3d filter."""
    if not cfg.aesthetic.lut:
        return Path(image_path)

    lut_path = _resolve_lut(cfg.aesthetic.lut)
    if lut_path is None:
        return Path(image_path)

    img = Path(image_path)
    out = img.with_name(img.stem + "_lut" + img.suffix)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(img),
        "-vf",
        f"lut3d={lut_path.as_posix()}",
        str(out),
    ]
    # Surface ffmpeg errors instead of crashing the whole pipeline — a
    # malformed .cube file or missing ffmpeg binary should degrade to
    # "skip LUT, keep raw image" with a visible log line.
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return out
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        stderr_tail = (
            getattr(e, "stderr", b"") or b""
        ).decode("utf-8", errors="replace")[-400:]
        from instagram_ai_agent.core.logging_setup import get_logger
        get_logger(__name__).warning(
            "apply_lut_image: ffmpeg failed on %s (LUT=%s) — returning raw image. %s",
            img.name, lut_path.name, stderr_tail.strip() or e,
        )
        return img


def apply_lut_video(video_path: str | Path, cfg: NicheConfig) -> Path:
    if not cfg.aesthetic.lut:
        return Path(video_path)
    lut_path = _resolve_lut(cfg.aesthetic.lut)
    if lut_path is None:
        return Path(video_path)

    v = Path(video_path)
    out = v.with_name(v.stem + "_lut" + v.suffix)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(v),
        "-vf",
        f"lut3d={lut_path.as_posix()}",
        "-c:a",
        "copy",
        str(out),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out


def _resolve_lut(ref: str) -> Path | None:
    p = Path(ref)
    if p.is_absolute() and p.exists():
        return p
    candidate = LUTS_DIR / ref
    if candidate.exists():
        return candidate
    return None


def _load_font(name: str, size: int) -> ImageFont.ImageFont:
    """Try fonts in data/fonts first, fall back to PIL default."""
    for candidate in (
        FONTS_DIR / f"{name}.ttf",
        FONTS_DIR / f"{name}.otf",
        FONTS_DIR / f"{name}-Regular.ttf",
        FONTS_DIR / f"{name}-Bold.ttf",
    ):
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    # Fall back to a common system font if present
    for system in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                   "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"):
        if Path(system).exists():
            return ImageFont.truetype(system, size=size)
    return ImageFont.load_default()
