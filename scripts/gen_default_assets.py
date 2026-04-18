#!/usr/bin/env python
"""Generate default starter assets: meme background, Google Fonts install hint.

Run once after clone; idempotent.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "src" / "instagram_ai_agent" / "content" / "templates" / "memes"
FONTS = ROOT / "data" / "fonts"


SIZE = (1080, 1080)
DARK = (18, 18, 20)
PANEL = (30, 30, 34)
DIVIDER = (80, 80, 90)
LIGHT = (240, 240, 240)
ACCENT = (201, 169, 97)


def _font(size: int):
    """Best-effort font fetch. Priority:

      1. data/fonts/ (installer downloads Archivo Black + Inter here)
      2. system DejaVu / Liberation / Arial fallbacks
      3. PIL default bitmap (tiny, ugly — last resort)
    """
    # 1. data/fonts/ — picks up the Google Fonts the installer downloads
    if FONTS.exists():
        for candidate in (
            "Archivo Black.ttf",
            "Inter.ttf",
            "ArchivoBlack-Regular.ttf",
        ):
            p = FONTS / candidate
            if p.exists():
                try:
                    return ImageFont.truetype(str(p), size)
                except OSError:
                    pass
        # Any .ttf / .otf under data/fonts/
        for p in FONTS.iterdir():
            if p.suffix.lower() in (".ttf", ".otf") and p.is_file():
                try:
                    return ImageFont.truetype(str(p), size)
                except OSError:
                    pass

    # 2. System fallbacks
    for f in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",  # macOS 10.15+
        "C:\\Windows\\Fonts\\arialbd.ttf",                    # Windows
    ):
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def make_twobox() -> Path:
    TEMPLATES.mkdir(parents=True, exist_ok=True)
    out = TEMPLATES / "twobox.jpg"
    bg = Image.new("RGB", SIZE, DARK)
    draw = ImageDraw.Draw(bg)
    for y in range(270, 810):
        draw.line([(0, y), (SIZE[0], y)], fill=PANEL)
    draw.line([(0, 270), (SIZE[0], 270)], fill=DIVIDER, width=3)
    draw.line([(0, 810), (SIZE[0], 810)], fill=DIVIDER, width=3)
    bg.save(out, "JPEG", quality=92)
    return out


def make_drake() -> Path:
    """Two-panel 'this not that': left column shows red-X / green-✓ markers,
    right column is the text canvas."""
    out = TEMPLATES / "drake.jpg"
    bg = Image.new("RGB", SIZE, LIGHT)
    draw = ImageDraw.Draw(bg)
    # Top half (rejection)
    draw.rectangle([(0, 0), (SIZE[0], 540)], fill=(245, 222, 222))
    # Bottom half (embrace)
    draw.rectangle([(0, 540), (SIZE[0], SIZE[1])], fill=(220, 240, 220))
    # Vertical divider between the marker and the text canvas
    draw.line([(520, 0), (520, SIZE[1])], fill=(60, 60, 60), width=4)
    # Horizontal divider between top + bottom
    draw.line([(0, 540), (SIZE[0], 540)], fill=(60, 60, 60), width=4)
    # Big × on top-left
    f_marker = _font(280)
    draw.text((180, 110), "✕", font=f_marker, fill=(180, 60, 60))
    # Big ✓ on bottom-left
    draw.text((180, 650), "✓", font=f_marker, fill=(60, 140, 60))
    bg.save(out, "JPEG", quality=92)
    return out


def make_expanding_brain() -> Path:
    """Four horizontal panels with brightening backdrops, text-canvas on right."""
    out = TEMPLATES / "expanding_brain.jpg"
    bg = Image.new("RGB", SIZE, DARK)
    draw = ImageDraw.Draw(bg)
    panel_h = SIZE[1] // 4
    shades = [(220, 220, 220), (210, 215, 230), (190, 220, 240), (255, 230, 180)]
    for i, shade in enumerate(shades):
        y0 = i * panel_h
        y1 = (i + 1) * panel_h
        draw.rectangle([(0, y0), (SIZE[0], y1)], fill=shade)
        # Brain glyph placeholder (numbered circle in left zone)
        cx, cy = 250, y0 + panel_h // 2
        rad = 100 + i * 12
        draw.ellipse([(cx - rad, cy - rad), (cx + rad, cy + rad)],
                     fill=(80, 80, 90), outline=(20, 20, 20), width=4)
        f = _font(rad)
        # Roman numeral-ish stage label
        label = ["I", "II", "III", "IV"][i]
        bbox = draw.textbbox((0, 0), label, font=f)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((cx - tw // 2, cy - th // 2 - bbox[1]), label,
                  font=f, fill=(245, 245, 245))
        # Vertical divider between glyph and text canvas
        draw.line([(520, y0), (520, y1)], fill=(60, 60, 60), width=3)
        # Panel-bottom divider (skip last)
        if i < 3:
            draw.line([(0, y1), (SIZE[0], y1)], fill=(60, 60, 60), width=4)
    bg.save(out, "JPEG", quality=92)
    return out


def make_stages() -> Path:
    """2x2 grid of dim panels — each tile is 540×540 with a thin gold border."""
    out = TEMPLATES / "stages.jpg"
    bg = Image.new("RGB", SIZE, DARK)
    draw = ImageDraw.Draw(bg)
    panel_w = SIZE[0] // 2
    panel_h = SIZE[1] // 2
    for col in range(2):
        for row in range(2):
            x0 = col * panel_w
            y0 = row * panel_h
            x1 = x0 + panel_w
            y1 = y0 + panel_h
            # Slightly different shade per tile so eye reads them as separate
            shade = (28 + (col + row) * 6, 28 + (col + row) * 6, 32 + (col + row) * 6)
            draw.rectangle([(x0 + 6, y0 + 6), (x1 - 6, y1 - 6)],
                           fill=shade, outline=ACCENT, width=3)
            # Stage number in the corner
            f = _font(54)
            label = f"{col + row * 2 + 1:02d}"
            draw.text((x0 + 24, y0 + 18), label, font=f, fill=ACCENT)
    bg.save(out, "JPEG", quality=92)
    return out


def make_expectation_reality() -> Path:
    """Two side-by-side panels with EXPECTATION / REALITY headers baked in."""
    out = TEMPLATES / "expectation_reality.jpg"
    bg = Image.new("RGB", SIZE, DARK)
    draw = ImageDraw.Draw(bg)
    half = SIZE[0] // 2
    # Left panel: cooler tone (the polished expectation)
    draw.rectangle([(0, 0), (half, SIZE[1])], fill=(40, 60, 90))
    # Right panel: warmer tone (gritty reality)
    draw.rectangle([(half, 0), (SIZE[0], SIZE[1])], fill=(90, 50, 40))
    # Header bars
    draw.rectangle([(0, 0), (half, 100)], fill=(20, 30, 50))
    draw.rectangle([(half, 0), (SIZE[0], 100)], fill=(50, 25, 20))
    f_hdr = _font(58)
    for label, x_anchor in (("EXPECTATION", half // 2), ("REALITY", half + half // 2)):
        bbox = draw.textbbox((0, 0), label, font=f_hdr)
        tw = bbox[2] - bbox[0]
        draw.text((x_anchor - tw // 2, 18), label, font=f_hdr, fill=LIGHT)
    # Divider between panels
    draw.line([(half, 0), (half, SIZE[1])], fill=(0, 0, 0), width=4)
    # Image-zone hint (centre rectangles where a visual would sit, optional)
    draw.rectangle([(60, 160), (half - 40, 660)],
                   outline=(255, 255, 255, 60), width=2)
    draw.rectangle([(half + 40, 160), (SIZE[0] - 60, 660)],
                   outline=(255, 255, 255, 60), width=2)
    bg.save(out, "JPEG", quality=92)
    return out


def main() -> int:
    for fn in (make_twobox, make_drake, make_expanding_brain, make_stages, make_expectation_reality):
        out = fn()
        print(f"[ok] wrote {out}")
    if not FONTS.exists() or not any(FONTS.glob("*.ttf")):
        print(
            "[hint] no fonts in data/fonts/. Recommended:\n"
            "   curl -L -o data/fonts/ArchivoBlack-Regular.ttf https://github.com/google/fonts/raw/main/ofl/archivoblack/ArchivoBlack-Regular.ttf\n"
            "   curl -L -o data/fonts/Inter-Regular.ttf https://github.com/google/fonts/raw/main/ofl/inter/Inter%5Bopsz%2Cwght%5D.ttf"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
