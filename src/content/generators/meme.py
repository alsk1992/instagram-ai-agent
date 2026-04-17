"""Meme generator — template-driven, LLM fills in the text."""
from __future__ import annotations

import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from src.content.generators.base import GeneratedContent, staging_path
from src.content.style import _load_font, apply_watermark
from src.core.config import NicheConfig, TEMPLATES_DIR
from src.core.llm import generate_json
from src.core.logging_setup import get_logger

log = get_logger(__name__)

MEME_DIR = TEMPLATES_DIR / "memes"


def list_templates() -> list[dict]:
    """Load all template JSONs. Each must pair with a PNG/JPG background."""
    out: list[dict] = []
    if not MEME_DIR.exists():
        return out
    for f in sorted(MEME_DIR.glob("*.json")):
        try:
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
            bg = MEME_DIR / data.get("background", f.stem + ".png")
            if not bg.exists():
                # Try .jpg
                alt = bg.with_suffix(".jpg")
                if alt.exists():
                    bg = alt
                else:
                    log.warning("Meme template %s has no background", f.name)
                    continue
            data["_path"] = f
            data["_background"] = bg
            out.append(data)
        except Exception as e:
            log.warning("Skipping bad meme template %s: %s", f, e)
    return out


async def _fill_text(
    cfg: NicheConfig, template: dict, trend_context: str, *, contrarian: bool = False,
) -> dict[str, str]:
    """Ask the LLM for a JSON object mapping each text box name to content."""
    boxes = template.get("text_boxes", [])
    box_names = [b["name"] for b in boxes]
    box_rules = template.get("rules", "")

    system = (
        f"You write meme text for an Instagram page about {cfg.niche}.\n"
        f"Audience: {cfg.target_audience}.\n"
        f"Voice: {cfg.voice.persona}. Tone: {', '.join(cfg.voice.tone)}.\n"
        f"Never: {', '.join(cfg.voice.forbidden) or 'n/a'}.\n"
        f"Template: {template.get('name')}. {template.get('description', '')}.\n"
        f"Rules: {box_rules}\n"
        "Keep each line punchy and niche-specific. No hashtags, no emojis."
    )
    if contrarian:
        system += (
            "\n\nCONTRARIAN MEME — contradict a popular niche belief.\n"
            f"Avoid: {', '.join(cfg.contrarian.avoid_topics) or 'none'}. "
            "No medical/conspiracy/self-harm/political-group takes."
        )
    prompt = (
        f"Fill these text boxes as a niche-specific meme: {box_names}.\n"
        f"Trend or context to riff on:\n{trend_context or '(general niche humor)'}\n\n"
        "Return JSON: { \"box_name\": \"text\", ... }"
    )
    data = await generate_json("caption", prompt, system=system, max_tokens=400)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object, got: {type(data).__name__}")
    # Enforce that every expected box has non-empty text
    for name in box_names:
        if not str(data.get(name, "")).strip():
            data[name] = "…"
    return {k: str(v).strip() for k, v in data.items() if k in box_names}


def _render(template: dict, fills: dict[str, str], cfg: NicheConfig) -> Path:
    bg = Image.open(template["_background"]).convert("RGB")
    draw = ImageDraw.Draw(bg)

    for box in template["text_boxes"]:
        name = box["name"]
        text = fills.get(name, "").upper() if box.get("uppercase", True) else fills.get(name, "")
        if not text:
            continue
        x, y, w, h = box["x"], box["y"], box["width"], box["height"]
        color = tuple(box.get("color", [255, 255, 255]))
        stroke_color = tuple(box.get("stroke_color", [0, 0, 0]))
        stroke_width = int(box.get("stroke_width", 3))
        font_family = box.get("font", cfg.aesthetic.heading_font)
        align = box.get("align", "center")
        _draw_text_box(
            draw,
            text,
            (x, y, w, h),
            font_family=font_family,
            color=color,
            stroke_color=stroke_color,
            stroke_width=stroke_width,
            align=align,
        )

    out = staging_path("meme", ".jpg")
    bg.save(out, "JPEG", quality=93)
    return out


def _draw_text_box(
    draw: ImageDraw.ImageDraw,
    text: str,
    box: tuple[int, int, int, int],
    *,
    font_family: str,
    color: tuple,
    stroke_color: tuple,
    stroke_width: int,
    align: str,
) -> None:
    x, y, w, h = box
    # Find the largest font size that fits using binary search.
    lo, hi = 16, max(18, h)
    best_size = lo
    best_lines: list[str] = []
    while lo <= hi:
        mid = (lo + hi) // 2
        font = _load_font(font_family, mid)
        lines = _wrap(text, font, w, draw)
        total_h = _total_height(lines, font, draw)
        max_line_w = max(_line_width(ln, font, draw) for ln in lines) if lines else 0
        if total_h <= h and max_line_w <= w:
            best_size = mid
            best_lines = lines
            lo = mid + 1
        else:
            hi = mid - 1

    font = _load_font(font_family, best_size)
    line_heights = [_line_height(ln, font, draw) for ln in best_lines]
    total_h = sum(line_heights) + max(0, (len(best_lines) - 1) * int(best_size * 0.08))
    cur_y = y + (h - total_h) // 2
    for ln, lh in zip(best_lines, line_heights, strict=False):
        lw = _line_width(ln, font, draw)
        if align == "left":
            lx = x
        elif align == "right":
            lx = x + w - lw
        else:
            lx = x + (w - lw) // 2
        draw.text(
            (lx, cur_y),
            ln,
            font=font,
            fill=color,
            stroke_width=stroke_width,
            stroke_fill=stroke_color,
        )
        cur_y += lh + int(best_size * 0.08)


def _line_width(s: str, font: ImageFont.ImageFont, draw: ImageDraw.ImageDraw) -> int:
    bbox = draw.textbbox((0, 0), s, font=font)
    return bbox[2] - bbox[0]


def _line_height(s: str, font: ImageFont.ImageFont, draw: ImageDraw.ImageDraw) -> int:
    bbox = draw.textbbox((0, 0), s, font=font)
    return bbox[3] - bbox[1]


def _total_height(lines: list[str], font: ImageFont.ImageFont, draw) -> int:
    if not lines:
        return 0
    h = sum(_line_height(ln, font, draw) for ln in lines)
    return h + max(0, (len(lines) - 1) * int(font.size * 0.08))


def _wrap(text: str, font: ImageFont.ImageFont, max_w: int, draw: ImageDraw.ImageDraw) -> list[str]:
    words = text.split()
    if not words:
        return []
    lines: list[str] = []
    current: list[str] = []
    for w in words:
        candidate = " ".join(current + [w])
        if _line_width(candidate, font, draw) <= max_w:
            current.append(w)
        else:
            if current:
                lines.append(" ".join(current))
                current = [w]
            else:
                # Single word too long — hard break it
                lines.append(w)
                current = []
    if current:
        lines.append(" ".join(current))
    return lines


async def generate(
    cfg: NicheConfig, trend_context: str = "", *, contrarian: bool = False,
) -> GeneratedContent:
    templates = list_templates()
    if not templates:
        raise RuntimeError(
            f"No meme templates found in {MEME_DIR}. Add at least one template JSON + image."
        )
    template = random.choice(templates)
    fills = await _fill_text(cfg, template, trend_context, contrarian=contrarian)
    path = _render(template, fills, cfg)
    final = apply_watermark(path, cfg)
    visible = " / ".join(v for v in fills.values() if v and v != "…")
    return GeneratedContent(
        format="meme",
        media_paths=[str(final)],
        visible_text=visible,
        caption_context=(
            f"Meme in the '{template['name']}' format with on-image text: {visible}. "
            "Caption should hook scrollers without spoiling the meme."
        ),
        generator="meme",
        meta={"template": template["name"], "fills": fills},
    )
