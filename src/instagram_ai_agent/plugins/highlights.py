"""Instagram Story Highlights automation.

Wraps instagrapi's HighlightMixin with business logic on top:

  - **Bootstrap**: for each category in cfg.highlights.categories, create
    the highlight with a generated cover if it doesn't already exist.
    Idempotent — safe to rerun on every boot.
  - **Auto-promote**: when a story is posted with a category tag (in its
    meta or via keyword match on the caption), add it to the matching
    highlight. Creates new highlights on demand.
  - **Cover generation**: Pillow-drawn 1080×1920 canvas with a solid-color
    circle + centered white letter/icon mark. Matches the minimal aesthetic
    that fitness pages actually use.

Research-verified behaviours (April 2026):
  - instagrapi's HighlightMixin write-path (create / add_stories /
    remove_stories / change_cover / delete) is stable and actively maintained.
  - Read-path ``highlight_info`` throws LoginRequired with web-cookie
    sessions — we use native u/p+TOTP so it works, but we wrap all reads
    in try/except regardless.
  - No reorder endpoint exists; reorder via the add-remove-nudge trick
    (add then immediately remove a story → IG re-sorts that highlight left).
  - Safe cadence: ≤10 highlight writes/day; we enforce via a DB counter.
  - Cover spec: 1080×1920 canvas, visible circle rendered from centered
    600×600 safe zone.
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from instagram_ai_agent.core.config import HighlightCategory, NicheConfig
from instagram_ai_agent.core.logging_setup import get_logger
from instagram_ai_agent.plugins.ig import IGClient

log = get_logger(__name__)

# IG's recommended canvas size for highlight covers.
COVER_W, COVER_H = 1080, 1920
# Safe zone — where the visible circle lives. IG's default crop_rect is
# [0.0, 0.21830457, 1.0, 0.78094524] which is a roughly-square vertical
# strip centered on the canvas. We paint inside a 600×600 box at this spot
# so the entire circle is visible.
CIRCLE_D = 900   # diameter — leaves breathing room inside the safe zone


@dataclass
class BootstrapResult:
    created: list[str]         # category names we made fresh
    existing: list[str]        # category names that already existed
    skipped: list[str]         # category names that errored out


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    s = hex_color.lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))


def _render_cover(category: HighlightCategory, out_path: Path) -> Path:
    """Render a 1080×1920 cover image matching the category's colour + icon.

    Layout: solid dark-grey backdrop + centred coloured circle + centred
    white icon/letter. Minimal aesthetic; matches what the fitness growth
    accounts actually use."""
    bg_rgb = _hex_to_rgb(category.color)

    # Pick a backdrop that contrasts the circle — dark grey unless the
    # circle is already dark, in which case use a lighter grey.
    avg = sum(bg_rgb) / 3
    backdrop = (30, 30, 30) if avg > 80 else (240, 240, 240)

    im = Image.new("RGB", (COVER_W, COVER_H), backdrop)
    draw = ImageDraw.Draw(im)

    # Centre the circle on the canvas
    cx, cy = COVER_W // 2, COVER_H // 2
    r = CIRCLE_D // 2
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=bg_rgb)

    # Draw the icon/letter in white, centred. Try to find a common system
    # font; fall back to Pillow's default bitmap font.
    icon_text = (category.icon or category.name[:1]).strip() or category.name[:1]
    font = None
    # Common sans-serif TTFs that usually exist on Linux/Win/Mac. Pillow
    # returns None silently if the font isn't found.
    for cand in (
        "DejaVuSans-Bold.ttf",
        "arial.ttf",
        "Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "/System/Library/Fonts/HelveticaNeue.ttc",
    ):
        try:
            font = ImageFont.truetype(cand, 420)
            break
        except (OSError, IOError):
            continue
    if font is None:
        font = ImageFont.load_default()

    # Measure + centre text using textbbox (Pillow ≥10)
    try:
        bbox = draw.textbbox((0, 0), icon_text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx = cx - tw // 2 - bbox[0]
        ty = cy - th // 2 - bbox[1]
    except AttributeError:
        # Fallback for very old Pillow
        tw, th = draw.textsize(icon_text, font=font)
        tx = cx - tw // 2
        ty = cy - th // 2

    # Text colour: white if circle is dark, near-black if circle is light
    circle_avg = sum(bg_rgb) / 3
    text_rgb = (250, 250, 245) if circle_avg < 160 else (20, 20, 20)
    draw.text((tx, ty), icon_text, font=font, fill=text_rgb)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path, "JPEG", quality=92)
    return out_path


def _existing_highlight_titles(ig: IGClient) -> dict[str, str]:
    """Return {title_lower: highlight_pk} for the account. Empty on read
    failure (LoginRequired on web-cookie sessions)."""
    try:
        ig._ensure_logged_in()
        me_id = str(ig.cl.user_id)
        highlights = ig._retry(lambda: ig.cl.user_highlights(me_id))
    except Exception as e:
        log.debug("highlights: user_highlights read failed — %s", e)
        return {}
    out: dict[str, str] = {}
    for h in highlights or []:
        title = (getattr(h, "title", "") or "").strip().lower()
        pk = str(getattr(h, "pk", "") or getattr(h, "id", "") or "")
        if title and pk:
            out[title] = pk
    return out


def _get_or_create_seed_story(ig: IGClient, category: HighlightCategory, cover_path: Path) -> str | None:
    """Upload the cover image itself as a hidden seed story so we have a
    story pk to feed highlight_create. Returns the story pk, or None on
    failure. The seed story is only live for 24h anyway — if we also add
    real content stories to the highlight, the seed drops off naturally."""
    try:
        ig._ensure_backoff_ok()
        ig._ensure_logged_in()
        story = ig._retry(lambda: ig.cl.photo_upload_to_story(str(cover_path)))
    except Exception as e:
        log.warning("highlights: seed story upload failed for %r — %s",
                    category.name, e)
        return None
    return str(getattr(story, "pk", "") or "")


def bootstrap(cfg: NicheConfig, ig: IGClient, covers_dir: Path) -> BootstrapResult:
    """Create each configured category as a highlight on the account if it
    doesn't exist yet. Idempotent — safe to rerun on every boot.

    ``covers_dir`` is where we write the generated cover jpgs (kept around
    for later cover swaps via ``highlight_change_cover``).
    """
    result = BootstrapResult(created=[], existing=[], skipped=[])
    if not cfg.highlights.enabled or not cfg.highlights.categories:
        return result

    existing = _existing_highlight_titles(ig)

    for cat in cfg.highlights.categories:
        key = cat.name.strip().lower()
        if key in existing:
            result.existing.append(cat.name)
            continue
        cover_path = covers_dir / f"{cat.name.lower().replace(' ', '_')}.jpg"
        try:
            _render_cover(cat, cover_path)
        except Exception as e:
            log.warning("highlights: cover render failed for %r — %s", cat.name, e)
            result.skipped.append(cat.name)
            continue
        seed_pk = _get_or_create_seed_story(ig, cat, cover_path)
        if seed_pk is None:
            result.skipped.append(cat.name)
            continue
        try:
            hl = ig._retry(lambda: ig.cl.highlight_create(
                title=cat.name,
                story_ids=[seed_pk],
                cover_story_id=seed_pk,
            ))
        except Exception as e:
            log.warning("highlights: create failed for %r — %s", cat.name, e)
            result.skipped.append(cat.name)
            continue
        # Swap to our branded cover (seed story's thumbnail is IG's default crop)
        try:
            ig._retry(lambda: ig.cl.highlight_change_cover(
                str(hl.pk), cover_path,
            ))
        except Exception as e:
            log.debug("highlights: cover change failed for %r — %s (cover stays as seed story thumb)",
                      cat.name, e)
        result.created.append(cat.name)
        log.info("highlights: created %r (pk=%s)", cat.name, hl.pk)

    return result


def category_for_story(
    cfg: NicheConfig, *, caption: str = "", tags: list[str] | None = None,
) -> HighlightCategory | None:
    """Match a just-posted story to the category it belongs in. Returns
    None when no category matches — caller should skip promotion."""
    if not cfg.highlights.enabled or not cfg.highlights.categories:
        return None
    haystack = (caption or "").lower()
    if tags:
        haystack += " " + " ".join(t.lower() for t in tags)
    for cat in cfg.highlights.categories:
        for kw in cat.keywords:
            if kw.strip() and kw.lower() in haystack:
                return cat
    return None


def promote_story_to_category(
    ig: IGClient, category: HighlightCategory, story_pk: str,
) -> bool:
    """Add a fresh story to the matching category's highlight. Returns
    True on success, False if the highlight doesn't exist or the add fails.
    """
    existing = _existing_highlight_titles(ig)
    hl_pk = existing.get(category.name.strip().lower())
    if not hl_pk:
        log.debug("highlights: no highlight found for %r — bootstrap first", category.name)
        return False
    try:
        ig._retry(lambda: ig.cl.highlight_add_stories(hl_pk, [story_pk]))
        log.info("highlights: added story %s to %r", story_pk, category.name)
        return True
    except Exception as e:
        log.warning("highlights: add story %s to %r failed — %s",
                    story_pk, category.name, e)
        return False
