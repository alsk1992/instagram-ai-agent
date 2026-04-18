"""Shared HTML→PNG renderer (Playwright) used by quote_card and carousel."""
from __future__ import annotations

import base64
from pathlib import Path
from string import Template

from playwright.async_api import async_playwright

from instagram_ai_agent.core.config import FONTS_DIR, TEMPLATES_DIR

_BASE_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  width: ${width}px;
  height: ${height}px;
  background: ${bg};
  color: ${fg};
  font-family: '${body_font}', -apple-system, BlinkMacSystemFont, 'Helvetica Neue', Arial, sans-serif;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 80px;
  overflow: hidden;
}
"""


_FONT_EXTS = (".ttf", ".otf", ".woff2", ".woff")
_FONT_MIMES = {
    ".ttf": "font/ttf",
    ".otf": "font/otf",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
}


def _find_font_file(family: str) -> Path | None:
    """Best-match font file in data/fonts/ — matches by family name prefix."""
    if not FONTS_DIR.exists():
        return None
    # Normalise family ("Archivo Black" → "archivoblack")
    key = "".join(ch for ch in family.lower() if ch.isalnum())
    candidates = []
    for p in FONTS_DIR.iterdir():
        if p.suffix.lower() not in _FONT_EXTS:
            continue
        stem = "".join(ch for ch in p.stem.lower() if ch.isalnum())
        if key in stem or stem.startswith(key):
            candidates.append(p)
    if not candidates:
        return None
    # Prefer regular over bold/italic when multiple match
    regular = [c for c in candidates if "regular" in c.stem.lower() or c.stem.lower() == key]
    return (regular or candidates)[0]


def font_face_css(*families: str) -> str:
    """Emit @font-face blocks that inline every matching font as base64.

    Playwright renders HTML headlessly, so absolute file:// paths fail on some
    Chromium builds. Inlining the font bytes is the only portable option.
    """
    seen: set[str] = set()
    blocks: list[str] = []
    for family in families:
        if not family or family in seen:
            continue
        seen.add(family)
        font_path = _find_font_file(family)
        if not font_path:
            continue
        mime = _FONT_MIMES.get(font_path.suffix.lower(), "font/ttf")
        data = base64.b64encode(font_path.read_bytes()).decode("ascii")
        blocks.append(
            "@font-face {\n"
            f"  font-family: '{family}';\n"
            f"  src: url(data:{mime};base64,{data}) format('{_mime_to_format(mime)}');\n"
            "  font-weight: normal;\n"
            "  font-style: normal;\n"
            "}"
        )
    return "\n".join(blocks)


def _mime_to_format(mime: str) -> str:
    return {
        "font/ttf": "truetype",
        "font/otf": "opentype",
        "font/woff": "woff",
        "font/woff2": "woff2",
    }.get(mime, "truetype")


async def render_html_to_png(
    html: str,
    out_path: Path,
    *,
    width: int = 1080,
    height: int = 1350,
    deviceScaleFactor: int = 2,
) -> Path:
    async with async_playwright() as p:
        browser = await p.chromium.launch(args=["--no-sandbox"])
        ctx = await browser.new_context(
            viewport={"width": width, "height": height},
            device_scale_factor=deviceScaleFactor,
        )
        page = await ctx.new_page()
        await page.set_content(html, wait_until="networkidle")
        await page.wait_for_timeout(150)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        await page.screenshot(
            path=str(out_path),
            full_page=False,
            omit_background=False,
            type="jpeg",
            quality=92,
        )
        await ctx.close()
        await browser.close()
    return out_path


def load_template(folder: str, name: str = "default.html") -> str:
    path = TEMPLATES_DIR / folder / name
    if not path.exists():
        raise FileNotFoundError(f"Missing template: {path}")
    return path.read_text(encoding="utf-8")


def list_templates(folder: str) -> list[Path]:
    """Every ``.html`` template available in ``folder`` (alphabetical)."""
    p = TEMPLATES_DIR / folder
    if not p.exists():
        return []
    return sorted(p.glob("*.html"))


def pick_template(folder: str, *, variant: str | None = None, rng=None) -> tuple[str, str]:
    """Return ``(template_name, template_content)`` for ``folder``.

    * ``variant``: if given, must match a file by stem (``"default"`` or
      ``"default.html"``). Falls back to ``default.html`` when not found.
    * Otherwise picks at random across every template in the folder so the
      feed visually varies post-to-post.
    """
    import random as _random

    templates = list_templates(folder)
    if not templates:
        raise FileNotFoundError(f"No HTML templates in {folder!r}")

    if variant:
        wanted = variant if variant.endswith(".html") else f"{variant}.html"
        for t in templates:
            if t.name == wanted:
                return t.stem, t.read_text(encoding="utf-8")
        # Variant requested but not found — warn so a typo in niche.yaml
        # doesn't silently randomise forever.
        from instagram_ai_agent.core.logging_setup import get_logger
        get_logger(__name__).warning(
            "pick_template: variant %r missing in %s (available: %s) — falling back to random",
            variant, folder, [t.stem for t in templates],
        )

    rng = rng or _random
    chosen = rng.choice(templates)
    return chosen.stem, chosen.read_text(encoding="utf-8")


def base_css(
    *,
    width: int,
    height: int,
    bg: str,
    fg: str,
    body_font: str,
    heading_font: str | None = None,
) -> str:
    faces = font_face_css(body_font, heading_font or body_font)
    return faces + "\n" + Template(_BASE_CSS).substitute(
        width=width,
        height=height,
        bg=bg,
        fg=fg,
        body_font=body_font,
    )
