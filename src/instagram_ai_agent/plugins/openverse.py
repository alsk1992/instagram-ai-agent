"""Openverse free image search — 800M+ CC / public-domain assets, no API key.

Provides a drop-in alternative when Pexels / Pixabay keys aren't configured
or when we want a broader pool. Every image returned has a clear CC/PD
licence attached via the ``license`` field, so downstream code can respect
commercial-only filtering when ``cfg.commercial=True``.

Example:
    from instagram_ai_agent.plugins import openverse
    img = await openverse.search("home calisthenics pullup")
    if img:
        # img = {"url": "...", "license": "cc0", "title": "..."}

2026 status: still open. Endpoint: ``https://api.openverse.org/v1/images/``
"""
from __future__ import annotations

from pathlib import Path

import httpx

from instagram_ai_agent.core.logging_setup import get_logger

log = get_logger(__name__)


ENDPOINT = "https://api.openverse.org/v1/images/"


# Licences that are safe for commercial IG pages. Exclude anything with
# the NC (non-commercial) tag per Creative Commons.
COMMERCIAL_SAFE_LICENSES = {
    "cc0", "pdm",            # public domain
    "by", "by-sa",           # free for commercial with attribution (handled by watermark/caption credit)
    "sampling+",
}


async def search(
    query: str,
    *,
    commercial_only: bool = True,
    page_size: int = 10,
    orientation: str | None = None,
) -> dict | None:
    """Return the top image match for ``query`` or None.

    ``commercial_only=True`` filters out NC-only licences. ``orientation``
    can be ``"tall"`` (portrait) | ``"wide"`` (landscape) | ``"square"``.
    """
    params: dict[str, str | int] = {"q": query, "page_size": page_size}
    if orientation in ("tall", "wide", "square"):
        params["aspect_ratio"] = orientation
    if commercial_only:
        params["license_type"] = "commercial"

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(20.0),
            headers={"User-Agent": "ig-agent/0.2 (+openverse)"},
        ) as client:
            r = await client.get(ENDPOINT, params=params)
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        log.warning("openverse search failed (q=%r): %s", query, e)
        return None

    results = data.get("results") or []
    for hit in results:
        url = hit.get("url") or ""
        lic = (hit.get("license") or "").lower()
        # Skip images without a direct URL or an unknown licence.
        if not url:
            continue
        if commercial_only and lic not in COMMERCIAL_SAFE_LICENSES:
            continue
        return {
            "url": url,
            "thumbnail": hit.get("thumbnail") or url,
            "license": lic,
            "license_url": hit.get("license_url") or "",
            "title": (hit.get("title") or "").strip(),
            "creator": (hit.get("creator") or "").strip(),
            "source": hit.get("source") or "openverse",
        }
    log.debug("openverse: no commercial-safe match for %r in %d results", query, len(results))
    return None


async def download(image: dict, out_path: Path) -> Path | None:
    """Download the image at ``image['url']`` to ``out_path``. Returns the
    path on success, None otherwise."""
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            headers={"User-Agent": "ig-agent/0.2 (+openverse)"},
        ) as client:
            r = await client.get(image["url"], follow_redirects=True)
            r.raise_for_status()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(r.content)
    except Exception as e:
        log.warning("openverse download failed (%s): %s", image.get("url"), e)
        return None
    return out_path


def attribution_line(image: dict) -> str:
    """Build a CC-By-compliant credit line (e.g. for the caption's hashtag
    block). Returns empty string for CC0 / Public Domain images."""
    lic = (image.get("license") or "").lower()
    if lic in ("cc0", "pdm"):
        return ""
    creator = image.get("creator") or ""
    title = image.get("title") or "image"
    lic_url = image.get("license_url") or ""
    line = f"📷 {title}"
    if creator:
        line += f" by {creator}"
    line += f" (CC {lic.upper()}"
    if lic_url:
        line += f" {lic_url}"
    line += ")"
    return line
