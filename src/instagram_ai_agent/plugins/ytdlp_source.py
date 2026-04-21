"""Anonymous public-video harvester via yt-dlp.

No API keys, no accounts. Downloads public YouTube Shorts / TikToks /
Reddit videos matching a niche keyword, returns the raw mp4 paths so the
transform pipeline can strip-metadata / crop / pitch-shift them into
originals-in-IG's-eyes.

Legal posture: all sources return publicly-available content. We add
attribution to the original creator in the caption layer (handled by
the transform generator), strip platform watermarks before upload, and
swap the audio track for our own TTS voiceover — which moves the result
from "plain rip" to "transformative commentary" in copyright terms.

yt-dlp handles the signature dance with YouTube/TikTok/Reddit; we just
hand it a URL or search query and it returns the best-quality vertical
mp4 available. When a platform changes extraction signatures (happens
every ~2 months for TikTok, monthly for YouTube), users just
``pip install --upgrade yt-dlp`` to pick up the fix.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from instagram_ai_agent.content.generators.base import staging_path
from instagram_ai_agent.core.logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class HarvestedClip:
    path: Path
    source_platform: str  # "youtube" | "tiktok" | "reddit"
    source_id: str
    source_url: str
    uploader: str  # @handle / /u/username
    title: str
    duration_s: float
    width: int
    height: int


def _ytdlp_available() -> bool:
    """yt-dlp might be installed via pip (preferred) or system binary."""
    return shutil.which("yt-dlp") is not None


async def _run_ytdlp(args: list[str], timeout: float = 120.0) -> subprocess.CompletedProcess:
    """Run yt-dlp in a thread so we don't block the asyncio loop."""
    def _sync():
        return subprocess.run(
            ["yt-dlp", *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    return await asyncio.to_thread(_sync)


async def search_youtube_shorts(keyword: str, limit: int = 8) -> list[dict[str, Any]]:
    """Enumerate trending YouTube Shorts matching a keyword. Returns a list
    of dicts with keys: id, title, uploader, duration, url. No download yet —
    caller picks which to fetch.

    Uses yt-dlp's `ytsearch` prefix which emits JSON metadata per match
    without pulling the actual video file.
    """
    if not _ytdlp_available():
        log.warning("ytdlp_source: yt-dlp not installed — run `pip install yt-dlp`")
        return []
    query = f"ytsearch{limit}:{keyword} shorts"
    res = await _run_ytdlp([
        "--flat-playlist",
        "--dump-single-json",
        "--match-filter", "duration<=90 & duration>=5",
        "--no-warnings",
        query,
    ], timeout=60)
    if res.returncode != 0:
        log.warning("ytdlp search failed: %s", (res.stderr or "")[:300])
        return []
    try:
        data = json.loads(res.stdout)
    except json.JSONDecodeError:
        return []
    entries = data.get("entries") or []
    out: list[dict[str, Any]] = []
    for e in entries:
        if not e or not e.get("id"):
            continue
        out.append({
            "id": e["id"],
            "title": e.get("title") or "",
            "uploader": e.get("uploader") or e.get("channel") or "",
            "duration": e.get("duration") or 0,
            "url": e.get("url") or f"https://www.youtube.com/shorts/{e['id']}",
        })
    return out


async def download_clip(url: str, *, max_height: int = 1920) -> HarvestedClip | None:
    """Download a single public video to the staging directory. Returns None on
    failure (private video, geo-restricted, platform signature broken, etc)."""
    if not _ytdlp_available():
        return None

    # Hash the URL to get a stable filename — lets us dedup across runs.
    h = hashlib.sha1(url.encode()).hexdigest()[:12]
    out_base = staging_path(f"src_{h}", "")
    # Template must resolve to a .mp4 — yt-dlp picks the extension
    out_tmpl = str(out_base.with_suffix("")) + ".%(ext)s"

    # Ask for the best vertical mp4 ≤ max_height. Merge audio+video if
    # they're separate streams. No playlist expansion.
    res = await _run_ytdlp([
        "--format", f"bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]/best[height<={max_height}][ext=mp4]/best",
        "--merge-output-format", "mp4",
        "--no-playlist",
        "--no-warnings",
        "--write-info-json",
        "--output", out_tmpl,
        url,
    ], timeout=180)

    if res.returncode != 0:
        log.warning("ytdlp download %s failed: %s", url, (res.stderr or "")[:300])
        return None

    # Find the resulting mp4
    mp4 = out_base.with_suffix(".mp4")
    if not mp4.exists():
        # yt-dlp sometimes outputs a different ext or suffix — scan
        for cand in out_base.parent.glob(f"src_{h}.*"):
            if cand.suffix.lower() in (".mp4", ".mov", ".webm", ".mkv"):
                mp4 = cand
                break
    if not mp4.exists():
        log.warning("ytdlp download %s produced no mp4", url)
        return None

    # Load metadata written alongside
    info_path = out_base.with_suffix(".info.json")
    meta: dict[str, Any] = {}
    if info_path.exists():
        try:
            meta = json.loads(info_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Detect source platform from URL
    platform = "unknown"
    if "youtube.com" in url or "youtu.be" in url:
        platform = "youtube"
    elif "tiktok.com" in url:
        platform = "tiktok"
    elif "reddit.com" in url or "redd.it" in url:
        platform = "reddit"

    uploader = (
        meta.get("uploader")
        or meta.get("channel")
        or meta.get("creator")
        or ""
    )
    if platform == "tiktok" and uploader and not uploader.startswith("@"):
        uploader = "@" + uploader

    return HarvestedClip(
        path=mp4,
        source_platform=platform,
        source_id=str(meta.get("id") or h),
        source_url=url,
        uploader=uploader,
        title=(meta.get("title") or "")[:300],
        duration_s=float(meta.get("duration") or 0),
        width=int(meta.get("width") or 0),
        height=int(meta.get("height") or 0),
    )


async def harvest_youtube_shorts(keyword: str, *, want: int = 1) -> list[HarvestedClip]:
    """Search YouTube Shorts by keyword, download the top ``want`` candidates.
    High-level convenience wrapper — the two lower-level functions let you
    scan metadata cheaply before committing to downloads."""
    candidates = await search_youtube_shorts(keyword, limit=max(8, want * 3))
    out: list[HarvestedClip] = []
    for c in candidates:
        if len(out) >= want:
            break
        clip = await download_clip(c["url"])
        if clip is None:
            continue
        # Only keep vertical-ish clips (aspect ≤ 1) so we don't import
        # landscape footage that'd letterbox ugly after composition.
        if clip.height > 0 and clip.width / max(clip.height, 1) > 1.05:
            log.debug("skipping landscape %s (%sx%s)", clip.source_url, clip.width, clip.height)
            continue
        out.append(clip)
    return out
