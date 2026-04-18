"""Music bed fetcher for reels + story videos.

Tries sources in the order configured by ``MusicConfig.sources``:

  1. ``local``         — any audio file under ``data/music/`` (user-populated
                         CC0 library, typically seeded from SoundSafari/
                         CC0-1.0-Music).
  2. ``pixabay``       — Pixabay Music API (commercial-OK, no attribution).
                         The public Pixabay API exposes images + videos
                         officially; a separate ``/api/music/`` endpoint is
                         used in practice. We call it defensively and fall
                         through on failure.
  3. ``freesound``     — Freesound.org API filtered to
                         ``license:"Creative Commons 0"`` so every downloaded
                         clip is commercial-safe without attribution.
                         Requires ``FREESOUND_API_KEY``.
  4. ``stable_audio``  — Generative music via Stable Audio Open Small
                         (Stability AI Community Licence). Requires
                         ``pip install .[stable-audio]`` AND
                         ``music.sao_license_acknowledged=True``. Runs
                         locally — no API calls, no external dependency.

Every fetched clip lands in ``data/music/cache/`` and is reused across posts.
Graceful no-op: if nothing resolves, return ``None`` — the caller continues
without a music bed.
"""
from __future__ import annotations

import hashlib
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from instagram_ai_agent.core.config import MUSIC_CACHE_DIR, MUSIC_DIR, MusicConfig, NicheConfig
from instagram_ai_agent.core.logging_setup import get_logger

log = get_logger(__name__)

_AUDIO_EXTS = {".mp3", ".m4a", ".wav", ".ogg", ".flac", ".aac", ".opus"}


@dataclass(frozen=True)
class MusicBed:
    path: Path
    title: str
    source: str
    license: str  # "CC0", "pixabay", "CC-BY", etc.
    duration_s: float | None = None


# ───── Source 1: local cache (always first, never fails) ─────
_LICENSE_FOLDER_HINTS = {
    "cc0": "CC0",
    "public-domain": "CC0",
    "freepd": "CC0",
    "soundsafari": "CC0",
    "pixabay": "pixabay",
    "cc-by": "CC-BY",
    "attribution": "CC-BY",
}


def _detect_local_license(path: Path) -> str:
    """Honest license detection — sidecar file > folder hint > ``user-declared``.

    We deliberately do NOT stamp "CC0" on content whose origin we cannot
    verify. Commercial pipelines reading this field shouldn't trust it
    beyond what the user told us.
    """
    # 1. Sidecar file: foo.mp3 + foo.mp3.license (or foo.license)
    for sidecar_name in (path.name + ".license", path.stem + ".license"):
        sidecar = path.with_name(sidecar_name)
        if sidecar.exists():
            try:
                first_line = sidecar.read_text(encoding="utf-8").strip().splitlines()
                if first_line:
                    return first_line[0].strip()[:40]
            except OSError:
                pass
    # 2. Parent-folder hint — walk up until we leave MUSIC_DIR
    for parent in path.parents:
        if parent == MUSIC_DIR.parent:
            break
        hint = _LICENSE_FOLDER_HINTS.get(parent.name.lower())
        if hint:
            return hint
    # 3. Unknown — honest label so no downstream component misreports it
    return "user-declared"


def _local_pick(query: str) -> MusicBed | None:
    if not MUSIC_DIR.exists():
        return None
    files = [p for p in MUSIC_DIR.rglob("*") if p.suffix.lower() in _AUDIO_EXTS]
    if not files:
        return None

    tokens = _tokenise(query)
    scored: list[tuple[int, Path]] = []
    for p in files:
        stem = p.stem.lower()
        score = sum(1 for t in tokens if t in stem)
        # Tag by parent folder too (e.g. data/music/lofi/xxx.mp3 scores for "lofi" queries)
        parent = p.parent.name.lower()
        score += sum(2 for t in tokens if t == parent)
        scored.append((score, p))

    scored.sort(key=lambda x: (x[0], random.random()), reverse=True)
    best = scored[0][1]
    return MusicBed(
        path=best,
        title=best.stem,
        source="local",
        license=_detect_local_license(best),
    )


# ───── Source 2: Pixabay Music API (commercial-OK, no attribution) ─────
async def _pixabay_fetch(query: str, *, min_duration: int = 5) -> MusicBed | None:
    key = os.environ.get("PIXABAY_API_KEY")
    if not key:
        return None
    # Pixabay exposes audio via the same API host; the documented endpoint
    # is the primary /api/ path with a type switch. Some accounts only have
    # image access. We try audio first, then bail on 400/403.
    endpoints = [
        "https://pixabay.com/api/audio/",
        "https://pixabay.com/api/music/",
    ]
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        for ep in endpoints:
            try:
                r = await client.get(
                    ep,
                    params={
                        "key": key,
                        "q": query,
                        "per_page": 20,
                        "safesearch": "true",
                    },
                )
                if r.status_code >= 400:
                    continue
                data = r.json()
                hits = data.get("hits") or []
                if not hits:
                    continue
                # Filter to reasonable durations
                pool = [h for h in hits if (h.get("duration") or 0) >= min_duration]
                if not pool:
                    pool = hits
                pick = random.choice(pool[:10])
                url = pick.get("audio") or pick.get("url") or pick.get("download_url")
                if not url:
                    continue
                dest = _cache_path(url, suffix=".mp3")
                await _download(client, url, dest)
                return MusicBed(
                    path=dest,
                    title=str(pick.get("title") or pick.get("tags") or "pixabay track"),
                    source="pixabay",
                    license="pixabay",
                    duration_s=float(pick.get("duration") or 0) or None,
                )
            except Exception as e:
                log.debug("pixabay %s failed: %s", ep, e)
                continue
    return None


# ───── Source 3: Freesound CC0 filter ─────
async def _freesound_fetch(query: str, *, min_duration: int = 8) -> MusicBed | None:
    key = os.environ.get("FREESOUND_API_KEY")
    if not key:
        return None
    search_url = "https://freesound.org/apiv2/search/text/"
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        try:
            r = await client.get(
                search_url,
                params={
                    "query": query,
                    "filter": f'license:"Creative Commons 0" duration:[{min_duration} TO 120]',
                    "sort": "downloads_desc",
                    "page_size": 15,
                    "fields": "id,name,previews,license,duration",
                    "token": key,
                },
            )
            r.raise_for_status()
            data = r.json()
            results = data.get("results") or []
            if not results:
                return None
            pick = random.choice(results[:8])
            previews = pick.get("previews") or {}
            # Preview HQ mp3 is commercial-OK for CC0 matches
            url = previews.get("preview-hq-mp3") or previews.get("preview-lq-mp3")
            if not url:
                return None
            dest = _cache_path(url, suffix=".mp3")
            await _download(client, url, dest)
            return MusicBed(
                path=dest,
                title=str(pick.get("name") or "freesound track"),
                source="freesound",
                license="CC0",
                duration_s=float(pick.get("duration") or 0) or None,
            )
        except Exception as e:
            log.debug("freesound failed: %s", e)
            return None


# ───── Source 4: Stable Audio Open Small (generative, commercial-safe) ─────
async def _stable_audio_fetch(cfg: NicheConfig, *, scene_context: str = "") -> MusicBed | None:
    """Synthesize a music bed via Stable Audio Open Small.

    Silently returns None when:
      * ``sao_enabled=False`` (user hasn't opted in)
      * the optional extra isn't installed (``stable-audio-tools`` + torch)
    so the source slot is just a no-op and the chain falls through to the
    next configured source. The licence acknowledgement gate is enforced
    at config-load time (NicheConfig._sao_license_gate) so by the time we
    reach here, the user has already opted in explicitly."""
    mc = cfg.music
    if not mc.sao_enabled:
        return None

    # Lazy import so the base install doesn't need torch
    try:
        from instagram_ai_agent.plugins import stable_audio as sao
    except Exception as e:
        log.debug("stable_audio module import failed: %s", e)
        return None

    if not sao.available():
        log.info(
            "music: stable_audio source configured but extra not installed. "
            "Run `pip install '.[stable-audio]'` to enable it."
        )
        return None

    try:
        prompt = await sao.build_prompt(cfg, scene_context=scene_context)
        target_s = float(mc.sao_duration_s)
        path = await sao.generate(prompt, target_s, cfg)
    except Exception as e:
        log.warning("stable_audio synthesis failed: %s", e)
        return None

    return MusicBed(
        path=path,
        title=f"stable-audio: {prompt[:60]}",
        source="stable_audio",
        license="Stability AI Community Licence",
        duration_s=target_s,
    )


# ───── Orchestrator ─────
async def find_music(cfg: NicheConfig, *, query_override: str | None = None) -> MusicBed | None:
    mc: MusicConfig = cfg.music
    if not mc.enabled:
        return None

    query = query_override or mc.query_template.format(niche=cfg.niche)
    # Attach one of the configured genres to widen the recall space
    if mc.genres:
        query = f"{query} {random.choice(mc.genres)}"

    for src in mc.sources:
        try:
            if src == "local":
                bed = _local_pick(query)
            elif src == "pixabay":
                bed = await _pixabay_fetch(query)
            elif src == "freesound":
                bed = await _freesound_fetch(query)
            elif src == "stable_audio":
                # We deliberately pass "" as scene_context — the SAO
                # prompt-builder already has the niche + voice from cfg
                # and the `query` variable here is just niche + genre
                # (duplicate signal). Callers with real scene-level
                # context should call stable_audio.generate() directly.
                bed = await _stable_audio_fetch(cfg, scene_context="")
            else:
                log.debug("music: unknown source %r", src)
                continue
        except Exception as e:
            log.warning("music source %s errored: %s", src, e)
            continue
        if bed is not None:
            log.info("music: picked %s from %s (%s)", bed.path.name, bed.source, bed.license)
            return bed
    log.info("music: no bed found — reel will use voiceover only (query=%r)", query)
    return None


# ───── utils ─────
def _cache_path(url: str, suffix: str = ".mp3") -> Path:
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return MUSIC_CACHE_DIR / f"{h}{suffix}"


async def _download(client: httpx.AsyncClient, url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 10_000:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    async with client.stream("GET", url) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            async for chunk in r.aiter_bytes():
                f.write(chunk)
    return dest


_TOKEN_RE = re.compile(r"[A-Za-z0-9]{3,}")


def _tokenise(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


# ───── metadata probe ─────
def probe_duration(path: Path) -> float:
    import subprocess

    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(out.stdout.strip() or 0)
    except Exception:
        return 0.0
