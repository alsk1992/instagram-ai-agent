"""Behavioural anti-detection layer.

instagrapi handles the transport (auth, endpoints, upload protocol).
This module handles the BEHAVIOUR layer — the things IG's 2026 ML
detectors look for beyond "is this a valid API call".

Bundled helpers:
  * ``pre_post_scroll(cl)`` — view 3-5 feed items + touch a story
    before posting, so the post looks like part of a normal session
    instead of a cold API write.
  * ``post_cooldown_ok() / stamp_post()`` — enforce a 30-90min silent
    window after each post before any other write action.
  * ``typing_delay(text)`` — sleep for a length-proportional duration
    before submitting a comment / DM so the timing looks human.
  * ``captions_too_similar(caption, recent)`` — Levenshtein-style
    near-duplicate guard on our own recent captions.
  * ``validate_aspect_ratio(media_path, format)`` — pre-flight check
    against IG's accepted dimensions so uploads aren't silently re-
    compressed + downranked.
  * ``should_rotate_client(cl_age_s)`` — advisory flag for recycling
    the instagrapi Client every 2-4 hours to reset the TCP connection
    pattern.
"""
from __future__ import annotations

import random
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from instagram_ai_agent.core import db
from instagram_ai_agent.core.logging_setup import get_logger

log = get_logger(__name__)


# ─── Pre-post scroll ──────────────────────────────────────────────
def pre_post_scroll(cl: Any, *, min_items: int = 3, max_items: int = 5) -> int:
    """Mimic a human opening IG before posting: fetch the timeline,
    mark a few posts as seen, maybe touch a story tray. Returns the
    number of items touched (mostly for logging).

    Silent on every failure — this is a cosmetic mimic, not a required
    step, and we mustn't block a post because a read failed."""
    n_touched = 0
    try:
        # instagrapi's Client exposes private_request directly; use the
        # public-facing helper if available.
        if hasattr(cl, "get_timeline_feed"):
            feed = cl.get_timeline_feed()
            time.sleep(random.uniform(2.0, 4.5))
            # feed may be a list[Media] or a raw dict — handle both
            items = feed if isinstance(feed, list) else feed.get("feed_items", []) if isinstance(feed, dict) else []
            sample = items[: random.randint(min_items, max_items)]
            for item in sample:
                # Humans read for 2-8s, not 0.1s
                time.sleep(random.uniform(2.0, 8.0))
                pk = getattr(item, "pk", None) or (item.get("pk") if isinstance(item, dict) else None)
                if pk and hasattr(cl, "media_seen"):
                    try:
                        cl.media_seen([pk])
                    except Exception as _seen_err:
                        log.debug("human_mimic: media_seen(%s) failed — non-fatal: %s",
                                  pk, _seen_err)
                n_touched += 1
    except Exception as e:
        log.debug("pre_post_scroll: %s (non-fatal)", e)

    # Occasionally peek at the story tray — humans do this ~1 in 3 sessions
    if random.random() < 0.33:
        try:
            if hasattr(cl, "user_story_feed_raw"):
                cl.user_story_feed_raw()
            elif hasattr(cl, "get_timeline_feed"):
                pass  # no-op if story API isn't exposed in this instagrapi build
            time.sleep(random.uniform(1.5, 3.0))
        except Exception as _story_err:
            log.debug("human_mimic: story-tray peek failed — non-fatal: %s", _story_err)
    return n_touched


# ─── Post cooldown ────────────────────────────────────────────────
_POST_COOLDOWN_MIN_S = 30 * 60  # 30 minutes
_POST_COOLDOWN_MAX_S = 90 * 60  # 90 minutes
_LAST_POST_KEY = "last_post_at_s"   # state K/V — unix seconds


def stamp_post(now_s: float | None = None) -> None:
    """Record that we just posted. Future write actions check this to
    enforce the mandatory silent window."""
    ts = now_s if now_s is not None else time.time()
    db.state_set(_LAST_POST_KEY, str(int(ts)))


def post_cooldown_remaining_s(now_s: float | None = None) -> float:
    """Seconds of mandatory silence left. Returns 0 when clear."""
    raw = db.state_get(_LAST_POST_KEY)
    if not raw:
        return 0.0
    try:
        last = float(raw)
    except ValueError:
        return 0.0
    now = now_s if now_s is not None else time.time()
    elapsed = now - last
    # Pick a deterministic required silence per post (based on the last
    # post's timestamp) so repeated checks in the same window return
    # the same threshold.
    rng = random.Random(int(last))
    required = rng.uniform(_POST_COOLDOWN_MIN_S, _POST_COOLDOWN_MAX_S)
    return max(0.0, required - elapsed)


def post_cooldown_ok() -> bool:
    """True when we're clear to perform another write action."""
    return post_cooldown_remaining_s() <= 0


# ─── Typing delay ─────────────────────────────────────────────────
def typing_delay_s(text: str, *, floor: float = 1.5, ceiling: float = 18.0) -> float:
    """Length-proportional pre-action sleep. Rough model: 3-5 chars/sec
    human typing (slower than average — accounts for think-time). Adds
    a small random jitter so consecutive calls don't look scripted."""
    if not text:
        return random.uniform(0.5, 1.2)
    base = len(text) / random.uniform(3.0, 5.0)
    jitter = random.uniform(-0.8, 2.5)
    return max(floor, min(ceiling, base + jitter))


def sleep_typing(text: str) -> float:
    """Call before submitting a comment/DM. Returns the seconds slept."""
    t = typing_delay_s(text)
    time.sleep(t)
    return t


# ─── Caption entropy check ────────────────────────────────────────
_WHITESPACE = re.compile(r"\s+")


def _normalize_caption(c: str) -> str:
    c = _WHITESPACE.sub(" ", (c or "").lower().strip())
    # Strip trailing hashtag block — it's noise for similarity
    if "\n\n#" in c:
        c = c.split("\n\n#", 1)[0].strip()
    return c


def _similarity(a: str, b: str) -> float:
    """Ratio of matching chars via difflib. 1.0 = identical.
    We use SequenceMatcher (stdlib) to keep this stdlib-only."""
    if not a or not b:
        return 0.0
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def captions_too_similar(caption: str, recent: list[str], *, threshold: float = 0.85) -> bool:
    """True when ``caption`` is ≥ ``threshold`` similar to ANY recent
    caption. 0.85 catches "same template, slightly different names"
    without rejecting natural niche-word overlap."""
    target = _normalize_caption(caption)
    if len(target) < 20:
        return False   # too short to call a collision meaningfully
    for c in recent:
        other = _normalize_caption(c)
        if not other or abs(len(other) - len(target)) > max(len(other), len(target)) * 0.5:
            continue
        if _similarity(target, other) >= threshold:
            return True
    return False


# ─── Aspect ratio pre-flight check ────────────────────────────────
# IG's accepted ratios + tolerances (2026):
#   feed portrait: 1080x1350  (0.8)
#   feed square:   1080x1080  (1.0)
#   feed landscape:1080x566   (1.91)  — still allowed but downranked
#   reel / story:  1080x1920  (0.5625) — 9:16
_ALLOWED_RATIOS: dict[str, tuple[tuple[float, float], ...]] = {
    "feed":   ((0.78, 0.82), (0.98, 1.02), (1.89, 1.93)),
    "reel":   ((0.55, 0.57),),
    "story":  ((0.55, 0.57),),
}


def _probe_dimensions(path: Path) -> tuple[int, int] | None:
    """(width, height) for an image or video. Uses PIL for images,
    ffprobe for videos. Returns None on any failure."""
    suffix = path.suffix.lower()
    if suffix in (".mp4", ".mov", ".m4v", ".webm"):
        try:
            r = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-select_streams", "v:0",
                    "-show_entries", "stream=width,height",
                    "-of", "csv=s=,:p=0",
                    str(path),
                ],
                capture_output=True, text=True, check=True, timeout=10,
            )
            w, h = (int(x) for x in r.stdout.strip().split(","))
            return w, h
        except Exception:
            return None
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size   # (width, height)
    except Exception:
        return None


def validate_aspect_ratio(media_path: str | Path, *, kind: str = "feed") -> bool:
    """True when the media matches an IG-compliant ratio for the given
    kind ("feed" | "reel" | "story"). Non-fatal — on probe failure we
    return True (don't block upload); caller logs.
    """
    p = Path(media_path)
    if not p.exists():
        return True
    dims = _probe_dimensions(p)
    if dims is None:
        return True
    w, h = dims
    if h == 0:
        return False
    ratio = w / h
    tolerances = _ALLOWED_RATIOS.get(kind, _ALLOWED_RATIOS["feed"])
    return any(lo <= ratio <= hi for lo, hi in tolerances)


# ─── Client recycling ────────────────────────────────────────────
_CLIENT_MAX_AGE_MIN_S = 2 * 3600   # 2 hours
_CLIENT_MAX_AGE_MAX_S = 4 * 3600   # 4 hours


def should_rotate_client(client_age_s: float, *, seed_ts: float | None = None) -> bool:
    """Return True when the instagrapi Client has been alive long enough
    that recreating it would reset the TCP connection pool + session
    idiom. Randomised per-process so re-roll windows don't all fall on
    the same minute across accounts."""
    seed = int(seed_ts if seed_ts is not None else _process_start_s())
    rng = random.Random(seed)
    threshold = rng.uniform(_CLIENT_MAX_AGE_MIN_S, _CLIENT_MAX_AGE_MAX_S)
    return client_age_s >= threshold


_PROCESS_START: float | None = None


def _process_start_s() -> float:
    """Cache process start time for consistent client-rotation seeding."""
    global _PROCESS_START
    if _PROCESS_START is None:
        _PROCESS_START = time.time()
    return _PROCESS_START
