"""Pick the next format to generate — feed vs story, then variant-within-pool."""
from __future__ import annotations

import os
import random
from collections import Counter

from instagram_ai_agent.core import db
from instagram_ai_agent.core.config import FEED_FORMATS, STORY_FORMATS, NicheConfig
from instagram_ai_agent.core.logging_setup import get_logger

log = get_logger(__name__)

# Formats with hard external-API requirements. When the API keys aren't
# present, the format would consume a generation slot just to fail — so
# we zero its weight before the picker runs.
_FORMAT_REQUIRES_ENV: dict[str, tuple[str, ...]] = {
    # reel_stock needs Pexels OR Pixabay to fetch footage. Either key
    # alone is sufficient; we only skip when BOTH are missing.
    "reel_stock": ("PEXELS_API_KEY", "PIXABAY_API_KEY"),
}


def _format_is_runnable(format_name: str) -> bool:
    """Cheap env-based readiness check. Returns True when the format
    either has no API requirements OR at least one required key is set."""
    required = _FORMAT_REQUIRES_ENV.get(format_name)
    if not required:
        return True
    return any(os.environ.get(k) for k in required)


def _prune_unrunnable(weights: dict[str, float]) -> dict[str, float]:
    """Strip formats that can't run under the current env. If that
    empties the pool, return the original weights — caller will fall
    back deterministically."""
    pruned = {f: w for f, w in weights.items() if _format_is_runnable(f)}
    if not pruned and weights:
        log.warning(
            "format_picker: all formats blocked by missing env vars — "
            "generation will likely fail. Check PEXELS_API_KEY / PIXABAY_API_KEY."
        )
        return weights
    dropped = [f for f in weights if f not in pruned and weights[f] > 0]
    if dropped:
        log.info("format_picker: skipping unrunnable formats %s (missing env)", dropped)
    return pruned


def pick_next(cfg: NicheConfig, *, kind: str | None = None) -> str:
    """Weighted random pick that corrects for current queue imbalance.

    ``kind`` restricts to "feed" or "story". When None, we choose the pool
    based on the relative demand ratio between the two schedules.
    """
    if kind is None:
        kind = _pick_pool(cfg)

    target: dict[str, float]
    if kind == "story":
        target = cfg.stories.normalized()
        pending_filter = STORY_FORMATS
    else:
        target = cfg.formats.normalized()
        pending_filter = FEED_FORMATS

    items = db.content_list(status=None, limit=200)
    pending = [i for i in items if i["status"] in ("pending_review", "approved", "pending_gen")]
    counts = Counter(i["format"] for i in pending if i["format"] in pending_filter)
    total_pending = max(1, sum(counts.values()))
    current_share = {f: counts.get(f, 0) / total_pending for f in target}

    effective = {
        f: target[f] / max(0.05, current_share.get(f, 0.0) + 0.05)
        for f in target
        if target[f] > 0
    }
    # Remove formats whose required API keys aren't set so we don't
    # burn a full generation cycle on a guaranteed failure.
    effective = _prune_unrunnable(effective)
    if not effective:
        # User disabled every format in this pool — fall back deterministically
        return "meme" if kind == "feed" else "story_quote"

    total_w = sum(effective.values())
    r = random.uniform(0, total_w)
    acc = 0.0
    for f, w in effective.items():
        acc += w
        if r <= acc:
            return f
    return next(iter(effective))


def _pick_pool(cfg: NicheConfig) -> str:
    """Decide whether to work on a feed post or a story next.

    Ratio of caps × (1 - progress) approximates "how behind is each pool right now".
    """
    feed_cap = max(1, cfg.schedule.posts_per_day)
    story_cap = max(0, cfg.schedule.stories_per_day)
    if story_cap == 0:
        return "feed"
    if feed_cap == 0:
        return "story"

    feed_used = db.action_count_today("post")
    story_used = db.action_count_today("story_post")
    feed_gap = max(0, feed_cap - feed_used)
    story_gap = max(0, story_cap - story_used)

    # Small bias toward stories since they're higher-volume by design
    weights = {
        "feed": feed_gap,
        "story": story_gap * 1.1,
    }
    total = weights["feed"] + weights["story"]
    if total == 0:
        return "feed"
    r = random.uniform(0, total)
    return "feed" if r < weights["feed"] else "story"
