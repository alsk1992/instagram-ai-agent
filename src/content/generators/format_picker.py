"""Pick the next format to generate — feed vs story, then variant-within-pool."""
from __future__ import annotations

import random
from collections import Counter

from src.core import db
from src.core.config import FEED_FORMATS, STORY_FORMATS, NicheConfig


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
