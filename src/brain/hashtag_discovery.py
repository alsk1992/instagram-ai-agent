"""Dynamic hashtag discovery — mine competitor top posts for tags worth stealing.

We already scrape competitor posts into ``competitor_posts``. This module
parses hashtags out of those captions, scores them by (a) appearance
frequency across competitors, (b) total like volume on posts that used
them, then proposes the top N as candidates for the ``growth`` pool.

Discovered tags land in ``state`` as pending suggestions — the user can
auto-merge via ``ig-agent hashtag-review`` or accept them manually.
"""
from __future__ import annotations

import re
from collections import defaultdict

from src.core import db
from src.core.config import NicheConfig
from src.core.logging_setup import get_logger

log = get_logger(__name__)

_TAG_RE = re.compile(r"#([A-Za-z0-9_]{2,32})")


def mine_from_competitors(cfg: NicheConfig, lookback_days: int = 21) -> list[dict]:
    """Return a list of {tag, count, likes_sum, sample_users} for competitor-used tags."""
    conn = db.get_conn()
    rows = conn.execute(
        """
        SELECT username, caption, likes FROM competitor_posts
        WHERE scraped_at >= datetime('now', ?)
        """,
        (f"-{int(lookback_days)} days",),
    ).fetchall()

    # Exclude tags we already use in any pool
    our_pool = {
        t.lstrip("#").lower()
        for t in (*cfg.hashtags.core, *cfg.hashtags.growth, *cfg.hashtags.long_tail)
    }

    count: dict[str, int] = defaultdict(int)
    likes_sum: dict[str, int] = defaultdict(int)
    users: dict[str, set[str]] = defaultdict(set)

    for r in rows:
        caption = (r["caption"] or "").lower()
        user = r["username"] or ""
        likes = int(r["likes"] or 0)
        seen_in_post: set[str] = set()
        for m in _TAG_RE.finditer(caption):
            tag = m.group(1).lower()
            if tag in our_pool or tag in seen_in_post:
                continue
            seen_in_post.add(tag)
            count[tag] += 1
            likes_sum[tag] += likes
            users[tag].add(user)

    suggestions = [
        {
            "tag": tag,
            "count": count[tag],
            "likes_sum": likes_sum[tag],
            "users": sorted(users[tag]),
            "score": count[tag] * 100 + likes_sum[tag],
        }
        for tag in count
    ]
    suggestions.sort(key=lambda s: s["score"], reverse=True)
    return suggestions


def persist_suggestions(cfg: NicheConfig, max_n: int = 30) -> list[dict]:
    """Compute and store the top N tag suggestions in state for review."""
    suggestions = mine_from_competitors(cfg)[:max_n]
    db.state_set_json("hashtag_suggestions", suggestions)
    if suggestions:
        log.info(
            "hashtag_discovery: %d suggestions (top: %s)",
            len(suggestions),
            ", ".join(s["tag"] for s in suggestions[:8]),
        )
    return suggestions


def approve_into_growth(cfg: NicheConfig, tags: list[str]) -> int:
    """Merge approved tags into cfg.hashtags.growth and persist to niche.yaml."""
    from src.core.config import save_niche

    current = {t.lower() for t in cfg.hashtags.growth}
    added: list[str] = []
    for tag in tags:
        norm = tag.lstrip("#").lower()
        if norm and norm not in current:
            cfg.hashtags.growth.append(norm)
            current.add(norm)
            added.append(norm)
    if added:
        save_niche(cfg)
    # Remove approved from pending
    pending = db.state_get_json("hashtag_suggestions", default=[]) or []
    remaining = [s for s in pending if s.get("tag") not in {t.lstrip("#").lower() for t in tags}]
    db.state_set_json("hashtag_suggestions", remaining)
    return len(added)


def run_once(cfg: NicheConfig) -> int:
    suggestions = persist_suggestions(cfg)
    return len(suggestions)
