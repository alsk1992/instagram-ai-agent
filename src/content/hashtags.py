"""Hashtag assembly — rotating pools, size-mixed, dedup vs recent posts."""
from __future__ import annotations

import random
from collections import Counter

from src.core import db
from src.core.config import NicheConfig


def build_hashtags(cfg: NicheConfig, *, seed: int | None = None) -> list[str]:
    """Pick hashtags_per_post tags mixing core/growth/long_tail.

    Strategy:
      - 40% core (niche foundation, always-on)
      - 40% growth (bigger pools where you can get discovered)
      - 20% long-tail (low-competition)
    Tags are shuffled; we avoid any tag used in the user's last 5 posts to
    keep the tag-set varied (IG penalises identical tag blocks).
    """
    rnd = random.Random(seed)
    n = cfg.hashtags.per_post
    n_core = max(1, round(n * 0.4))
    n_growth = max(1, round(n * 0.4))
    n_long = max(0, n - n_core - n_growth)

    recent = _recently_used()

    def pick(pool: list[str], k: int) -> list[str]:
        fresh = [t for t in pool if t not in recent]
        # If fresh < k, fall back to the full pool
        source = fresh if len(fresh) >= k else pool
        if not source:
            return []
        rnd.shuffle(source)
        return source[:k]

    chosen = (
        pick(list(cfg.hashtags.core), n_core)
        + pick(list(cfg.hashtags.growth), n_growth)
        + pick(list(cfg.hashtags.long_tail), n_long)
    )

    # De-duplicate while preserving order
    seen: set[str] = set()
    uniq: list[str] = []
    for tag in chosen:
        normalized = tag.lower().lstrip("#")
        if normalized in seen:
            continue
        seen.add(normalized)
        uniq.append(normalized)

    # Backfill from the merged pool if any sub-pool was short-stocked
    if len(uniq) < n:
        merged = (
            list(cfg.hashtags.core)
            + list(cfg.hashtags.growth)
            + list(cfg.hashtags.long_tail)
        )
        rnd.shuffle(merged)
        for tag in merged:
            normalized = tag.lower().lstrip("#")
            if normalized in seen:
                continue
            seen.add(normalized)
            uniq.append(normalized)
            if len(uniq) >= n:
                break

    return uniq[:n]


def format_hashtags(tags: list[str]) -> str:
    """Render tags with leading #, one per line for the caption tail."""
    return " ".join("#" + t for t in tags if t)


def _recently_used(lookback: int = 5) -> set[str]:
    """Tags used across the last N posted items. Returns empty set if DB not ready."""
    try:
        rows = db.content_list(status="posted", limit=lookback)
    except Exception:
        # DB not initialised yet — skip the "avoid recent tags" optimisation.
        return set()
    used: Counter[str] = Counter()
    for r in rows:
        for t in r.get("hashtags") or []:
            used[t.lower().lstrip("#")] += 1
    # Only suppress tags used in *every* recent post — otherwise we'd starve
    # the pool on pages with short rotations.
    return {t for t, c in used.items() if c >= max(1, lookback - 1)}
