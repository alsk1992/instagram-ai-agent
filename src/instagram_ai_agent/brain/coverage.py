"""Sub-topic coverage rotator — keeps the feed spread across all niche angles.

Every enqueued post logs which ``sub_topic`` it represented. At generation
time we pick the sub-topic whose last coverage is stalest, with a small
random bias so the feed doesn't become mechanical.

Store format (in state K/V as JSON):
    {"topic": "iso-timestamp", ...}
"""
from __future__ import annotations

import random

from instagram_ai_agent.core import db
from instagram_ai_agent.core.config import NicheConfig

_KEY = "sub_topic_coverage"


def _load() -> dict[str, str]:
    return db.state_get_json(_KEY, default={}) or {}


def _save(d: dict[str, str]) -> None:
    db.state_set_json(_KEY, d)


def pick_sub_topic(cfg: NicheConfig) -> str | None:
    topics = list(cfg.sub_topics or [])
    if not topics:
        return None
    coverage = _load()
    # Any never-covered topic wins
    uncovered = [t for t in topics if t not in coverage]
    if uncovered:
        return random.choice(uncovered)

    # Otherwise pick staler ones more often
    ordered = sorted(topics, key=lambda t: coverage.get(t, ""))
    # Weighted head bias: first half of ordered list gets ~70% pick share
    cutoff = max(1, len(ordered) // 2)
    if random.random() < 0.7:
        return random.choice(ordered[:cutoff])
    return random.choice(ordered)


def record_coverage(topic: str) -> None:
    if not topic:
        return
    coverage = _load()
    coverage[topic] = db.now_iso()
    _save(coverage)


def coverage_report(cfg: NicheConfig) -> list[tuple[str, str]]:
    """Return (topic, last_seen_iso_or_never) sorted by freshness ascending."""
    coverage = _load()
    return sorted(
        [(t, coverage.get(t, "never")) for t in cfg.sub_topics],
        key=lambda x: x[1] if x[1] != "never" else "",
    )
