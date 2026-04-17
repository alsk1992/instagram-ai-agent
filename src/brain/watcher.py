"""Multi-target watcher — polls N IG accounts and pushes new posts as high-priority context."""
from __future__ import annotations

from src.brain.scraper import PublicScraper
from src.core import db
from src.core.config import NicheConfig
from src.core.logging_setup import get_logger

log = get_logger(__name__)


def run_once(cfg: NicheConfig, scraper: PublicScraper | None = None) -> int:
    """Returns total newly-detected posts across all watch targets."""
    targets = cfg.all_watch_targets()
    if not targets:
        return 0
    s = scraper or PublicScraper()
    total_new = 0
    for target in targets:
        total_new += _watch_one(s, target)
    return total_new


def _watch_one(scraper: PublicScraper, target: str) -> int:
    try:
        posts = scraper.profile_posts(target, limit=8)
    except Exception as e:
        log.warning("watcher: failed %s: %s", target, e)
        return 0

    new = 0
    for p in posts:
        if not db.target_feed_upsert(p.ig_pk, p.username, "post", p.caption, p.likes, p.posted_at):
            continue
        new += 1
        db.push_context(
            "watch_target",
            f"@{p.username} just posted: {p.caption[:220]}",
            priority=4,
        )
    if new:
        log.info("Watcher: %d new post(s) from @%s", new, target)
    return new
