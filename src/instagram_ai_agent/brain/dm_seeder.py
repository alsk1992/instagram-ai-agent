"""DM seeder — populates dm_contacts with ``discovered`` candidates.

Sources (ranked by signal strength):
  1. People who engaged with our target account's posts (high priority).
  2. Top hashtag post authors.
  3. Competitor commenters (pulled when available).

LLM curation (``dm_worker.curate_discovered``) is what actually promotes
to ``targeted`` — seeding is just widening the funnel.
"""
from __future__ import annotations

from instagram_ai_agent.brain.scraper import PublicScraper
from instagram_ai_agent.core import db
from instagram_ai_agent.core.config import NicheConfig
from instagram_ai_agent.core.logging_setup import get_logger

log = get_logger(__name__)


def seed_from_hashtag_authors(cfg: NicheConfig, scraper: PublicScraper | None = None) -> int:
    s = scraper or PublicScraper()
    seeded = 0
    for tag in (cfg.hashtags.core or [])[:3]:
        for post in s.hashtag_posts(tag, limit=15):
            if not post.username:
                continue
            # Skip our own posts + configured competitors (we don't DM competitors)
            if post.username == cfg.watch_target:
                continue
            if post.username in cfg.competitors:
                continue
            db.dm_upsert_contact(
                post.username,
                source=f"hashtag:{tag}",
                priority=2,
                notes=(post.caption or "")[:200],
            )
            seeded += 1
    if seeded:
        log.info("dm seed: %d from hashtag authors", seeded)
    return seeded


def seed_from_target_feed(cfg: NicheConfig) -> int:
    """Promote authors of our target feed's recent posts.

    target_feed stores posts BY the target; we also want people who reply to
    target's posts — future enhancement via comments scrape. For now we seed
    the target itself's visible followers-worth of engagement.
    """
    if not cfg.watch_target:
        return 0
    # For MVP: surface the target's username as a high-priority contact so we
    # have SOMETHING in the funnel even on a fresh account. Future: scrape
    # target's commenters and seed them.
    db.dm_upsert_contact(cfg.watch_target, source="target.self", priority=4)
    return 1


def run_once(cfg: NicheConfig) -> int:
    scraper = PublicScraper()
    return (
        seed_from_target_feed(cfg)
        + seed_from_hashtag_authors(cfg, scraper)
    )
