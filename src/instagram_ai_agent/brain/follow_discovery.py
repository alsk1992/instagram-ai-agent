"""Follow-target discovery — finds accounts worth following.

The engager already knows how to execute follows, and the per-day budget
sits at 25. Nothing was populating the pipeline though: `engagement_seeder`
only queued `like` and `comment` actions. This module bridges that gap by
scraping two high-ROI sources:

  1. **Likers of hashtag-top posts** — people who just liked a popular
     post in our niche are active niche engagers right now. High follow-back
     rate because the recommendation signal matches.
  2. **Followers of competitors** — if they follow a similar page, they'll
     follow us too for the same reason. Requires `competitors` configured
     in niche.yaml.

Candidates land in the ``follow_candidates`` table. The seeder reads from
there and pushes into ``engagement_queue``; the engager drains the queue
respecting the daily follow budget + warmup gates.

Dedup: ``follow_candidates`` has ``user_id`` as PRIMARY KEY so re-discovering
the same user is idempotent. We also skip users the account is already
following via the action_log history.
"""
from __future__ import annotations

import random

from instagram_ai_agent.core import db
from instagram_ai_agent.core.config import NicheConfig
from instagram_ai_agent.core.logging_setup import get_logger
from instagram_ai_agent.plugins.ig import IGClient

log = get_logger(__name__)


def _already_followed_ids() -> set[str]:
    """User IDs we've already followed (via action_log) — skip re-adding."""
    rows = db.get_conn().execute(
        "SELECT target FROM action_log WHERE action='follow' AND result='ok'"
    ).fetchall()
    return {str(r["target"]) for r in rows if r["target"]}


def discover_from_hashtag_likers(
    cfg: NicheConfig, ig: IGClient, *,
    posts_per_tag: int = 2, likers_per_post: int = 8,
) -> int:
    """For the top posts in each core/growth hashtag, scrape likers and
    store them as follow candidates."""
    tags = list({*cfg.hashtags.core, *cfg.hashtags.growth})
    if not tags:
        return 0
    random.shuffle(tags)
    tags = tags[:4]

    # Pull already-discovered post PKs so we don't re-scrape the same ones
    already_followed = _already_followed_ids()
    added = 0
    for tag in tags:
        rows = db.get_conn().execute(
            """
            SELECT ig_pk FROM hashtag_top
            WHERE hashtag = ?
            ORDER BY likes DESC LIMIT ?
            """,
            (tag, posts_per_tag),
        ).fetchall()
        for r in rows:
            pk = r["ig_pk"]
            try:
                likers = ig.media_likers(pk, amount=likers_per_post)
            except Exception as e:
                log.debug("media_likers for %s failed: %s", pk, e)
                continue
            for liker in likers:
                uid = liker["user_id"]
                if not uid or uid in already_followed:
                    continue
                db.follow_candidate_upsert(
                    user_id=uid,
                    username=liker["username"],
                    source=f"hashtag_liker:{tag}",
                    # Use follower count as a proxy: prefer mid-size accounts
                    # (1k–50k) which are most likely to follow back.
                    score=_score_follower_count(liker["follower_count"]),
                )
                added += 1
    if added:
        log.info("follow_discovery/hashtag_likers: %d candidates added", added)
    return added


def discover_from_competitor_followers(
    cfg: NicheConfig, ig: IGClient, *,
    per_competitor: int = 15,
) -> int:
    """Pull each competitor's recent followers. They already follow a
    similar page — high intent signal."""
    if not cfg.competitors:
        return 0

    already_followed = _already_followed_ids()
    added = 0
    for handle in cfg.competitors:
        handle = handle.lstrip("@")
        try:
            uid = ig.user_id_from_username(handle)
        except Exception as e:
            log.debug("user_id_from_username %s failed: %s", handle, e)
            continue
        try:
            followers = ig.followers_of(uid, amount=per_competitor)
        except Exception as e:
            log.debug("followers_of %s failed: %s", handle, e)
            continue
        for f in followers:
            fuid = f["user_id"]
            if not fuid or fuid in already_followed:
                continue
            db.follow_candidate_upsert(
                user_id=fuid,
                username=f["username"],
                source=f"competitor:{handle}",
                score=_score_follower_count(f["follower_count"]),
            )
            added += 1
    if added:
        log.info("follow_discovery/competitor_followers: %d candidates added", added)
    return added


def _score_follower_count(n: int) -> int:
    """Bias toward mid-tier accounts (1k-50k): they're active enough to
    notice a new follow but small enough to reciprocate. Micro-accounts
    (<1k) score lower because they're often inactive; mega-accounts are
    filtered out entirely in the ig.py scrapers."""
    if n >= 50_000:
        return 30
    if n >= 10_000:
        return 80
    if n >= 1_000:
        return 100
    if n >= 100:
        return 60
    return 30


async def run_once(cfg: NicheConfig, ig: IGClient | None = None) -> int:
    """One pass of both discovery sources. Called by the orchestrator on
    an interval. Returns total candidates added."""
    ig = ig or IGClient()
    try:
        ig.login()
    except Exception as e:
        log.warning("follow_discovery: login failed — skipping this cycle (%s)", e)
        return 0

    n1 = discover_from_hashtag_likers(cfg, ig)
    n2 = discover_from_competitor_followers(cfg, ig)
    return n1 + n2
