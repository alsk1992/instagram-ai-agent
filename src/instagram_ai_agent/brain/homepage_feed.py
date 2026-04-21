"""Homepage-feed engagement — engage with what accounts you follow are posting.

Before this module, engagement was hashtag-only: the agent found posts via
``#calisthenics`` et al. and liked them. That's scraper behaviour — fine
for early exposure, but a real human also scrolls the home feed and
engages with accounts they already follow.

This brain job fetches the timeline feed via ``cl.get_timeline_feed()``
and queues:
  - likes on a few recent posts (conservative: ~3-5 per run)
  - occasional LLM-written comments on high-engagement posts from the feed
    (rarer — ~1 per run, feed posts are from accounts we already care about)

Dedup: skip posts we've already queued / liked (action_log lookup). Skip
our own posts (handled inside ``ig.timeline_feed_posts``).

Cadence: runs on its own scheduler interval (20 min). Budget respect is
enforced downstream by the engager's daily cap — this module just offers
candidates.
"""
from __future__ import annotations

import random

from instagram_ai_agent.brain.engagement_seeder import _clean_comment, _niche_comment
from instagram_ai_agent.core import db
from instagram_ai_agent.core.config import NicheConfig
from instagram_ai_agent.core.logging_setup import get_logger
from instagram_ai_agent.plugins.ig import IGClient

log = get_logger(__name__)


def _already_liked(media_pk: str) -> bool:
    """True if we've queued or executed a like on this media already."""
    row = db.get_conn().execute(
        """
        SELECT 1 FROM engagement_queue
        WHERE action IN ('like','comment') AND target_media=?
        LIMIT 1
        """,
        (media_pk,),
    ).fetchone()
    if row:
        return True
    row = db.get_conn().execute(
        "SELECT 1 FROM action_log WHERE action='like' AND target=? LIMIT 1",
        (media_pk,),
    ).fetchone()
    return bool(row)


async def run_once(
    cfg: NicheConfig,
    ig: IGClient | None = None,
    *,
    like_budget: int = 4,
    comment_budget: int = 1,
) -> dict[str, int]:
    """Fetch the timeline, queue likes + one comment on fresh posts.
    Returns {"likes": n, "comments": n}."""
    ig = ig or IGClient()
    try:
        ig.login()
    except Exception as e:
        log.debug("homepage_feed: login failed — %s", e)
        return {"likes": 0, "comments": 0}

    posts = ig.timeline_feed_posts(amount=18)
    if not posts:
        return {"likes": 0, "comments": 0}

    likes_queued = 0
    comments_queued = 0

    # Shuffle so we don't always pick the first 4 — more natural behaviour
    random.shuffle(posts)

    for post in posts:
        if likes_queued >= like_budget and comments_queued >= comment_budget:
            break
        pk = post["media_pk"]
        if _already_liked(pk):
            continue

        # Every candidate gets a like queue attempt (cheap, high-volume).
        if likes_queued < like_budget:
            db.engagement_enqueue(
                "like",
                target_user=post["user_id"],
                target_media=pk,
                payload={
                    "source": "homepage_feed",
                    "username": post["username"],
                },
            )
            likes_queued += 1

        # ~1 in 4 likes gets a comment too — higher-signal engagement on
        # posts from accounts we actually follow.
        if (
            comments_queued < comment_budget
            and random.random() < 0.25
            and post.get("caption")
        ):
            try:
                text = await _niche_comment(cfg, post["caption"])
                text = _clean_comment(text)
            except Exception as e:
                log.debug("homepage_feed: comment LLM failed — %s", e)
                text = ""
            if text:
                db.engagement_enqueue(
                    "comment",
                    target_user=post["user_id"],
                    target_media=pk,
                    payload={
                        "text": text,
                        "source": "homepage_feed",
                        "username": post["username"],
                    },
                )
                comments_queued += 1

    if likes_queued or comments_queued:
        log.info(
            "homepage_feed: queued %d likes + %d comments from timeline",
            likes_queued, comments_queued,
        )
    return {"likes": likes_queued, "comments": comments_queued}
