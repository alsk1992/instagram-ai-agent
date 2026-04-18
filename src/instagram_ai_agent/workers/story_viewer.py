"""Story-view pass — cheap warming signals toward our highest-signal contacts.

For each of our recent posts we fetch the likers + commenters, and queue a
``story_view`` action for anyone we haven't already viewed today. Story
views are the cheapest action IG offers: they land us in the viewer list of
those people's stories, which prompts them to check our profile.

Priorities:
  commenters  > repeat likers > one-time likers
"""
from __future__ import annotations

from collections import Counter

from instagram_ai_agent.core import db
from instagram_ai_agent.core.config import NicheConfig
from instagram_ai_agent.core.logging_setup import get_logger
from instagram_ai_agent.plugins.ig import BackoffActive, IGClient

log = get_logger(__name__)


def _recent_post_pks(limit: int = 5) -> list[str]:
    conn = db.get_conn()
    rows = conn.execute(
        "SELECT ig_media_pk FROM posts ORDER BY posted_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [r["ig_media_pk"] for r in rows]


def _already_viewed_today(user_id: str) -> bool:
    row = db.get_conn().execute(
        """
        SELECT 1 FROM action_log
        WHERE action='story_view' AND target=?
          AND at >= strftime('%Y-%m-%dT00:00:00Z','now')
        """,
        (user_id,),
    ).fetchone()
    return bool(row)


def _already_queued_today(user_id: str) -> bool:
    row = db.get_conn().execute(
        """
        SELECT 1 FROM engagement_queue
        WHERE action='story_view' AND target_user=?
          AND created_at >= strftime('%Y-%m-%dT00:00:00Z','now')
        """,
        (user_id,),
    ).fetchone()
    return bool(row)


def run_pass(cfg: NicheConfig, ig: IGClient | None = None, *, max_queue: int = 30) -> int:
    """Scrape engagers on our recent posts, queue prioritised story views."""
    if cfg.budget.story_views <= 0:
        return 0

    post_pks = _recent_post_pks()
    if not post_pks:
        return 0

    cl = ig or IGClient()
    try:
        cl.login()
    except BackoffActive as e:
        log.warning("story_viewer: cooldown — %s", e)
        return 0
    except Exception as e:
        log.error("story_viewer: login failed: %s", e)
        return 0

    # Build a scored map of user_id → priority (commenter > multi-liker > liker)
    scores: Counter[str] = Counter()
    usernames: dict[str, str] = {}

    # Commenters (priority 3 apiece) — re-use our own inbound_comments table
    conn = db.get_conn()
    rows = conn.execute(
        """
        SELECT DISTINCT user_id, username FROM inbound_comments
        WHERE media_pk IN (%s) AND user_id IS NOT NULL AND user_id != '' AND is_own=0
        """ % ",".join("?" * len(post_pks)),
        post_pks,
    ).fetchall()
    for r in rows:
        uid = r["user_id"]
        scores[uid] += 3
        if r["username"]:
            usernames[uid] = r["username"]

    # Likers (priority 1, capped to recent posts)
    for pk in post_pks:
        try:
            likers = cl.cl.media_likers(pk)
        except BackoffActive:
            return 0
        except Exception as e:
            log.debug("story_viewer: likers fetch failed on %s: %s", pk, e)
            continue
        for user in likers or []:
            uid = str(getattr(user, "pk", "") or "")
            if not uid:
                continue
            scores[uid] += 1
            if getattr(user, "username", None):
                usernames.setdefault(uid, user.username)

    if not scores:
        return 0

    # Sort by score desc, queue up to max_queue entries we haven't acted on today
    queued = 0
    for uid, _score in scores.most_common():
        if queued >= max_queue:
            break
        if _already_viewed_today(uid) or _already_queued_today(uid):
            continue
        db.engagement_enqueue(
            "story_view",
            target_user=uid,
            payload={
                "source": "post_engagers",
                "username": usernames.get(uid, ""),
            },
        )
        queued += 1

    if queued:
        log.info("story_viewer: queued %d views for post engagers", queued)
    return queued
