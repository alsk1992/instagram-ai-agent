"""Engagement seeder — populates the engagement_queue from brain intel.

Three sources feed the queue, ordered by ROI:

  1. **Target watcher**: when the watched account posts, queue a thoughtful
     comment + like on that post (highest priority — early-comment boost).
  2. **Competitor hot posts**: queue likes on top-performing competitor posts
     so their followers see us in the like-list.
  3. **Hashtag top**: queue likes on high-engagement posts in our core tags,
     so the tag viewers see us.

Daily caps are enforced downstream by the engager worker. The seeder's job is
to *offer* more than we'll execute, and let budget enforcement trim.
"""
from __future__ import annotations

import random

from src.core import db
from src.core.config import NicheConfig
from src.core.llm import generate
from src.core.logging_setup import get_logger

log = get_logger(__name__)


# ─── Comment generator ───
async def _niche_comment(cfg: NicheConfig, post_caption: str) -> str:
    """Produce a short, on-voice comment that reacts to a specific post."""
    system = (
        f"You leave short, specific, on-voice comments on other people's Instagram posts.\n"
        f"Niche: {cfg.niche}. Voice: {cfg.voice.persona}. Tone: {', '.join(cfg.voice.tone)}.\n"
        f"Never: {', '.join(cfg.voice.forbidden) or 'n/a'}.\n"
        "Rules: 4–14 words. Specific to the post content. No hashtags. No emoji spam. "
        "Sounds like a real person, not a bot template. No \"great post!\". No generic."
    )
    prompt = f"Post caption:\n{post_caption[:500]}\n\nReturn the comment text only."
    out = await generate("caption", prompt, system=system, max_tokens=120, temperature=0.9)
    return _clean_comment(out)


def _clean_comment(raw: str) -> str:
    s = raw.strip()
    for q in ('"', "'", "“", "”"):
        if s.startswith(q) and s.endswith(q):
            s = s[1:-1].strip()
    # Strip leading labels some models add
    for lead in ("Comment:", "comment:", "Reply:", "reply:"):
        if s.startswith(lead):
            s = s[len(lead):].strip()
    # Single-line only
    s = s.splitlines()[0].strip() if s else s
    return s[:200]


# ─── Sources ───
async def seed_from_target(cfg: NicheConfig) -> int:
    """Queue a comment+like per new target-account post (highest priority)."""
    if not cfg.watch_target:
        return 0
    conn = db.get_conn()
    rows = conn.execute(
        """
        SELECT ig_pk, username, caption FROM target_feed
        WHERE engaged = 0
        ORDER BY seen_at DESC LIMIT 5
        """
    ).fetchall()
    if not rows:
        return 0

    seeded = 0
    for row in rows:
        pk = row["ig_pk"]
        caption = row["caption"] or ""
        # Like immediately (cheap)
        db.engagement_enqueue("like", target_media=pk)
        # Comment — ask the LLM for a specific line
        try:
            text = await _niche_comment(cfg, caption)
        except Exception as e:
            log.warning("Target comment LLM failed: %s", e)
            text = ""
        if text:
            db.engagement_enqueue(
                "comment",
                target_media=pk,
                payload={"text": text, "source": "target"},
            )
        conn.execute("UPDATE target_feed SET engaged=1 WHERE ig_pk=?", (pk,))
        seeded += 1
    log.info("seeder/target: %d new interactions queued", seeded)
    return seeded


def seed_from_competitors(cfg: NicheConfig, per_competitor: int = 2) -> int:
    """Queue likes on the top-performing recent posts of each competitor."""
    if not cfg.competitors:
        return 0
    conn = db.get_conn()
    seeded = 0
    for username in cfg.competitors:
        rows = conn.execute(
            """
            SELECT ig_pk FROM competitor_posts
            WHERE username=? AND engaged=0
            ORDER BY likes DESC LIMIT ?
            """,
            (username, per_competitor),
        ).fetchall()
        for r in rows:
            db.engagement_enqueue(
                "like",
                target_media=r["ig_pk"],
                payload={"source": f"competitor:{username}"},
            )
            conn.execute("UPDATE competitor_posts SET engaged=1 WHERE ig_pk=?", (r["ig_pk"],))
            seeded += 1
    if seeded:
        log.info("seeder/competitors: %d likes queued", seeded)
    return seeded


def seed_from_hashtags(cfg: NicheConfig, per_tag: int = 3) -> int:
    """Queue likes on the hottest posts pulled from hashtag_top."""
    if not cfg.hashtags.core:
        return 0
    conn = db.get_conn()
    tags = list({*cfg.hashtags.core, *cfg.hashtags.growth})
    random.shuffle(tags)
    tags = tags[:4]

    seeded = 0
    seen_pks: set[str] = set()
    for tag in tags:
        rows = conn.execute(
            """
            SELECT ig_pk FROM hashtag_top
            WHERE hashtag=?
            ORDER BY likes DESC LIMIT ?
            """,
            (tag, per_tag * 2),
        ).fetchall()
        picks = 0
        for r in rows:
            pk = r["ig_pk"]
            if pk in seen_pks:
                continue
            # Skip if we've already queued this post today
            already = conn.execute(
                """
                SELECT 1 FROM engagement_queue
                WHERE action='like' AND target_media=?
                  AND created_at >= strftime('%Y-%m-%dT00:00:00Z','now')
                """,
                (pk,),
            ).fetchone()
            if already:
                continue
            db.engagement_enqueue(
                "like",
                target_media=pk,
                payload={"source": f"hashtag:{tag}"},
            )
            seen_pks.add(pk)
            seeded += 1
            picks += 1
            if picks >= per_tag:
                break
    if seeded:
        log.info("seeder/hashtags: %d likes queued across %d tags", seeded, len(tags))
    return seeded


async def seed_comments_on_hashtags(cfg: NicheConfig, per_tag: int = 1) -> int:
    """Queue LLM-written comments on 1-2 hashtag-top posts per core tag.

    Comments are much higher-signal engagement than likes but also riskier;
    we cap per run conservatively and only select top-engagement posts.
    """
    if not cfg.hashtags.core:
        return 0
    conn = db.get_conn()
    queued = 0
    for tag in cfg.hashtags.core[:3]:
        rows = conn.execute(
            """
            SELECT ig_pk, caption FROM hashtag_top
            WHERE hashtag=?
            ORDER BY likes DESC LIMIT ?
            """,
            (tag, per_tag * 2),
        ).fetchall()
        picked = 0
        for r in rows:
            # Skip if we already commented / queued a comment for this pk today
            already = conn.execute(
                """
                SELECT 1 FROM engagement_queue
                WHERE action='comment' AND target_media=?
                  AND created_at >= strftime('%Y-%m-%dT00:00:00Z','now')
                """,
                (r["ig_pk"],),
            ).fetchone()
            if already:
                continue
            try:
                text = await _niche_comment(cfg, r["caption"] or "")
            except Exception as e:
                log.warning("comment LLM failed for hashtag post: %s", e)
                continue
            if not text:
                continue
            db.engagement_enqueue(
                "comment",
                target_media=r["ig_pk"],
                payload={"text": text, "source": f"hashtag:{tag}"},
            )
            queued += 1
            picked += 1
            if picked >= per_tag:
                break
    if queued:
        log.info("seeder/hashtag_comments: %d comments queued", queued)
    return queued


async def run_once(cfg: NicheConfig) -> dict[str, int]:
    """One full seeder pass. Safe to call repeatedly."""
    results = {
        "target": await seed_from_target(cfg),
        "competitors": seed_from_competitors(cfg),
        "hashtags": seed_from_hashtags(cfg),
        "hashtag_comments": await seed_comments_on_hashtags(cfg),
    }
    return results
