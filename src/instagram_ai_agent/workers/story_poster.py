"""Story poster — separate budget + separate schedule from feed posts.

Pulls items where ``content_queue.format`` starts with ``story_`` and status
is ``approved``. Uses ``stories_per_day`` for the budget check.
"""
from __future__ import annotations

from instagram_ai_agent.core import alerts, db, storage
from instagram_ai_agent.core.budget import allowed
from instagram_ai_agent.core.config import NicheConfig
from instagram_ai_agent.core.logging_setup import get_logger
from instagram_ai_agent.plugins.ig import BackoffActive, IGClient

log = get_logger(__name__)

STORY_FORMATS = (
    "story_quote",
    "story_announcement",
    "story_photo",
    "story_video",
    "story_human",
)


def _next_story() -> dict | None:
    conn = db.get_conn()
    # Parameterised IN-clause via formatted placeholders
    qs = ",".join("?" * len(STORY_FORMATS))
    row = conn.execute(
        f"""
        SELECT * FROM content_queue
        WHERE status='approved'
          AND format IN ({qs})
          AND (scheduled_for IS NULL OR scheduled_for <= strftime('%Y-%m-%dT%H:%M:%SZ','now'))
        ORDER BY COALESCE(scheduled_for, created_at) ASC
        LIMIT 1
        """,
        STORY_FORMATS,
    ).fetchone()
    if row is None:
        return None
    return db._row_to_content(row)


def _dispatch_upload(cl: IGClient, item: dict) -> str:
    fmt = item["format"]
    paths = item["media_paths"]
    if not paths:
        raise RuntimeError("Story item has no media paths")
    stickers = (item.get("meta") or {}).get("stickers") or {}
    mention = stickers.get("mention")
    hashtag = stickers.get("hashtag")
    link = stickers.get("link")

    if fmt in ("story_quote", "story_announcement", "story_photo", "story_human"):
        return cl.upload_story_image(paths[0], item["caption"], mention=mention, hashtag=hashtag, link=link)
    if fmt == "story_video":
        return cl.upload_story_video(paths[0], item["caption"], mention=mention, hashtag=hashtag, link=link)
    raise ValueError(f"Unknown story format: {fmt}")


async def post_next(cfg: NicheConfig, ig: IGClient | None = None) -> int | None:
    ok, used, cap = allowed("story_post", cfg)
    if not ok:
        log.info("story budget exhausted (%d/%d)", used, cap)
        return None

    item = _next_story()
    if not item:
        return None

    cl = ig or IGClient()
    try:
        cl.login()
    except BackoffActive as e:
        log.warning("story: cooldown — %s", e)
        return None
    except Exception as e:
        log.error("story: login failed: %s", e)
        await alerts.send(f"Story login failed: {e}", level="err")
        return None

    cid = int(item["id"])
    try:
        pk = _dispatch_upload(cl, item)
    except BackoffActive as e:
        log.warning("story: cooldown mid-post — %s", e)
        return None
    except Exception as e:
        log.exception("Story upload failed for id=%d", cid)
        db.content_update_status(cid, "failed", str(e)[:500])
        db.action_log("story_post", None, "failed", 0)
        await alerts.send(f"Story upload failed id={cid}: {e}", level="err")
        return None

    db.content_mark_posted(cid, pk)
    db.post_record(pk, cid, item["format"], item["caption"])
    db.action_log("story_post", pk, "ok", 0)

    if storage.configured():
        try:
            storage.archive_posted_media(cid, item["media_paths"])
            storage.cleanup_local(item["media_paths"])
        except Exception as e:
            log.warning("R2 archive failed for story (non-fatal): %s", e)

    log.info("STORY POSTED id=%d format=%s ig_pk=%s", cid, item["format"], pk)
    await alerts.send(f"Story posted <b>{item['format']}</b> — <code>{pk}</code>", level="ok")
    return cid
