"""Poster worker — pulls approved content and uploads it to Instagram."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.core import alerts, db, storage
from src.core.budget import allowed
from src.core.config import FEED_FORMATS, STORY_FORMATS, NicheConfig
from src.core.logging_setup import get_logger
from src.plugins.ig import BackoffActive, IGClient

log = get_logger(__name__)


def _thumbnail_for_reel(video_path: str) -> str | None:
    """Pick a thumbnail by extracting the middle frame (best effort)."""
    import subprocess

    v = Path(video_path)
    if not v.exists():
        return None
    out = v.with_suffix(".thumb.jpg")
    try:
        dur_cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(v),
        ]
        dur = float(subprocess.run(dur_cmd, capture_output=True, text=True, check=True).stdout.strip() or "2")
        ts = max(0.1, dur / 2.0)
        subprocess.run(
            [
                "ffmpeg", "-y", "-ss", f"{ts:.2f}", "-i", str(v),
                "-frames:v", "1", "-q:v", "3", str(out),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return str(out)
    except Exception:
        return None


def _dispatch_upload(cl: IGClient, item: dict) -> str:
    fmt = item["format"]
    caption = item["caption"]
    paths = item["media_paths"]
    if not paths:
        raise RuntimeError("No media paths on content item")

    if fmt == "carousel":
        return cl.upload_album(paths, caption)
    if fmt in ("reel_stock", "reel_ai"):
        video = paths[0]
        thumb = _thumbnail_for_reel(video)
        return cl.upload_reel(video, caption, thumbnail=thumb)
    if fmt in ("meme", "quote_card", "photo"):
        return cl.upload_photo(paths[0], caption)
    raise ValueError(f"Unknown format for upload: {fmt}")


async def post_next(cfg: NicheConfig, ig: IGClient | None = None) -> int | None:
    ok, used, cap = allowed("post", cfg)
    if not ok:
        log.info("post budget exhausted (%d/%d) — skipping", used, cap)
        return None

    item = db.content_next_to_post()
    if not item or item["format"] in STORY_FORMATS:
        # Story items are handled by story_poster — re-query for feed-only
        item = _next_feed_item()
    if not item:
        return None

    cl = ig or IGClient()
    try:
        cl.login()
    except BackoffActive as e:
        log.warning("Skipping post: %s", e)
        return None
    except Exception as e:
        log.error("Login failed: %s", e)
        await alerts.send(f"Login failed: {e}", level="err")
        return None

    cid = int(item["id"])
    try:
        media_pk = _dispatch_upload(cl, item)
    except BackoffActive as e:
        log.warning("Cooldown hit mid-post: %s", e)
        return None
    except Exception as e:
        log.exception("Upload failed for content id=%d", cid)
        db.content_update_status(cid, "failed", str(e)[:500])
        db.action_log("post", None, "failed", 0)
        await alerts.send(f"Upload failed for id={cid}: {e}", level="err")
        return None

    db.content_mark_posted(cid, media_pk)
    db.post_record(media_pk, cid, item["format"], item["caption"])
    db.action_log("post", media_pk, "ok", 0)

    # Archive to R2 (no-op if unconfigured), then clean up local stage
    if storage.configured():
        try:
            storage.archive_posted_media(cid, item["media_paths"])
            storage.cleanup_local(item["media_paths"])
        except Exception as e:
            log.warning("R2 archive failed (non-fatal): %s", e)

    log.info("POSTED id=%d format=%s ig_pk=%s", cid, item["format"], media_pk)
    await alerts.send(
        f"Posted <b>{item['format']}</b> — ig_pk=<code>{media_pk}</code>",
        level="ok",
    )
    return cid


def _next_feed_item() -> dict | None:
    conn = db.get_conn()
    qs = ",".join("?" * len(FEED_FORMATS))
    row = conn.execute(
        f"""
        SELECT * FROM content_queue
        WHERE status='approved'
          AND format IN ({qs})
          AND (scheduled_for IS NULL OR scheduled_for <= strftime('%Y-%m-%dT%H:%M:%SZ','now'))
        ORDER BY COALESCE(scheduled_for, created_at) ASC
        LIMIT 1
        """,
        tuple(FEED_FORMATS),
    ).fetchone()
    return db._row_to_content(row) if row else None


def schedule_approved_items(cfg: NicheConfig) -> int:
    """Assign scheduled_for times to approved items that don't have one yet.

    Feed items and story items get their own slot streams so they don't
    compete for posting hours.
    """
    n = 0
    n += _schedule_pool(cfg, pool=FEED_FORMATS, per_day=cfg.schedule.posts_per_day,
                       best_hours=cfg.schedule.best_hours_utc, action_key="post")
    # Stories spread across the waking-hours window rather than peak-only hours
    story_hours = _story_hours(cfg.schedule.best_hours_utc)
    n += _schedule_pool(cfg, pool=STORY_FORMATS, per_day=cfg.schedule.stories_per_day,
                       best_hours=story_hours, action_key="story_post")
    return n


def _schedule_pool(
    cfg: NicheConfig,
    *,
    pool: frozenset[str],
    per_day: int,
    best_hours: list[int],
    action_key: str,
) -> int:
    if per_day <= 0:
        return 0
    now = datetime.now(timezone.utc)
    items = [
        i for i in db.content_list(status="approved", limit=100)
        if i["format"] in pool and not i.get("scheduled_for")
    ]
    if not items:
        return 0
    hours = list(best_hours) or [15, 19, 21]

    def next_slot(after: datetime) -> datetime:
        for day_offset in range(0, 14):
            day = after.date() + timedelta(days=day_offset)
            for h in hours:
                candidate = datetime(day.year, day.month, day.day, h, 0, 0, tzinfo=timezone.utc)
                if candidate > after:
                    return candidate
        return after + timedelta(hours=1)

    scheduled = 0
    cursor = now
    daily_used = {now.date(): db.action_count_today(action_key)}

    for item in items:
        while True:
            slot = next_slot(cursor)
            d = slot.date()
            used = daily_used.get(d, 0)
            if used >= per_day:
                cursor = datetime(d.year, d.month, d.day, 23, 59, 59, tzinfo=timezone.utc)
                continue
            daily_used[d] = used + 1
            cursor = slot
            break
        db.content_schedule(int(item["id"]), cursor.strftime("%Y-%m-%dT%H:%M:%SZ"))
        scheduled += 1
    return scheduled


def _story_hours(best_hours: list[int]) -> list[int]:
    """Spread stories more evenly across waking hours than feed posts do."""
    if not best_hours:
        return [9, 12, 15, 18, 21]
    # Fan out by ±2 hours around each best hour, dedup + clamp
    spread: set[int] = set()
    for h in best_hours:
        for off in (-3, -1, 0, 1, 3):
            candidate = (h + off) % 24
            if 8 <= candidate <= 23:
                spread.add(candidate)
    return sorted(spread)
