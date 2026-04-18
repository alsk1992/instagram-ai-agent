"""Poster worker — pulls approved content and uploads it to Instagram."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from instagram_ai_agent.core import alerts, db, storage
from instagram_ai_agent.core.budget import allowed
from instagram_ai_agent.core.config import FEED_FORMATS, STORY_FORMATS, NicheConfig
from instagram_ai_agent.core.logging_setup import get_logger
from instagram_ai_agent.plugins import human_mimic
from instagram_ai_agent.plugins.ig import BackoffActive, IGClient

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


async def post_next(
    cfg: NicheConfig,
    ig: IGClient | None = None,
    *,
    drain: bool = False,
) -> int | None:
    """Post the next queued item.

    When ``drain=True``, bypasses the ``scheduled_for`` best-hours
    filter — the caller is explicitly asking for an immediate post.
    Used by ``ig-agent drain`` for first-post-proof on fresh installs.
    """
    ok, used, cap = allowed("post", cfg)
    if not ok:
        # Surface the actual cause loudly — the default is "warmup blocks
        # posts for 7 days" which silently left users confused on first
        # run. Tell them about the IG_SKIP_WARMUP escape hatch.
        from instagram_ai_agent.core.warmup import current_phase, current_day
        phase = current_phase()
        day = current_day()
        if cap == 0 and phase is not None and not phase.allow_posts:
            log.warning(
                "post blocked by warmup — day %d/%d (%s phase). "
                "Fresh accounts get ZERO posts for the first 7 days to avoid "
                "ban patterns. Set IG_SKIP_WARMUP=1 in .env if this account "
                "is established + has real post history already.",
                day or 0, 14, phase.label,
            )
        else:
            log.info("post budget exhausted (%d/%d) — skipping", used, cap)
        return None

    if drain:
        item = db.content_next_to_drain()
    else:
        item = db.content_next_to_post()
    if not item or item["format"] in STORY_FORMATS:
        # Story items are handled by story_poster — re-query for feed-only
        item = _next_feed_item(drain=drain)
    if not item:
        return None

    # Aspect-ratio pre-flight: IG silently re-compresses off-spec media
    # and downranks the result. Refuse ahead of time so we can regen.
    if cfg.human_mimic.aspect_ratio_check:
        kind = "reel" if item["format"] in ("reel_stock", "reel_ai") else "feed"
        for media in item["media_paths"]:
            if not human_mimic.validate_aspect_ratio(media, kind=kind):
                log.warning(
                    "Aspect-ratio check failed for %s (%s) — marking failed, generator will regen",
                    media, kind,
                )
                db.content_update_status(
                    int(item["id"]), "failed",
                    f"aspect_ratio_check: {media} not IG-compliant for {kind}",
                )
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

    # Pre-post scroll — mimic "user opened IG" instead of cold API write
    if cfg.human_mimic.pre_post_scroll:
        n = await _maybe_scroll(cl)
        if n:
            log.debug("pre_post_scroll touched %d items before posting", n)

    cid = int(item["id"])
    # If first-comment hashtags is enabled, post the caption WITHOUT the
    # hashtag block and re-attach hashtags as a self-reply after upload.
    caption_to_post, hashtag_comment = _split_hashtags_if_configured(item, cfg)
    if caption_to_post != item["caption"]:
        # Shadow the mutated caption into the dispatch item so every
        # uploader sees the stripped text.
        item = {**item, "caption": caption_to_post}

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

    # First-comment hashtag drop — post a self-reply immediately after
    # the upload lands, using the same client so the cookie context is
    # carried over.
    if hashtag_comment:
        try:
            await _post_first_comment(cl, media_pk, hashtag_comment, cfg)
        except Exception as e:
            log.warning("First-comment hashtag drop failed (non-fatal): %s", e)

    # Stamp the post-cooldown clock — any follow-up write action will
    # be silently skipped until the mandatory window elapses.
    if cfg.human_mimic.post_cooldown:
        human_mimic.stamp_post()

    db.content_mark_posted(cid, media_pk)
    db.post_record(media_pk, cid, item["format"], item["caption"])
    db.action_log("post", media_pk, "ok", 0)

    # Persist the session AFTER the write — IG rotates cookies
    # (mid / rur / x-ig-www-claim / occasionally csrftoken) on write
    # responses. Losing them on crash = silent session drift.
    try:
        cl.persist_settings()
    except Exception:
        pass

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


async def _maybe_scroll(cl: IGClient) -> int:
    """Run the pre-post scroll off the event loop (instagrapi is sync)."""
    import asyncio
    try:
        return await asyncio.to_thread(human_mimic.pre_post_scroll, cl.cl)
    except Exception as e:
        log.debug("pre_post_scroll helper errored: %s (non-fatal)", e)
        return 0


def _split_hashtags_if_configured(
    item: dict, cfg: NicheConfig,
) -> tuple[str, str | None]:
    """When ``first_comment_hashtags`` is enabled, return (caption_without_tags,
    hashtag_block). Otherwise (caption_unchanged, None).

    Detection: the caption pipeline appends hashtags via ``\\n\\n`` +
    space-joined tags prefixed with ``#``. We split on that sentinel."""
    if not cfg.human_mimic.first_comment_hashtags:
        return item["caption"], None
    caption = item.get("caption") or ""
    if "\n\n#" not in caption:
        return caption, None
    body, _, tags = caption.rpartition("\n\n")
    if not tags.strip().startswith("#"):
        return caption, None
    return body.strip(), tags.strip()


async def _post_first_comment(
    cl: IGClient, media_pk: str, comment_text: str, cfg: NicheConfig,
) -> None:
    """Drop a self-reply on our own post with the hashtag block.

    Uses typing_delay if enabled so the comment timing reads human
    (still fast — hashtag blocks aren't typed, they're pasted — 3-6s
    is a realistic paste delay)."""
    import asyncio
    if not comment_text:
        return
    # Short, deliberate delay — humans paste a pre-prepared comment
    # block within a few seconds of posting.
    import random as _rand
    await asyncio.sleep(_rand.uniform(3.0, 7.0))
    # Further type-style delay if enabled (paste is faster than type,
    # so cap low).
    if cfg.human_mimic.typing_delays:
        await asyncio.sleep(_rand.uniform(0.5, 2.0))
    await asyncio.to_thread(cl.comment, media_pk, comment_text)
    log.info("first-comment hashtag drop ok (%d chars) on %s", len(comment_text), media_pk)


def _next_feed_item(drain: bool = False) -> dict | None:
    conn = db.get_conn()
    qs = ",".join("?" * len(FEED_FORMATS))
    if drain:
        sql = f"""
        SELECT * FROM content_queue
        WHERE status='approved'
          AND format IN ({qs})
        ORDER BY COALESCE(scheduled_for, created_at) ASC
        LIMIT 1
        """
    else:
        sql = f"""
        SELECT * FROM content_queue
        WHERE status='approved'
          AND format IN ({qs})
          AND (scheduled_for IS NULL OR scheduled_for <= strftime('%Y-%m-%dT%H:%M:%SZ','now'))
        ORDER BY COALESCE(scheduled_for, created_at) ASC
        LIMIT 1
        """
    row = conn.execute(sql, tuple(FEED_FORMATS)).fetchone()
    if row is None:
        return None
    if drain:
        # Clear the slot so the scheduler doesn't double-book after post
        conn.execute(
            "UPDATE content_queue SET scheduled_for=NULL WHERE id=?",
            (int(row["id"]),),
        )
    return db._row_to_content(row)


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
