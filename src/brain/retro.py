"""Retrospective learning — scrape our own posts for fresh metrics and feed
top-performing patterns back into the pipeline.

Two mechanisms:
  1. :func:`refresh_metrics` keeps the ``posts`` table current (likes /
     comments / reach) for every post from the last ~30 days.
  2. :func:`push_retro_context` summarises the highest-ER posts and pushes
     the pattern (format, caption opener, sub-topic) to context_feed so the
     next generation riffs on what's already working.

Everything falls open — retro is an advisory context signal, never a gate.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from src.core import db
from src.core.config import NicheConfig
from src.core.logging_setup import get_logger
from src.plugins.ig import BackoffActive, IGClient

log = get_logger(__name__)


# ───── Metrics refresh ─────
def refresh_metrics(cl: IGClient, *, days: int = 30, limit: int = 30) -> int:
    """Update like/comment/reach on our posts from the last ``days``."""
    conn = db.get_conn()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = conn.execute(
        """
        SELECT ig_media_pk FROM posts
        WHERE posted_at >= ?
        ORDER BY posted_at DESC LIMIT ?
        """,
        (cutoff, limit),
    ).fetchall()
    updated = 0
    for row in rows:
        pk = row["ig_media_pk"]
        try:
            metrics = cl.media_metrics(pk)
        except BackoffActive:
            break
        except Exception as e:
            log.debug("retro: metrics fetch failed for %s: %s", pk, e)
            continue
        reach = max(int(metrics.get("view_count") or 0), int(metrics.get("play_count") or 0))
        db.post_update_metrics(
            pk,
            likes=int(metrics.get("likes") or 0),
            comments=int(metrics.get("comments") or 0),
            reach=reach,
        )
        updated += 1
    if updated:
        log.info("retro: refreshed metrics on %d recent posts", updated)
    return updated


# ───── Performance analytics ─────
def top_posts(days: int = 30, limit: int = 5) -> list[dict[str, Any]]:
    """Return our top posts by engagement (likes + comments) in the window."""
    conn = db.get_conn()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = conn.execute(
        """
        SELECT ig_media_pk, format, caption, posted_at, likes, comments, reach
        FROM posts
        WHERE posted_at >= ?
        ORDER BY (COALESCE(likes, 0) + COALESCE(comments, 0) * 3) DESC
        LIMIT ?
        """,
        (cutoff, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def performance_by_format(days: int = 30) -> dict[str, dict[str, float]]:
    """Average likes/comments per format over the window."""
    conn = db.get_conn()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = conn.execute(
        """
        SELECT format,
               COUNT(*) AS n,
               AVG(COALESCE(likes, 0))    AS avg_likes,
               AVG(COALESCE(comments, 0)) AS avg_comments
        FROM posts
        WHERE posted_at >= ?
        GROUP BY format
        """,
        (cutoff,),
    ).fetchall()
    out: dict[str, dict[str, float]] = {}
    for r in rows:
        out[r["format"]] = {
            "n": int(r["n"]),
            "avg_likes": float(r["avg_likes"] or 0),
            "avg_comments": float(r["avg_comments"] or 0),
        }
    return out


# ───── Feedback to generator ─────
def push_retro_context(limit: int = 3) -> int:
    """Push a distilled 'what's working for us' signal into the context_feed."""
    tops = top_posts(limit=limit)
    if not tops:
        return 0
    parts: list[str] = []
    for p in tops:
        opener = (p.get("caption") or "").splitlines()[0][:140]
        parts.append(
            f"{p['format']} ({p['likes']}👍 / {p['comments']}💬): \"{opener}\""
        )
    summary = "top-performing recent posts: " + " | ".join(parts)
    db.push_context("retro", summary, priority=2)

    # Also push the best-performing formats so the picker learns
    perf = performance_by_format()
    if perf:
        best = sorted(
            perf.items(),
            key=lambda kv: kv[1]["avg_likes"] + kv[1]["avg_comments"] * 3,
            reverse=True,
        )[:3]
        order = ", ".join(f"{fmt} (avg {v['avg_likes']:.0f}👍)" for fmt, v in best)
        db.push_context("retro.format", f"formats outperforming for us: {order}", priority=1)
    log.info("retro: pushed %d top-post patterns to context", len(tops))
    return len(tops)


async def run_once(cfg: NicheConfig) -> dict[str, int]:
    cl = IGClient()
    try:
        cl.login()
    except BackoffActive as e:
        log.warning("retro: cooldown — %s", e)
        return {"metrics": 0, "context": 0}
    except Exception as e:
        log.error("retro: login failed: %s", e)
        return {"metrics": 0, "context": 0}
    metrics = refresh_metrics(cl)
    context = push_retro_context()
    return {"metrics": metrics, "context": context}
