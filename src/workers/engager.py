"""Engager worker — likes, follows, comments, story views from the engagement queue."""
from __future__ import annotations

from src.core import db
from src.core.budget import allowed
from src.core.config import NicheConfig
from src.core.logging_setup import get_logger
from src.plugins.ig import BackoffActive, IGClient

log = get_logger(__name__)


def _execute(cl: IGClient, row: dict) -> tuple[str, str | None]:
    action = row["action"]
    payload = row.get("payload") or {}
    if action == "like":
        ok = cl.like(row["target_media"] or payload.get("media_pk", ""))
        return ("ok" if ok else "failed"), None
    if action == "follow":
        uid = row["target_user"] or payload.get("user_id")
        if not uid:
            return "failed", "missing user_id"
        ok = cl.follow(str(uid))
        return ("ok" if ok else "failed"), None
    if action == "unfollow":
        uid = row["target_user"] or payload.get("user_id")
        if not uid:
            return "failed", "missing user_id"
        ok = cl.unfollow(str(uid))
        return ("ok" if ok else "failed"), None
    if action == "comment":
        pk = row["target_media"] or payload.get("media_pk")
        text = payload.get("text", "")
        if not (pk and text):
            return "failed", "missing media_pk or text"
        cid = cl.comment(str(pk), text)
        return "ok", cid
    if action == "story_view":
        uid = row["target_user"] or payload.get("user_id")
        if not uid:
            return "failed", "missing user_id"
        n = cl.view_stories(str(uid))
        return "ok", str(n)
    return "failed", f"unknown action {action}"


def run_pass(cfg: NicheConfig, ig: IGClient | None = None, batch: int = 4) -> int:
    """Drain up to `batch` ready actions from the engagement queue."""
    cl = ig or IGClient()
    try:
        cl.login()
    except BackoffActive as e:
        log.warning("engager: cooldown — %s", e)
        return 0
    except Exception as e:
        log.error("engager: login failed — %s", e)
        return 0

    done = 0
    for row in db.engagement_next(limit=batch):
        action = row["action"]
        ok, used, cap = allowed(action, cfg)
        if not ok:
            log.info("budget exhausted for %s (%d/%d)", action, used, cap)
            continue

        try:
            result, extra = _execute(cl, row)
        except BackoffActive as e:
            log.warning("engager: cooldown mid-action — %s", e)
            return done
        except Exception as e:
            db.engagement_mark(int(row["id"]), "failed", str(e)[:500])
            db.action_log(action, row.get("target_user") or row.get("target_media"), "failed", 0)
            continue

        db.engagement_mark(int(row["id"]), result, extra if result != "ok" else None)
        db.action_log(action, row.get("target_user") or row.get("target_media"), result, 0)
        if result == "ok":
            done += 1
    return done
