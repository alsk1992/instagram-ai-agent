"""Follow-back worker + reciprocal engagement bookkeeping.

Polls our latest followers, triages each via LLM against the niche filter,
follows back the ones that look like genuine niche-aligned accounts, and
queues a light reciprocal action (like on a recent post) so the new
follower notices.
"""
from __future__ import annotations

from instagram_ai_agent.core import alerts, db
from instagram_ai_agent.core.budget import allowed
from instagram_ai_agent.core.config import NicheConfig
from instagram_ai_agent.core.llm import generate_json
from instagram_ai_agent.core.logging_setup import get_logger
from instagram_ai_agent.plugins.ig import BackoffActive, IGClient

log = get_logger(__name__)


async def _triage(cfg: NicheConfig, candidates: list[dict]) -> dict[str, dict]:
    """LLM decides follow-back (keeps spam / ill-fit accounts out)."""
    if not candidates:
        return {}
    sample = "\n".join(
        f"- @{c['username']}  |  {c.get('full_name') or ''}  "
        f"|  verified={c.get('is_verified')} private={c.get('is_private')}"
        for c in candidates
    )
    system = (
        f"You triage new Instagram followers for a page about {cfg.niche}.\n"
        f"Audience: {cfg.target_audience}.\n"
        "Follow back only accounts that plausibly match the niche or audience. "
        "Reject obvious spam, engagement-pod bots, off-topic brands, adult-content, "
        "crypto scammers, or accounts that look like automation farms."
    )
    prompt = (
        f"Candidates:\n{sample}\n\n"
        "Return JSON: {\"decisions\": [{\"username\": str, \"action\": \"follow|skip|reject\", \"reason\": str}]}"
    )
    try:
        data = await generate_json("analyze", prompt, system=system, max_tokens=600)
    except Exception as e:
        log.warning("follow-back triage LLM failed: %s", e)
        return {}
    out: dict[str, dict] = {}
    for d in (data.get("decisions") or []):
        u = str(d.get("username") or "").lstrip("@")
        if u:
            out[u] = {
                "action": str(d.get("action") or "skip").lower(),
                "reason": str(d.get("reason") or "")[:200],
            }
    return out


async def run_pass(cfg: NicheConfig, ig: IGClient | None = None, batch: int = 8) -> int:
    ok, used, cap = allowed("follow", cfg)
    if not ok or cap == 0:
        return 0

    cl = ig or IGClient()
    try:
        cl.login()
    except BackoffActive as e:
        log.warning("follow_back: cooldown — %s", e)
        return 0
    except Exception as e:
        log.error("follow_back: login failed — %s", e)
        return 0

    # 1. Scrape our latest followers not yet in the followback table
    try:
        pending = cl.pending_followers(amount=80)
    except BackoffActive:
        return 0
    except Exception as e:
        log.warning("pending_followers fetch failed: %s", e)
        return 0

    new = 0
    for u in pending:
        if db.follower_upsert(u["user_id"], u["username"], u.get("full_name") or ""):
            new += 1
    if new:
        log.info("follow_back: %d new inbound followers seen", new)

    # 2. Triage pending rows via LLM in small batches
    to_triage = db.followers_pending(limit=batch)
    if not to_triage:
        return 0
    decisions = await _triage(cfg, to_triage)

    remaining_budget = max(0, cap - used)
    if remaining_budget == 0:
        return 0

    followed = 0
    for row in to_triage:
        if followed >= remaining_budget:
            break
        d = decisions.get(row["username"]) or {"action": "skip", "reason": "no triage"}
        action = d.get("action")
        if action == "reject":
            db.follower_triage(row["user_id"], "rejected", d.get("reason"))
            continue
        if action != "follow":
            db.follower_triage(row["user_id"], "pending", "deferred by triage")
            continue

        try:
            ok_follow = cl.follow(row["user_id"])
        except BackoffActive:
            break
        except Exception as e:
            log.warning("follow failed on @%s: %s", row["username"], e)
            db.action_log("follow", row["username"], "failed", 0)
            continue

        if ok_follow:
            db.follower_triage(row["user_id"], "followed_back", d.get("reason"))
            db.action_log("follow", row["username"], "ok", 0)
            followed += 1

            # Reciprocal: queue a light like on their most recent post
            db.engagement_enqueue(
                "story_view",
                target_user=row["user_id"],
                payload={"source": "reciprocal:followback", "note": d.get("reason")},
            )

    if followed:
        await alerts.send(f"Followed back {followed} niche-aligned accounts.", level="info")
    return followed


# ───── Reciprocal: engage with users who engaged with us ─────
def queue_reciprocal_from_recent_comments(batch: int = 5) -> int:
    """For users who recently commented on our posts, queue a story-view back.

    Story views are cheap signals that show up in the viewer list — a low-cost
    reciprocal action that strengthens warm leads without burning the comment
    budget.
    """
    conn = db.get_conn()
    rows = conn.execute(
        """
        SELECT DISTINCT user_id, username FROM inbound_comments
        WHERE is_own=0 AND user_id IS NOT NULL AND user_id != ''
          AND scraped_at >= datetime('now', '-2 days')
        ORDER BY scraped_at DESC LIMIT ?
        """,
        (batch * 3,),
    ).fetchall()
    queued = 0
    for r in rows:
        # Skip if we already queued a recent story_view for them today
        dup = conn.execute(
            """
            SELECT 1 FROM engagement_queue
            WHERE action='story_view' AND target_user=?
              AND created_at >= strftime('%Y-%m-%dT00:00:00Z','now')
            """,
            (r["user_id"],),
        ).fetchone()
        if dup:
            continue
        db.engagement_enqueue(
            "story_view",
            target_user=r["user_id"],
            payload={"source": f"reciprocal:commenter:{r['username']}"},
        )
        queued += 1
        if queued >= batch:
            break
    if queued:
        log.info("reciprocal: queued %d story views for recent commenters", queued)
    return queued
