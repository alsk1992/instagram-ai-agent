"""Reply-to-own-comments worker.

Polls the last N of our own posts, collects any comments we haven't
responded to yet, asks the LLM to write a specific reply, sends it.

Filters out:
  - comments by us
  - spam-looking comments (links, mass emoji, very short ``🔥``-only)
  - non-niche / off-topic comments (LLM triage)
"""
from __future__ import annotations

import re
from typing import Any

from instagram_ai_agent.core import alerts, db
from instagram_ai_agent.core.budget import allowed
from instagram_ai_agent.core.config import NicheConfig
from instagram_ai_agent.core.llm import generate, generate_json
from instagram_ai_agent.core.logging_setup import get_logger
from instagram_ai_agent.plugins.ig import BackoffActive, IGClient

log = get_logger(__name__)


# ───── Scraping ─────
def scrape_recent_posts(cl: IGClient, limit: int = 5) -> int:
    """Pull comments from our most recent posts into inbound_comments."""
    conn = db.get_conn()
    rows = conn.execute(
        "SELECT ig_media_pk FROM posts ORDER BY posted_at DESC LIMIT ?", (limit,)
    ).fetchall()
    new = 0
    own_user_id = str(getattr(cl.cl, "user_id", "") or "")
    for row in rows:
        media_pk = row["ig_media_pk"]
        try:
            comments = cl.media_comments(media_pk, limit=40)
        except Exception as e:
            log.warning("fetch comments failed %s: %s", media_pk, e)
            continue
        for c in comments:
            is_own = own_user_id and c.get("user_id") == own_user_id
            if db.inbound_comment_upsert(
                c["pk"],
                media_pk=media_pk,
                username=c.get("username"),
                user_id=c.get("user_id"),
                text=c.get("text") or "",
                created_at=c.get("created_at") or "",
                is_own=is_own,
            ):
                new += 1
    if new:
        log.info("comment scraper: %d new inbound comments", new)
    return new


# ───── Filters ─────
_LINK = re.compile(r"https?://\S+|\b\w+\.\w{2,}/\S+", re.IGNORECASE)
_MENTION_ONLY = re.compile(r"^(@\S+\s*)+$")


def _looks_like_spam(text: str) -> bool:
    if not text or len(text.strip()) < 2:
        return True
    stripped = text.strip()
    if _LINK.search(stripped):
        return True
    if _MENTION_ONLY.match(stripped):
        return True
    # Non-alphanumeric ratio > 70% → probably emoji-only spam
    alpha = sum(ch.isalnum() for ch in stripped)
    if alpha / max(1, len(stripped)) < 0.3:
        return True
    # Trading / scam spam patterns
    for bad in ("crypto", "profit", "whatsapp", "dm me", "trader", "investment"):
        if bad in stripped.lower():
            return True
    return False


# ───── Reply composition ─────
async def _compose_reply(cfg: NicheConfig, comment_text: str, username: str) -> str:
    system = (
        f"You reply to comments on our Instagram posts about {cfg.niche}.\n"
        f"Voice: {cfg.voice.persona}. Tone: {', '.join(cfg.voice.tone)}.\n"
        f"Never: {', '.join(cfg.voice.forbidden) or 'n/a'}.\n"
        "Rules: ≤18 words. Specific to the comment. Warm but not sucky-uppy. "
        "Never generic 'thanks!'. No hashtags. One emoji max. Never link out."
    )
    prompt = (
        f"Commenter: @{username}\n"
        f"Their comment:\n{comment_text}\n\n"
        "Return the reply text only."
    )
    out = await generate("caption", prompt, system=system, max_tokens=160, temperature=0.9)
    return _clean(out)


def _clean(s: str) -> str:
    s = (s or "").strip()
    for q in ('"', "'", "“", "”"):
        if s.startswith(q) and s.endswith(q):
            s = s[1:-1].strip()
    for pref in ("Reply:", "Response:", "reply:", "response:"):
        if s.startswith(pref):
            s = s[len(pref):].lstrip(" ,.:—")
    return s.splitlines()[0].strip()[:300] if s else s


# ───── Optional LLM triage for borderline comments ─────
async def _is_worth_replying(cfg: NicheConfig, comment_text: str) -> bool:
    system = (
        f"You triage comments on an Instagram page about {cfg.niche}. "
        "Decide if replying is worth the effort (ignore spam/off-topic, reply when on-niche)."
    )
    prompt = (
        f"Comment: {comment_text}\n\n"
        "Return JSON: {\"reply\": bool, \"reason\": str}"
    )
    try:
        data = await generate_json("critic", prompt, system=system, max_tokens=160, temperature=0.1)
        return bool(data.get("reply"))
    except Exception:
        return True  # Fail open — better to reply than miss


# ───── Worker entry ─────
async def run_pass(cfg: NicheConfig, ig: IGClient | None = None, batch: int = 4) -> int:
    ok, used, cap = allowed("comment", cfg)
    if not ok or cap == 0:
        return 0

    cl = ig or IGClient()
    try:
        cl.login()
    except BackoffActive as e:
        log.warning("comment_replier: cooldown — %s", e)
        return 0
    except Exception as e:
        log.error("comment_replier: login failed — %s", e)
        return 0

    # Fresh scrape of our last 5 posts' comments
    scrape_recent_posts(cl, limit=5)

    remaining = max(0, cap - used)
    if remaining == 0:
        return 0

    todo = db.inbound_comments_to_reply(limit=batch * 2)
    sent = 0
    import asyncio as _asyncio
    import random as _random
    for i, c in enumerate(todo):
        if sent >= remaining:
            break
        text = c.get("text") or ""
        if _looks_like_spam(text):
            db.inbound_comment_ignore(c["comment_pk"])
            continue
        if not await _is_worth_replying(cfg, text):
            db.inbound_comment_ignore(c["comment_pk"])
            continue

        # Comment-reply stagger — if enabled, wait 5-60 min-ish BETWEEN
        # replies in the same batch so IG doesn't see 4 replies all
        # landing within 30 seconds. First reply of the batch fires
        # immediately; subsequent ones space out.
        if cfg.human_mimic.comment_reply_delay and i > 0:
            wait_s = _random.uniform(5 * 60, 60 * 60)  # 5..60 min
            log.info(
                "comment_replier: staggering next reply by %d min (anti-burst)",
                int(wait_s / 60),
            )
            await _asyncio.sleep(wait_s)

        try:
            reply = await _compose_reply(cfg, text, c.get("username") or "")
            if not reply:
                db.inbound_comment_ignore(c["comment_pk"])
                continue
            reply_pk = cl.reply_to_comment(c["media_pk"], c["comment_pk"], reply)
            db.inbound_comment_mark_replied(c["comment_pk"], reply_pk)
            db.action_log("comment", c["comment_pk"], "ok", 0)
            sent += 1
            log.info("replied to @%s on %s: %s", c.get("username"), c["media_pk"], reply[:80])
        except BackoffActive as e:
            log.warning("comment_replier: cooldown mid-pass — %s", e)
            break
        except Exception as e:
            log.warning("reply failed on %s: %s", c["comment_pk"], e)
            db.action_log("comment", c["comment_pk"], "failed", 0)
    if sent:
        await alerts.send(f"Replied to {sent} comment(s) on our posts.", level="info")
        # Persist post-write so IG's rotated cookies (mid/rur/x-ig-www-
        # claim/csrftoken) don't get lost on crash.
        try:
            cl.persist_settings()
        except Exception as _persist_err:
            log.debug("comment_replier: persist_settings after replies failed — "
                      "rotated cookies may be lost on crash: %s", _persist_err)
    return sent
