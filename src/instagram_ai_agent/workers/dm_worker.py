"""DM funnel worker — graduates contacts through the sequence.

Stages:
  discovered → targeted → contacted → replied → converted / dropped

A separate ``dm_seeder`` pushes people into ``discovered``. This worker
moves contacts forward one step per cycle, respecting the per-day ``dm``
budget and warmup constraints. Cooldowns between messages to the same
contact prevent over-outreach.
"""
from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

from instagram_ai_agent.core import alerts, db
from instagram_ai_agent.core.budget import allowed
from instagram_ai_agent.core.config import NicheConfig
from instagram_ai_agent.core.llm import generate
from instagram_ai_agent.core.logging_setup import get_logger
from instagram_ai_agent.plugins.ig import BackoffActive, IGClient

log = get_logger(__name__)

COOLDOWN_HOURS = 48   # minimum gap between outbound messages to same contact
MAX_STEPS = 3         # intro + 2 follow-ups before drop


# ───── LLM ─────
async def _compose_intro(cfg: NicheConfig, contact: dict) -> str:
    system = (
        f"You write short opening DMs from a niche Instagram page to niche-aligned users.\n"
        f"Niche: {cfg.niche}. Voice: {cfg.voice.persona}. Tone: {', '.join(cfg.voice.tone)}.\n"
        f"Never: {', '.join(cfg.voice.forbidden) or 'n/a'}.\n"
        "Rules: 10–28 words. One specific compliment or observation + one low-stakes question. "
        "No sales pitch. No emojis. No generic templates. No \"hey checking out your profile\". "
        "Sound like a real human who noticed something."
    )
    prompt = (
        f"Contact: @{contact['username']}.\n"
        f"How we found them: {contact.get('source') or 'unknown'}.\n"
        f"Notes: {contact.get('notes') or 'none'}.\n\n"
        "Return the message text only."
    )
    return _sanitise(await generate("caption", prompt, system=system, max_tokens=160, temperature=0.9))


async def _compose_followup(cfg: NicheConfig, contact: dict, step: int) -> str:
    system = (
        f"You write a SHORT follow-up DM (step {step + 1} of {MAX_STEPS}) in the persona of a niche IG page.\n"
        f"Niche: {cfg.niche}. Voice: {cfg.voice.persona}. Tone: {', '.join(cfg.voice.tone)}.\n"
        "Rules: ≤20 words. Gentle, not pushy. Reference a different angle from the intro. "
        "Ask a tiny question or share a micro-insight. No \"just following up\". No emojis."
    )
    last = db.dm_last_out(contact["username"])
    prompt = (
        f"Contact: @{contact['username']}.\n"
        f"Prior outbound: {last['body'] if last else '(none)'}.\n"
        f"Notes: {contact.get('notes') or 'none'}.\n\n"
        "Return the message text only."
    )
    return _sanitise(await generate("caption", prompt, system=system, max_tokens=160, temperature=0.9))


def _sanitise(text: str) -> str:
    s = (text or "").strip()
    for q in ('"', "'", "“", "”"):
        if s.startswith(q) and s.endswith(q):
            s = s[1:-1].strip()
    for prefix in ("Message:", "DM:", "Hey!", "Hi!"):
        if s.lower().startswith(prefix.lower()):
            s = s[len(prefix):].lstrip(" ,.—:")
    return s.splitlines()[0].strip()[:500] if s else s


# ───── Dispatch ─────
async def run_pass(cfg: NicheConfig, ig: IGClient | None = None, batch: int = 3) -> int:
    """Process up to ``batch`` contacts due for an outbound message."""
    ok, used, cap = allowed("dm", cfg)
    if not ok or cap == 0:
        return 0
    budget_remaining = max(0, cap - used)
    if budget_remaining == 0:
        return 0

    targets = db.dm_contacts_due("targeted", limit=budget_remaining * 2)
    contacted = db.dm_contacts_due("contacted", limit=budget_remaining * 2)
    # Mix: half new intros, half follow-ups (if both supplied)
    candidates = _interleave(targets, contacted)[: min(batch, budget_remaining)]
    if not candidates:
        return 0

    cl = ig or IGClient()
    try:
        cl.login()
    except BackoffActive as e:
        log.warning("dm: cooldown — %s", e)
        return 0
    except Exception as e:
        log.error("dm: login failed: %s", e)
        await alerts.send(f"DM worker login failed: {e}", level="err")
        return 0

    sent = 0
    for contact in candidates:
        if not _cooldown_ok(contact):
            continue
        try:
            # Resolve user id if missing
            uid = contact.get("ig_user_id")
            if not uid:
                uid = cl.user_id_from_username(contact["username"])
                # Persist for later
                db.dm_upsert_contact(contact["username"], ig_user_id=uid)

            step = db.dm_step_count(contact["username"], direction="out")
            if step >= MAX_STEPS:
                db.dm_advance(contact["username"], "dropped")
                continue

            body = (
                await _compose_intro(cfg, contact)
                if step == 0
                else await _compose_followup(cfg, contact, step)
            )
            if not body:
                log.warning("dm: empty body for @%s — skipping", contact["username"])
                continue

            thread_id = cl.send_dm(str(uid), body)
            db.dm_record_message(contact["username"], "out", body, step=step, ig_thread_id=thread_id or None)
            next_at = (datetime.now(UTC) + timedelta(hours=COOLDOWN_HOURS + random.randint(-6, 6)))
            db.dm_advance(
                contact["username"],
                "contacted",
                next_after_iso=next_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
            )
            db.action_log("dm", contact["username"], "ok", 0)
            sent += 1
            log.info("dm OUT → @%s step=%d", contact["username"], step)
        except BackoffActive as e:
            log.warning("dm: cooldown mid-send — %s", e)
            break
        except Exception as e:
            log.warning("dm: send failed for @%s: %s", contact["username"], e)
            db.action_log("dm", contact["username"], "failed", 0)
            # Don't advance stage on failure — retry on next pass after cooldown.
            next_at = (datetime.now(UTC) + timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%SZ")
            db.dm_advance(contact["username"], contact["stage"], next_after_iso=next_at)
    return sent


def _cooldown_ok(contact: dict) -> bool:
    last = contact.get("last_action_at")
    if not last:
        return True
    try:
        last_dt = datetime.fromisoformat(last.replace("Z", "+00:00"))
    except ValueError:
        return True
    return (datetime.now(UTC) - last_dt) >= timedelta(hours=COOLDOWN_HOURS - 6)


def _interleave(a: list[dict], b: list[dict]) -> list[dict]:
    out: list[dict] = []
    for x, y in zip(a, b, strict=False):
        out.append(x)
        out.append(y)
    out.extend(a[len(b):])
    out.extend(b[len(a):])
    return out


# ───── Promote discovered → targeted via LLM curation (bulk filter) ─────
async def curate_discovered(cfg: NicheConfig, limit: int = 20) -> int:
    """Ask the LLM which discovered contacts are worth DMing.

    Called periodically to gate the funnel — keeps us from spraying every
    scraped hashtag commenter.
    """
    pool = db.dm_contacts_due("discovered", limit=limit)
    if not pool:
        return 0
    sample = "\n".join(
        f"- @{c['username']} (from {c.get('source') or '?'}): {c.get('notes') or ''}"
        for c in pool
    )
    system = (
        f"You triage a list of Instagram usernames for outreach for a page about {cfg.niche}.\n"
        f"Audience: {cfg.target_audience}.\n"
        "Rules: approve only users who plausibly match the niche + audience. "
        "Reject spam, bots, clearly-unrelated accounts, and inactive-looking profiles."
    )
    prompt = (
        f"Contacts:\n{sample}\n\n"
        "Return JSON: { \"approved\": [usernames...], \"rejected\": [usernames...] }"
    )
    from instagram_ai_agent.core.llm import generate_json

    try:
        data = await generate_json("analyze", prompt, system=system, max_tokens=400)
    except Exception as e:
        log.warning("DM curation LLM failed: %s", e)
        return 0

    approved = {str(u).lstrip("@") for u in (data.get("approved") or [])}
    rejected = {str(u).lstrip("@") for u in (data.get("rejected") or [])}

    promoted = 0
    for c in pool:
        u = c["username"]
        if u in approved:
            db.dm_advance(u, "targeted")
            promoted += 1
        elif u in rejected:
            db.dm_advance(u, "dropped")
    log.info("dm curate: %d→targeted, %d→dropped from %d discovered", promoted, len(rejected), len(pool))
    return promoted
