"""Dev.to trend feed — free, keyless, tag-queryable.

Pulls top articles of the past week for each tag in ``cfg.devto_tags`` and
pushes LLM-filtered picks to the context_feed. Useful for tech / career /
productivity / web-dev niches where Dev.to is a focused firehose of what
practitioners are discussing right now.

2026 status: still open. Endpoint: ``https://dev.to/api/articles`` with
``tag`` + ``top=7`` query params. Empty ``devto_tags`` disables the feed.
"""
from __future__ import annotations

import httpx

from instagram_ai_agent.core import db
from instagram_ai_agent.core.config import NicheConfig
from instagram_ai_agent.core.llm import generate_json
from instagram_ai_agent.core.logging_setup import get_logger

log = get_logger(__name__)


ENDPOINT = "https://dev.to/api/articles"


async def _fetch_tag(tag: str, *, hits: int = 20) -> list[dict]:
    """Fetch ``hits`` top articles for a tag from the past 7 days."""
    params = {"tag": tag, "top": 7, "per_page": hits}
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(15.0),
            headers={"User-Agent": "ig-agent/0.2 (+devto)"},
        ) as client:
            r = await client.get(ENDPOINT, params=params)
            r.raise_for_status()
            raw = r.json()
    except Exception as e:
        log.warning("devto fetch failed (tag=%r): %s", tag, e)
        return []
    out: list[dict] = []
    for it in raw or []:
        title = (it.get("title") or "").strip()
        if not title:
            continue
        out.append({
            "title": title,
            "link": it.get("url") or "",
            "description": (it.get("description") or "").strip(),
            "reactions": int(it.get("public_reactions_count") or 0),
            "comments": int(it.get("comments_count") or 0),
            "tag": tag,
        })
    return out


async def _filter_picks(cfg: NicheConfig, items: list[dict]) -> list[dict]:
    if not items:
        return []
    sample = "\n".join(
        f"- [{i}] [{it['tag']}] {it['title']} ({it['reactions']}❤️) — {it['description'][:160]}"
        for i, it in enumerate(items[:25])
    )
    system = (
        f"You triage Dev.to trending articles for an Instagram page about {cfg.niche}.\n"
        f"Sub-topics: {', '.join(cfg.sub_topics)}.\n"
        "Return only items that would inspire a concrete niche-specific post."
    )
    prompt = (
        f"{sample}\n\n"
        "Return JSON: {\"picks\": [ {\"i\": int, \"angle\": str}, ... ]} (at most 4).\n"
        "``angle`` = one-line suggestion for how our niche page should riff on it."
    )
    try:
        data = await generate_json("analyze", prompt, system=system, max_tokens=500)
    except Exception as e:
        log.warning("devto filter LLM failed: %s", e)
        return []
    out: list[dict] = []
    for p in (data.get("picks") or [])[:4]:
        try:
            idx = int(p.get("i"))
        except (TypeError, ValueError):
            continue
        if not 0 <= idx < len(items):
            continue
        item = items[idx]
        out.append({
            "title": item["title"],
            "link": item["link"],
            "reactions": item["reactions"],
            "angle": str(p.get("angle") or "").strip(),
        })
    return out


async def run_once(cfg: NicheConfig) -> int:
    tags = [t for t in cfg.devto_tags if t]
    if not tags:
        return 0
    items: list[dict] = []
    for tag in tags[:5]:
        items.extend(await _fetch_tag(tag, hits=15))
    seen: set[str] = set()
    unique: list[dict] = []
    for it in items:
        if it["link"] and it["link"] not in seen:
            seen.add(it["link"])
            unique.append(it)
    if not unique:
        return 0
    picks = await _filter_picks(cfg, unique)
    for p in picks:
        db.push_context(
            "devto",
            f"dev.to ({p['reactions']}❤️): {p['title']} → {p['angle']} ({p['link']})",
            priority=3,
        )
        db.narrative_bump(p["title"][:80], sample_ref=p["link"] or None)
    log.info("devto: %d items scanned (%d tags), %d picks queued", len(unique), len(tags), len(picks))
    return len(picks)
