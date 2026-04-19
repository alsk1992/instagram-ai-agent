"""HackerNews trend feed — free, keyless, fires off Algolia's public API.

Pulls items matching ``cfg.hackernews_keywords`` (or front-page posts when no
keywords are set) and pushes LLM-filtered picks to the context_feed. Perfect
for tech / AI / startup / productivity niches where HN surfaces the trends
before Twitter catches up.

2026 status: still open + no rate limit in practice. Zero setup — empty
``hackernews_keywords`` in niche.yaml simply disables the feed.
"""
from __future__ import annotations

import httpx

from instagram_ai_agent.core import db
from instagram_ai_agent.core.config import NicheConfig
from instagram_ai_agent.core.llm import generate_json
from instagram_ai_agent.core.logging_setup import get_logger

log = get_logger(__name__)


BASE = "https://hn.algolia.com/api/v1"


async def _fetch(query: str | None, *, hits: int = 30) -> list[dict]:
    """Pull ``hits`` stories. When query is None, returns front-page ranked by points."""
    params: dict[str, str | int] = {"hitsPerPage": hits, "tags": "front_page"}
    if query:
        params = {"query": query, "hitsPerPage": hits, "tags": "story"}
    url = f"{BASE}/search_by_date" if query else f"{BASE}/search"
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(15.0),
            headers={"User-Agent": "ig-agent/0.2 (+hn)"},
        ) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        log.warning("hackernews fetch failed (query=%r): %s", query, e)
        return []
    out: list[dict] = []
    for hit in (data.get("hits") or [])[:hits]:
        title = (hit.get("title") or "").strip()
        if not title:
            continue
        out.append({
            "title": title,
            "link": hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID')}",
            "points": int(hit.get("points") or 0),
            "comments": int(hit.get("num_comments") or 0),
        })
    return out


async def _filter_picks(cfg: NicheConfig, items: list[dict]) -> list[dict]:
    if not items:
        return []
    sample = "\n".join(
        f"- [{i}] {it['title']} ({it['points']}↑, {it['comments']}💬) — {it['link']}"
        for i, it in enumerate(items[:25])
    )
    system = (
        f"You triage HackerNews front-page stories for an Instagram page about {cfg.niche}.\n"
        f"Sub-topics: {', '.join(cfg.sub_topics)}.\n"
        "Return only stories that could inspire a concrete niche-specific post."
    )
    prompt = (
        f"{sample}\n\n"
        "Return JSON: {\"picks\": [ {\"i\": int, \"angle\": str}, ... ]} (at most 4).\n"
        "``angle`` = one-line suggestion for how our niche page should riff on it."
    )
    try:
        data = await generate_json("analyze", prompt, system=system, max_tokens=500)
    except Exception as e:
        log.warning("hackernews filter LLM failed: %s", e)
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
            "points": item["points"],
            "angle": str(p.get("angle") or "").strip(),
        })
    return out


async def run_once(cfg: NicheConfig) -> int:
    keywords = [k for k in cfg.hackernews_keywords if k]
    items: list[dict] = []
    if keywords:
        for kw in keywords[:5]:
            items.extend(await _fetch(kw, hits=15))
    else:
        items = await _fetch(None, hits=30)
    # Dedupe by link
    seen: set[str] = set()
    unique: list[dict] = []
    for it in items:
        if it["link"] not in seen:
            seen.add(it["link"])
            unique.append(it)
    if not unique:
        return 0
    picks = await _filter_picks(cfg, unique)
    for p in picks:
        db.push_context(
            "hackernews",
            f"HN ({p['points']}↑): {p['title']} → {p['angle']} ({p['link']})",
            priority=3,
        )
        db.narrative_bump(p["title"][:80], sample_ref=p["link"] or None)
    log.info("hackernews: %d items scanned, %d picks queued", len(unique), len(picks))
    return len(picks)
