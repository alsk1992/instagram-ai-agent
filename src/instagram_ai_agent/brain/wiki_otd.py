"""Wikipedia 'On This Day' seed feed — free, keyless, perfectly scheduled.

Wikipedia exposes a REST feed of historical events, births, and deaths for
every MM/DD, updated daily. Perfect for an Instagram page that wants an
evergreen anchor: "on this day in 1969..." carousels that cost $0 to source.

Runs once a day — pushes 3-5 LLM-filtered niche-relevant events into
context_feed. Enable via ``wiki_otd_enabled: true`` in niche.yaml.

2026 status: still open, unauthenticated, served from Wikimedia CDN.
Endpoint: ``https://en.wikipedia.org/api/rest_v1/feed/onthisday/events/MM/DD``
"""
from __future__ import annotations

from datetime import datetime, timezone

import httpx

from instagram_ai_agent.core import db
from instagram_ai_agent.core.config import NicheConfig
from instagram_ai_agent.core.llm import generate_json
from instagram_ai_agent.core.logging_setup import get_logger

log = get_logger(__name__)


async def _fetch_events(month: int, day: int) -> list[dict]:
    url = (
        f"https://en.wikipedia.org/api/rest_v1/feed/onthisday/events/"
        f"{month:02d}/{day:02d}"
    )
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(15.0),
            headers={"User-Agent": "ig-agent/0.2 (+otd)"},
        ) as client:
            r = await client.get(url)
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        log.warning("wiki_otd fetch failed for %02d/%02d: %s", month, day, e)
        return []
    out: list[dict] = []
    for ev in (data.get("events") or [])[:50]:
        text = (ev.get("text") or "").strip()
        year = ev.get("year")
        if not text or not year:
            continue
        pages = ev.get("pages") or []
        link = ""
        if pages:
            link = pages[0].get("content_urls", {}).get("desktop", {}).get("page", "") or ""
        out.append({"year": int(year), "text": text, "link": link})
    # Sort most recent first — fresher history usually picks better for IG
    out.sort(key=lambda e: -e["year"])
    return out


async def _filter_picks(cfg: NicheConfig, events: list[dict]) -> list[dict]:
    if not events:
        return []
    sample = "\n".join(
        f"- [{i}] {ev['year']}: {ev['text']}"
        for i, ev in enumerate(events[:30])
    )
    system = (
        f"You triage 'on this day in history' events for an Instagram page about {cfg.niche}.\n"
        f"Sub-topics: {', '.join(cfg.sub_topics)}.\n"
        "Only pick events that could inspire a niche-relevant post — ignore pure "
        "political / military anniversaries unless they connect to our niche."
    )
    prompt = (
        f"{sample}\n\n"
        "Return JSON: {\"picks\": [ {\"i\": int, \"angle\": str}, ... ]} (at most 3).\n"
        "``angle`` = one-line suggestion for how our niche page could riff on this anniversary."
    )
    try:
        data = await generate_json("analyze", prompt, system=system, max_tokens=400)
    except Exception as e:
        log.warning("wiki_otd filter LLM failed: %s", e)
        return []
    out: list[dict] = []
    for p in (data.get("picks") or [])[:3]:
        try:
            idx = int(p.get("i"))
        except (TypeError, ValueError):
            continue
        if not 0 <= idx < len(events):
            continue
        ev = events[idx]
        out.append({
            "year": ev["year"],
            "text": ev["text"],
            "link": ev["link"],
            "angle": str(p.get("angle") or "").strip(),
        })
    return out


async def run_once(cfg: NicheConfig) -> int:
    if not cfg.wiki_otd_enabled:
        return 0
    today = datetime.now(timezone.utc)
    events = await _fetch_events(today.month, today.day)
    if not events:
        return 0
    picks = await _filter_picks(cfg, events)
    for p in picks:
        db.push_context(
            "wiki_otd",
            f"on this day {p['year']}: {p['text']} → {p['angle']}"
            + (f" ({p['link']})" if p["link"] else ""),
            priority=4,  # evergreen — lower than fresh trends
        )
    log.info(
        "wiki_otd: %d events fetched for %02d/%02d, %d picks queued",
        len(events), today.month, today.day, len(picks),
    )
    return len(picks)
