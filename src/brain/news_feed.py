"""RSS / Atom feed miner — turns niche news into context_feed entries.

Works with any public RSS feed (Google News niche queries, Reddit RSS, blog
feeds, YouTube channel feeds). No API key required.
"""
from __future__ import annotations

from xml.etree import ElementTree as ET

import httpx

from src.core import db
from src.core.config import NicheConfig
from src.core.llm import generate_json
from src.core.logging_setup import get_logger

log = get_logger(__name__)


async def _fetch(url: str, *, timeout: float = 20.0) -> str | None:
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            headers={"User-Agent": "ig-agent/0.1 (+rss)"},
        ) as client:
            r = await client.get(url, follow_redirects=True)
            r.raise_for_status()
            return r.text
    except Exception as e:
        log.warning("rss fetch failed %s: %s", url, e)
        return None


def _parse_items(xml_text: str) -> list[dict]:
    """Best-effort RSS 2.0 + Atom parser. Returns list of {title, link, summary}."""
    items: list[dict] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        log.debug("rss parse failed: %s", e)
        return items

    tag = root.tag.lower()
    if tag.endswith("rss") or root.find("channel") is not None:
        for it in root.findall(".//item"):
            items.append({
                "title": (it.findtext("title") or "").strip(),
                "link": (it.findtext("link") or "").strip(),
                "summary": (it.findtext("description") or "").strip(),
            })
    elif tag.endswith("feed"):
        # Atom
        ns = {"a": "http://www.w3.org/2005/Atom"}
        for it in root.findall("a:entry", ns):
            link_el = it.find("a:link", ns)
            items.append({
                "title": (it.findtext("a:title", default="", namespaces=ns) or "").strip(),
                "link": (link_el.get("href") if link_el is not None else "").strip(),
                "summary": (
                    it.findtext("a:summary", default="", namespaces=ns)
                    or it.findtext("a:content", default="", namespaces=ns)
                    or ""
                ).strip(),
            })
    return items[:30]


async def _filter_and_summarise(cfg: NicheConfig, items: list[dict]) -> list[dict]:
    """LLM filter — only items that are actually on-niche; summarise each."""
    if not items:
        return []
    sample = "\n".join(
        f"- [{i}] {it['title']} — {it['summary'][:220]}" for i, it in enumerate(items[:20])
    )
    system = (
        f"You triage RSS headlines for an Instagram page about {cfg.niche}.\n"
        f"Sub-topics: {', '.join(cfg.sub_topics)}.\n"
        "Return only items that would inspire a concrete niche-specific post."
    )
    prompt = (
        f"{sample}\n\n"
        "Return JSON: {\"picks\": [ {\"i\": int, \"angle\": str}, ... ]}\n"
        "For each pick, ``angle`` is a one-line suggestion for how our page could "
        "riff on the story in our niche voice. Pick at most 5."
    )
    try:
        data = await generate_json("analyze", prompt, system=system, max_tokens=600)
    except Exception as e:
        log.warning("news filter LLM failed: %s", e)
        return []
    out: list[dict] = []
    for p in (data.get("picks") or [])[:5]:
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
            "angle": str(p.get("angle") or "").strip(),
        })
    return out


async def run_once(cfg: NicheConfig) -> int:
    """Pull every configured RSS feed, filter via LLM, push picks to context_feed."""
    feeds = [u for u in cfg.rss_feeds if u]
    if not feeds:
        return 0

    all_items: list[dict] = []
    for url in feeds:
        text = await _fetch(url)
        if not text:
            continue
        for it in _parse_items(text):
            # Deduplicate by link within this run
            if not any(existing["link"] == it["link"] for existing in all_items):
                all_items.append(it)

    if not all_items:
        return 0

    picks = await _filter_and_summarise(cfg, all_items)
    for p in picks:
        db.push_context(
            "news",
            f"news: {p['title']} → {p['angle']} ({p['link']})",
            priority=3,
        )
        db.narrative_bump(p["title"][:80], sample_ref=p["link"] or None)

    log.info("news: %d items fetched, %d picks pushed to context", len(all_items), len(picks))
    return len(picks)
