"""Competitor intel — pull top recent posts from each competitor, analyse for patterns."""
from __future__ import annotations

from src.brain.scraper import PublicScraper
from src.core import db
from src.core.config import NicheConfig
from src.core.llm import generate_json
from src.core.logging_setup import get_logger

log = get_logger(__name__)


async def _analyse(cfg: NicheConfig, posts: list[dict]) -> dict:
    """Ask the LLM what's working across these posts."""
    sample = "\n---\n".join(
        f"@{p['username']} ({p['likes']} likes): {p['caption'][:200]}"
        for p in posts[:20]
    )
    system = (
        f"You analyse competitor Instagram posts for a page about {cfg.niche}.\n"
        f"Audience: {cfg.target_audience}.\n"
        "Identify patterns a niche-specific Instagram page should steal (format, hook style, topic)."
    )
    prompt = (
        f"Top competitor posts:\n{sample}\n\n"
        "Return JSON: {\n"
        "  \"working_formats\": [str, ...],  // e.g. \"listicle carousel\", \"before/after reel\"\n"
        "  \"hot_topics\":      [str, ...],  // specific niche sub-topics blowing up\n"
        "  \"hooks\":           [str, ...],  // example opening lines worth riffing on\n"
        "  \"avoid\":           [str, ...]   // patterns our page should NOT copy\n"
        "}"
    )
    return await generate_json("analyze", prompt, system=system, max_tokens=800)


async def run_once(cfg: NicheConfig, scraper: PublicScraper | None = None) -> int:
    """One pass over the competitor list. Returns number of posts stored."""
    if not cfg.competitors:
        log.info("No competitors configured — skipping")
        return 0

    s = scraper or PublicScraper()
    stored = 0
    posts_for_llm: list[dict] = []

    for username in cfg.competitors:
        scraped = s.profile_posts(username, limit=8)
        for p in scraped:
            db.competitor_upsert(
                p.ig_pk, p.username, p.caption, p.likes, p.comments, p.posted_at
            )
            posts_for_llm.append(p.__dict__)
            stored += 1

    if not posts_for_llm:
        return 0

    try:
        analysis = await _analyse(cfg, posts_for_llm)
    except Exception as e:
        log.warning("Competitor LLM analysis failed: %s", e)
        return stored

    # Push actionable signals to the context feed
    for fmt in analysis.get("working_formats") or []:
        db.push_context("competitor.format", f"competitors winning with: {fmt}", priority=2)
    for topic in analysis.get("hot_topics") or []:
        db.push_context("competitor.topic", f"hot in niche: {topic}", priority=3)
        db.narrative_bump(topic)
    for hook in (analysis.get("hooks") or [])[:5]:
        db.push_context("competitor.hook", f"hook pattern: {hook}", priority=1)

    log.info("Competitor intel: %d posts, %d topics, %d formats",
             stored,
             len(analysis.get("hot_topics") or []),
             len(analysis.get("working_formats") or []))
    return stored
