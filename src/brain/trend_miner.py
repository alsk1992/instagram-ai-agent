"""Trend miner — scrape niche hashtags' top posts, cluster captions by theme."""
from __future__ import annotations

import random

from src.brain.scraper import PublicScraper
from src.core import db
from src.core.config import NicheConfig
from src.core.llm import generate_json
from src.core.logging_setup import get_logger

log = get_logger(__name__)


async def _cluster(cfg: NicheConfig, samples: list[dict]) -> dict:
    joined = "\n---\n".join(
        f"#{s['hashtag']} · {s['likes']}👍: {s['caption'][:180]}" for s in samples[:30]
    )
    system = (
        f"You do trend analysis for a niche Instagram page about {cfg.niche}.\n"
        f"Sub-topics: {', '.join(cfg.sub_topics)}.\n"
        "You cluster top hashtag posts into sub-topic themes and pick the most post-worthy ones."
    )
    prompt = (
        f"Sample posts across niche hashtags:\n{joined}\n\n"
        "Return JSON: {\n"
        "  \"themes\":       [ {\"label\": str, \"angle\": str, \"post_idea\": str}, ... ],\n"
        "  \"banned_tags\":  [ str, ... ]   // tags that looked like spam / off-topic\n"
        "}\n"
        "Exactly 3–5 themes. `angle` is ≤18 words. `post_idea` is a concrete single post suggestion."
    )
    return await generate_json("analyze", prompt, system=system, max_tokens=1000)


async def run_once(cfg: NicheConfig, scraper: PublicScraper | None = None) -> int:
    """One mining pass. Returns number of samples gathered."""
    s = scraper or PublicScraper()
    # Pool tags to scan each cycle — rotate to avoid hammering the same handful
    pool = list({*cfg.hashtags.core, *cfg.hashtags.growth})
    if not pool:
        log.info("No hashtag pools configured — skipping trend miner")
        return 0
    random.shuffle(pool)
    scan = pool[:4]

    samples: list[dict] = []
    for tag in scan:
        posts = s.hashtag_posts(tag, limit=8)
        for p in posts:
            db.hashtag_upsert(tag, p.ig_pk, p.caption, p.likes, p.posted_at)
            samples.append({
                "hashtag": tag,
                "caption": p.caption,
                "likes": p.likes,
            })

    if not samples:
        return 0
    try:
        analysis = await _cluster(cfg, samples)
    except Exception as e:
        log.warning("Trend clustering failed: %s", e)
        return len(samples)

    for t in (analysis.get("themes") or [])[:5]:
        label = str(t.get("label") or "").strip()
        angle = str(t.get("angle") or "").strip()
        idea = str(t.get("post_idea") or "").strip()
        if not label:
            continue
        db.narrative_bump(label, sample_ref=idea[:120] or None)
        db.push_context(
            "trend",
            f"theme={label}. angle={angle}. idea={idea}",
            priority=3,
        )

    log.info("Trend miner: %d samples, %d themes", len(samples), len(analysis.get("themes") or []))
    return len(samples)
