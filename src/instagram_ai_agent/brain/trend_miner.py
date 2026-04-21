"""Trend miner — scrape niche hashtags' top posts, cluster captions by theme."""
from __future__ import annotations

import random

from pydantic import BaseModel, Field

from instagram_ai_agent.brain.scraper import PublicScraper
from instagram_ai_agent.core import db
from instagram_ai_agent.core.config import NicheConfig
from instagram_ai_agent.core.llm import generate_json_model
from instagram_ai_agent.core.logging_setup import get_logger

log = get_logger(__name__)


class _Theme(BaseModel):
    label: str = Field(..., description="Short theme name — 2-5 words.")
    angle: str = Field(..., description="The niche-specific angle, ≤18 words.")
    post_idea: str = Field(..., description="Concrete single-post suggestion.")


class _TrendClusterResponse(BaseModel):
    themes: list[_Theme] = Field(..., min_length=3, max_length=5, description="3-5 distinct themes.")
    banned_tags: list[str] = Field(default_factory=list, description="Tags that looked like spam / off-topic.")


async def _cluster(cfg: NicheConfig, samples: list[dict]) -> _TrendClusterResponse:
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
        "Cluster the posts into 3-5 distinct themes. Each theme has a short label, "
        "a niche-specific angle (≤18 words), and a concrete post_idea. "
        "Also list any hashtags that looked like spam or off-topic."
    )
    return await generate_json_model(
        "analyze", prompt, _TrendClusterResponse,
        system=system, max_tokens=2500,
    )


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
        log.warning("Trend clustering failed: %s", str(e)[:200])
        return len(samples)

    for t in analysis.themes[:5]:
        label = t.label.strip()
        angle = t.angle.strip()
        idea = t.post_idea.strip()
        if not label:
            continue
        db.narrative_bump(label, sample_ref=idea[:120] or None)
        db.push_context(
            "trend",
            f"theme={label}. angle={angle}. idea={idea}",
            priority=3,
        )

    log.info("Trend miner: %d samples, %d themes", len(samples), len(analysis.themes))
    return len(samples)
