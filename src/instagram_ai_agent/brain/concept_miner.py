"""Concept miner — distil {hook, structure, payoff} from viral posts.

Scrapes-we-already-run (hashtag_top via trend_miner, reddit_harvester,
hackernews) leave raw content in the DB but the generator doesn't see
it as *structural* inspiration — only as trend signal. This module closes
that gap: we iterate the recent top-performers, ask the LLM to extract
the abstract pattern each one used (hook pattern, narrative structure,
payoff shape), and store that in ``concept_bank``.

The caption generator then pulls 3-6 concept_bank entries as context:
"here's what's working in your niche right now structurally — use it
as inspiration, not verbatim template."

Why this matters: inventing a hook from scratch on every post makes
generic content. Having 6 proven structural patterns in the context
means the LLM grounds its hook in what actually converts.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from instagram_ai_agent.core import db
from instagram_ai_agent.core.config import NicheConfig
from instagram_ai_agent.core.llm import generate_json_model
from instagram_ai_agent.core.logging_setup import get_logger

log = get_logger(__name__)


class _ConceptPattern(BaseModel):
    hook: str = Field(..., description="The abstracted hook pattern. Not verbatim — the structural class. E.g. 'unexpected number + common activity + surprising reveal' rather than the actual headline.")
    structure: str = Field(..., description="The narrative scaffold: how the post flows from opening to payoff. 2-3 sentences describing the shape.")
    payoff: str = Field(..., description="What makes the reader save/share — the value or emotional beat delivered at the end.")


class _ConceptExtraction(BaseModel):
    patterns: list[_ConceptPattern] = Field(..., min_length=1, max_length=4, description="1-4 extracted patterns from the sampled posts.")


def _recent_top_samples(cfg: NicheConfig, limit: int = 8) -> list[dict]:
    """Pull the highest-engagement recent posts from hashtag_top, our best
    concrete signal of what's working in the niche right now."""
    if not cfg.hashtags.core:
        return []
    tags = list({*cfg.hashtags.core, *cfg.hashtags.growth})
    if not tags:
        return []
    qs = ",".join("?" * len(tags))
    rows = db.get_conn().execute(
        f"""
        SELECT hashtag, ig_pk, caption, likes
        FROM hashtag_top
        WHERE hashtag IN ({qs})
          AND likes > 100
        ORDER BY likes DESC, scraped_at DESC
        LIMIT ?
        """,
        (*tags, limit),
    ).fetchall()
    return [dict(r) for r in rows]


async def run_once(cfg: NicheConfig) -> int:
    """One mining pass. Returns the number of patterns added to concept_bank."""
    samples = _recent_top_samples(cfg, limit=8)
    if not samples:
        log.debug("concept_miner: no samples to mine (hashtag_top empty)")
        return 0

    # Truncate captions so the whole batch fits in the prompt budget.
    joined = "\n---\n".join(
        f"[likes={s['likes']}] #{s['hashtag']}: {s['caption'][:240]}"
        for s in samples if s.get("caption")
    )
    if not joined:
        return 0

    system = (
        f"You are a growth analyst for an Instagram page about {cfg.niche}.\n"
        f"Sub-topics: {', '.join(cfg.sub_topics)}.\n"
        "You read samples of high-engagement posts in this niche and extract "
        "the STRUCTURAL PATTERNS they use — not the verbatim content. Future "
        "posts on the page will be built around these patterns so they "
        "converge on what converts.\n\n"
        "For each pattern you extract:\n"
        "- hook: the shape of the opening line (NOT the actual headline). "
        "  e.g., 'specific number + timeframe + contrarian reveal' not 'I did "
        "  100 pullups in 30 days and X'.\n"
        "- structure: the narrative scaffold — how it flows from open to payoff.\n"
        "- payoff: what makes it save-worthy or share-worthy at the end."
    )
    prompt = (
        f"Niche sample posts (top by engagement):\n{joined}\n\n"
        "Extract 2-4 distinct structural patterns from these. "
        "Abstract — don't quote the original posts verbatim."
    )

    try:
        response = await generate_json_model(
            "analyze", prompt, _ConceptExtraction,
            system=system, max_tokens=1500,
        )
    except Exception as e:
        log.warning("concept_miner: extraction failed — %s", str(e)[:200])
        return 0

    # Compute an aggregate source score from the sampled posts' engagement,
    # so fresher mining passes (newer signals) get a higher priority.
    agg_score = sum(s.get("likes", 0) for s in samples) // max(len(samples), 1)

    added = 0
    for p in response.patterns:
        if not p.hook or not p.structure or not p.payoff:
            continue
        db.concept_append(
            hook=p.hook, structure=p.structure, payoff=p.payoff,
            source=f"hashtag_top:{','.join(sorted({s['hashtag'] for s in samples}))}",
            source_score=agg_score,
        )
        added += 1
    if added:
        log.info("concept_miner: +%d patterns (avg source_score=%d)", added, agg_score)
    return added


def build_concept_block(max_concepts: int = 5) -> str:
    """Render the top concept_bank entries as a system-prompt block for
    caption generation. Marks them as used so fresh patterns get rotated in."""
    concepts = db.concept_top(limit=max_concepts)
    if not concepts:
        return ""
    db.concept_touch([c["id"] for c in concepts])
    lines = []
    for c in concepts:
        lines.append(
            f"  - HOOK SHAPE: {c['hook']}\n"
            f"    STRUCTURE: {c['structure']}\n"
            f"    PAYOFF:    {c['payoff']}"
        )
    return (
        "\n\nCONCEPT BANK — patterns currently working in this niche. Use "
        "these as STRUCTURAL inspiration; do not copy verbatim. Pick the "
        "one that best fits the content you're about to write:\n"
        + "\n".join(lines)
    )
