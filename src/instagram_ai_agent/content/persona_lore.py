"""Persona lore — running character memory across posts.

Problem this solves: every post was generated independently with no
knowledge of previous posts. A page claiming "I'm training for a muscle-up
by June" in post 3 would have no way to reference that commitment in
post 20 — the persona drifted post-to-post.

Mechanism:
  1. After a post ships, the extractor distils any NEW claim/commitment/
     preference/milestone/incident about the "character" from the caption,
     stores it in persona_lore.
  2. Before the next post generates, the injector pulls the top-weighted
     facts and prepends them to the system prompt as a LORE block.
  3. The caption generator sees a running timeline of what the page has
     already said — callbacks become possible, contradictions get avoided.

Fact extraction is best-effort. A post caption may contain zero facts
(pure tip content) — that's fine; we skip. Weights decay naturally via
`lore_top` preferring higher-weight entries for injection.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from instagram_ai_agent.core import db
from instagram_ai_agent.core.config import NicheConfig
from instagram_ai_agent.core.llm import generate_json_model
from instagram_ai_agent.core.logging_setup import get_logger

log = get_logger(__name__)


class _LoreFact(BaseModel):
    category: str = Field(..., description="One of: claim, commitment, preference, milestone, incident.")
    fact: str = Field(..., description="Short first-person statement. Reuseable in future captions.")
    weight: float = Field(0.7, ge=0.0, le=1.0, description="How load-bearing this fact is, 0-1.")


class _LoreExtraction(BaseModel):
    facts: list[_LoreFact] = Field(default_factory=list, description="0-3 new facts from this post. Empty is valid.")


_VALID_CATEGORIES = {"claim", "commitment", "preference", "milestone", "incident"}


async def extract_from_post(cfg: NicheConfig, *, caption: str, post_id: int | None = None) -> int:
    """Extract 0-3 new persona facts from a caption and append to lore.
    Returns the number of facts added."""
    if not caption or len(caption) < 20:
        return 0

    system = (
        f"You maintain the running character memory for a faceless Instagram page about "
        f"{cfg.niche}. Persona: {cfg.voice.persona}.\n\n"
        "Your job is to extract any NEW facts this caption reveals about the character — "
        "claims they've made, commitments they've declared, preferences they've stated, "
        "milestones they've hit, or incidents they've shared. These will be used as memory "
        "in future captions so the persona stays coherent across posts.\n\n"
        "Rules:\n"
        "- Extract 0-3 facts. Zero is valid — many captions are pure tips with no personal fact.\n"
        "- Only facts that could reasonably reappear in future captions. Skip trivia.\n"
        "- Facts must be first-person statements ('I trained for 8 weeks to get 10 pullups').\n"
        "- Skip the hook/CTA — those are rhetorical, not character facts.\n"
        "- Skip facts that are generic niche advice (those belong to the niche, not the persona).\n"
        "- category: claim | commitment | preference | milestone | incident."
    )
    prompt = f"Caption:\n{caption[:1200]}\n\nExtract any new persona facts."

    try:
        response = await generate_json_model(
            "analyze", prompt, _LoreExtraction,
            system=system, max_tokens=600,
        )
    except Exception as e:
        log.debug("persona_lore: extraction failed (non-fatal) — %s", str(e)[:150])
        return 0

    added = 0
    for f in response.facts[:3]:
        cat = f.category.strip().lower()
        if cat not in _VALID_CATEGORIES:
            # Coerce unknown categories to "claim" — safest default
            cat = "claim"
        fact = f.fact.strip()
        if not fact or len(fact) < 10:
            continue
        db.lore_append(category=cat, fact=fact, post_id=post_id, weight=f.weight)
        added += 1
    if added:
        log.info("persona_lore: +%d facts from post_id=%s", added, post_id)
    return added


def build_lore_block(max_facts: int = 8) -> str:
    """Render the top-weighted lore entries as a system-prompt block.
    Returns empty string on fresh accounts with no lore yet."""
    facts = db.lore_top(limit=max_facts)
    if not facts:
        return ""
    # Mark these as used so unused facts decay in priority over time
    db.lore_touch([f["id"] for f in facts])

    lines = []
    for f in facts:
        prefix = {
            "claim":       "claim",
            "commitment":  "committed to",
            "preference":  "prefers",
            "milestone":   "hit milestone",
            "incident":    "shared incident",
        }.get(f["category"], "said")
        lines.append(f"  - [{prefix}] {f['fact']}")
    return (
        "\n\nPERSONA MEMORY — the account has already shared these facts "
        "in previous posts. Keep this caption consistent with them; "
        "callbacks are welcome, contradictions are not:\n"
        + "\n".join(lines)
    )
