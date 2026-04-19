"""Specificity rewrite — strip generic filler, ground draft in concrete detail.

Runs on every caption candidate after generation, before the critic ranks.
One fast free-tier LLM call takes (draft, context) and returns a tightened
version where vague/generic phrases are replaced by specific alternatives
drawn from the research context + RAG knowledge.

Example transforms (illustrative):
  "pro tips for better pullups"
    → "the 3 shoulder-position fixes that unlocked my first pullup"
  "game-changer workout"
    → "the 4-minute CNS primer I run before every session"
  "unlock your potential"
    → "the neural adaptation that kicks in around week 3"

Silent fallback: on ANY LLM failure or degenerate output (empty, far longer
than original, identical), return the original text unchanged. This stage
is a quality lift, not a dependency — the pipeline stays green without it.
"""
from __future__ import annotations

from instagram_ai_agent.core.config import NicheConfig
from instagram_ai_agent.core.llm import generate
from instagram_ai_agent.core.logging_setup import get_logger

log = get_logger(__name__)


GENERIC_PATTERNS = (
    "pro tip", "pro tips", "game-changer", "game changer",
    "next level", "level up", "unlock", "secret", "hack", "hacks",
    "master this", "you won't believe", "mind-blowing", "jaw-dropping",
    "the ultimate", "the best way", "the right way", "trust me",
    "take your X to the next", "elevate your", "reach new heights",
    "crush it", "smash your goals", "grind", "hustle",
)


def has_generic_filler(text: str) -> bool:
    """Cheap pre-check — skip the LLM call when the draft is already clean."""
    if not text:
        return False
    low = text.lower()
    return any(p in low for p in GENERIC_PATTERNS)


async def concretize(
    cfg: NicheConfig,
    draft: str,
    *,
    context: str = "",
    knowledge: str = "",
) -> str:
    """Rewrite ``draft`` to be more specific. Returns original on any failure.

    We only invoke the LLM when the draft actually contains generic filler —
    saves the call on already-clean drafts, which compounds at n=3 candidates.
    """
    if not draft or not draft.strip():
        return draft
    if not has_generic_filler(draft):
        return draft

    system = (
        "You are a ruthless copy editor for Instagram niche content. "
        "Your ONLY job is replacing generic/vague phrases with specific, "
        "concrete ones drawn from the provided CONTEXT and KNOWLEDGE. "
        "You never invent facts. You preserve the author's voice, tone, "
        "length, and structure. You return ONLY the rewritten caption — "
        "no preamble, no labels, no markdown, no quotes."
    )

    context_block = context.strip() or "(no specific context — use the niche expertise below)"
    knowledge_block = knowledge.strip() or "(no knowledge — don't invent)"

    prompt = f"""NICHE: {cfg.niche}
VOICE: {", ".join(cfg.voice.tone) if cfg.voice.tone else "unspecified"}
PERSONA: {cfg.voice.persona or "unspecified"}

CONTEXT (real signals — use these to ground specific rewrites):
{context_block}

KNOWLEDGE (facts you MAY cite — never invent beyond these):
{knowledge_block}

ORIGINAL CAPTION:
{draft}

REWRITE RULES
1. Replace every generic/vague phrase with a specific one grounded in CONTEXT or KNOWLEDGE.
2. Prefer numbers over adjectives ("3 shoulder-position fixes" beats "some fixes").
3. Prefer concrete nouns over abstractions ("the CNS primer" beats "the routine").
4. BAN these phrases entirely: {", ".join(GENERIC_PATTERNS)}.
5. Keep the voice, tone, and length (within ±15%).
6. Keep the original hook line if it's already specific — don't rewrite for its own sake.
7. Output the rewritten caption only. No preamble, quotes, labels, or markdown.
"""

    try:
        raw = await generate(
            "caption",
            prompt,
            system=system,
            max_tokens=400,
            temperature=0.3,
        )
    except Exception as e:
        log.debug("specificity_pass: LLM failed — keeping original: %s", e)
        return draft

    cleaned = _clean(raw)
    if not cleaned:
        return draft

    # Guardrails against bad rewrites. Specificity naturally expands length
    # (a vague adjective becomes a concrete phrase), so we allow generous
    # upward drift. Upper bound catches actual rambling.
    if cleaned == draft:
        return draft
    max_ok = max(len(draft) * 2.0, len(draft) + 80)
    min_ok = max(len(draft) * 0.4, 15)
    if len(cleaned) > max_ok:
        log.debug("specificity_pass: rewrite ran long (%d → %d) — keeping original",
                  len(draft), len(cleaned))
        return draft
    if len(cleaned) < min_ok:
        log.debug("specificity_pass: rewrite ran short (%d → %d) — keeping original",
                  len(draft), len(cleaned))
        return draft

    # If the rewrite still contains banned phrases, it failed its job — keep original
    if has_generic_filler(cleaned):
        log.debug("specificity_pass: rewrite still had filler — keeping original")
        return draft

    log.debug("specificity_pass: rewrote (%d → %d chars)", len(draft), len(cleaned))
    return cleaned


def _clean(raw: str) -> str:
    s = raw.strip()
    # Some models wrap with label ("Rewrite:" / "Caption:")
    for prefix in ("Rewrite:", "Caption:", "Rewritten caption:", "REWRITE:", "Output:"):
        if s.lower().startswith(prefix.lower()):
            s = s[len(prefix):].strip()
    # Strip matching quote wrappers
    for q in ('"', "'", "“", "”"):
        if s.startswith(q) and s.endswith(q):
            s = s[1:-1].strip()
    return s
