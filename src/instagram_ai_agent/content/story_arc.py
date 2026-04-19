"""Personal-story-arc validator — prescriptive → lived-experience framing.

Top creator research is consistent: "I did X for 6 weeks and Y happened"
posts out-perform "you should do X" by a wide margin on save + share
metrics. Prescriptive framing reads as "AI assistant", lived-experience
reads as "real human with real results".

This validator runs per-candidate BEFORE the critic. Flow:

  1. Cheap regex scan flags prescriptive density (you should, here's how,
     imperative opener, the right way...).
  2. If the draft is heavy-prescriptive AND the format benefits from
     story arcs (carousel, reel_stock, reel_ai, photo), trigger ONE LLM
     rewrite pass that converts prescription → first-person story
     (time marker + concrete outcome).
  3. If the rewrite still reads heavy-prescriptive, drop the candidate
     rather than ship generic how-to content.
  4. Memes/quotes/stories are skipped — prescription works fine there.

Silent fallback to original on any LLM failure.
"""
from __future__ import annotations

import re

from instagram_ai_agent.core.config import NicheConfig
from instagram_ai_agent.core.llm import generate
from instagram_ai_agent.core.logging_setup import get_logger

log = get_logger(__name__)


# Formats where lived-experience genuinely outperforms prescription.
# Meme + quote + story_* stay untouched — prescription fits those.
STORY_ARC_FORMATS = {"carousel", "reel_stock", "reel_ai", "photo", "human_photo"}


_PRESCRIPTIVE_PATTERNS = (
    r"\byou (?:should|need to|must|have to|gotta|ought to)\b",
    r"\bhere(?:'s| is) (?:how|what|why)\b",
    r"\bfollow these\b",
    r"\bthe (?:right|only|best) way\b",
    r"\b(?:stop|start|try|do|make) (?:doing|being)\b",
    r"^(?:stop|start|try|do|use|make|follow|avoid)\s",  # imperative openers
)

_LIVED_EXPERIENCE_PATTERNS = (
    r"\bI (?:did|tried|spent|trained|ran|used|built|made|noticed|realized|figured)\b",
    r"\bmy (?:experience|result|routine|take|mistake)\b",
    r"\bfor \d+ (?:week|month|day|year)s?\b",
    r"\bafter \d+ (?:week|month|day|year)s?\b",
    r"\bat week \d+\b",
    r"\bturns out\b",
)

_PRESCRIPTIVE_RE = re.compile("|".join(_PRESCRIPTIVE_PATTERNS), re.IGNORECASE | re.MULTILINE)
_LIVED_RE = re.compile("|".join(_LIVED_EXPERIENCE_PATTERNS), re.IGNORECASE | re.MULTILINE)


def score_prescription(text: str) -> float:
    """Prescription density: 0 = pure lived-experience, 1 = pure how-to.

    Returns a float in [0, 1]. We divide by ``sentence_count`` to keep
    the score length-invariant so a 3-sentence draft with 2 prescriptive
    phrases scores higher than a 10-sentence draft with 3.
    """
    if not text or not text.strip():
        return 0.0
    presc_hits = len(_PRESCRIPTIVE_RE.findall(text))
    lived_hits = len(_LIVED_RE.findall(text))
    sentence_count = max(1, len(re.findall(r"[.!?]+", text)))
    presc_density = presc_hits / sentence_count
    lived_density = lived_hits / sentence_count
    # Lived-experience cancels prescription roughly 1:1. Clamp to [0, 1].
    net = max(0.0, min(1.0, presc_density - lived_density * 0.8))
    return net


def is_heavy_prescriptive(text: str, threshold: float = 0.35) -> bool:
    return score_prescription(text) >= threshold


async def convert_to_story(
    cfg: NicheConfig,
    draft: str,
    *,
    context: str = "",
    knowledge: str = "",
) -> str:
    """One LLM rewrite: prescriptive draft → first-person story arc.

    On any failure (LLM error, rewrite still prescriptive, degenerate length),
    returns the ORIGINAL draft. Callers handle the still-prescriptive case
    themselves.
    """
    if not draft.strip():
        return draft
    if not is_heavy_prescriptive(draft):
        return draft

    system = (
        "You rewrite prescriptive Instagram captions ('you should X', "
        "'here's how to Y') into first-person lived-experience arcs "
        "('I did X for 6 weeks — week 5 was where it clicked, because Z'). "
        "You preserve the topic, the CTA, and the voice. You add a time "
        "marker and a concrete outcome. You never invent facts beyond what "
        "the CONTEXT / KNOWLEDGE supports. You return ONLY the rewritten "
        "caption — no preamble, no labels, no quotes, no markdown."
    )

    context_block = context.strip() or "(no specific context — draw on niche expertise)"
    knowledge_block = knowledge.strip() or "(no indexed knowledge — do not invent)"

    prompt = f"""NICHE: {cfg.niche}
VOICE tone: {", ".join(cfg.voice.tone) if cfg.voice.tone else "(none)"}
PERSONA: {cfg.voice.persona or "(none)"}

CONTEXT (real signals):
{context_block}

KNOWLEDGE:
{knowledge_block}

ORIGINAL PRESCRIPTIVE DRAFT:
{draft}

REWRITE RULES
1. Replace prescription with first-person story: "I did X for Y time, Z happened".
2. Add ONE time marker (e.g., "for 6 weeks", "at week 3", "after 2 months").
3. Add ONE concrete outcome or insight from the experience.
4. Preserve the existing CTA line — don't rewrite it, move it to the end.
5. Keep the length within ±25% of the original.
6. Output the rewritten caption ONLY. No preamble, labels, markdown, or quotes.
"""

    try:
        raw = await generate(
            "caption",
            prompt,
            system=system,
            max_tokens=500,
            temperature=0.6,
        )
    except Exception as e:
        log.debug("story_arc: LLM failed — keeping original: %s", e)
        return draft

    cleaned = _clean(raw)
    if not cleaned or cleaned == draft:
        return draft

    # Guardrails — bad rewrite = keep original. Story arcs legitimately
    # expand length (they add time markers + outcomes), so we allow 2× or
    # +120 chars, whichever is larger.
    max_ok = max(len(draft) * 2.0, len(draft) + 120)
    min_ok = max(len(draft) * 0.5, 30)
    if len(cleaned) > max_ok or len(cleaned) < min_ok:
        log.debug("story_arc: rewrite length drift (%d → %d) — keeping original",
                  len(draft), len(cleaned))
        return draft

    # Still prescriptive after rewrite = model failed its job.
    # Return the ORIGINAL so the pipeline can score and possibly drop it.
    if is_heavy_prescriptive(cleaned):
        log.debug("story_arc: rewrite still prescriptive — keeping original")
        return draft

    log.debug("story_arc: rewrote prescriptive → story (%d → %d chars)",
              len(draft), len(cleaned))
    return cleaned


def _clean(raw: str) -> str:
    s = raw.strip()
    for prefix in ("Rewrite:", "Caption:", "Rewritten caption:", "STORY:", "Output:"):
        if s.lower().startswith(prefix.lower()):
            s = s[len(prefix):].strip()
    for q in ('"', "'", "“", "”"):
        if s.startswith(q) and s.endswith(q):
            s = s[1:-1].strip()
    return s
