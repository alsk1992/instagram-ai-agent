"""Slide 1 / reel-frame-1 scroll-stop hook generator.

The single biggest leverage point in the feed. Instagram users decide in
~0.3 seconds whether to swipe past or stop. That decision is made on
slide 1 (carousel), thumbnail/first-frame (reel), or the image (meme).
If slide 1 is generic, nobody sees slide 2.

Current gap: the carousel generator writes all 7 slides in ONE JSON call,
so slide 1 gets the same LLM attention as slide 4. This stage runs ONE
upstream call that asks a free-tier model to brainstorm 8 slide-1 hook
candidates, self-score each on:

  * scroll_stop: would a human actually stop on this in a dense feed?
  * specificity: concrete nouns / numbers / names vs vague
  * visual_dominance: would the headline text POP at thumbnail size?

The winner feeds downstream as a hard constraint ("slide 1 MUST use
this exact hook") — slides 2..N get written to fulfil its promise.

Silent fallback on any LLM failure — downstream carousel generator's
existing slide-1 prompt takes over, so this stage is a quality lift
not a dependency.
"""
from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field

from instagram_ai_agent.content import voice_fingerprint
from instagram_ai_agent.core.config import NicheConfig
from instagram_ai_agent.core.llm import generate_json_model
from instagram_ai_agent.core.logging_setup import get_logger


# ─── Pydantic schema ─────────────────────────────────────────────────
class _Slide1Candidate(BaseModel):
    title: str = Field(..., description="Headline ≤9 words — the scroll-stop line with a number, name, or concrete noun.")
    body: str = Field(..., description="Supporting line ≤14 words sharpening the promise.")
    pattern: str = Field("", description="Hook pattern type: stat_shock|contrarian|curiosity|claim|list|question|reveal|before_after.")
    scroll_stop: int = Field(..., ge=0, le=10, description="0-10, would a human stop in a dense feed.")
    specificity: int = Field(..., ge=0, le=10, description="0-10, concrete vs vague.")
    visual_dominance: int = Field(..., ge=0, le=10, description="0-10, would the text POP at thumbnail size.")
    total: int = Field(0, description="scroll_stop + specificity + visual_dominance.")


class _Slide1Winner(BaseModel):
    title: str = Field(..., description="The chosen headline, verbatim.")
    body: str = Field(..., description="The chosen supporting line, verbatim.")
    why: str = Field(..., description="ONE sentence rationale.")


class _Slide1Response(BaseModel):
    candidates: list[_Slide1Candidate] = Field(..., min_length=3, description="At least 3 scored candidates.")
    winner: _Slide1Winner

log = get_logger(__name__)


@dataclass(frozen=True)
class Slide1Hook:
    title: str          # headline — ≤9 words, the scroll-stop line
    body: str           # supporting line — ≤14 words, sharpens the promise
    why: str            # one-sentence rationale (kept for retro/debug)

    def as_constraint_block(self) -> str:
        return (
            "[SLIDE 1 HARD CONSTRAINT — use this exact hook, do not paraphrase]\n"
            f"title: {self.title}\n"
            f"body: {self.body}\n"
            "Slides 2..N must deliver on the specific promise of slide 1."
        )


BANNED_HOOK_PHRASES = (
    "pro tip", "pro tips", "game-changer", "game changer", "next level",
    "level up", "unlock", "secret", "hack", "hacks", "master this",
    "you won't believe", "mind-blowing", "the ultimate", "crush it",
)


async def best_slide1_hook(
    cfg: NicheConfig,
    *,
    trend_context: str,
    angle_hook: str | None = None,
    contrarian: bool = False,
    n_candidates: int = 8,
) -> Slide1Hook | None:
    """Brainstorm N slide-1 hook candidates, self-rank, return winner.

    ``angle_hook`` is the upstream ``angle_brainstorm`` winner if available —
    when provided, the stage is constrained to variants of that angle. When
    absent, candidates are generated free-form from the trend context.

    Returns None on LLM failure or degenerate output — caller must not
    treat that as fatal.
    """
    voice = cfg.voice
    voice_line = ", ".join(voice.tone) if voice.tone else "(unspecified)"
    persona_line = voice.persona or "(unspecified)"
    forbidden_line = ", ".join(voice.forbidden) or "n/a"

    contrarian_block = (
        "CONTRARIAN MODE — the slide 1 hook must name or allude to a widely-held "
        "niche belief that's about to be inverted. Build curiosity about the flip, "
        "don't spoil the body."
        if contrarian else ""
    )

    angle_block = (
        f"UPSTREAM WINNING ANGLE (all candidates must serve this angle):\n{angle_hook}"
        if angle_hook else "UPSTREAM ANGLE: none — generate from trend context."
    )

    system = (
        "You are a senior Instagram carousel designer. You think in terms of "
        "FEED VIEW — the audience sees a 1080×1350 thumbnail next to 5 other "
        "posts. Your slide-1 headline has 0.3 seconds to stop the scroll. "
        "You hate generic phrases that could run on any niche account. You "
        "deploy specific numbers, names, and concrete nouns. You know that "
        "curiosity gaps, contrarian statements, and unexpected claims are the "
        "three patterns that actually work. You know ALL-CAPS headlines and "
        "character-dense lines die at thumbnail size."
    )
    # Voice anchor — hooks should match the page's shipped style, not LLM-default voice
    examples = voice_fingerprint.pick_voice_examples(n=5)
    system += voice_fingerprint.build_voice_block(examples)

    prompt = f"""NICHE: {cfg.niche}
AUDIENCE: {cfg.target_audience}
VOICE tone: {voice_line}
PERSONA: {persona_line}
NEVER use words: {forbidden_line}

{angle_block}

{contrarian_block}

TREND CONTEXT (real signals to anchor in):
{trend_context.strip() or "(no fresh signals — lean on niche expertise)"}

TASK
1. Generate {n_candidates} DISTINCT slide-1 hook candidates.
   * Each = (title ≤9 words, body ≤14 words).
   * Title is THE scroll-stop line. Body sharpens what the carousel delivers.
   * BAN these phrases: {", ".join(BANNED_HOOK_PHRASES)}.
   * Each hook must contain at least one: specific number, name, concrete noun, or unexpected claim.
   * Vary patterns — don't submit 8 question-format hooks. Mix: stat-shock, contrarian-statement, curiosity-gap, counter-intuitive claim, list-promise, question, reveal, before/after.
2. Self-score each candidate (0–10 integers) on three axes:
   - scroll_stop: would a human actually STOP in a dense feed?
   - specificity: concrete (numbers, names, nouns) vs generic
   - visual_dominance: does the title POP at 1080×1350 thumbnail size? Short + high-contrast > long + dense.
3. Compute total = scroll_stop + specificity + visual_dominance.
4. Pick the highest-total candidate. Break ties by highest scroll_stop.
5. Explain WHY the winner wins in ONE sentence.

OUTPUT JSON EXACTLY:
{{
  "candidates": [
    {{"title": "...", "body": "...", "pattern": "stat_shock|contrarian|curiosity|claim|list|question|reveal|before_after", "scroll_stop": 8, "specificity": 9, "visual_dominance": 7, "total": 24}}
  ],
  "winner": {{"title": "...", "body": "...", "why": "..."}}
}}
"""

    try:
        response = await generate_json_model(
            "bulk",
            prompt,
            _Slide1Response,
            system=system,
            max_tokens=2048,
            temperature=0.85,
        )
    except Exception as e:
        log.warning("slide1_hook: all JSON_CHAIN endpoints failed schema — %s", str(e)[:200])
        return None

    # Prefer rubric over model-declared winner when they disagree — models
    # sometimes pick a random candidate despite the instruction to take the
    # highest-total one.
    if response.candidates:
        best = max(
            response.candidates,
            key=lambda c: (c.total or c.scroll_stop + c.specificity + c.visual_dominance),
        )
        best_total = best.total or (best.scroll_stop + best.specificity + best.visual_dominance)
        if best.title != response.winner.title and best_total > 0:
            log.debug("slide1_hook: overriding model winner with highest-total candidate")
            title = best.title.strip()
            body = best.body.strip()
            why = f"highest-total candidate ({best_total})"
        else:
            title = response.winner.title.strip()
            body = response.winner.body.strip()
            why = response.winner.why.strip() or f"rubric pick ({best_total})"
    else:
        title = response.winner.title.strip()
        body = response.winner.body.strip()
        why = response.winner.why.strip()

    if not title or not body:
        log.warning("slide1_hook: winner missing title/body (%r / %r)", title, body)
        return None

    # Enforce length ceilings from the prompt — models ignore them sometimes.
    if len(title.split()) > 12 or len(body.split()) > 20:
        log.warning("slide1_hook: winner too long (title=%d body=%d words)",
                    len(title.split()), len(body.split()))
        return None

    low = f"{title} {body}".lower()
    if any(bad in low for bad in BANNED_HOOK_PHRASES):
        log.warning("slide1_hook: winner contained a banned phrase — %r", title)
        return None

    n_gen = len(response.candidates)
    log.info(
        "slide1_hook: %d candidates → winner %r",
        n_gen, title[:80],
    )
    return Slide1Hook(title=title, body=body, why=why)
