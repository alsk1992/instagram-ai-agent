"""Pre-generation angle + hook brainstorm.

Runs ONCE per post before the generator. One LLM call asks a fast free
model (Gemini Flash → Llama 3.3 → Groq) to:

  1. Produce N distinct angles grounded in the research context,
  2. Write a scroll-stopping hook for each,
  3. Self-score each on specificity / hook_strength / evidence_anchor,
  4. Pick the winner.

The winner becomes a hard constraint prepended to ``trend_context`` so
the downstream generator builds the piece around a sharp, research-anchored
claim rather than a vibes-based topic. Caption best-of-N can't rescue a
weak skeleton — this stage fixes the skeleton.
"""
from __future__ import annotations

from dataclasses import dataclass

from instagram_ai_agent.core.config import NicheConfig
from instagram_ai_agent.core.llm import generate_json
from instagram_ai_agent.core.logging_setup import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class WinningAngle:
    angle: str
    hook: str
    why: str

    def as_context_block(self) -> str:
        return (
            "[WINNING_ANGLE — build the piece around this EXACT hook]\n"
            f"hook: {self.hook}\n"
            f"angle: {self.angle}\n"
            f"why: {self.why}\n"
            "body rules: specific (names, numbers, concrete nouns). "
            "no clichés — never 'pro tips', 'game-changer', 'next level', "
            "'unlock', 'secret', 'hack', 'level up'."
        )


FORBIDDEN_HOOK_WORDS = (
    "game-changer", "game changer", "next level", "level up", "pro tips",
    "unlock", "secret", "hack", "hacks", "master this", "you won't believe",
)


async def brainstorm_angle(
    cfg: NicheConfig,
    *,
    format_name: str,
    sub_topic: str | None,
    research_context: str,
    archetype_hook: str | None = None,
    contrarian: bool = False,
    n_angles: int = 15,
) -> WinningAngle | None:
    """Brainstorm N angles, self-rank, return the winner.

    Returns ``None`` on any failure (LLM error, malformed JSON, empty
    winner). Callers must be defensive — a ``None`` return means "fall
    through to the prior research-only context path" and is NOT fatal.
    """
    voice = cfg.voice
    voice_lines: list[str] = []
    if voice.tone:
        voice_lines.append(f"tone: {', '.join(voice.tone)}")
    if voice.persona:
        voice_lines.append(f"persona: {voice.persona}")
    if voice.forbidden:
        voice_lines.append(f"never use these words: {', '.join(voice.forbidden)}")
    voice_block = "\n".join(voice_lines) or "(no voice constraints)"

    contrarian_block = (
        "CONTRARIAN MODE — the winning angle must invert a mainstream niche belief. "
        "Structure: widely-believed claim → 'actually the opposite' → specific evidence. "
        "Generic disagreement without evidence is NOT contrarian — it's outrage-bait. Reject it."
        if contrarian else ""
    )

    archetype_block = (
        f"ARCHETYPE HOOK TEMPLATE (use as a seed, riff hard): {archetype_hook}"
        if archetype_hook else "ARCHETYPE: free choice."
    )

    research_block = research_context.strip() or "(no fresh signals this cycle — lean on niche expertise)"

    system = (
        "You are a senior content strategist for an Instagram niche account. "
        "Your job: reject generic, vibes-based content before it gets written. "
        "You hate tip lists without a through-line, hooks that could fit any niche, "
        "claims with no evidence, and filler words. A good angle is a specific claim "
        "or framing — not a topic. A good hook contains a number, a name, or a concrete "
        "noun. Anything that could run on 50 other accounts fails."
    )

    prompt = f"""NICHE: {cfg.niche}
SUB-TOPIC FOCUS: {sub_topic or "(rotate freely)"}
FORMAT: {format_name}

{voice_block}

{archetype_block}

{contrarian_block}

RESEARCH CONTEXT (real signals from reddit, competitors, trends, events, RAG):
{research_block}

TASK
1. Brainstorm {n_angles} DISTINCT angles for a single {format_name} post on the niche+sub-topic.
   * An angle is a specific CLAIM or FRAMING, not a topic.
   * GOOD examples (fitness niche): "most guys quit pullups one rep before CNS adapts",
     "the 5-minute mobility warmup costs you 40% of your workout".
   * BAD examples (reject these shapes): "pullup tips", "mobility matters", "5 ways to X".
   * Ground each angle in the research context where possible — cite which signal inspired it.
2. For each angle, write ONE scroll-stopping hook under 12 words.
   * Must contain a specific number, name, or concrete noun.
   * NO clichés: {", ".join(FORBIDDEN_HOOK_WORDS)}.
   * If your hook could run on any niche, it fails — rewrite.
3. Self-score each angle (0–10 integers) on:
   - specificity: concrete vs generic
   - hook_strength: scroll-stop potential (curiosity gap, contrarian, emotion)
   - evidence_anchor: can the body cite a study, data point, quote, or lived experience
4. Compute total = specificity + hook_strength + evidence_anchor.
5. Pick the highest-total angle as "winner". Break ties by highest hook_strength.
6. Explain WHY the winner wins in ONE sentence.

OUTPUT JSON EXACTLY:
{{
  "angles": [
    {{"angle": "...", "hook": "...", "source_signal": "...", "specificity": 8, "hook_strength": 9, "evidence_anchor": 7, "total": 24}}
  ],
  "winner": {{"angle": "...", "hook": "...", "why": "..."}}
}}
"""

    try:
        data = await generate_json(
            "bulk",
            prompt,
            system=system,
            max_tokens=2048,
            temperature=0.85,
        )
    except Exception as e:
        log.warning("angle_brainstorm: LLM call failed — %s", e)
        return None

    if not isinstance(data, dict):
        log.warning("angle_brainstorm: expected dict, got %s", type(data).__name__)
        return None

    winner = data.get("winner")
    if not isinstance(winner, dict):
        log.warning("angle_brainstorm: no winner field in response")
        return None

    angle = str(winner.get("angle") or "").strip()
    hook = str(winner.get("hook") or "").strip()
    why = str(winner.get("why") or "").strip()

    if not angle or not hook:
        log.warning("angle_brainstorm: winner missing angle/hook (angle=%r hook=%r)", angle, hook)
        return None

    # Hard cliché check — some models still smuggle them through despite the prompt.
    lower = hook.lower()
    if any(bad in lower for bad in FORBIDDEN_HOOK_WORDS):
        log.warning("angle_brainstorm: winner hook %r contains a forbidden cliché", hook)
        return None

    n_generated = len(data.get("angles") or [])
    log.info(
        "angle_brainstorm: %d angles → winner hook=%r",
        n_generated, hook[:80],
    )
    return WinningAngle(angle=angle, hook=hook, why=why)
