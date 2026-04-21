"""Caption generation — niche-voice calibrated via niche.yaml."""
from __future__ import annotations

from instagram_ai_agent.content import voice_fingerprint
from instagram_ai_agent.core.config import NicheConfig
from instagram_ai_agent.core.llm import generate

_SYSTEM_TEMPLATE = """\
You are writing an Instagram caption for a page about: {niche}.
Sub-topics: {sub_topics}.
Target audience: {audience}.

Voice persona: {persona}
Tone: {tone}
Never: {forbidden}

Constraints:
- 1–3 short lines, maximum ~180 characters of body text (hashtags added separately).
- No cringe hustle clichés, no hashtags in the caption body.
- No emojis unless they earn their place — one max.
- End with a single call-to-action pulled from: {ctas}.
- Match the post format: {format_hint}

Specificity rules (hard):
- Every line must contain a concrete noun, number, or name. "3 mobility drills" beats "some drills".
- BAN these phrases: pro tips, game-changer, next level, level up, unlock, secret, hack, hacks, master this, you won't believe, mind-blowing, the ultimate, the best way, crush it, grind, hustle.
- If a sentence could run on 50 other niche accounts, rewrite it to one that couldn't.
- Prefer lived-experience framing ("I did X for N weeks, result was Y") over prescription ("you should X").

OUTPUT FORMAT — CRITICAL:
- Your response is pasted DIRECTLY into the Instagram caption field, as-is.
- DO NOT explain what you are about to write. DO NOT say "We need to...", "Let me think...", "The caption is:", "Output:", "Here is the caption:", or any similar preamble.
- DO NOT wrap the caption in quotes. DO NOT add a label. DO NOT emit a JSON object.
- Emit ONLY the final caption text. Your first character must be the first character of the caption.
"""


_CONTRARIAN_BLOCK = """

CONTRARIAN MODE — this post is a hot take:
- Open by naming a widely-held niche belief, then flip it.
- Lead with a specific claim, not a moral lecture.
- Back it with ONE concrete reason, example, or number — vague opinion
  sounds like a rant.
- Avoid contrarian takes on: {avoid_topics}.
- Never touch: medical advice, vaccines, cancer cures, extreme diets,
  self-harm, political candidates, ethnic/religious generalisations.
"""


def build_system(cfg: NicheConfig, format_name: str, *, contrarian: bool = False) -> str:
    format_hint = {
        "meme": "A meme — punchy, funny, minimal caption (often just a setup line).",
        "quote_card": "A quote card — caption should reinforce or expand the quote without repeating it.",
        "carousel": "A carousel — caption is the hook; content lives on the slides.",
        "reel_stock": "A reel with stock footage and voiceover — caption is the hook to keep them watching.",
        "reel_ai": "An AI-generated reel — caption hooks, hints at the visual payoff.",
        "photo": "A photo post — caption sets context for the image, sparks a save.",
        "human_photo": "A portrait-style post of a real person in a niche moment — caption speaks directly or observationally about the moment, never describes the subject visually.",
        "story_human": "A portrait story — ≤10 word line that plays off the scene.",
        # Stories use short captions or none at all — big, conversational, stickers do the real work
        "story_quote": "A story quote card — caption is a single short line (≤10 words) or empty.",
        "story_announcement": "A story announcement — short CTA line (≤8 words).",
        "story_photo": "A photo story — optional one-liner (≤10 words) describing the moment.",
        "story_video": "A video story — caption is a short hook (≤10 words) layered atop the clip.",
    }.get(format_name, "A niche Instagram post.")

    base = _SYSTEM_TEMPLATE.format(
        niche=cfg.niche,
        sub_topics=", ".join(cfg.sub_topics),
        audience=cfg.target_audience,
        persona=cfg.voice.persona,
        tone=", ".join(cfg.voice.tone),
        forbidden=", ".join(cfg.voice.forbidden) or "none",
        ctas=", ".join(cfg.voice.cta_styles),
        format_hint=format_hint,
    )
    if contrarian:
        base += _CONTRARIAN_BLOCK.format(
            avoid_topics=", ".join(cfg.contrarian.avoid_topics) or "none specified",
        )

    # Voice fingerprint — anchor to shipped style, not keyword interpretation.
    # Empty block on fresh accounts (<2 posts of history) → fallback to persona.
    examples = voice_fingerprint.pick_voice_examples(n=5)
    base += voice_fingerprint.build_voice_block(examples)
    return base


async def generate_caption(
    cfg: NicheConfig,
    format_name: str,
    *,
    context: str,
    knowledge: str | None = None,
    contrarian: bool = False,
) -> str:
    """Generate a caption given a content context (e.g., the meme top text,
    the reel script).

    ``knowledge`` is the pre-computed RAG snippet block. Pipeline computes
    it once per post and passes it to every caption candidate + critic so
    we don't burn N × M Gemini embedding calls per generation. When None,
    we fall back to fetching it here (still safe but slower).
    """
    system = build_system(cfg, format_name, contrarian=contrarian)

    if knowledge is None:
        try:
            from instagram_ai_agent.brain import rag
            knowledge = await rag.context_for(context or cfg.niche, cfg.rag)
        except Exception as e:
            knowledge = ""
            from instagram_ai_agent.core.logging_setup import get_logger
            get_logger(__name__).debug("captions: RAG retrieval failed: %s", e)

    if knowledge:
        system = (
            system
            + "\n\nNiche knowledge (cite concretely when it fits; never invent):\n"
            + knowledge
        )

    prompt = (
        f"Write the caption. Context for this specific post:\n{context}\n\n"
        f"Output the caption text only — no hashtags, no quotes, no labels."
    )
    raw = await generate("caption", prompt, system=system, max_tokens=400)
    return _clean_caption(raw)


def _clean_caption(raw: str) -> str:
    # Strip leading/trailing quotes some models wrap around outputs
    s = raw.strip()
    for q in ('"', "'", "“", "”"):
        if s.startswith(q) and s.endswith(q):
            s = s[1:-1].strip()
    # Kill accidentally-included hashtags
    s = "\n".join(line for line in s.splitlines() if not line.strip().startswith("#"))
    return s.strip()
