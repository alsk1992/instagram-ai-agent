"""Comment-bait CTA engineer — engineer the last line for replies.

Instagram's algorithm weights comment RATE (comments per reach) heavily,
especially in the first hour. Pro accounts don't wait for comments — they
engineer them.

The default niche.yaml ``cta_styles`` list is save/tag/follow oriented
("save for later", "tag a mate", "follow for more"). Useful, but not
comment-triggering. This module runs AFTER caption generation on formats
where comments are the engagement target (quote_card, carousel, reel_*,
photo) and rewrites the final line into one of eight known-working
comment-triggering patterns tailored to the post.

The 8 patterns (Hormozi/Koe/Welsh playbook, 2024–26):
  binary_pick        "1 or 2?"          — tiny-commit vote
  fill_in_blank      "my biggest ___"    — low barrier, personal
  emoji_react        "✅ if this hit"    — smallest possible commit
  number_drop        "drop your rep count" — specific, niche-signal
  single_tag         "tag someone who"   — networking amplifier
  story_invite       "what was YOUR day-5?" — mirrors lived-experience
  unpopular_roast    "roast this if you disagree" — contrarian
  one_word_reply     "one word for how this feels"  — minimal commit

Silent fallback: on any LLM failure OR degenerate output, return the
original caption unchanged. Never break the pipeline with a rewrite.
"""
from __future__ import annotations

from instagram_ai_agent.core.config import NicheConfig
from instagram_ai_agent.core.llm import generate
from instagram_ai_agent.core.logging_setup import get_logger

log = get_logger(__name__)


# Formats where comment-engineering beats save-optimisation. Memes
# amplify via shares, not comments — leave those alone.
COMMENT_OPTIMISED_FORMATS = {
    "quote_card", "carousel", "reel_stock", "reel_ai", "photo", "human_photo",
}


PATTERNS = (
    "binary_pick",
    "fill_in_blank",
    "emoji_react",
    "number_drop",
    "single_tag",
    "story_invite",
    "unpopular_roast",
    "one_word_reply",
)


def pick_pattern(
    format_name: str, *, contrarian: bool = False, has_numbers: bool = False,
) -> str:
    """Deterministically pick the CTA pattern that best fits the post.

    Rules:
      * contrarian posts → ``unpopular_roast`` (disagree-bait)
      * carousels with numbers in the body → ``number_drop``
      * story-arc carousels → ``story_invite`` (mirrors the arc)
      * quote cards → ``emoji_react`` (lowest-barrier vote on the line)
      * reels + photos → ``binary_pick`` (simplest engagement ask)
      * fallback → ``fill_in_blank``
    """
    if contrarian:
        return "unpopular_roast"
    if format_name == "carousel" and has_numbers:
        return "number_drop"
    if format_name == "carousel":
        return "story_invite"
    if format_name == "quote_card":
        return "emoji_react"
    if format_name in ("reel_stock", "reel_ai", "photo", "human_photo"):
        return "binary_pick"
    return "fill_in_blank"


def _pattern_brief(pattern: str) -> str:
    return {
        "binary_pick": (
            "Binary-pick CTA: end with a single question offering two choices "
            "labelled 1 and 2 that readers can pick in one character. Example: "
            "\"pullups or mobility first thing in the morning? 1 or 2?\""
        ),
        "fill_in_blank": (
            "Fill-in-the-blank CTA: end with a short personal prompt the reader "
            "can complete in 2–6 words. Example: \"my biggest plateau lately: ___\""
        ),
        "emoji_react": (
            "Emoji-react CTA: end with a single-emoji reaction ask tied to the "
            "post's insight. Example: \"✅ if this is where you're stuck\""
        ),
        "number_drop": (
            "Number-drop CTA: end asking the reader to drop a SPECIFIC number "
            "(their current rep count, week, weight, etc.). Example: \"drop your "
            "current pullup max 👇\""
        ),
        "single_tag": (
            "Single-tag CTA: end asking for ONE tag of a specific niche persona "
            "(not \"tag a friend\"). Example: \"tag the mate who skips warmups\""
        ),
        "story_invite": (
            "Story-invite CTA: end asking the reader for their specific mirror "
            "experience. Example: \"what week did yours click? drop it below\""
        ),
        "unpopular_roast": (
            "Roast-invite CTA: end inviting disagreement explicitly. Example: "
            "\"roast this take if you think I'm wrong\""
        ),
        "one_word_reply": (
            "One-word CTA: end asking for a single word reply. Example: "
            "\"one word for how this week felt\""
        ),
    }.get(pattern, "")


async def engineer(
    cfg: NicheConfig,
    caption: str,
    *,
    format_name: str,
    angle_hook: str | None = None,
    contrarian: bool = False,
) -> str:
    """Rewrite the caption's final line into a comment-triggering CTA.

    Returns the original caption if:
      * the format isn't comment-optimised (meme/quote/story_*)
      * the caption already ends with a ``?`` question (assume engineered)
      * the LLM call fails or produces a degenerate output
    """
    if format_name not in COMMENT_OPTIMISED_FORMATS:
        return caption
    if not caption.strip():
        return caption
    # Already ends in a question — assume it's already engineered
    if caption.rstrip().endswith("?"):
        return caption

    has_numbers = any(c.isdigit() for c in caption)
    pattern = pick_pattern(format_name, contrarian=contrarian, has_numbers=has_numbers)

    system = (
        "You are a CTA engineer for Instagram niche content. You rewrite the "
        "FINAL LINE of a caption to trigger a specific comment reply pattern. "
        "You NEVER modify the body — only the last line (CTA). You preserve "
        "the voice. You keep the CTA under 14 words. You NEVER use generic "
        "phrases: 'save for later', 'tag a mate', 'follow for more', 'let "
        "me know', 'what do you think', 'thoughts?', 'comment below'. "
        "Output the FULL caption (body + rewritten CTA) — no preamble, labels, or markdown."
    )

    angle_line = f"WINNING ANGLE: {angle_hook}" if angle_hook else "ANGLE: (derive from body)"
    prompt = f"""NICHE: {cfg.niche}
VOICE: {", ".join(cfg.voice.tone) if cfg.voice.tone else "(none)"}
{angle_line}

CTA PATTERN TO USE: {pattern}
{_pattern_brief(pattern)}

FULL CAPTION (body + current CTA):
{caption}

TASK
Rewrite ONLY the final line into a CTA that follows the {pattern!r} pattern,
specific to the post's actual content. Body stays unchanged. Output the
full caption — body lines + new final CTA line. No preamble or labels.
"""

    try:
        raw = await generate(
            "caption", prompt, system=system,
            max_tokens=400, temperature=0.7,
        )
    except Exception as e:
        log.debug("comment_bait: LLM failed — keeping original: %s", e)
        return caption

    cleaned = _clean(raw)
    if not cleaned or cleaned == caption:
        return caption

    # Length guardrail — body shouldn't shift much
    if len(cleaned) > len(caption) * 1.35 or len(cleaned) < len(caption) * 0.5:
        log.debug("comment_bait: rewrite length drift (%d → %d) — keeping original",
                  len(caption), len(cleaned))
        return caption

    # Must now end in a question or emoji (signals engineered CTA)
    last_line = cleaned.strip().splitlines()[-1].strip()
    if not last_line:
        return caption
    if not (last_line.endswith("?") or last_line.endswith("👇") or _ends_with_emoji(last_line)):
        log.debug("comment_bait: rewrite didn't produce a triggering CTA — keeping original")
        return caption

    log.debug("comment_bait: engineered %s CTA", pattern)
    return cleaned


def _clean(raw: str) -> str:
    s = raw.strip()
    for prefix in ("Rewrite:", "Caption:", "FULL CAPTION:", "Output:"):
        if s.lower().startswith(prefix.lower()):
            s = s[len(prefix):].strip()
    for q in ('"', "'", "“", "”"):
        if s.startswith(q) and s.endswith(q):
            s = s[1:-1].strip()
    return s


def _ends_with_emoji(s: str) -> bool:
    if not s:
        return False
    # Rough heuristic: last char outside the ASCII range AND not whitespace.
    last = s.rstrip()[-1]
    return ord(last) > 127 and not last.isspace()
