"""Voice fingerprint — anchor every new draft to the page's shipped style.

niche.yaml is the brief (keywords like "direct, dry humour"). LLMs reinterpret
those keywords fresh on every call, so captions drift across weeks — a reader
feels three different humans.

This module pulls the N most-recent posted captions, ranks by critic_score,
and formats them as few-shot voice examples. The caption generator and angle
brainstormer both inject this block so every new post anchors to the same
shipped style rather than the keyword-interpretation-of-the-day.

On a fresh account with no posted content, the block is empty and generation
falls back to the niche.yaml persona — the fingerprint kicks in naturally
once history accumulates.
"""
from __future__ import annotations

from instagram_ai_agent.core import db
from instagram_ai_agent.core.logging_setup import get_logger

log = get_logger(__name__)


MIN_EXAMPLE_LEN = 40
MAX_EXAMPLE_LEN = 400


def pick_voice_examples(n: int = 5) -> list[str]:
    """Return up to ``n`` shipped captions suitable as voice anchors.

    Preference order:
      1. status='posted' items ranked by critic_score DESC, posted_at DESC
      2. status='approved' as fallback when not enough posted items exist
    Captions that are too short or too long are filtered — few-shot works
    best when the examples are a representative length.
    """
    try:
        rows = db.get_conn().execute(
            """
            SELECT caption, critic_score, posted_at
            FROM content_queue
            WHERE status='posted' AND caption IS NOT NULL AND caption != ''
            ORDER BY critic_score DESC NULLS LAST, posted_at DESC
            LIMIT ?
            """,
            (n * 3,),
        ).fetchall()
    except Exception as e:
        log.debug("voice_fingerprint: posted query failed — %s", e)
        rows = []

    examples: list[str] = []
    for r in rows:
        cap = _strip_caption(r["caption"])
        if _is_usable(cap):
            examples.append(cap)
        if len(examples) >= n:
            break

    if len(examples) < n:
        try:
            extra = db.get_conn().execute(
                """
                SELECT caption, critic_score
                FROM content_queue
                WHERE status='approved' AND caption IS NOT NULL AND caption != ''
                ORDER BY critic_score DESC NULLS LAST, created_at DESC
                LIMIT ?
                """,
                (n * 3,),
            ).fetchall()
        except Exception as e:
            log.debug("voice_fingerprint: approved fallback query failed — %s", e)
            extra = []

        for r in extra:
            cap = _strip_caption(r["caption"])
            if _is_usable(cap) and cap not in examples:
                examples.append(cap)
            if len(examples) >= n:
                break

    return examples


def build_voice_block(examples: list[str]) -> str:
    """Format ``examples`` as a few-shot voice anchor for the system prompt.

    Returns an empty string when there are fewer than 2 examples — a single
    example is noise, not a fingerprint.
    """
    if len(examples) < 2:
        return ""

    numbered = "\n\n".join(f"example {i + 1}:\n{cap}" for i, cap in enumerate(examples))
    return (
        "\n\nVOICE FINGERPRINT — the below captions already shipped on this "
        "page and are the definitive style anchor. Match their sentence "
        "rhythm, openers, how numbers and names are deployed, how humour "
        "lands, and how CTAs are phrased. This is HOW we sound here. Do "
        "NOT copy their specific topics — borrow the cadence and texture "
        "only.\n\n"
        f"{numbered}"
    )


def _strip_caption(raw: str) -> str:
    """Drop the trailing hashtag block — we want the caption BODY only."""
    if not raw:
        return ""
    body_lines: list[str] = []
    for line in raw.splitlines():
        if line.strip().startswith("#"):
            break
        body_lines.append(line)
    return "\n".join(body_lines).strip()


def _is_usable(cap: str) -> bool:
    if not cap:
        return False
    length = len(cap)
    return MIN_EXAMPLE_LEN <= length <= MAX_EXAMPLE_LEN
