"""Content pipeline — format-pick → gen → best-of-N caption → critic → dedup → enqueue.

Flow per cycle:
  1. Sub-topic rotator picks the under-covered angle.
  2. Format picker chooses feed vs story + specific variant.
  3. Brain context (trend + competitor + watcher) is joined + prioritised.
  4. Generator produces media (optionally best-of-N image variants).
  5. We spawn N caption candidates (parallel LLM calls).
  6. Critic 2.0 ranks every (media, caption) combination.
  7. Best candidate goes through PHash dedup, then lands on the queue.
"""
from __future__ import annotations

import asyncio
import random
import traceback

from src.content import captions as caption_mod
from src.content import critic as critic_mod
from src.content import dedup as dedup_mod
from src.content import hashtags as hashtag_mod
from src.content.generators import (
    carousel as carousel_gen,
    format_picker,
    meme as meme_gen,
    photo as photo_gen,
    quote_card as quote_gen,
)
from src.content.generators.base import GeneratedContent
from src.core import db
from src.core.config import NicheConfig
from src.core.logging_setup import get_logger
from src.brain import idea_bank
from src.brain.coverage import pick_sub_topic, record_coverage

log = get_logger(__name__)


async def generate_one(
    cfg: NicheConfig,
    *,
    format_override: str | None = None,
    contrarian_override: bool | None = None,
) -> int | None:
    """Generate, critic-rank, and enqueue one piece of content. Returns queue row id.

    ``contrarian_override`` forces contrarian mode on/off for this cycle
    regardless of the configured dice roll — handy for CLI one-shots
    that want to preview a hot take without changing niche.yaml.
    """
    format_name = format_override or format_picker.pick_next(cfg)
    sub_topic = pick_sub_topic(cfg)

    # Contrarian mode: dice roll per cycle. Rolls even when disabled so
    # the test surface is deterministic via contrarian_override.
    if contrarian_override is not None:
        contrarian_active = contrarian_override
    else:
        contrarian_active = (
            cfg.contrarian.enabled
            and random.random() < cfg.contrarian.frequency
        )

    # Pull any fresh context from the brain (priority-ordered)
    context_rows = db.pop_context(limit=6)
    trend_context_parts = [f"[{r['source']}] {r['text']}" for r in context_rows]
    if sub_topic:
        trend_context_parts.insert(0, f"[sub_topic] focus on: {sub_topic}")
    if contrarian_active:
        trend_context_parts.insert(
            0,
            f"[contrarian_mode:{cfg.contrarian.intensity}] "
            "this post must challenge a mainstream niche belief — be specific, "
            "evidence-backed, no cheap outrage.",
        )

    # Draw an archetype from the idea bank so every post rides a proven
    # hook formula instead of re-deriving "what kind of post is this?".
    chosen_idea = idea_bank.pick_for(
        cfg,
        format_name=format_name,
        commercial_only=cfg.commercial,
        prefer_contrarian=contrarian_active,
    )
    if chosen_idea is not None:
        trend_context_parts.insert(
            0,
            f"[archetype:{chosen_idea.archetype}] hook: {chosen_idea.hook_formula}\n"
            f"approach: {chosen_idea.body_template}",
        )
        # NOTE: we only `mark_used` after a successful enqueue below — a
        # failed generation must NOT burn the recency slot or bump use_count.

    trend_context = "\n".join(trend_context_parts)

    attempts = cfg.safety.critic_max_regens + 1
    last_error: Exception | None = None

    for attempt in range(attempts):
        try:
            content = await _dispatch(format_name, cfg, trend_context, contrarian=contrarian_active)
        except Exception as e:
            last_error = e
            log.warning(
                "Generator %s failed (attempt %d/%d): %s\n%s",
                format_name, attempt + 1, attempts, e, traceback.format_exc(limit=3),
            )
            continue

        # Best-of-N caption candidates — generate in parallel, critic-rank.
        # RAG is fetched ONCE per post and shared across every caption +
        # critic call to avoid burning N×M embedding API quota.
        knowledge = ""
        try:
            from src.brain import rag
            knowledge = await rag.context_for(
                content.caption_context or trend_context or cfg.niche, cfg.rag,
            )
        except Exception as e:
            log.debug("pipeline: RAG retrieval failed: %s", e)

        n = max(1, cfg.safety.caption_candidates)
        candidates = await _build_caption_candidates(
            cfg, format_name, content, n=n, knowledge=knowledge,
            contrarian=contrarian_active,
        )

        # Contrarian hard-safety gate — runs on every candidate after
        # caption generation. A match means the model produced a claim
        # in one of our refuse-to-ship categories (medical, conspiracy,
        # self-harm, etc.). We drop the offending candidate; if ALL
        # candidates are unsafe, fall through to the regen loop.
        if contrarian_active:
            from src.content import contrarian_safety
            safe_candidates = []
            for cand in candidates:
                result = contrarian_safety.check(
                    cand.get("caption") or "",
                    cand.get("visible_text") or "",
                )
                if result.safe:
                    safe_candidates.append(cand)
                else:
                    log.warning(
                        "contrarian-safety: dropping candidate (pattern=%s)",
                        result.reason,
                    )
            if not safe_candidates:
                log.warning("All contrarian candidates hit the safety gate; regenerating")
                continue
            candidates = safe_candidates

        recent = [r["caption"] for r in db.content_list(status="posted", limit=10)]
        ranked = await critic_mod.rank_candidates(
            cfg,
            format_name=format_name,
            candidates=candidates,
            recent_captions=recent,
            knowledge=knowledge,
            contrarian=contrarian_active,
        )
        if not ranked:
            log.warning("No caption candidates produced for %s", format_name)
            continue
        best = ranked[0]
        score = best["critique"]

        log.info(
            "critic %s best of %d: overall=%.2f verdict=%s — %s",
            format_name, n, score.get("overall", 0.0),
            score.get("verdict"), (score.get("reasons") or "")[:140],
        )

        # Dedup against recent posts
        phash = dedup_mod.compute_phash(content.media_paths[0])
        is_dup, match = dedup_mod.is_duplicate(phash, cfg.safety.dedup_hamming_threshold)
        if is_dup:
            log.warning("Dedup skip — phash %s matches recent %s", phash, match)
            if attempt < attempts - 1:
                continue
            return None

        # Caption-entropy guard — refuse captions that are too similar
        # to any of our last 10 posts. Catches LLM-template drift that
        # phash dedup misses (different image, same caption template).
        if cfg.human_mimic.caption_entropy_check:
            from src.plugins import human_mimic as _hm
            recent_captions = [r["caption"] for r in recent if r.get("caption")]
            if _hm.captions_too_similar(best["caption"], recent_captions):
                log.warning(
                    "caption_entropy: %r too similar to recent — regenerating",
                    best["caption"][:80],
                )
                if attempt < attempts - 1:
                    continue
                return None

        verdict = score.get("verdict", "regen")
        if verdict == "reject":
            log.info("Critic rejected; dropping")
            if attempt < attempts - 1:
                continue
            return None
        if verdict == "regen" and attempt < attempts - 1:
            log.info("Critic says regen — weak spots=%s", score.get("weak_spots"))
            continue

        # Approve path
        caption_full = best["caption"]
        tags_used = best.get("hashtags") or []
        initial_status = "pending_review" if cfg.safety.require_review else "approved"
        cid = db.content_enqueue(
            format=content.format,
            caption=caption_full,
            hashtags=tags_used,
            media_paths=content.media_paths,
            phash=phash,
            critic_score=score.get("overall", 0.0),
            critic_notes=score.get("reasons"),
            generator=content.generator,
            status=initial_status,
            meta={
                **content.meta,
                "candidates_considered": len(candidates),
                "sub_topic": sub_topic,
                "archetype": chosen_idea.archetype if chosen_idea else None,
                "archetype_id": chosen_idea.id if chosen_idea else None,
                "contrarian_mode": contrarian_active,
            },
        )
        if sub_topic:
            record_coverage(sub_topic)
        # Only now — after a row actually lands on the queue — do we mark
        # the archetype used. Failed / rejected generations above would
        # have `continue`d before reaching this point.
        if chosen_idea is not None and chosen_idea.id is not None:
            idea_bank.mark_used(chosen_idea.id)
        log.info(
            "enqueued content id=%d format=%s status=%s (score=%.2f, sub=%s)",
            cid, format_name, initial_status, score.get("overall", 0.0), sub_topic or "?",
        )
        return cid

    log.error(
        "generate_one exhausted attempts for %s (last error: %s)",
        format_name, last_error,
    )
    return None


async def _build_caption_candidates(
    cfg: NicheConfig,
    format_name: str,
    content: GeneratedContent,
    *,
    n: int,
    knowledge: str | None = None,
    contrarian: bool = False,
) -> list[dict]:
    """Produce n caption candidates (caption + hashtags + image_path) in parallel."""

    async def _one() -> dict:
        caption_body = await caption_mod.generate_caption(
            cfg, format_name, context=content.caption_context, knowledge=knowledge,
            contrarian=contrarian,
        )
        if format_name.startswith("story_"):
            tags: list[str] = []
            full = caption_body.strip()
        else:
            tags = hashtag_mod.build_hashtags(cfg)
            full = caption_body + "\n\n" + hashtag_mod.format_hashtags(tags)
        return {
            "caption": full,
            "hashtags": tags,
            "visible_text": content.visible_text,
            "image_path": content.media_paths[0] if content.media_paths else None,
        }

    if n == 1:
        return [await _one()]
    results = await asyncio.gather(*[_one() for _ in range(n)], return_exceptions=True)
    out: list[dict] = []
    for r in results:
        if isinstance(r, Exception):
            log.warning("caption candidate failed: %s", r)
            continue
        out.append(r)
    return out


async def _dispatch(
    format_name: str,
    cfg: NicheConfig,
    trend_context: str,
    *,
    contrarian: bool = False,
) -> GeneratedContent:
    if format_name == "meme":
        return await meme_gen.generate(cfg, trend_context, contrarian=contrarian)
    if format_name == "quote_card":
        return await quote_gen.generate(cfg, trend_context, contrarian=contrarian)
    if format_name == "carousel":
        return await carousel_gen.generate(cfg, trend_context, slides=7, contrarian=contrarian)
    if format_name == "story_carousel":
        from src.content.generators import story_carousel as story_gen
        return await story_gen.generate(cfg, trend_context)
    if format_name == "photo":
        return await photo_gen.generate_image(cfg, trend_context)
    if format_name == "reel_stock":
        from src.content.generators import reel_stock as reel_mod
        return await reel_mod.generate(cfg, trend_context, contrarian=contrarian)
    if format_name == "reel_ai":
        from src.content.generators import reel_ai as reel_ai_mod
        return await reel_ai_mod.generate(cfg, trend_context)
    if format_name == "human_photo":
        from src.content.generators import human_photo as human_mod
        return await human_mod.generate(cfg, trend_context)
    if format_name in ("story_quote", "story_announcement", "story_photo"):
        from src.content.generators import story_image as story_img_mod
        return await story_img_mod.generate(cfg, trend_context, variant=format_name)
    if format_name == "story_video":
        from src.content.generators import story_video as story_vid_mod
        return await story_vid_mod.generate(cfg, trend_context)
    if format_name == "story_human":
        from src.content.generators import human_photo as human_mod
        return await human_mod.generate_story(cfg, trend_context)
    raise ValueError(f"Unknown format: {format_name}")
