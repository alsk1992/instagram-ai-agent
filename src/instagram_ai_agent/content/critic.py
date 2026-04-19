"""LLM critic — scores generated content against niche voice, trends, competitors.

Critic 2.0 differences from v1:
  - Broader rubric (relevance_now, originality, competitor_edge).
  - Optional vision pass that inspects the rendered image with caption.
  - Compares against top-performing competitor captions pulled from brain.db.
  - Returns verdict + structured regen hints (which dimension to fix).
"""
from __future__ import annotations

import base64
from pathlib import Path

from instagram_ai_agent.core import db
from instagram_ai_agent.core.config import NicheConfig
from instagram_ai_agent.core.llm import describe_image, generate_json, providers_configured
from instagram_ai_agent.core.logging_setup import get_logger

log = get_logger(__name__)


_RUBRIC = """\
Score a proposed Instagram post on these dimensions (0.0–1.0):

  on_niche         specifically about this niche's actual topics (not adjacent)
  on_voice         sounds like the persona, avoids forbidden phrases
  hook             first line / visible text makes a scroll-stop
  originality      not a rehash of prior posts or standard niche tropes
  relevance_now    connects to a current narrative or trend the niche cares about
  competitor_edge  as good or better than what top competitors ship in this niche
  save_potential   WOULD A USER HIT THE SAVE/BOOKMARK BUTTON?
                   HIGH (0.8+): practical takeaway, reference-worthy, list or framework,
                     specific steps/numbers users will come back to.
                   MID (0.5–0.7): useful but ephemeral, opinion-driven, one-time read.
                   LOW (<0.4): vibes-only motivational ("you got this"), pure aesthetic
                     with no takeaway, like-bait content. These MUST score low even if
                     they "feel good" — Instagram's 2026 algorithm weights saves ~3×
                     likes, so this dimension is 3× weighted in the overall score.
  dedup_risk       1.0 = clearly distinct from priors; 0.0 = too similar

Also return:
  overall     weighted mean: (on_niche + on_voice + hook + originality +
              relevance_now + competitor_edge + 3 × save_potential) / 9
  reasons     one-sentence justification
  weak_spots  list of dimension names below 0.55 (for targeted regen)
  verdict     "approve" if overall >= {threshold:.2f}
              "regen"   if 0.40 <= overall < {threshold:.2f}
              "reject"  if overall < 0.40
"""


_CONTRARIAN_RUBRIC_ADDENDUM = """

CONTRARIAN MODE is ON — evaluate the post as a hot take:
- Replace "originality" scoring with: does it genuinely contradict a
  mainstream niche belief? (vague disagreement scores 0.3; a specific
  named-belief-flipped scores 0.9+)
- Score ``claim_defensible`` 0.0–1.0 — is the contrarian claim backed by
  a specific reason, number, mechanism, or lived experience? Vague
  moralising scores < 0.4.
- INCLUDE ``claim_defensible`` in weak_spots when it's < 0.55 so the
  pipeline regenerates with more evidence.
- Factor ``claim_defensible`` into the overall mean alongside the other
  six dimensions.
- Reject outright if the post veers into medical advice, conspiracy
  tropes, self-harm, or blanket ethnic/political generalisations.
"""


def _rubric(threshold: float, *, contrarian: bool = False) -> str:
    base = _RUBRIC.format(threshold=threshold)
    if contrarian:
        base += _CONTRARIAN_RUBRIC_ADDENDUM
    return base


def _persona_block(cfg: NicheConfig) -> str:
    return (
        f"Niche: {cfg.niche}\n"
        f"Sub-topics: {', '.join(cfg.sub_topics)}\n"
        f"Audience: {cfg.target_audience}\n"
        f"Voice: {cfg.voice.persona}\n"
        f"Tone: {', '.join(cfg.voice.tone)}\n"
        f"Forbidden: {', '.join(cfg.voice.forbidden) or 'none'}\n"
    )


def _competitor_block(cfg: NicheConfig, limit: int = 5) -> str:
    if not cfg.competitors:
        return ""
    samples: list[str] = []
    for username in cfg.competitors[:5]:
        for p in db.competitor_top_recent(username, limit=3):
            caption = (p.get("caption") or "")[:200].replace("\n", " ").strip()
            if caption:
                samples.append(f"@{username} ({p.get('likes', 0)}👍): {caption}")
            if len(samples) >= limit:
                break
        if len(samples) >= limit:
            break
    if not samples:
        return ""
    return "Recent top competitor posts — our work should match or exceed these:\n" + "\n".join(samples)


def _narrative_block(limit: int = 5) -> str:
    conn = db.get_conn()
    rows = conn.execute(
        """
        SELECT topic, mentions FROM narratives
        WHERE last_seen >= datetime('now', '-7 days')
        ORDER BY mentions DESC LIMIT ?
        """,
        (limit,),
    ).fetchall()
    if not rows:
        return ""
    tops = ", ".join(f"{r['topic']} ({r['mentions']})" for r in rows)
    return f"Current niche narratives trending in the last 7 days: {tops}"


async def critique(
    cfg: NicheConfig,
    *,
    format_name: str,
    caption: str,
    visible_text: str = "",
    recent_captions: list[str] | None = None,
    image_path: str | None = None,
    knowledge: str | None = None,
    contrarian: bool = False,
) -> dict:
    """Return a scored critique with structured regen hints.

    The optional ``image_path`` triggers a vision pass (Gemini/Llama vision).
    When ``contrarian=True`` the rubric picks up a ``claim_defensible``
    dimension so posts aren't penalised for challenging consensus.
    """
    system = (
        _persona_block(cfg)
        + "\n"
        + _rubric(cfg.safety.critic_min_score, contrarian=contrarian)
        + "\n"
        + _competitor_block(cfg)
        + "\n"
        + _narrative_block()
    )

    # Niche knowledge for fact-check. Pipeline normally pre-computes and
    # passes it in to avoid N RAG calls per best-of-N. When None, fall back.
    if knowledge is None:
        try:
            from instagram_ai_agent.brain import rag
            knowledge = await rag.context_for(caption or visible_text or cfg.niche, cfg.rag)
        except Exception as e:
            knowledge = ""
            log.debug("critic: RAG retrieval failed: %s", e)
    if knowledge:
        system += (
            "\n\nIndexed niche knowledge — use as fact-check reference:\n"
            + knowledge
        )

    recent_block = ""
    if recent_captions:
        joined = "\n---\n".join(c[:220] for c in recent_captions[:8] if c)
        recent_block = f"\nOur recent captions (avoid rehashing):\n{joined}\n"

    # Optional vision pass
    image_summary = ""
    if image_path and Path(image_path).exists() and "vision" in _vision_available():
        try:
            image_summary = await _summarize_image(image_path, cfg)
            image_summary = f"\nVision summary of rendered image:\n{image_summary}\n"
        except Exception as e:
            log.debug("critic vision pass failed: %s", e)

    prompt = (
        f"Format: {format_name}\n"
        f"Visible on-image/on-video text:\n{visible_text or '(none)'}\n"
        f"{image_summary}\n"
        f"Caption:\n{caption}\n"
        f"{recent_block}\n"
        "Return JSON with keys: on_niche, on_voice, hook, originality, relevance_now, "
        "competitor_edge, save_potential, dedup_risk, overall, reasons, weak_spots, verdict."
    )

    data = await generate_json("critic", prompt, system=system, max_tokens=600)
    dims = (
        "on_niche", "on_voice", "hook", "originality",
        "relevance_now", "competitor_edge", "save_potential", "dedup_risk",
    )
    out: dict = {d: _clamp(data.get(d)) for d in dims}
    if contrarian:
        out["claim_defensible"] = _clamp(data.get("claim_defensible"))
    # Recompute overall defensively — LLMs are bad at weighted means in their
    # heads, so Python computes the canonical score from the individual dims.
    # save_potential gets 3× weight because Instagram's algorithm weights
    # saves ~3× likes since 2024, and this is the single biggest free lever.
    core_weighted = (
        out["on_niche"]
        + out["on_voice"]
        + out["hook"]
        + out["originality"]
        + out["relevance_now"]
        + out["competitor_edge"]
        + 3 * out["save_potential"]
    )
    divisor = 9  # 6 core dims × 1 + save_potential × 3
    if contrarian:
        core_weighted += out["claim_defensible"]
        divisor += 1
    out["overall"] = core_weighted / divisor
    out["reasons"] = str(data.get("reasons") or "")[:500]
    weak = data.get("weak_spots") or []
    if isinstance(weak, str):
        weak = [w.strip() for w in weak.split(",") if w.strip()]
    allowed_dims = set(dims)
    if contrarian:
        allowed_dims.add("claim_defensible")
    out["weak_spots"] = [w for w in weak if w in allowed_dims]
    v = str(data.get("verdict") or "").lower().strip()
    if v not in ("approve", "regen", "reject"):
        v = (
            "approve" if out["overall"] >= cfg.safety.critic_min_score
            else "regen" if out["overall"] >= 0.40
            else "reject"
        )
    out["verdict"] = v
    return out


async def rank_candidates(
    cfg: NicheConfig,
    *,
    format_name: str,
    candidates: list[dict],
    recent_captions: list[str] | None = None,
    knowledge: str | None = None,
    contrarian: bool = False,
) -> list[dict]:
    """Critic-rank a list of {caption, visible_text, image_path} candidates.

    Each returned dict gains a ``critique`` key with the full score. The list
    is sorted best-first. ``knowledge`` is the pre-computed RAG snippet pool
    so all N critic calls reuse the same retrieval result. ``contrarian``
    routes every critique through the hot-take rubric.
    """
    if not candidates:
        return []
    scored: list[dict] = []
    for cand in candidates:
        try:
            critique_data = await critique(
                cfg,
                format_name=format_name,
                caption=cand.get("caption", ""),
                visible_text=cand.get("visible_text", ""),
                recent_captions=recent_captions,
                image_path=cand.get("image_path"),
                knowledge=knowledge,
                contrarian=contrarian,
            )
        except Exception as e:
            log.warning("critique failed for candidate: %s", e)
            critique_data = {
                "overall": 0.0, "verdict": "reject",
                "reasons": f"critique error: {e}",
            }
        cand = {**cand, "critique": critique_data}
        scored.append(cand)
    scored.sort(key=lambda c: c["critique"].get("overall", 0.0), reverse=True)
    return scored


def _clamp(v, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return max(lo, min(hi, f))


def _vision_available() -> set[str]:
    """Which providers support vision in our configured set."""
    providers = set(providers_configured())
    # Gemini + OpenRouter (with Llama-3.2-vision) are our current vision lanes.
    if providers & {"gemini", "openrouter"}:
        return {"vision"}
    return set()


async def _summarize_image(image_path: str, cfg: NicheConfig) -> str:
    """Ask vision LLM to describe the rendered image from a niche POV.

    Instagram hosts vision APIs expect a URL, but we only have a local file.
    We base64-encode as a data URL, which Gemini/OpenRouter both accept.
    """
    p = Path(image_path)
    if not p.exists():
        return ""
    ext = p.suffix.lower().lstrip(".")
    mime = {
        "jpg": "image/jpeg", "jpeg": "image/jpeg",
        "png": "image/png", "webp": "image/webp",
    }.get(ext)
    if mime is None:
        return ""
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    data_url = f"data:{mime};base64,{b64}"
    question = (
        f"This image is a proposed Instagram post for a page about {cfg.niche}. "
        "In 40-60 words describe what's actually visible, whether it reads as on-niche, "
        "and note any quality issues (artifacts, bad anatomy, off-brand palette, text errors)."
    )
    return await describe_image(data_url, question=question)
