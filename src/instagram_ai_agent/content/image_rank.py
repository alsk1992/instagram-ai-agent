"""Vision-critic image ranker — picks the best of N generated images.

Generators that produce multiple candidates call :func:`rank` with a list of
paths; the vision LLM scores each on realism + on-niche + quality, and the
best path is returned. When the vision LLM is unavailable the first image
wins (silent fallback — never block a post on ranking).
"""
from __future__ import annotations

import base64
import re
from pathlib import Path

from instagram_ai_agent.core.config import NicheConfig
from instagram_ai_agent.core.llm import describe_image, providers_configured
from instagram_ai_agent.core.logging_setup import get_logger

log = get_logger(__name__)


_MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
_SCORE_RE = re.compile(r"score\s*[:=]\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)


def _vision_ready() -> bool:
    return bool(providers_configured() & {"gemini", "openrouter"})


def _data_url(path: Path) -> str | None:
    mime = _MIME.get(path.suffix.lower())
    if mime is None or not path.exists():
        return None
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


async def _score_image(path: Path, cfg: NicheConfig, *, subject_is_human: bool) -> tuple[float, str]:
    """Ask the vision LLM for a 0–1 score and one-line justification."""
    data_url = _data_url(path)
    if data_url is None:
        return 0.0, "unsupported format"

    human_rider = (
        " Flag any bad anatomy, extra fingers, plastic-skin AI tells, "
        "uncanny facial asymmetry, or warped hands — these should score low."
        if subject_is_human else ""
    )
    palette_hint = ", ".join(cfg.aesthetic.palette[:3])
    question = (
        f"This is a candidate Instagram image for a page about {cfg.niche}.\n"
        f"Aesthetic palette hint: {palette_hint}.\n"
        f"Score from 0.00 (unusable) to 1.00 (excellent) on a single combined scale "
        f"that blends: (1) technical quality, (2) on-niche relevance, "
        f"(3) aesthetic match to the palette.{human_rider}\n"
        "Reply in EXACTLY this format:\n"
        "score: <float>\n"
        "reason: <one short sentence>"
    )
    try:
        out = await describe_image(data_url, question=question)
    except Exception as e:
        log.debug("image_rank: vision call failed: %s", e)
        return 0.0, f"vision error: {e}"

    m = _SCORE_RE.search(out or "")
    try:
        score = float(m.group(1)) if m else 0.0
    except ValueError:
        score = 0.0
    # Clamp + capture reason
    score = max(0.0, min(1.0, score))
    reason = ""
    for line in (out or "").splitlines():
        if line.lower().startswith("reason"):
            reason = line.split(":", 1)[-1].strip()[:240]
            break
    return score, reason


async def _local_prefilter(paths: list[str | Path]) -> list[dict]:
    """Run the local ensemble over every candidate. Returns a list of
    ``{path, local_score, local_raw, local_model}`` entries sorted best-first.
    Falls back to equal scores when no local scorer is available."""
    from instagram_ai_agent.content import local_aesthetic

    ens = local_aesthetic.get_ensemble()
    if not ens.is_available():
        return [
            {"path": str(p), "local_score": 0.5, "local_raw": {}, "local_model": "none"}
            for p in paths
        ]

    scored: list[dict] = []
    for p in paths:
        try:
            ls = await ens.score(Path(p))
        except Exception as e:
            log.debug("local_aesthetic: %s failed — %s", Path(p).name, e)
            ls = None
        if ls is None:
            scored.append({
                "path": str(p), "local_score": 0.0,
                "local_raw": {}, "local_model": "failed",
            })
        else:
            scored.append({
                "path": str(p), "local_score": ls.score,
                "local_raw": dict(ls.raw), "local_model": ls.model_used,
            })
    scored.sort(key=lambda r: r["local_score"], reverse=True)
    return scored


async def rank(
    cfg: NicheConfig,
    paths: list[str | Path],
    *,
    subject_is_human: bool = False,
) -> list[dict]:
    """Two-pass rank of candidates.

    1. **Local pass** (if ``safety.local_aesthetic``): every path gets a
       cheap 0-1 score from the ensemble (LAION + PyIQA by default).
    2. **Vision pass** (if ``safety.vision_critic`` and a provider configured):
       only the top ``safety.vision_top_k`` by local score go to the
       vision LLM. Others inherit their local score.

    Final per-candidate ``score`` is the vision score when available, else
    the local score. Returns best-first.
    """
    if not paths:
        return []
    if len(paths) == 1:
        return [{"path": str(paths[0]), "score": 1.0, "reason": "only candidate"}]

    # ── Pass 1: local prefilter ──
    use_local = cfg.safety.local_aesthetic
    local_rows = (
        await _local_prefilter(paths) if use_local else [
            {"path": str(p), "local_score": 0.5, "local_raw": {}, "local_model": "skipped"}
            for p in paths
        ]
    )

    # ── Pass 2: vision on top-K ──
    want_vision = cfg.safety.vision_critic and _vision_ready()
    if not want_vision:
        # Use local-only ranking (first-candidate fallback when local absent).
        if not use_local:
            return [
                {"path": r["path"], "score": 1.0 if i == 0 else 0.0,
                 "reason": "no vision provider; local disabled", **r}
                for i, r in enumerate(local_rows)
            ]
        return [
            {**r, "score": r["local_score"],
             "reason": f"local-only ({r['local_model']})"}
            for r in local_rows
        ]

    top_k = max(1, min(cfg.safety.vision_top_k, len(local_rows)))
    head = local_rows[:top_k]
    tail = local_rows[top_k:]

    vision_scored: list[dict] = []
    for row in head:
        v_score, v_reason = await _score_image(
            Path(row["path"]), cfg, subject_is_human=subject_is_human,
        )
        vision_scored.append({
            **row,
            "vision_score": v_score,
            "vision_reason": v_reason,
            "score": v_score,
            "reason": f"vision:{v_reason}",
        })
    # Non-top-K inherit local score (they were never considered by the vision pass)
    for row in tail:
        vision_scored.append({
            **row,
            "vision_score": None,
            "vision_reason": "below vision_top_k",
            "score": row["local_score"],
            "reason": f"local-only ({row['local_model']})",
        })
    vision_scored.sort(key=lambda r: r["score"], reverse=True)
    return vision_scored


async def pick_best(
    cfg: NicheConfig,
    paths: list[str | Path],
    *,
    subject_is_human: bool = False,
) -> tuple[str, dict]:
    """Convenience: return (best_path, full_rank_metadata)."""
    ranked = await rank(cfg, paths, subject_is_human=subject_is_human)
    if not ranked:
        raise ValueError("No candidates to rank")
    return ranked[0]["path"], {"ranked": ranked}
