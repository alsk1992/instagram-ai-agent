"""Open-Carrusel repurpose: reel → carousel.

Picks a previously-posted reel, extracts a scene keyframe per slide,
compresses the voiceover line into on-slide copy, and renders a photo
carousel riding the reel's original visual language. Lets a proven reel
have a second life on the feed without paying for a fresh generation.

Safety / licensing:
  * Uses only assets already on disk from a prior reel — no new image
    generation, no external scraping.
  * Same Playwright + template stack as the vanilla carousel generator.
  * Standalone scheduled job; does NOT participate in format_picker.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path
from string import Template
from typing import Any

from instagram_ai_agent.content.generators.base import GeneratedContent, staging_path
from instagram_ai_agent.content.generators.playwright_render import base_css, pick_template, render_html_to_png
from instagram_ai_agent.content.style import apply_lut_image
from instagram_ai_agent.core import db
from instagram_ai_agent.core.config import NicheConfig
from instagram_ai_agent.core.llm import generate_json
from instagram_ai_agent.core.logging_setup import get_logger

log = get_logger(__name__)

_SEEN_KEY = "repurposed_reels_seen"
_REEL_FORMATS = ("reel_stock", "reel_ai")


# ───── dedup state ─────
def _seen() -> dict[str, str]:
    raw = db.state_get_json(_SEEN_KEY, default={}) or {}
    return raw if isinstance(raw, dict) else {}


def _mark_repurposed(source_reel_id: int) -> None:
    store = _seen()
    store[str(source_reel_id)] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    # Prune anything older than 365d so the store can't grow unbounded.
    cutoff = datetime.now(timezone.utc) - timedelta(days=365)
    pruned = {}
    for k, v in store.items():
        try:
            ts = datetime.strptime(v, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            continue
        if ts >= cutoff:
            pruned[k] = v
    db.state_set_json(_SEEN_KEY, pruned)


# ───── candidate selection ─────
@dataclass(frozen=True)
class RepurposeCandidate:
    content_id: int
    format: str
    mp4_path: Path
    scenes: list[dict]
    posted_at: str
    variant: str | None
    original_caption: str


def _parse_iso(ts: str) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.strptime(ts.replace("Z", ""), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def pick_candidate(cfg: NicheConfig) -> RepurposeCandidate | None:
    """Find a posted reel eligible for repurposing.

    Eligibility:
      * format is reel_stock or reel_ai
      * status = 'posted'
      * posted_at older than min_reel_age_days, newer than max_reel_age_days
      * content_id not in _seen() dedup store
      * mp4 file still exists on disk
      * meta.scenes is a non-empty list (we need the scene metadata to
        decide per-slide keyframe positions)
    """
    rc = cfg.reel_repurpose
    if not rc.enabled:
        return None

    now = datetime.now(timezone.utc)
    max_cutoff = now - timedelta(days=rc.min_reel_age_days)
    min_cutoff = now - timedelta(days=rc.max_reel_age_days)
    seen = _seen()

    rows = db.get_conn().execute(
        """
        SELECT id, format, media_paths, meta, posted_at, caption
        FROM content_queue
        WHERE status='posted'
          AND format IN (?, ?)
          AND posted_at IS NOT NULL
        ORDER BY posted_at DESC
        LIMIT 100
        """,
        _REEL_FORMATS,
    ).fetchall()

    for row in rows:
        cid = int(row["id"])
        if str(cid) in seen:
            continue
        posted_at = _parse_iso(row["posted_at"])
        if posted_at is None:
            continue
        if posted_at > max_cutoff or posted_at < min_cutoff:
            continue
        try:
            paths = json.loads(row["media_paths"] or "[]")
            meta = json.loads(row["meta"] or "{}")
        except (TypeError, ValueError):
            continue
        if not paths:
            continue
        mp4 = Path(paths[0])
        if not mp4.exists():
            log.debug("repurpose: skipping cid=%d — mp4 vanished (%s)", cid, mp4)
            continue
        scenes = meta.get("scenes") or []
        if not isinstance(scenes, list) or not scenes:
            continue
        return RepurposeCandidate(
            content_id=cid,
            format=str(row["format"]),
            mp4_path=mp4,
            scenes=[s for s in scenes if isinstance(s, dict) and s.get("line")],
            posted_at=row["posted_at"],
            variant=str(meta.get("variant") or "") or None,
            original_caption=str(row["caption"] or ""),
        )
    return None


# ───── keyframe extraction ─────
def _probe_duration(mp4_path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(mp4_path),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(r.stdout.strip() or 0.0)
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError) as e:
        log.warning("ffprobe failed on %s: %s", mp4_path, e)
        return 0.0


def _scene_midpoints(scenes: list[dict], duration_s: float) -> list[float]:
    """Compute a midpoint timestamp (seconds) for each scene using the same
    word-count-proportional algorithm the reel generators used at render
    time. Returns timestamps clamped to [0.1, duration_s - 0.1]."""
    if duration_s <= 0 or not scenes:
        return []
    lines = [str(s.get("line") or "") for s in scenes]
    word_counts = [max(1, len(line.split())) for line in lines]
    total_words = sum(word_counts)
    cursor = 0.0
    mids: list[float] = []
    for wc in word_counts:
        scene_dur = duration_s * (wc / total_words)
        mid = cursor + scene_dur / 2.0
        mids.append(max(0.1, min(duration_s - 0.1, mid)))
        cursor += scene_dur
    return mids


def _extract_frame(mp4_path: Path, timestamp_s: float, out_path: Path) -> bool:
    """Grab a single frame at ``timestamp_s`` into ``out_path`` (JPEG).

    Returns True on success. ``-ss`` BEFORE ``-i`` uses fast keyframe
    seek; for an occasional "not quite on the exact second" inaccuracy
    this is a fine trade — we want the scene's vibe, not millisecond
    precision.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{timestamp_s:.3f}",
        "-i", str(mp4_path),
        "-frames:v", "1",
        "-q:v", "2",
        str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return out_path.exists() and out_path.stat().st_size > 0
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        log.warning("ffmpeg frame-grab failed at t=%.2f of %s: %s", timestamp_s, mp4_path, e)
        return False


def extract_keyframes(
    mp4_path: Path, scenes: list[dict], out_dir: Path, *, max_frames: int,
) -> list[Path]:
    """Extract one keyframe per scene (capped at ``max_frames``).

    If there are fewer scenes than ``max_frames``, all scenes get a frame
    and we stop. If there are more scenes, we evenly subsample so the
    picked frames span the full reel.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    duration = _probe_duration(mp4_path)
    if duration <= 0:
        return []
    mids = _scene_midpoints(scenes, duration)
    if not mids:
        return []
    if len(mids) > max_frames:
        step = len(mids) / max_frames
        picked_idx = [min(len(mids) - 1, int(i * step)) for i in range(max_frames)]
    else:
        picked_idx = list(range(len(mids)))

    frames: list[Path] = []
    for n, i in enumerate(picked_idx):
        ts = mids[i]
        out = out_dir / f"keyframe_{n:02d}.jpg"
        if _extract_frame(mp4_path, ts, out):
            frames.append(out)
    return frames


# ───── slide copy generation ─────
def _heuristic_slide(line: str, *, max_title_words: int = 6) -> dict[str, str]:
    """Fallback: split a voiceover line into title/body without LLM.

    Takes the first ``max_title_words`` as title; the rest (or a trimmed
    version of the same line) as body. If the line is short we repeat
    it as both so we never ship an empty field.
    """
    words = line.strip().split()
    if not words:
        return {"title": "", "body": ""}
    title = " ".join(words[:max_title_words]).rstrip(",.;:!?")
    body = " ".join(words[max_title_words:]).strip() or line.strip()
    return {"title": title, "body": body}


async def _llm_compress(
    cfg: NicheConfig, scene_lines: list[str], *, n_slides: int,
) -> list[dict[str, str]]:
    """Ask the LLM to compress scene lines into slide-ready title/body.

    Returns a list of length ``n_slides``. Falls back to the heuristic
    slide builder on any failure so we still ship a usable carousel.
    """
    if not scene_lines:
        return []
    system = (
        f"You turn Instagram reel voiceover lines into carousel slide text for a page about {cfg.niche}.\n"
        f"Audience: {cfg.target_audience}.\n"
        f"Voice: {cfg.voice.persona}. Tone: {', '.join(cfg.voice.tone)}.\n"
        f"Never: {', '.join(cfg.voice.forbidden) or 'n/a'}.\n"
        "Rules:\n"
        "- Slide 1 is the HOOK: ≤9 words, scroll-stopping.\n"
        "- Final slide is a CTA tied to the niche (save, follow, share, tag).\n"
        "- Middle slides: ONE idea each.\n"
        "- Title ≤6 words. Body ≤22 words. Zero emojis. No hashtags."
    )
    numbered = "\n".join(f"{i + 1}. {line}" for i, line in enumerate(scene_lines))
    prompt = (
        f"Source voiceover, line-by-line ({len(scene_lines)} lines):\n{numbered}\n\n"
        f"Produce exactly {n_slides} slides that cover the key ideas above.\n"
        "You may merge, re-order, or sharpen lines — but stay faithful to the content.\n"
        "Return JSON: {\"slides\":[{\"kind\":\"hook|content|cta\",\"title\":str,\"body\":str}, ...]}"
    )
    try:
        data = await generate_json("script", prompt, system=system, max_tokens=1200)
        slides = data.get("slides") or []
        if not isinstance(slides, list) or len(slides) < n_slides:
            raise ValueError(f"LLM returned {len(slides)} slides, expected {n_slides}")
    except Exception as e:
        log.warning("repurpose: LLM compress failed, using heuristic fallback: %s", e)
        return _heuristic_fallback(scene_lines, n_slides)

    cleaned = [
        {
            "kind": str(s.get("kind") or "content"),
            "title": str(s.get("title") or "").strip(),
            "body": str(s.get("body") or "").strip(),
        }
        for s in slides[:n_slides]
    ]
    # Enforce shape invariants even if the LLM disobeyed the rules:
    # slide 0 is the hook, final slide is the CTA.
    if cleaned:
        cleaned[0]["kind"] = "hook"
    if len(cleaned) > 1:
        cleaned[-1]["kind"] = "cta"
    return cleaned


def _heuristic_fallback(scene_lines: list[str], n_slides: int) -> list[dict[str, str]]:
    """Produce n_slides from scene_lines without any LLM call.

    Subsamples or pads evenly, then assigns hook/content/cta kinds.
    """
    if not scene_lines:
        return []
    if len(scene_lines) >= n_slides:
        step = len(scene_lines) / n_slides
        picked = [scene_lines[min(len(scene_lines) - 1, int(i * step))] for i in range(n_slides)]
    else:
        # Pad by repeating the last line
        picked = scene_lines + [scene_lines[-1]] * (n_slides - len(scene_lines))
    out: list[dict[str, str]] = []
    for i, line in enumerate(picked):
        slide = _heuristic_slide(line)
        if i == 0:
            slide["kind"] = "hook"
        elif i == n_slides - 1:
            slide["kind"] = "cta"
            # Nudge the last line into a CTA shape if it doesn't already read like one
            if not any(w in slide["body"].lower() for w in ("save", "follow", "share", "tag", "try")):
                slide["body"] = (slide["body"].rstrip(".") + ". Save this and try it.").strip()
        else:
            slide["kind"] = "content"
        out.append(slide)
    return out


# ───── slide rendering ─────
def _image_data_url(path: Path) -> str:
    """Base64-inline an image — Playwright in headless chromium can't reach
    arbitrary file:// paths reliably, so we embed the bytes."""
    suffix = path.suffix.lower().lstrip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}.get(suffix, "image/jpeg")
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def _render_slide_html(
    cfg: NicheConfig,
    slide: dict,
    *,
    keyframe: Path,
    index: int,
    total: int,
    tpl: str,
) -> str:
    palette = cfg.aesthetic.palette
    bg = palette[0]
    fg = palette[1] if len(palette) > 1 else "#ffffff"
    accent = palette[2] if len(palette) > 2 else fg

    css = base_css(
        width=1080,
        height=1350,
        bg=bg,
        fg=fg,
        body_font=cfg.aesthetic.body_font,
        heading_font=cfg.aesthetic.heading_font,
    )
    is_hook = slide.get("kind") == "hook"
    is_cta = slide.get("kind") == "cta"
    return Template(tpl).safe_substitute(
        css=css,
        heading_font=cfg.aesthetic.heading_font,
        body_font=cfg.aesthetic.body_font,
        bg=bg,
        accent=accent,
        fg=fg,
        title=escape(slide.get("title") or ""),
        body=escape(slide.get("body") or ""),
        index=f"{index:02d}",
        total=f"{total:02d}",
        hook_class="hook" if is_hook else "",
        cta_class="cta" if is_cta else "",
        watermark=escape(cfg.aesthetic.watermark or ""),
        background_image=_image_data_url(keyframe),
    )


# ───── public entry point ─────
async def generate(cfg: NicheConfig) -> GeneratedContent | None:
    """Full repurpose pipeline. Returns GeneratedContent or None if no
    eligible reel exists. Never raises — logs and returns None on error
    so the scheduled job stays quiet."""
    rc = cfg.reel_repurpose
    if not rc.enabled:
        return None

    candidate = pick_candidate(cfg)
    if candidate is None:
        log.debug("repurpose: no eligible reel")
        return None

    work_dir = staging_path("repurpose_work", "").with_suffix("")
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        try:
            keyframes = await asyncio.to_thread(
                extract_keyframes,
                candidate.mp4_path, candidate.scenes, work_dir,
                max_frames=rc.max_slides,
            )
        except Exception as e:
            log.warning("repurpose: keyframe extraction crashed for cid=%d: %s", candidate.content_id, e)
            return None

        if not keyframes:
            log.warning(
                "repurpose: no keyframes extracted from cid=%d (%s) — skipping",
                candidate.content_id, candidate.mp4_path.name,
            )
            return None

        n_slides = len(keyframes)
        scene_lines = [str(s.get("line") or "") for s in candidate.scenes if s.get("line")]
        slides_copy = await _llm_compress(cfg, scene_lines, n_slides=n_slides)
        if not slides_copy:
            slides_copy = _heuristic_fallback(scene_lines, n_slides)
        if not slides_copy:
            log.warning("repurpose: no slide copy produced for cid=%d", candidate.content_id)
            return None

        template_name, tpl = pick_template("carousels", variant=rc.template_variant)
        # If variant was missing, pick_template falls back to a random template.
        # That fallback won't know about $background_image and the keyframe
        # work would be wasted. Refuse politely instead.
        if "$background_image" not in tpl:
            log.warning(
                "repurpose: template %r does not declare $background_image — "
                "skipping (check niche.yaml reel_repurpose.template_variant)",
                template_name,
            )
            return None

        paths: list[str] = []
        failed_mid_render = False
        for i, (slide, kf) in enumerate(zip(slides_copy, keyframes, strict=False), start=1):
            html = _render_slide_html(
                cfg, slide, keyframe=kf,
                index=i, total=n_slides, tpl=tpl,
            )
            out = staging_path(f"repurp_slide{i:02d}_{template_name}", ".jpg")
            try:
                await render_html_to_png(html, out, width=1080, height=1350)
            except Exception as e:
                log.warning("repurpose: slide %d render failed: %s", i, e)
                failed_mid_render = True
                break
            out = apply_lut_image(out, cfg)
            paths.append(str(out))

        if failed_mid_render:
            # Clean up partial slides so they don't linger in staging
            for p in paths:
                try:
                    Path(p).unlink(missing_ok=True)
                except OSError:
                    pass
            return None
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    hook = slides_copy[0].get("title") or ""
    body_preview = " / ".join((s.get("body") or "") for s in slides_copy[:3])
    return GeneratedContent(
        format="carousel",
        media_paths=paths,
        visible_text=f"Hook: {hook}. Slides: {body_preview}",
        caption_context=(
            f"A {n_slides}-slide carousel repurposed from a prior reel. "
            f"Hook: '{hook}'. Caption must stand on its own — assume the reader hasn't seen the reel."
        ),
        generator=f"carousel_repurpose:{template_name}",
        meta={
            "slides": slides_copy,
            "template": template_name,
            "source": "repurpose_reel",
            "source_reel_content_id": candidate.content_id,
            "source_reel_format": candidate.format,
            "source_reel_variant": candidate.variant,
        },
    )


async def run_once(cfg: NicheConfig) -> int | None:
    """Scheduled-job entry point. Generates a repurposed carousel and
    enqueues it straight to content_queue (bypassing the main pipeline —
    the captions + critic cycle runs independently via the poster)."""
    if not cfg.reel_repurpose.enabled:
        return None

    try:
        content = await generate(cfg)
    except Exception as e:
        log.exception("repurpose: generate crashed: %s", e)
        return None
    if content is None:
        return None

    # Draft a minimal caption so the row is postable even if the full
    # caption pipeline is skipped. The poster's review UI can still
    # replace it before publishing.
    from instagram_ai_agent.content import captions as caption_mod
    from instagram_ai_agent.content import hashtags as hashtag_mod

    try:
        caption_body = await caption_mod.generate_caption(
            cfg, "carousel", context=content.caption_context,
        )
    except Exception as e:
        log.warning("repurpose: caption gen failed, using stub: %s", e)
        first_slide = (content.meta.get("slides") or [{}])[0]
        caption_body = str(first_slide.get("title") or cfg.niche)
    try:
        tags = hashtag_mod.build_hashtags(cfg)
    except Exception:
        tags = []
    caption_full = caption_body + ("\n\n" + hashtag_mod.format_hashtags(tags) if tags else "")

    from instagram_ai_agent.content import dedup as dedup_mod

    phash = dedup_mod.compute_phash(content.media_paths[0]) if content.media_paths else None
    if phash:
        is_dup, match = dedup_mod.is_duplicate(phash, cfg.safety.dedup_hamming_threshold)
        if is_dup:
            log.info("repurpose: phash collision with %s — skipping enqueue", match)
            # Still mark the source so we don't re-attempt on every cycle.
            source_id = content.meta.get("source_reel_content_id")
            if isinstance(source_id, int):
                _mark_repurposed(source_id)
            return None

    status = "pending_review" if cfg.safety.require_review else "approved"
    cid = db.content_enqueue(
        format=content.format,
        caption=caption_full,
        hashtags=tags,
        media_paths=content.media_paths,
        phash=phash,
        critic_score=None,
        critic_notes=None,
        generator=content.generator,
        status=status,
        meta=content.meta,
    )
    source_id = content.meta.get("source_reel_content_id")
    if isinstance(source_id, int):
        _mark_repurposed(source_id)
    log.info(
        "repurpose: enqueued cid=%d (source reel cid=%s, %d slides, status=%s)",
        cid, source_id, len(content.media_paths), status,
    )
    return cid
