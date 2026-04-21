"""HuggingFace Spaces video rotator — free T2V/I2V via gradio_client.

No paid API. No credit card. Zero signin for unauthenticated use (tighter
quota); authenticated HF token bumps to the free-tier ZeroGPU pool (~3.5
min H200/day per account — still free).

The pool rotates across community-hosted Spaces for Wan 2.2 / LTX-2 /
SVD that I verified were running with public APIs on 2026-04-21:

  - alexnasa/ltx-2-TURBO (LTX-2 Turbo, T2V+audio, ZeroGPU)
  - zerogpu-aoti/wan2-2-fp8da-aoti-faster (Wan2.2 14B I2V, ZeroGPU)
  - r3gm/wan2-2-fp8da-aoti-preview (Wan2.2 mirror fallback)
  - dream2589632147/Dream-wan2-2-faster-Pro (Wan2.2 I2V alt)
  - linoyts/LTX-2-3-sync (LTX 2.3 with motion conditioning)
  - Pyramid-Flow/pyramid-flow (older architecture fallback)
  - multimodalart/stable-video-diffusion (SVD 1.1 legacy, rock-solid last resort)

Strategy:
  1. Try Spaces in order; first success wins.
  2. On GradioError / queue-full / 429, park the Space for 10 min and
     move to the next one.
  3. Optional HF_TOKEN env var enables authenticated calls → ZeroGPU
     priority tier. Unauthenticated still works but slower + lower quota.

Integration: this module is called by reel_ai.py's generator path
(replacing the old Pollinations image→Ken-Burns hack).
"""
from __future__ import annotations

import asyncio
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from instagram_ai_agent.content.generators.base import staging_path
from instagram_ai_agent.core.logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class VideoSpace:
    """One HuggingFace Space endpoint."""
    space_id: str              # e.g. "alexnasa/ltx-2-TURBO"
    api_name: str              # "/predict" or "/generate" — the Gradio API endpoint
    mode: str                  # "t2v" | "i2v"
    notes: str = ""


# Ordered rotation pool. First = highest preference; we work through on failure.
VIDEO_POOL: list[VideoSpace] = [
    VideoSpace(
        space_id="alexnasa/ltx-2-TURBO",
        api_name="/generate",
        mode="t2v",
        notes="LTX-2 Turbo — fast distilled, native 9:16 capable, audio-producing",
    ),
    VideoSpace(
        space_id="zerogpu-aoti/wan2-2-fp8da-aoti-faster",
        api_name="/generate",
        mode="i2v",
        notes="Wan2.2 14B FP8 — image→video, 5s clips",
    ),
    VideoSpace(
        space_id="r3gm/wan2-2-fp8da-aoti-preview",
        api_name="/generate",
        mode="i2v",
        notes="Wan2.2 mirror — fallback when primary is queued",
    ),
    VideoSpace(
        space_id="dream2589632147/Dream-wan2-2-faster-Pro",
        api_name="/generate",
        mode="i2v",
        notes="Wan2.2 I2V alternative mirror",
    ),
    VideoSpace(
        space_id="Pyramid-Flow/pyramid-flow",
        api_name="/predict",
        mode="t2v",
        notes="Pyramid Flow — older architecture but lower queue contention",
    ),
    VideoSpace(
        space_id="multimodalart/stable-video-diffusion",
        api_name="/predict",
        mode="i2v",
        notes="SVD 1.1 legacy — rock-solid last resort",
    ),
]


# Per-space cooldown state (in-memory, reset on process restart).
_cooldown: dict[str, float] = {}
_DEFAULT_COOLDOWN_S = 600.0   # 10 min when a Space reports queue-full / overload


def _is_cooling(space_id: str) -> float:
    until = _cooldown.get(space_id, 0.0)
    remaining = until - time.monotonic()
    return remaining if remaining > 0 else 0.0


def _park(space_id: str, seconds: float = _DEFAULT_COOLDOWN_S) -> None:
    _cooldown[space_id] = time.monotonic() + seconds


def _hf_token() -> str | None:
    """Optional HF token from env. Authenticated calls get ZeroGPU priority;
    unauthenticated still works at lower quota."""
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    return tok or None


def _call_t2v(space: VideoSpace, prompt: str, duration_s: float, out_path: Path) -> bool:
    """Synchronous call into a text-to-video Space via gradio_client.
    Saves the returned mp4 to ``out_path``. Returns True on success.

    Runs in a thread — caller should wrap with asyncio.to_thread."""
    try:
        from gradio_client import Client  # type: ignore[import-not-found]
    except ImportError:
        log.warning("hf_video: gradio_client not installed — cannot call Spaces")
        return False

    try:
        client = Client(space.space_id, hf_token=_hf_token())
    except Exception as e:
        log.warning("hf_video: Client init failed for %s — %s", space.space_id, e)
        return False

    # Gradio Spaces have wildly different API signatures — we try the most
    # common T2V argument shape first; if that fails, we fall back. The
    # returned value is usually a filepath to an mp4, or a dict with the
    # path under 'video' / 'value'.
    try:
        # Common signature: (prompt, [duration]) → mp4 path
        result = client.predict(
            prompt,
            int(max(2.0, min(10.0, duration_s))),
            api_name=space.api_name,
        )
    except Exception:
        try:
            result = client.predict(prompt, api_name=space.api_name)
        except Exception as e:
            log.warning("hf_video: predict failed for %s — %s", space.space_id, e)
            return False

    # Normalise the result to a source file path
    src_path: str | None = None
    if isinstance(result, str):
        src_path = result
    elif isinstance(result, dict):
        src_path = result.get("video") or result.get("path") or result.get("value")
    elif isinstance(result, (list, tuple)) and result:
        first = result[0]
        if isinstance(first, str):
            src_path = first
        elif isinstance(first, dict):
            src_path = first.get("video") or first.get("path") or first.get("value")

    if not src_path or not Path(src_path).exists():
        log.warning("hf_video: %s returned non-file result: %r",
                    space.space_id, str(result)[:150])
        return False

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, out_path)
        return True
    except Exception as e:
        log.warning("hf_video: copy result failed — %s", e)
        return False


async def generate_t2v(prompt: str, *, duration_s: float = 5.0) -> Path | None:
    """Try each T2V/I2V Space in the pool until one returns a video.
    Returns the local mp4 path, or None when the whole pool fails.

    Image-to-video Spaces are invoked via t2v-style prompt too — the
    Wan mirrors accept prompt-only calls that generate both keyframes
    and motion internally."""
    out = staging_path("hfv", ".mp4")
    for space in VIDEO_POOL:
        cd = _is_cooling(space.space_id)
        if cd > 0:
            log.debug("hf_video: skip %s (cooldown %.0fs)", space.space_id, cd)
            continue
        log.info("hf_video: trying %s (%s) for '%s…'",
                 space.space_id, space.mode, prompt[:60])
        try:
            ok = await asyncio.to_thread(_call_t2v, space, prompt, duration_s, out)
        except Exception as e:
            log.warning("hf_video: thread error on %s — %s", space.space_id, e)
            _park(space.space_id)
            continue
        if ok and out.exists() and out.stat().st_size > 10_000:
            log.info("hf_video: ✓ %s produced %s (%d bytes)",
                     space.space_id, out.name, out.stat().st_size)
            return out
        _park(space.space_id)

    log.warning("hf_video: entire pool failed or cooling down")
    return None
