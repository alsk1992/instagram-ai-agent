"""Upscale + face-restore finish pass (Real-ESRGAN + GFPGAN).

Two execution paths, picked at call time by availability:

  1. **Local** — real torch + realesrgan + gfpgan. First call lazy-loads
     weights (auto-downloaded on first import by the packages themselves),
     subsequent calls reuse the cached models.
  2. **HF Space** — gradio_client calls a hosted Space. Slower, queued,
     but free and requires no local GPU.

Both paths are *entirely optional*. Every import is guarded and every failure
downgrades to passing the original image through unchanged. The caller sees
a single `enhance(path, cfg, subject_is_human)` surface and never has to
care about the backend.

Commercial-safe licensing notes baked in:
  * Real-ESRGAN  — BSD-3-Clause        — OK
  * GFPGAN       — Apache-2.0          — OK
  * CodeFormer   — S-Lab NC            — **never used here**
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.core.config import MEDIA_STAGED, FinishPass
from src.core.logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class FinishResult:
    path: Path
    upscaled: bool
    face_restored: bool
    backend: str  # "local" | "hf" | "noop"
    notes: str = ""


# ─────────── Local backend (lazy-loaded singletons) ───────────
_local_upscaler: Any = None
_local_restorer: Any = None


def _local_available() -> bool:
    try:
        import realesrgan  # noqa: F401
        import gfpgan      # noqa: F401
        import torch       # noqa: F401
        return True
    except Exception:
        return False


def _get_local_upscaler(factor: int):
    """Lazy-load Real-ESRGAN. Caches by factor — we usually stick with one."""
    global _local_upscaler
    if _local_upscaler is not None and _local_upscaler.get("factor") == factor:
        return _local_upscaler["instance"]

    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    # Pick model architecture matching the factor. x2 and x4 are the canonical
    # commercial-safe weights shipped with the realesrgan package.
    if factor == 4:
        model_name = "RealESRGAN_x4plus"
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    else:
        # 2x model has 6-block RRDB
        model_name = "RealESRGAN_x2plus"
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"

    import torch
    use_half = torch.cuda.is_available()
    instance = RealESRGANer(
        scale=factor,
        model_path=url,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=use_half,
        gpu_id=0 if torch.cuda.is_available() else None,
    )
    _local_upscaler = {"factor": factor, "instance": instance, "name": model_name}
    return instance


def _get_local_restorer(upscale: int):
    """Lazy-load GFPGAN. ``upscale`` here is 1 — restoration only; upscale
    is handled separately by Real-ESRGAN."""
    global _local_restorer
    if _local_restorer is not None:
        return _local_restorer

    from gfpgan import GFPGANer

    url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth"
    instance = GFPGANer(
        model_path=url,
        upscale=upscale,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,       # Real-ESRGAN already handled bg upscale
    )
    _local_restorer = instance
    return instance


def _local_upscale(src: Path, dst: Path, factor: int) -> Path:
    import cv2

    upscaler = _get_local_upscaler(factor)
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"cv2 could not read {src}")
    out, _ = upscaler.enhance(img, outscale=factor)
    cv2.imwrite(str(dst), out, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
    return dst


def _local_restore_face(src: Path, dst: Path) -> Path:
    """Restore faces in-place via GFPGAN.

    Raises ``_NoFaceDetected`` when the detector finds nothing to restore,
    so the outer caller records ``face_restored=False`` honestly instead
    of shipping telemetry that claims restoration on an unmodified copy.
    """
    import cv2

    restorer = _get_local_restorer(upscale=1)
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"cv2 could not read {src}")
    _cropped, _restored_faces, restored_img = restorer.enhance(
        img,
        has_aligned=False,
        only_center_face=False,
        paste_back=True,
    )
    if restored_img is None:
        raise _NoFaceDetected("GFPGAN detected no faces to restore")
    cv2.imwrite(str(dst), restored_img, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
    return dst


class _NoFaceDetected(RuntimeError):
    """Signals the subject was flagged human but no face could be detected."""


def _staged_sibling(src: Path, suffix: str) -> Path:
    """Intermediate output path — always under MEDIA_STAGED, never polluting
    the caller's directory (which could be the user's library for add-content).
    """
    MEDIA_STAGED.mkdir(parents=True, exist_ok=True)
    return MEDIA_STAGED / f"{src.stem}{suffix}{src.suffix}"


# ─────────── HF Space backend ───────────
def _hf_available() -> bool:
    try:
        from gradio_client import Client  # noqa: F401
        return True
    except Exception:
        return False


def _hf_upscale(src: Path, dst: Path, factor: int, space: str) -> Path | None:
    from gradio_client import Client, file

    try:
        client = Client(space)
    except Exception as e:
        log.warning("HF upscale: cannot open space %s: %s", space, e)
        return None
    try:
        # Nick088/Real-ESRGAN_Pytorch exposes predict(image, outscale)
        result = client.predict(
            image=file(str(src)),
            outscale=float(factor),
            api_name="/predict",
        )
    except Exception as e:
        log.warning("HF upscale call failed on %s: %s", space, e)
        return None
    if not result:
        return None
    # Result is the path of the produced file in the gradio tmp
    try:
        shutil.copy2(result, dst)
    except Exception as e:
        log.warning("HF upscale: result copy failed: %s", e)
        return None
    return dst


def _hf_restore_face(src: Path, dst: Path, space: str) -> Path | None:
    from gradio_client import Client, file

    try:
        client = Client(space)
    except Exception as e:
        log.warning("HF face restore: cannot open space %s: %s", space, e)
        return None
    try:
        result = client.predict(img=file(str(src)), api_name="/predict")
    except Exception as e:
        log.warning("HF face restore call failed: %s", e)
        return None
    if not result:
        return None
    try:
        shutil.copy2(result, dst)
    except Exception as e:
        log.warning("HF face restore: result copy failed: %s", e)
        return None
    return dst


# ─────────── Public entry ───────────
def enhance(
    image_path: str | Path,
    cfg_finish: FinishPass,
    *,
    subject_is_human: bool = False,
) -> FinishResult:
    """Run the finish pass on ``image_path``.

    Always returns a FinishResult. On no-op the returned path equals the
    input. Never raises — every failure degrades to a notes-tagged no-op.
    """
    src = Path(image_path)
    if not cfg_finish.enabled:
        return FinishResult(src, False, False, "noop", "disabled")
    if not src.exists():
        return FinishResult(src, False, False, "noop", "missing input")

    # Cap input size by megapixels to avoid OOM on giant sources.
    # If PIL can't even probe the header, refuse to run — upscaling an
    # unreadable file will only waste GPU time or crash the worker.
    try:
        from PIL import Image
        with Image.open(src) as im:
            mp = (im.width * im.height) / 1_000_000
            if mp > cfg_finish.max_input_megapixels:
                return FinishResult(src, False, False, "noop", f"input {mp:.1f}MP > cap")
    except Exception as e:
        log.warning("finish: cannot probe %s (%s) — skipping", src.name, e)
        return FinishResult(src, False, False, "noop", f"probe failed: {e}")

    want_upscale = cfg_finish.upscale_factor > 1
    want_face = cfg_finish.face_restore and subject_is_human

    # Prefer local if available and permitted
    backend = "noop"
    upscaled = False
    restored = False
    notes: list[str] = []

    current = src

    if cfg_finish.use_local and _local_available():
        backend = "local"
        if want_upscale:
            try:
                up_path = _staged_sibling(src, "_up")
                current = _local_upscale(current, up_path, cfg_finish.upscale_factor)
                upscaled = True
            except Exception as e:
                notes.append(f"local upscale failed: {e}")
                log.warning("local upscale failed: %s", e)
        if want_face:
            try:
                face_path = _staged_sibling(src, "_face")
                current = _local_restore_face(current, face_path)
                restored = True
            except _NoFaceDetected:
                # Honest reporting: face restore was attempted but had nothing
                # to do. Current image is unchanged; do NOT claim restoration.
                notes.append("no face detected")
            except Exception as e:
                notes.append(f"local face restore failed: {e}")
                log.warning("local face restore failed: %s", e)

    elif cfg_finish.use_hf_fallback and _hf_available():
        backend = "hf"
        if want_upscale:
            up_path = _staged_sibling(src, "_up")
            out = _hf_upscale(current, up_path, cfg_finish.upscale_factor, cfg_finish.hf_upscale_space)
            if out is not None:
                current = out
                upscaled = True
            else:
                notes.append("hf upscale unavailable")
        if want_face:
            face_path = _staged_sibling(src, "_face")
            out = _hf_restore_face(current, face_path, cfg_finish.hf_face_space)
            if out is not None:
                current = out
                restored = True
            else:
                notes.append("hf face restore unavailable")
    else:
        notes.append("no backend available — install [finish] extra or enable hf fallback")

    return FinishResult(
        current,
        upscaled=upscaled,
        face_restored=restored,
        backend=backend if (upscaled or restored) else "noop",
        notes="; ".join(notes),
    )


def reset_caches() -> None:
    """Drop cached model instances — useful for tests."""
    global _local_upscaler, _local_restorer
    _local_upscaler = None
    _local_restorer = None
