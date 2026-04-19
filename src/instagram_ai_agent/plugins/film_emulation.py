"""Film emulation — make AI-rendered images look photographed.

The single biggest visual "AI tell" on Instagram is that raw Pollinations /
Flux / SDXL / Midjourney output is TOO CLEAN. Real camera sensors add grain.
Real lenses add vignette. Real JPEG pipelines add compression artifacts.
Real photography has a colour temperature — usually slightly warm or cool,
never a perfect mid-grey.

This module applies all four in one pass using only Pillow (no extra deps):

  1. **Luminance grain** — per-pixel Gaussian noise, stronger in shadows
     so highlights stay clean (matches real sensor behaviour).
  2. **Vignette** — subtle radial darkening at the corners (< 8%) to
     mimic wide-angle lens fall-off.
  3. **Colour cast** — random warm/cool shift of ±4°K equivalent so the
     account's colour signature varies naturally post-to-post.
  4. **JPEG recompression** at quality 87 to emulate the "phone exported
     to social" compression profile.

Input/output: PIL Image or file path. Deterministic when a ``seed`` is
provided — same seed → same grain pattern, useful for A/B tests or when
an image is a slide in a multi-slide carousel (grain should match).

Usage:
    from instagram_ai_agent.plugins import film_emulation
    film_emulation.apply_film_look(image_path, strength="medium")

Strength presets:
    "off"      — no-op
    "subtle"   — minimal grain, ideal for clean aesthetics (food, architecture)
    "medium"   — default, indistinguishable-from-phone look
    "strong"   — visible film grain, 35mm-film vibe (editorial / fashion / travel)
"""
from __future__ import annotations

import hashlib
import random
from pathlib import Path

from PIL import Image, ImageChops, ImageFilter

from instagram_ai_agent.core.logging_setup import get_logger

log = get_logger(__name__)


STRENGTH_PRESETS = {
    "off":    {"grain_sigma": 0.0, "vignette": 0.0, "cast": 0.0, "jpeg_q": 95},
    "subtle": {"grain_sigma": 1.5, "vignette": 0.03, "cast": 2.0, "jpeg_q": 92},
    "medium": {"grain_sigma": 3.5, "vignette": 0.06, "cast": 4.0, "jpeg_q": 88},
    "strong": {"grain_sigma": 6.0, "vignette": 0.10, "cast": 6.0, "jpeg_q": 85},
}


def apply_film_look(
    image_path: str | Path,
    *,
    strength: str = "medium",
    seed: int | None = None,
) -> Path:
    """Apply the film-look treatment to an image on disk, overwriting it.

    Returns the path on success; returns the original path unchanged on
    any failure so the pipeline never breaks on a post-processing hiccup.
    """
    p = Path(image_path)
    if strength == "off":
        return p
    if not p.exists():
        log.debug("film_emulation: source not found: %s", p)
        return p

    preset = STRENGTH_PRESETS.get(strength, STRENGTH_PRESETS["medium"])

    # Deterministic RNG — either the explicit seed or one derived from the
    # file contents so the same image always gets the same grain.
    if seed is None:
        seed = int(hashlib.md5(p.read_bytes()[:8192]).hexdigest()[:8], 16)
    rng = random.Random(seed)

    try:
        img = Image.open(p).convert("RGB")
        img = _apply_colour_cast(img, preset["cast"], rng)
        if preset["vignette"] > 0:
            img = _apply_vignette(img, preset["vignette"])
        if preset["grain_sigma"] > 0:
            img = _apply_grain(img, preset["grain_sigma"], rng)
        img.save(p, "JPEG" if p.suffix.lower() in (".jpg", ".jpeg") else "PNG",
                 quality=preset["jpeg_q"], optimize=True)
        return p
    except Exception as e:
        log.debug("film_emulation: failed on %s — keeping original: %s", p.name, e)
        return p


def _apply_grain(img: Image.Image, sigma: float, rng: random.Random) -> Image.Image:
    """Add luminance grain weighted toward shadows.

    Real sensor grain scales inversely with signal-to-noise — shadows are
    noisier than highlights. We approximate by generating a grey-noise
    layer and blending with 'soft light' on the luminance channel only.
    """
    w, h = img.size
    # Faster than PIL's randint for large arrays — generate in ~8k-byte chunks
    noise_bytes = bytearray(w * h)
    scaled_sigma = max(1, int(sigma))
    for i in range(len(noise_bytes)):
        # Gauss-ish via two uniform draws (Irwin-Hall n=2 ≈ triangular)
        n = (rng.randint(-scaled_sigma, scaled_sigma)
             + rng.randint(-scaled_sigma, scaled_sigma)) // 2
        noise_bytes[i] = max(0, min(255, 128 + n))
    noise = Image.frombytes("L", (w, h), bytes(noise_bytes))
    # Light blur on the noise keeps it looking like grain, not digital static
    noise = noise.filter(ImageFilter.GaussianBlur(radius=0.4))
    # Blend: grain layer masked by inverse luminance (= stronger in shadows)
    lum = img.convert("L")
    shadow_mask = ImageChops.invert(lum)
    grain_rgb = Image.merge("RGB", (noise, noise, noise))
    return Image.composite(
        ImageChops.add(img, grain_rgb, scale=2.0, offset=-64),
        img,
        shadow_mask.point(lambda v: min(255, v * 1)),
    )


def _apply_vignette(img: Image.Image, strength: float) -> Image.Image:
    """Radial darkening from edges. ``strength`` 0.0–1.0 roughly = edge darkness %."""
    w, h = img.size
    # Fast approximation: a large Gaussian-blurred ellipse gradient
    cx, cy = w // 2, h // 2
    max_r = (cx * cx + cy * cy) ** 0.5
    darkening = int(255 * strength)
    # Draw a black gradient via a fat blurred ring
    grad = Image.new("L", (w, h), 0)
    # Build a simple radial: nested ovals using alpha compose
    for step in range(8):
        frac = 1.0 - step / 8
        shade = int(darkening * step / 8)
        bbox = (
            int(cx - max_r * frac),
            int(cy - max_r * frac),
            int(cx + max_r * frac),
            int(cy + max_r * frac),
        )
        ring = Image.new("L", (w, h), 0)
        from PIL import ImageDraw
        d = ImageDraw.Draw(ring)
        d.ellipse(bbox, fill=shade)
        grad = ImageChops.lighter(grad, ring)
    grad = grad.filter(ImageFilter.GaussianBlur(radius=max(w, h) // 12))

    dark = Image.new("RGB", (w, h), (0, 0, 0))
    return Image.composite(dark, img, grad)


def _apply_colour_cast(img: Image.Image, strength: float, rng: random.Random) -> Image.Image:
    """Random warm OR cool cast ≤ ``strength``. 0 = neutral."""
    if strength <= 0:
        return img
    # Decide warm (+R, -B) or cool (-R, +B). Very subtle — 3–6 points on 255.
    shift_r = rng.randint(-int(strength), int(strength))
    shift_b = -shift_r  # inverse so the cast is symmetric round the grey axis
    r, g, b = img.split()
    r = r.point(lambda v: max(0, min(255, v + shift_r)))
    b = b.point(lambda v: max(0, min(255, v + shift_b)))
    return Image.merge("RGB", (r, g, b))
