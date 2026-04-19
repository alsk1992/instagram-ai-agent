"""Stable Audio Open Small generative music backend.

Produces original, commercial-safe music beds for reels. Gated behind
the ``[stable-audio]`` optional extra (torch + stable-audio-tools) and
Stability's Community Licence acknowledgement in niche.yaml.

Flow:
  1. ``build_prompt(cfg, scene_context)`` — asks the LLM to draft a
     short English music-direction prompt from the reel's scenes +
     niche voice ("upbeat gym motivation beat, 120 BPM, minor key").
  2. ``generate(prompt, duration_s, cfg)`` — lazy-imports
     stable-audio-tools, loads the model (cached per-process),
     synthesises one clip, tiles+crossfades to fill ``duration_s``,
     writes a WAV into ``MUSIC_CACHE_DIR``. Cached by SHA256 of the
     (prompt, duration, seed) tuple so the same reel rerun reuses it.
  3. ``music.py`` wires this in as a source — users add
     ``"stable_audio"`` to ``music.sources`` to activate the chain slot.

Licensing: outputs are owned by the user per the Community Licence.
We never ship model weights; the user downloads them on first run
(stable-audio-tools pulls from HF ``stabilityai/stable-audio-open-small``).
"""
from __future__ import annotations

import asyncio
import hashlib
import math
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from instagram_ai_agent.core.config import MUSIC_CACHE_DIR, NicheConfig
from instagram_ai_agent.core.logging_setup import get_logger

log = get_logger(__name__)

# Native model output is ~11s — longer beds are tiled. We cap a single
# synth at 10s to leave crossfade headroom.
MAX_SINGLE_CLIP_S = 10.0
# Crossfade length between tile repeats
_TILE_CROSSFADE_S = 0.5

# Module-level cache so we don't reload the model on every call within
# a process. Test fixtures monkey-patch this to a stub.
_MODEL_CACHE: dict[str, Any] = {}


# ───── availability ─────
def available() -> bool:
    """True when the optional extra is installed. Does NOT download or
    load the model — that happens lazily at first generate()."""
    try:
        import stable_audio_tools  # noqa: F401
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def _resolve_device(requested: str) -> str:
    """Map 'auto' → cuda|mps|cpu; pass explicit values through."""
    if requested != "auto":
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception as _probe_err:
        log.debug("stable_audio: device probe failed — defaulting to cpu: %s",
                  _probe_err)
    return "cpu"


# ───── prompt building ─────
@dataclass(frozen=True)
class MusicDirection:
    """The English music description we feed to SAO. Structured so we
    can test prompt-building separately from the LLM call."""
    bpm: int
    mood: str
    genre: str
    key: str
    extra: str = ""

    def render(self) -> str:
        parts = [self.genre, self.mood, f"{self.bpm} BPM", f"{self.key} key"]
        if self.extra:
            parts.append(self.extra)
        return ", ".join(p for p in parts if p)


# Niche → default music direction when the LLM route is unavailable.
# Lightweight, deterministic, always works offline.
_NICHE_DEFAULTS: dict[str, MusicDirection] = {
    "calisthenics": MusicDirection(bpm=120, mood="motivational", genre="upbeat electronic", key="minor"),
    "fitness":      MusicDirection(bpm=125, mood="motivational", genre="upbeat electronic", key="minor"),
    "mindfulness":  MusicDirection(bpm=70,  mood="calm",         genre="ambient",           key="major"),
    "cooking":      MusicDirection(bpm=95,  mood="playful",      genre="lofi hip-hop",      key="major"),
    "finance":      MusicDirection(bpm=105, mood="focused",      genre="cinematic",         key="minor"),
    "productivity": MusicDirection(bpm=100, mood="focused",      genre="lofi hip-hop",      key="major"),
    "fashion":      MusicDirection(bpm=110, mood="stylish",      genre="electronic",        key="minor"),
    "travel":       MusicDirection(bpm=100, mood="uplifting",    genre="indie electronic",  key="major"),
}
_DEFAULT_DIRECTION = MusicDirection(bpm=110, mood="upbeat", genre="instrumental", key="minor")


def direction_from_niche(niche: str) -> MusicDirection:
    """Best-match music direction for a niche string. Uses lowercased
    substring matching so 'home calisthenics' picks up the calisthenics
    preset, 'travel blogger' picks travel, etc."""
    n = (niche or "").lower()
    for key, direction in _NICHE_DEFAULTS.items():
        if key in n:
            return direction
    return _DEFAULT_DIRECTION


async def build_prompt(cfg: NicheConfig, scene_context: str = "") -> str:
    """Return a one-line English music prompt for SAO Small.

    If ``sao_prompt_override`` is set on the config, returns that
    unchanged. Otherwise asks the LLM for a direction-shaped prompt
    seeded from the niche's default and falls back to the deterministic
    default on any LLM failure.
    """
    mc = cfg.music
    if mc.sao_prompt_override.strip():
        return mc.sao_prompt_override.strip()

    direction = direction_from_niche(cfg.niche)
    # Try LLM — short, bounded tokens, failure-tolerant.
    try:
        from instagram_ai_agent.core.llm import (
            generate as llm_generate,  # lazy to keep this module light
        )
    except Exception:
        return direction.render()

    scene_hint = scene_context.strip() or f"(general {cfg.niche} content)"
    system = (
        "You write one-line English prompts for a generative music model.\n"
        "Output ONE line, ≤20 words, comma-separated descriptors only.\n"
        "Include: genre, mood, BPM, key. NO lyrics, NO song titles, NO artist names."
    )
    prompt = (
        f"Niche: {cfg.niche}. Audience: {cfg.target_audience}.\n"
        f"Voice persona: {cfg.voice.persona}.\n"
        f"Default direction: {direction.render()}.\n"
        f"Scene hint: {scene_hint}\n"
        "Return the music prompt only."
    )
    try:
        out = await llm_generate("bulk", prompt, system=system, max_tokens=80)
        cleaned = out.strip().strip('"').splitlines()[0] if out else ""
        if cleaned:
            return cleaned
    except Exception as e:
        log.debug("SAO prompt LLM failed, using niche default: %s", e)
    return direction.render()


# ───── cache keys ─────
def _cache_key(
    prompt: str,
    duration_s: float,
    *,
    seed: int | None,
    steps: int,
    cfg_scale: float,
) -> str:
    """Hash every knob that changes the waveform. ``sao_device`` is
    deliberately excluded — the same prompt/seed on CPU vs GPU produces
    the same numeric output (modulo float determinism) so caching
    across device switches is fine."""
    h = hashlib.sha256()
    h.update(prompt.encode("utf-8"))
    h.update(
        f"|dur={duration_s:.2f}|seed={seed}|steps={steps}|cfg={cfg_scale:.3f}".encode()
    )
    return h.hexdigest()[:20]


def _cache_path(
    prompt: str,
    duration_s: float,
    *,
    seed: int | None,
    steps: int,
    cfg_scale: float,
) -> Path:
    return MUSIC_CACHE_DIR / (
        f"sao_{_cache_key(prompt, duration_s, seed=seed, steps=steps, cfg_scale=cfg_scale)}.wav"
    )


# ───── tiling ─────
def _probe_duration(path: Path) -> float:
    try:
        r = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True, text=True, check=True,
        )
        return float(r.stdout.strip() or 0.0)
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return 0.0


def tile_with_crossfade(
    src: Path, dest: Path, *, target_s: float, crossfade_s: float = _TILE_CROSSFADE_S,
) -> Path:
    """Loop ``src`` to fill ``target_s`` seconds, crossfading each
    loop seam with ``acrossfade=d=crossfade_s``. Writes PCM WAV to
    ``dest``. Returns dest.

    The filter graph ``asplits`` the source into N copies and chains
    them through ``acrossfade`` pairwise so every seam gets a real
    crossfade (not just start/end fades). N is computed so the
    effective run length (``N * (src_s - crossfade_s) + crossfade_s``)
    is at least ``target_s``; the ``-t`` flag then trims to exactly
    ``target_s``."""
    if target_s <= 0:
        raise ValueError("target_s must be positive")
    if not src.exists() or src.stat().st_size == 0:
        raise FileNotFoundError(f"Source clip missing: {src}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    crossfade_s = max(0.0, min(crossfade_s, 2.0))

    src_dur = _probe_duration(src)
    # Fallback when ffprobe can't tell us: assume the full target fits
    # in one pass (cheapest and safe — the `-t` trim still holds).
    if src_dur <= 0 or crossfade_s == 0 or target_s <= src_dur + 0.05:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(src),
            "-t", f"{target_s:.2f}",
            "-af", (
                f"afade=t=in:st=0:d={crossfade_s:.2f},"
                f"afade=t=out:st={max(0.0, target_s - crossfade_s):.2f}:d={crossfade_s:.2f}"
            ),
            "-c:a", "pcm_s16le", "-ar", "44100", "-ac", "2",
            str(dest),
        ]
    else:
        # How many tiles so effective duration ≥ target_s?
        effective_per_tile = max(0.1, src_dur - crossfade_s)
        n_tiles = max(2, math.ceil((target_s - crossfade_s) / effective_per_tile))
        # Cap so we don't build a graph with 100+ nodes on absurd inputs
        n_tiles = min(n_tiles, 64)

        split = f"[0:a]asplit={n_tiles}" + "".join(f"[a{i}]" for i in range(n_tiles))
        chain_parts = [split]
        prev_label = "a0"
        for i in range(1, n_tiles):
            out_label = f"x{i}" if i < n_tiles - 1 else "out"
            chain_parts.append(
                f"[{prev_label}][a{i}]acrossfade=d={crossfade_s:.2f}[{out_label}]"
            )
            prev_label = out_label
        filter_complex = ";".join(chain_parts)

        cmd = [
            "ffmpeg", "-y",
            "-i", str(src),
            "-filter_complex", filter_complex,
            "-map", "[out]",
            "-t", f"{target_s:.2f}",
            "-c:a", "pcm_s16le", "-ar", "44100", "-ac", "2",
            str(dest),
        ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise RuntimeError(f"ffmpeg tiling failed: {e}") from e
    return dest


# ───── model loading ─────
def _load_model(device: str) -> tuple[Any, dict[str, Any]]:
    """Lazy-load stable-audio-tools + the SAO Small model. Cached per
    (device) so warm calls are instant.

    Raises RuntimeError if the extra isn't installed — callers should
    check ``available()`` first."""
    if device in _MODEL_CACHE:
        return _MODEL_CACHE[device]

    try:
        from stable_audio_tools import get_pretrained_model
    except Exception as e:
        raise RuntimeError(
            "Stable Audio Open Small requires `pip install .[stable-audio]`. "
            f"Underlying error: {e}"
        ) from e

    log.info("SAO: loading stabilityai/stable-audio-open-small on %s (first call is slow)", device)
    model, model_config = get_pretrained_model("stabilityai/stable-audio-open-small")
    model = model.to(device)
    _MODEL_CACHE[device] = (model, model_config)
    return model, model_config


def _synthesise_sync(
    prompt: str, duration_s: float, *, device: str, steps: int, cfg_scale: float, seed: int | None,
) -> object:
    """Sync wrapper — invokes SAO inference and returns the float32
    waveform as a numpy-compatible array. Runs in a worker thread via
    asyncio.to_thread().

    Kept as a plain function so tests can monkey-patch it without
    needing a real torch install."""
    import torch
    from stable_audio_tools.inference.generation import generate_diffusion_cond

    model, model_config = _load_model(device)
    sample_rate = int(model_config["sample_rate"])
    sample_size = int(model_config["sample_size"])

    cond = [{"prompt": prompt, "seconds_start": 0, "seconds_total": int(duration_s)}]
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(int(seed))

    with torch.inference_mode():
        output = generate_diffusion_cond(
            model,
            steps=steps,
            cfg_scale=cfg_scale,
            conditioning=cond,
            sample_size=sample_size,
            sampler_type="dpmpp-3m-sde",
            device=device,
            generator=generator,
        )
    # output shape: [batch, channels, samples] — take first batch
    wave = output[0].to(torch.float32).cpu().numpy()
    return {"wave": wave, "sample_rate": sample_rate}


def _write_wav(wave: object, sample_rate: int, dest: Path) -> Path:
    """Write a (channels, samples) float32 numpy array as 16-bit PCM
    WAV. Uses torchaudio if available (already pulled by stable-audio-
    tools) and falls back to the stdlib ``wave`` module otherwise."""
    import numpy as np
    arr = np.asarray(wave)
    # Normalise to [-1, 1] defensively; SAO usually stays in range but
    # a clipped tensor would wrap to garbage in int16.
    peak = float(np.max(np.abs(arr))) if arr.size else 1.0
    if peak > 1.0:
        arr = arr / peak
    int16 = (arr * 32767.0).clip(-32768, 32767).astype(np.int16)
    # arr is (channels, samples) — WAV expects interleaved (samples, channels)
    if int16.ndim == 2:
        int16 = int16.T
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        import torch
        import torchaudio
        t = torch.from_numpy(int16.T if int16.ndim == 2 else int16[np.newaxis, :])
        torchaudio.save(str(dest), t, sample_rate, encoding="PCM_S", bits_per_sample=16)
    except Exception:
        # stdlib fallback
        import wave as _wave
        with _wave.open(str(dest), "wb") as wf:
            wf.setnchannels(2 if int16.ndim == 2 else 1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(int16.tobytes())
    return dest


# ───── public entry point ─────
async def generate(
    prompt: str,
    duration_s: float,
    cfg: NicheConfig,
) -> Path:
    """Generate a music bed. Returns the cached path on disk.

    ``duration_s`` may exceed SAO Small's native 11s cap — we synthesise
    a sub-cap clip once and tile with crossfades to hit the target
    length.

    Licence gate: re-asserts ``sao_enabled`` + ``sao_license_acknowledged``
    at the call site even though ``NicheConfig._sao_license_gate`` also
    fires at load time — Pydantic model validators don't re-run on
    ``.model_copy(update=...)`` so a runtime check closes that seam.
    """
    if not available():
        raise RuntimeError(
            "Stable Audio Open Small is not installed. "
            "Run `pip install '.[stable-audio]'` to enable it."
        )
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")

    mc = cfg.music
    # Defence in depth — the config-load validator is the primary gate,
    # but model_copy() bypasses it and direct MusicConfig construction
    # produces a dormant-but-dangerous state.
    if not mc.sao_enabled:
        raise RuntimeError("music.sao_enabled is False — refusing to generate.")
    if not mc.sao_license_acknowledged:
        raise RuntimeError(
            "music.sao_license_acknowledged is False — refusing to generate. "
            "Set it to True in niche.yaml after reading the Stability AI "
            "Community Licence at https://stability.ai/community-license-agreement."
        )

    # Seed-None means "re-roll on every call" — skip the cache entirely
    # so the user gets fresh audio instead of a stale hit from a prior
    # run of the same prompt.
    use_cache = mc.sao_seed is not None
    if use_cache:
        cached = _cache_path(
            prompt, duration_s,
            seed=mc.sao_seed, steps=mc.sao_steps, cfg_scale=mc.sao_cfg_scale,
        )
        if cached.exists() and cached.stat().st_size > 10_000:
            log.info("SAO: cache hit %s", cached.name)
            return cached
    else:
        cached = MUSIC_CACHE_DIR / f"sao_{uuid.uuid4().hex[:16]}.wav"

    device = _resolve_device(mc.sao_device)
    single_s = min(duration_s, MAX_SINGLE_CLIP_S)

    # Write to a staging path first so concurrent callers can't see
    # partial bytes, then atomic-replace into the cache slot.
    staging = cached.with_name(cached.stem + f"_stage_{uuid.uuid4().hex[:6]}.wav")
    try:
        result = await asyncio.to_thread(
            _synthesise_sync,
            prompt, single_s,
            device=device, steps=mc.sao_steps,
            cfg_scale=mc.sao_cfg_scale, seed=mc.sao_seed,
        )
    except Exception as e:
        log.warning("SAO synthesis failed: %s", e)
        raise

    _write_wav(result["wave"], int(result["sample_rate"]), staging)

    try:
        if duration_s <= single_s + 0.05:
            staging.replace(cached)          # atomic on POSIX
        else:
            tiled = cached.with_name(cached.stem + "_tiled.wav")
            tile_with_crossfade(staging, tiled, target_s=duration_s)
            tiled.replace(cached)
            staging.unlink(missing_ok=True)
    finally:
        # On any pre-replace failure, clean the staging file
        try:
            staging.unlink(missing_ok=True)
        except OSError as _cleanup_err:
            log.debug("stable_audio: staging cleanup of %s failed: %s",
                      staging, _cleanup_err)

    log.info(
        "SAO: generated %s (%.1fs, prompt=%r, cached=%s)",
        cached.name, duration_s, prompt[:60], use_cache,
    )
    return cached
