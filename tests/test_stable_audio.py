"""Stable Audio Open Small — config gate, prompt building, cache, tiling,
music.py integration."""
from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import pytest

from src.core import config as cfg_mod
from src.plugins import music as music_mod
from src.plugins import stable_audio as sao


def _mkcfg(**kwargs):
    base = dict(
        niche="home calisthenics",
        sub_topics=["pullups"],
        target_audience="dads 35+",
        commercial=True,
        voice=cfg_mod.Voice(tone=["direct"], forbidden=[], persona="ex-office worker rebuilding fitness."),
        aesthetic=cfg_mod.Aesthetic(palette=["#000", "#fff", "#c9a961"]),
        hashtags=cfg_mod.HashtagPools(core=["calisthenics", "homeworkout", "dadfit"]),
    )
    base.update(kwargs)
    return cfg_mod.NicheConfig(**base)


@pytest.fixture()
def patched_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cache = tmp_path / "music_cache"
    cache.mkdir()
    monkeypatch.setattr(cfg_mod, "MUSIC_CACHE_DIR", cache)
    monkeypatch.setattr(sao, "MUSIC_CACHE_DIR", cache)
    monkeypatch.setattr(music_mod, "MUSIC_CACHE_DIR", cache)
    yield cache


@pytest.fixture(autouse=True)
def _isolate_model_cache():
    """SAO caches the loaded model in a module-level dict. Wipe between
    tests so monkey-patched loaders don't leak."""
    sao._MODEL_CACHE.clear()
    yield
    sao._MODEL_CACHE.clear()


# ─── Config ───
def test_sao_config_defaults_sane():
    cfg = _mkcfg()
    assert cfg.music.sao_enabled is False
    assert cfg.music.sao_license_acknowledged is False
    assert cfg.music.sao_duration_s == 30
    assert 4 <= cfg.music.sao_steps <= 100
    assert cfg.music.sao_device == "auto"
    assert cfg.music.sao_prompt_override == ""


def test_sao_config_roundtrips():
    import yaml
    cfg = _mkcfg(music=cfg_mod.MusicConfig(
        sao_enabled=True,
        sao_license_acknowledged=True,
        sao_duration_s=45,
        sao_device="cpu",
        sao_prompt_override="lofi study beat, 90 BPM, major key",
    ))
    d = cfg.model_dump(mode="json")
    loaded = cfg_mod.NicheConfig.model_validate(yaml.safe_load(yaml.safe_dump(d)))
    assert loaded.music.sao_enabled is True
    assert loaded.music.sao_duration_s == 45
    assert loaded.music.sao_prompt_override == "lofi study beat, 90 BPM, major key"


def test_sao_license_gate_blocks_when_unacknowledged():
    with pytest.raises(Exception) as exc:
        _mkcfg(music=cfg_mod.MusicConfig(sao_enabled=True, sao_license_acknowledged=False))
    assert "license" in str(exc.value).lower() or "licence" in str(exc.value).lower()


def test_sao_license_gate_allows_when_acknowledged():
    cfg = _mkcfg(music=cfg_mod.MusicConfig(sao_enabled=True, sao_license_acknowledged=True))
    assert cfg.music.sao_enabled is True


def test_sao_license_gate_ignores_disabled():
    """A license_acknowledged=False config is fine when sao_enabled=False."""
    cfg = _mkcfg(music=cfg_mod.MusicConfig(sao_enabled=False, sao_license_acknowledged=False))
    assert cfg.music.sao_enabled is False


# ─── availability ───
def test_available_returns_false_when_extra_missing():
    """The CI env doesn't install `[stable-audio]` so this must be False."""
    # stable_audio_tools and torch aren't in dev deps
    assert sao.available() is False


# ─── device resolution ───
def test_resolve_device_passes_explicit_through():
    assert sao._resolve_device("cpu") == "cpu"
    assert sao._resolve_device("cuda") == "cuda"
    assert sao._resolve_device("mps") == "mps"


def test_resolve_device_auto_without_torch_returns_cpu(monkeypatch):
    # torch isn't installed; auto → cpu is the safe default
    assert sao._resolve_device("auto") == "cpu"


# ─── prompt / direction ───
@pytest.mark.parametrize("niche,expected_genre", [
    ("home calisthenics", "upbeat electronic"),
    ("mindfulness coach", "ambient"),
    ("finance for founders", "cinematic"),
    ("cooking with herbs", "lofi hip-hop"),
    ("travel blogger", "indie electronic"),
])
def test_direction_from_niche_matches_substring(niche, expected_genre):
    direction = sao.direction_from_niche(niche)
    assert direction.genre == expected_genre


def test_direction_from_niche_falls_back_to_default():
    direction = sao.direction_from_niche("aardvark diaries")
    assert direction == sao._DEFAULT_DIRECTION


def test_direction_render_produces_comma_separated_line():
    direction = sao.MusicDirection(bpm=120, mood="upbeat", genre="electronic", key="minor")
    rendered = direction.render()
    assert rendered == "electronic, upbeat, 120 BPM, minor key"


def test_direction_render_includes_extra():
    direction = sao.MusicDirection(bpm=120, mood="upbeat", genre="electronic", key="minor", extra="dark synth")
    assert "dark synth" in direction.render()


# ─── build_prompt ───
@pytest.mark.asyncio
async def test_build_prompt_uses_override_when_set():
    cfg = _mkcfg(music=cfg_mod.MusicConfig(
        sao_enabled=True, sao_license_acknowledged=True,
        sao_prompt_override="ambient drones, 60 BPM, minor key",
    ))
    out = await sao.build_prompt(cfg, scene_context="anything")
    assert out == "ambient drones, 60 BPM, minor key"


@pytest.mark.asyncio
async def test_build_prompt_falls_back_to_niche_default_when_llm_fails(monkeypatch):
    cfg = _mkcfg()

    async def broken_generate(*a, **k):
        raise RuntimeError("LLM down")

    import src.core.llm as llm_mod
    monkeypatch.setattr(llm_mod, "generate", broken_generate, raising=False)
    out = await sao.build_prompt(cfg, scene_context="pullup struggle")
    # Matches the calisthenics niche default
    assert "electronic" in out
    assert "120 BPM" in out


@pytest.mark.asyncio
async def test_build_prompt_uses_llm_output_when_available(monkeypatch):
    cfg = _mkcfg()
    captured: dict = {}

    async def fake_generate(route, prompt, *, system, max_tokens=80):
        captured["route"] = route
        captured["system"] = system
        return '"heavy industrial techno, 140 BPM, minor key"'

    import src.core.llm as llm_mod
    monkeypatch.setattr(llm_mod, "generate", fake_generate, raising=False)
    out = await sao.build_prompt(cfg, scene_context="gym montage")
    assert "industrial" in out
    # Should strip the surrounding quotes
    assert not out.startswith('"')
    assert captured["route"] == "bulk"


@pytest.mark.asyncio
async def test_build_prompt_strips_trailing_newlines(monkeypatch):
    cfg = _mkcfg()

    async def fake_generate(*a, **k):
        return "lofi study beat, 90 BPM\n\nwith additional content we don't want"

    import src.core.llm as llm_mod
    monkeypatch.setattr(llm_mod, "generate", fake_generate, raising=False)
    out = await sao.build_prompt(cfg)
    assert "\n" not in out
    assert out == "lofi study beat, 90 BPM"


# ─── cache keys ───
def test_cache_key_is_deterministic():
    a = sao._cache_key("prompt", 30.0, seed=42, steps=8, cfg_scale=6.0)
    b = sao._cache_key("prompt", 30.0, seed=42, steps=8, cfg_scale=6.0)
    assert a == b


def test_cache_key_differs_by_prompt():
    a = sao._cache_key("p1", 30.0, seed=42, steps=8, cfg_scale=6.0)
    b = sao._cache_key("p2", 30.0, seed=42, steps=8, cfg_scale=6.0)
    assert a != b


def test_cache_key_differs_by_duration():
    a = sao._cache_key("p", 30.0, seed=42, steps=8, cfg_scale=6.0)
    b = sao._cache_key("p", 45.0, seed=42, steps=8, cfg_scale=6.0)
    assert a != b


def test_cache_key_differs_by_seed():
    a = sao._cache_key("p", 30.0, seed=42, steps=8, cfg_scale=6.0)
    b = sao._cache_key("p", 30.0, seed=43, steps=8, cfg_scale=6.0)
    a_none = sao._cache_key("p", 30.0, seed=None, steps=8, cfg_scale=6.0)
    assert a != b
    assert a != a_none


def test_cache_key_differs_by_steps():
    """Audit fix: changing sao_steps must invalidate the cache — the
    waveform genuinely differs between 8 and 16 steps."""
    a = sao._cache_key("p", 30.0, seed=42, steps=8, cfg_scale=6.0)
    b = sao._cache_key("p", 30.0, seed=42, steps=16, cfg_scale=6.0)
    assert a != b


def test_cache_key_differs_by_cfg_scale():
    a = sao._cache_key("p", 30.0, seed=42, steps=8, cfg_scale=6.0)
    b = sao._cache_key("p", 30.0, seed=42, steps=8, cfg_scale=7.5)
    assert a != b


# ─── tile_with_crossfade (requires ffmpeg, skip if absent) ───
def _have_ffmpeg() -> bool:
    import shutil
    return shutil.which("ffmpeg") is not None


@pytest.mark.skipif(not _have_ffmpeg(), reason="ffmpeg not installed")
def test_tile_with_crossfade_produces_target_length(patched_cache, tmp_path: Path):
    """Generate a 1s sine as the source and tile it to 5s."""
    import subprocess
    src = tmp_path / "beep.wav"
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
        "-ar", "44100", "-ac", "2",
        str(src),
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    dest = tmp_path / "tiled.wav"
    sao.tile_with_crossfade(src, dest, target_s=5.0, crossfade_s=0.3)
    assert dest.exists() and dest.stat().st_size > 0
    # Check duration is approximately 5.0s
    probe = subprocess.run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(dest),
    ], capture_output=True, text=True, check=True)
    duration = float(probe.stdout.strip())
    assert 4.9 <= duration <= 5.1


def test_tile_crossfade_builds_acrossfade_chain_for_long_target(monkeypatch, tmp_path: Path):
    """Audit fix: verify the filter_complex emitted for a long target
    actually chains multiple acrossfade stages (one per seam), instead
    of only applying fade-in/out on a hard loop."""
    src = tmp_path / "src.wav"
    src.write_bytes(b"RIFF" + b"\x00" * 1000)

    # Stub probe + capture the ffmpeg command so we can inspect it
    monkeypatch.setattr(sao, "_probe_duration", lambda p: 10.0)
    captured: dict = {}

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        class _R: returncode = 0
        return _R()

    monkeypatch.setattr(sao.subprocess, "run", fake_run)
    sao.tile_with_crossfade(src, tmp_path / "dest.wav", target_s=30.0, crossfade_s=0.5)
    # Find the filter_complex string in the command
    cmd = captured["cmd"]
    assert "-filter_complex" in cmd
    flt = cmd[cmd.index("-filter_complex") + 1]
    # Need >=2 acrossfade stages for a 30s target from a 10s source
    # (10s - 0.5 crossfade = 9.5s effective per tile → 30 / 9.5 ≈ 4 tiles → 3 acrossfades)
    assert flt.count("acrossfade") >= 2
    assert "asplit=" in flt
    # Must -map the final [out] label
    assert "-map" in cmd
    assert "[out]" in cmd


def test_tile_crossfade_no_filter_complex_for_short_target(monkeypatch, tmp_path: Path):
    """When target fits in one clip, no tile chain needed — plain afade."""
    src = tmp_path / "src.wav"
    src.write_bytes(b"RIFF" + b"\x00" * 1000)
    monkeypatch.setattr(sao, "_probe_duration", lambda p: 10.0)
    captured: dict = {}

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        class _R: returncode = 0
        return _R()

    monkeypatch.setattr(sao.subprocess, "run", fake_run)
    sao.tile_with_crossfade(src, tmp_path / "dest.wav", target_s=8.0, crossfade_s=0.5)
    cmd = captured["cmd"]
    assert "-filter_complex" not in cmd
    assert "-af" in cmd


def test_tile_with_crossfade_rejects_zero_target(tmp_path: Path):
    src = tmp_path / "s.wav"
    src.write_bytes(b"\x00" * 100)
    with pytest.raises(ValueError):
        sao.tile_with_crossfade(src, tmp_path / "d.wav", target_s=0)


def test_tile_with_crossfade_rejects_missing_source(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        sao.tile_with_crossfade(tmp_path / "ghost.wav", tmp_path / "d.wav", target_s=5.0)


def test_sao_duration_max_matches_ig_reel_cap():
    """Audit fix: sao_duration_s cap tightened to 90s (IG reels cap at 90s)."""
    with pytest.raises(Exception):
        cfg_mod.MusicConfig(sao_duration_s=180)
    # 90 is still allowed
    assert cfg_mod.MusicConfig(sao_duration_s=90).sao_duration_s == 90


# ─── generate() — fully mocked (no real torch / no real model) ───
@pytest.mark.asyncio
async def test_generate_refuses_when_extra_missing(patched_cache):
    """available()=False → RuntimeError pointing at the install command."""
    cfg = _mkcfg(music=cfg_mod.MusicConfig(
        sao_enabled=True, sao_license_acknowledged=True,
    ))
    with pytest.raises(RuntimeError) as exc:
        await sao.generate("upbeat, 120 BPM", 30.0, cfg)
    assert "stable-audio" in str(exc.value).lower()


@pytest.mark.asyncio
async def test_generate_rejects_zero_duration(patched_cache, monkeypatch):
    monkeypatch.setattr(sao, "available", lambda: True)
    cfg = _mkcfg(music=cfg_mod.MusicConfig(
        sao_enabled=True, sao_license_acknowledged=True,
    ))
    with pytest.raises(ValueError):
        await sao.generate("p", 0.0, cfg)


@pytest.mark.asyncio
async def test_generate_runtime_gate_rejects_bypass_via_model_copy(patched_cache, monkeypatch):
    """Audit fix: a user bypasses _sao_license_gate by constructing
    MusicConfig directly (or via model_copy). generate() must refuse
    at runtime."""
    monkeypatch.setattr(sao, "available", lambda: True)
    # Build a cfg through the validator with sao_enabled=False, then
    # flip the bit via model_copy to simulate the bypass.
    base = _mkcfg()
    poisoned_music = base.music.model_copy(update={
        "sao_enabled": True,
        "sao_license_acknowledged": False,
    })
    cfg = base.model_copy(update={"music": poisoned_music})
    with pytest.raises(RuntimeError) as exc:
        await sao.generate("p", 10.0, cfg)
    assert "license" in str(exc.value).lower() or "licence" in str(exc.value).lower()


@pytest.mark.asyncio
async def test_generate_runtime_gate_rejects_disabled(patched_cache, monkeypatch):
    monkeypatch.setattr(sao, "available", lambda: True)
    cfg = _mkcfg()  # sao_enabled=False
    with pytest.raises(RuntimeError):
        await sao.generate("p", 10.0, cfg)


@pytest.mark.asyncio
async def test_generate_seed_none_skips_cache(patched_cache, monkeypatch):
    """Audit fix: with sao_seed=None, every call must produce a fresh
    file — no silent stale-cache hit under the 'None' key."""
    monkeypatch.setattr(sao, "available", lambda: True)
    cfg = _mkcfg(music=cfg_mod.MusicConfig(
        sao_enabled=True, sao_license_acknowledged=True, sao_seed=None,
    ))

    synth_calls = {"n": 0}

    def fake_synth(prompt, duration_s, *, device, steps, cfg_scale, seed):
        synth_calls["n"] += 1
        import numpy as np
        wave = np.zeros((2, int(44100 * duration_s)), dtype="float32")
        return {"wave": wave, "sample_rate": 44100}

    def fake_write(wave, sr, dest):
        dest.write_bytes(b"\x00" * 200_000)
        return dest

    monkeypatch.setattr(sao, "_synthesise_sync", fake_synth)
    monkeypatch.setattr(sao, "_write_wav", fake_write)
    monkeypatch.setattr(sao, "tile_with_crossfade", lambda src, dest, **k: dest.write_bytes(src.read_bytes()) or dest)

    p1 = await sao.generate("p", 8.0, cfg)
    p2 = await sao.generate("p", 8.0, cfg)
    # Two distinct files
    assert p1 != p2
    # And two synth calls — no cache hit for seed=None
    assert synth_calls["n"] == 2


@pytest.mark.asyncio
async def test_generate_caches_output(patched_cache, monkeypatch):
    """Second call with same prompt+duration+seed returns the cached file
    without re-invoking the synthesiser."""
    monkeypatch.setattr(sao, "available", lambda: True)
    cfg = _mkcfg(music=cfg_mod.MusicConfig(
        sao_enabled=True, sao_license_acknowledged=True, sao_seed=123,
    ))

    synth_calls = {"n": 0}

    def fake_synth(prompt, duration_s, *, device, steps, cfg_scale, seed):
        synth_calls["n"] += 1
        # Return something _write_wav can serialise
        import numpy as np
        wave = np.zeros((2, int(44100 * min(duration_s, 10.0))), dtype="float32")
        return {"wave": wave, "sample_rate": 44100}

    def fake_write_wav(wave, sr, dest):
        dest.write_bytes(b"\x00" * 200_000)  # > 10k so cache check passes
        return dest

    # Skip tiling — write a fully-sized file directly
    def fake_tile(src, dest, *, target_s, crossfade_s=0.5):
        dest.write_bytes(src.read_bytes())
        return dest

    monkeypatch.setattr(sao, "_synthesise_sync", fake_synth)
    monkeypatch.setattr(sao, "_write_wav", fake_write_wav)
    monkeypatch.setattr(sao, "tile_with_crossfade", fake_tile)

    p1 = await sao.generate("a prompt", 30.0, cfg)
    p2 = await sao.generate("a prompt", 30.0, cfg)
    assert p1 == p2
    # Second call must hit the cache, not re-synth
    assert synth_calls["n"] == 1


@pytest.mark.asyncio
async def test_generate_different_prompts_different_files(patched_cache, monkeypatch):
    monkeypatch.setattr(sao, "available", lambda: True)
    cfg = _mkcfg(music=cfg_mod.MusicConfig(
        sao_enabled=True, sao_license_acknowledged=True, sao_seed=7,
    ))

    def fake_synth(prompt, duration_s, *, device, steps, cfg_scale, seed):
        import numpy as np
        wave = np.zeros((2, 44100), dtype="float32")
        return {"wave": wave, "sample_rate": 44100}

    def fake_write_wav(wave, sr, dest):
        dest.write_bytes(b"\x00" * 200_000)
        return dest

    def fake_tile(src, dest, *, target_s, crossfade_s=0.5):
        dest.write_bytes(src.read_bytes())
        return dest

    monkeypatch.setattr(sao, "_synthesise_sync", fake_synth)
    monkeypatch.setattr(sao, "_write_wav", fake_write_wav)
    monkeypatch.setattr(sao, "tile_with_crossfade", fake_tile)

    p_a = await sao.generate("prompt A", 30.0, cfg)
    p_b = await sao.generate("prompt B", 30.0, cfg)
    assert p_a != p_b
    assert p_a.exists() and p_b.exists()


@pytest.mark.asyncio
async def test_generate_short_duration_skips_tiling(patched_cache, monkeypatch):
    """When duration ≤ MAX_SINGLE_CLIP_S, we must NOT call tile_with_crossfade."""
    monkeypatch.setattr(sao, "available", lambda: True)
    cfg = _mkcfg(music=cfg_mod.MusicConfig(
        sao_enabled=True, sao_license_acknowledged=True, sao_seed=1,
    ))

    def fake_synth(prompt, duration_s, *, device, steps, cfg_scale, seed):
        import numpy as np
        wave = np.zeros((2, int(44100 * duration_s)), dtype="float32")
        return {"wave": wave, "sample_rate": 44100}

    def fake_write(wave, sr, dest):
        dest.write_bytes(b"\x00" * 200_000)
        return dest

    called = {"tile": False}

    def tile_should_not_be_called(*a, **k):
        called["tile"] = True

    monkeypatch.setattr(sao, "_synthesise_sync", fake_synth)
    monkeypatch.setattr(sao, "_write_wav", fake_write)
    monkeypatch.setattr(sao, "tile_with_crossfade", tile_should_not_be_called)

    out = await sao.generate("p", 8.0, cfg)
    assert out.exists()
    assert called["tile"] is False


@pytest.mark.asyncio
async def test_generate_long_duration_invokes_tiling(patched_cache, monkeypatch):
    monkeypatch.setattr(sao, "available", lambda: True)
    cfg = _mkcfg(music=cfg_mod.MusicConfig(
        sao_enabled=True, sao_license_acknowledged=True, sao_seed=1,
    ))

    def fake_synth(prompt, duration_s, *, device, steps, cfg_scale, seed):
        import numpy as np
        wave = np.zeros((2, int(44100 * duration_s)), dtype="float32")
        return {"wave": wave, "sample_rate": 44100}

    def fake_write(wave, sr, dest):
        dest.write_bytes(b"\x00" * 200_000)
        return dest

    called = {"tile": False, "target_s": 0.0}

    def fake_tile(src, dest, *, target_s, crossfade_s=0.5):
        called["tile"] = True
        called["target_s"] = target_s
        dest.write_bytes(b"\x00" * 300_000)
        return dest

    monkeypatch.setattr(sao, "_synthesise_sync", fake_synth)
    monkeypatch.setattr(sao, "_write_wav", fake_write)
    monkeypatch.setattr(sao, "tile_with_crossfade", fake_tile)

    out = await sao.generate("p", 30.0, cfg)
    assert out.exists()
    assert called["tile"] is True
    assert called["target_s"] == 30.0


@pytest.mark.asyncio
async def test_generate_passes_through_sao_config_knobs(patched_cache, monkeypatch):
    """steps, cfg_scale, seed — all must reach the synthesiser exactly."""
    monkeypatch.setattr(sao, "available", lambda: True)
    cfg = _mkcfg(music=cfg_mod.MusicConfig(
        sao_enabled=True, sao_license_acknowledged=True,
        sao_seed=999, sao_steps=16, sao_cfg_scale=7.5, sao_device="cpu",
    ))

    captured: dict = {}

    def fake_synth(prompt, duration_s, *, device, steps, cfg_scale, seed):
        captured.update(
            prompt=prompt, duration_s=duration_s, device=device,
            steps=steps, cfg_scale=cfg_scale, seed=seed,
        )
        import numpy as np
        wave = np.zeros((2, 44100), dtype="float32")
        return {"wave": wave, "sample_rate": 44100}

    def fake_write(wave, sr, dest):
        dest.write_bytes(b"\x00" * 200_000)
        return dest

    monkeypatch.setattr(sao, "_synthesise_sync", fake_synth)
    monkeypatch.setattr(sao, "_write_wav", fake_write)
    monkeypatch.setattr(sao, "tile_with_crossfade", lambda *a, **k: k["target_s"])

    await sao.generate("marked prompt", 8.0, cfg)
    assert captured["device"] == "cpu"
    assert captured["steps"] == 16
    assert captured["cfg_scale"] == 7.5
    assert captured["seed"] == 999
    assert captured["prompt"] == "marked prompt"


# ─── music.py integration ───
@pytest.mark.asyncio
async def test_stable_audio_fetch_silent_when_disabled(patched_cache):
    cfg = _mkcfg()  # sao_enabled=False
    bed = await music_mod._stable_audio_fetch(cfg)
    assert bed is None


@pytest.mark.asyncio
async def test_stable_audio_fetch_returns_none_when_extra_missing(patched_cache, monkeypatch):
    cfg = _mkcfg(music=cfg_mod.MusicConfig(
        sao_enabled=True, sao_license_acknowledged=True,
    ))
    monkeypatch.setattr(sao, "available", lambda: False)
    bed = await music_mod._stable_audio_fetch(cfg)
    assert bed is None


@pytest.mark.asyncio
async def test_stable_audio_fetch_returns_musicbed_on_success(patched_cache, monkeypatch):
    cfg = _mkcfg(music=cfg_mod.MusicConfig(
        sao_enabled=True, sao_license_acknowledged=True, sao_duration_s=30,
    ))
    fake_wav = patched_cache / "sao_fake.wav"
    fake_wav.write_bytes(b"\x00" * 200_000)

    monkeypatch.setattr(sao, "available", lambda: True)

    async def fake_build(cfg_inner, scene_context=""):
        return "upbeat, 120 BPM, minor key"

    async def fake_gen(prompt, dur, cfg_inner):
        return fake_wav

    monkeypatch.setattr(sao, "build_prompt", fake_build)
    monkeypatch.setattr(sao, "generate", fake_gen)

    bed = await music_mod._stable_audio_fetch(cfg)
    assert bed is not None
    assert bed.path == fake_wav
    assert bed.source == "stable_audio"
    assert "stability" in bed.license.lower()
    assert bed.duration_s == 30.0


@pytest.mark.asyncio
async def test_stable_audio_fetch_swallows_synthesis_errors(patched_cache, monkeypatch):
    cfg = _mkcfg(music=cfg_mod.MusicConfig(
        sao_enabled=True, sao_license_acknowledged=True,
    ))
    monkeypatch.setattr(sao, "available", lambda: True)

    async def fake_build(cfg_inner, scene_context=""):
        return "x"

    async def broken_gen(prompt, dur, cfg_inner):
        raise RuntimeError("model oom")

    monkeypatch.setattr(sao, "build_prompt", fake_build)
    monkeypatch.setattr(sao, "generate", broken_gen)
    bed = await music_mod._stable_audio_fetch(cfg)
    assert bed is None   # graceful fallthrough


@pytest.mark.asyncio
async def test_find_music_routes_stable_audio_source(patched_cache, monkeypatch):
    """find_music with `stable_audio` in the sources chain must call our
    helper. Chain falls through to next source on None."""
    cfg = _mkcfg(music=cfg_mod.MusicConfig(
        sao_enabled=True,
        sao_license_acknowledged=True,
        sources=["stable_audio"],
    ))

    called = {"n": 0}

    async def fake_fetch(cfg_inner, *, scene_context=""):
        called["n"] += 1
        fake = patched_cache / "fake.wav"
        fake.write_bytes(b"\x00" * 200_000)
        return music_mod.MusicBed(
            path=fake, title="t", source="stable_audio",
            license="Stability AI Community Licence", duration_s=30.0,
        )

    monkeypatch.setattr(music_mod, "_stable_audio_fetch", fake_fetch)
    bed = await music_mod.find_music(cfg)
    assert bed is not None
    assert bed.source == "stable_audio"
    assert called["n"] == 1


@pytest.mark.asyncio
async def test_find_music_falls_through_sao_on_none(patched_cache, monkeypatch):
    """If SAO is first in the chain but returns None (not installed),
    find_music must continue to the next source."""
    cfg = _mkcfg(music=cfg_mod.MusicConfig(
        sao_enabled=True,
        sao_license_acknowledged=True,
        sources=["stable_audio", "local"],
    ))

    async def sao_returns_none(cfg_inner, *, scene_context=""):
        return None

    def local_returns_bed(query):
        fake = patched_cache / "local.mp3"
        fake.write_bytes(b"\x00" * 100_000)
        return music_mod.MusicBed(
            path=fake, title="local bed", source="local", license="CC0",
        )

    monkeypatch.setattr(music_mod, "_stable_audio_fetch", sao_returns_none)
    monkeypatch.setattr(music_mod, "_local_pick", local_returns_bed)

    bed = await music_mod.find_music(cfg)
    assert bed is not None
    assert bed.source == "local"
