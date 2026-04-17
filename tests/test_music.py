"""Unit tests for music source chain + audio mixer filter-graph shape."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from src.core import config as cfg_mod


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


# ─── Config ───
def test_music_defaults_safe():
    cfg = _mkcfg()
    m = cfg.music
    assert m.enabled is True
    assert m.sources[0] == "local"
    assert 0 <= m.duck_gain <= 1
    assert m.vo_gain > 0


def test_music_disabled_is_respected():
    cfg = _mkcfg(music=cfg_mod.MusicConfig(enabled=False))
    assert cfg.music.enabled is False


def test_music_query_template_substitutes_niche():
    cfg = _mkcfg(music=cfg_mod.MusicConfig(query_template="{niche} vibes"))
    assert "{niche}" in cfg.music.query_template
    filled = cfg.music.query_template.format(niche=cfg.niche)
    assert "home calisthenics" in filled


# ─── Tokeniser + local pick ───
def test_tokenise_strips_short_and_punct():
    from src.plugins.music import _tokenise
    assert _tokenise("Dad Strength Vibes!") == ["dad", "strength", "vibes"]
    assert _tokenise("") == []
    assert _tokenise("a 12 big") == ["big"]  # 2-char terms dropped


def test_local_pick_returns_none_for_empty_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from src.plugins import music
    monkeypatch.setattr(music, "MUSIC_DIR", tmp_path / "empty")
    assert music._local_pick("anything") is None


def test_local_pick_scores_by_filename_tokens(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from src.plugins import music
    music_root = tmp_path / "music"
    music_root.mkdir()
    # Three tracks
    (music_root / "upbeat_motivation_track.mp3").write_bytes(b"x" * 20_000)
    (music_root / "ambient_lofi.wav").write_bytes(b"x" * 20_000)
    (music_root / "random_whatever.mp3").write_bytes(b"x" * 20_000)
    monkeypatch.setattr(music, "MUSIC_DIR", music_root)
    bed = music._local_pick("upbeat motivation for dads")
    assert bed is not None
    assert "upbeat" in bed.path.name
    assert bed.source == "local"


def test_local_pick_parent_folder_scoring(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from src.plugins import music
    music_root = tmp_path / "music"
    (music_root / "lofi").mkdir(parents=True)
    (music_root / "upbeat").mkdir()
    # Filename doesn't mention the genre; folder name is the signal
    (music_root / "lofi" / "track_a.mp3").write_bytes(b"x" * 20_000)
    (music_root / "upbeat" / "track_b.mp3").write_bytes(b"x" * 20_000)
    monkeypatch.setattr(music, "MUSIC_DIR", music_root)
    bed = music._local_pick("some lofi vibes")
    assert bed is not None
    # Parent-folder match gives lofi priority
    assert bed.path.parent.name == "lofi"


# ─── Source chain ordering ───
def test_find_music_disabled_short_circuits(monkeypatch: pytest.MonkeyPatch):
    from src.plugins import music
    cfg = _mkcfg(music=cfg_mod.MusicConfig(enabled=False))
    out = asyncio.run(music.find_music(cfg))
    assert out is None


def test_find_music_uses_local_first(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from src.plugins import music
    music_root = tmp_path / "music"
    music_root.mkdir()
    (music_root / "calisthenics_upbeat.mp3").write_bytes(b"x" * 20_000)
    monkeypatch.setattr(music, "MUSIC_DIR", music_root)
    cfg = _mkcfg()

    # Force pixabay/freesound sources to be unreachable (no API key)
    monkeypatch.delenv("PIXABAY_API_KEY", raising=False)
    monkeypatch.delenv("FREESOUND_API_KEY", raising=False)

    bed = asyncio.run(music.find_music(cfg))
    assert bed is not None
    assert bed.source == "local"


def test_find_music_returns_none_when_all_fail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from src.plugins import music
    monkeypatch.setattr(music, "MUSIC_DIR", tmp_path / "empty")
    monkeypatch.delenv("PIXABAY_API_KEY", raising=False)
    monkeypatch.delenv("FREESOUND_API_KEY", raising=False)
    cfg = _mkcfg()
    assert asyncio.run(music.find_music(cfg)) is None


# ─── Audio mixer filter-graph ───
def test_mix_filter_graph_shape(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Verify the ffmpeg command we build is syntactically sensible
    and references every expected filter step."""
    from src.plugins import audio_mix

    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        # Simulate ffmpeg success
        from subprocess import CompletedProcess
        return CompletedProcess(cmd, 0, b"", b"")

    import subprocess as _sp
    monkeypatch.setattr(_sp, "run", fake_run)
    monkeypatch.setattr(audio_mix.subprocess, "run", fake_run)

    vo = tmp_path / "vo.mp3"
    music = tmp_path / "music.mp3"
    out = tmp_path / "out.m4a"
    vo.write_bytes(b"x" * 1000)
    music.write_bytes(b"x" * 1000)

    cfg = _mkcfg()
    audio_mix.mix_vo_and_music(vo, music, out, duration_s=25.0, music_cfg=cfg.music)
    cmd = captured["cmd"]
    assert "ffmpeg" in cmd[0]
    # Inputs
    assert str(vo) in cmd
    assert str(music) in cmd
    # Filter complex joined
    fc_idx = cmd.index("-filter_complex")
    fc = cmd[fc_idx + 1]
    for piece in ("aloop=loop=-1", "afade=t=in", "afade=t=out", "volume=", "amix="):
        assert piece in fc, f"missing {piece!r} in filter graph"
    # Duration + aac out
    assert "-t" in cmd
    assert "aac" in cmd


def test_mix_respects_fade_out_window():
    from src.plugins import audio_mix
    cfg = _mkcfg(music=cfg_mod.MusicConfig(fade_out_s=1.5))
    fade_out_start = max(0.0, 10.0 - cfg.music.fade_out_s)
    assert fade_out_start == pytest.approx(8.5)


# ─── Cache determinism + re-download short-circuit ───
def test_cache_path_is_deterministic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from src.plugins import music
    monkeypatch.setattr(music, "MUSIC_CACHE_DIR", tmp_path)
    a = music._cache_path("https://example.com/song.mp3", ".mp3")
    b = music._cache_path("https://example.com/song.mp3", ".mp3")
    c = music._cache_path("https://example.com/other.mp3", ".mp3")
    assert a == b
    assert a != c


def test_cache_hit_short_circuits_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from src.plugins import music
    monkeypatch.setattr(music, "MUSIC_CACHE_DIR", tmp_path)
    url = "https://example.com/pre-cached.mp3"
    dest = music._cache_path(url)
    dest.write_bytes(b"x" * 20_000)   # already-cached, > 10kB

    # httpx client that would explode if actually used
    class Boom:
        def stream(self, *_a, **_k): raise RuntimeError("must not hit network")

    async def go():
        return await music._download(Boom(), url, dest)

    result = asyncio.run(go())
    assert result == dest
    assert result.read_bytes().startswith(b"x")


# ─── Network-shape: Pixabay + Freesound request contract ───
def test_pixabay_sends_key_and_query(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from src.plugins import music as music_mod
    calls: list[dict] = []

    class FakeResp:
        def __init__(self, status, payload):
            self.status_code = status
            self._payload = payload
        def raise_for_status(self):  # not used in _pixabay_fetch
            pass
        def json(self):
            return self._payload

    class FakeClient:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return None
        async def get(self, url, params=None, **kw):
            calls.append({"url": url, "params": params})
            return FakeResp(404, {})  # force Pixabay fetcher to return None

    monkeypatch.setenv("PIXABAY_API_KEY", "dummy-key")
    monkeypatch.setattr(music_mod.httpx, "AsyncClient", lambda *a, **k: FakeClient())

    result = asyncio.run(music_mod._pixabay_fetch("dads calisthenics upbeat"))
    assert result is None
    # Should have tried both candidate endpoints with key + q
    assert len(calls) >= 1
    params = calls[0]["params"]
    assert params["key"] == "dummy-key"
    assert params["q"] == "dads calisthenics upbeat"


def test_freesound_filter_requires_cc0_license(monkeypatch: pytest.MonkeyPatch):
    from src.plugins import music as music_mod
    calls: list[dict] = []

    class FakeResp:
        def raise_for_status(self): pass
        def json(self): return {"results": []}

    class FakeClient:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return None
        async def get(self, url, params=None, **kw):
            calls.append({"url": url, "params": params})
            return FakeResp()

    monkeypatch.setenv("FREESOUND_API_KEY", "fs-dummy")
    monkeypatch.setattr(music_mod.httpx, "AsyncClient", lambda *a, **k: FakeClient())
    result = asyncio.run(music_mod._freesound_fetch("dad strength"))
    assert result is None
    assert len(calls) == 1
    # Explicit CC0 enforcement in the filter string is the whole point
    assert 'license:"Creative Commons 0"' in calls[0]["params"]["filter"]
    assert calls[0]["params"]["token"] == "fs-dummy"


# ─── License detection (the MEDIUM fix) ───
def test_detect_license_folder_cc0(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from src.plugins import music
    root = tmp_path / "music"
    (root / "cc0").mkdir(parents=True)
    track = root / "cc0" / "a.mp3"
    track.write_bytes(b"x" * 20_000)
    monkeypatch.setattr(music, "MUSIC_DIR", root)
    assert music._detect_local_license(track) == "CC0"


def test_detect_license_sidecar_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from src.plugins import music
    root = tmp_path / "music"
    (root / "cc0").mkdir(parents=True)
    track = root / "cc0" / "a.mp3"
    track.write_bytes(b"x" * 20_000)
    (root / "cc0" / "a.mp3.license").write_text("CC-BY-4.0 — AuthorName\nnote: credit in caption")
    monkeypatch.setattr(music, "MUSIC_DIR", root)
    # Sidecar beats folder hint
    lic = music._detect_local_license(track)
    assert lic.startswith("CC-BY")


def test_detect_license_unknown_is_labelled_honestly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from src.plugins import music
    root = tmp_path / "music"
    (root / "misc").mkdir(parents=True)
    track = root / "misc" / "a.mp3"
    track.write_bytes(b"x" * 20_000)
    monkeypatch.setattr(music, "MUSIC_DIR", root)
    assert music._detect_local_license(track) == "user-declared"


def test_local_pick_stamps_detected_license(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from src.plugins import music
    root = tmp_path / "music"
    (root / "cc0").mkdir(parents=True)
    (root / "cc0" / "dad_upbeat.mp3").write_bytes(b"x" * 20_000)
    monkeypatch.setattr(music, "MUSIC_DIR", root)
    bed = music._local_pick("dad upbeat")
    assert bed is not None
    assert bed.license == "CC0"
