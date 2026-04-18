"""Hook overlay — wrapping, font resolution, alpha expr, no-op guards."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from instagram_ai_agent.core import config as cfg_mod
from instagram_ai_agent.plugins import video_overlay


def _mkcfg(**kwargs):
    base = dict(
        niche="home calisthenics",
        sub_topics=["pullups"],
        target_audience="dads 35+",
        commercial=True,
        voice=cfg_mod.Voice(tone=["direct"], forbidden=[], persona="ex-office worker rebuilding fitness."),
        aesthetic=cfg_mod.Aesthetic(palette=["#0a0a0a", "#f5f5f0", "#c9a961"]),
        hashtags=cfg_mod.HashtagPools(core=["calisthenics", "homeworkout", "dadfit"]),
    )
    base.update(kwargs)
    return cfg_mod.NicheConfig(**base)


# ─── Config ───
def test_hook_defaults_sane():
    cfg = _mkcfg()
    ho = cfg.hook_overlay
    assert ho.enabled is True
    assert 0 < ho.duration_s <= 5
    assert 0 < ho.font_scale <= 0.12
    assert 0 < ho.y_ratio < 0.5
    assert ho.box is True


# ─── Text wrapping ───
def test_wrap_hook_trims_to_max_words():
    lines = video_overlay.wrap_hook(
        "one two three four five six seven eight nine ten",
        max_words=4, max_chars_per_line=20, max_lines=3,
    )
    # Only the first 4 words survive the max_words cap.
    # Since input had 10 words, the ellipsis is appended to the last line.
    joined = " ".join(lines)
    tokens = joined.replace("…", "").split()
    assert tokens == ["one", "two", "three", "four"]
    assert joined.endswith("…")


def test_wrap_hook_respects_line_width():
    lines = video_overlay.wrap_hook(
        "home calisthenics pull-ups made easy today",
        max_words=8, max_chars_per_line=18, max_lines=3,
    )
    for line in lines:
        assert len(line) <= 18, line
    assert len(lines) <= 3


def test_wrap_hook_empty_returns_empty():
    assert video_overlay.wrap_hook("", max_words=8, max_chars_per_line=22, max_lines=3) == []
    assert video_overlay.wrap_hook("   ", max_words=8, max_chars_per_line=22, max_lines=3) == []


def test_wrap_hook_single_long_word_fits_one_line():
    # A single word that exceeds max_chars — must still occupy a line
    lines = video_overlay.wrap_hook(
        "pullupprogressionmasterclass",
        max_words=3, max_chars_per_line=10, max_lines=2,
    )
    assert len(lines) == 1


# ─── Colour helper ───
def test_hex_to_ff():
    assert video_overlay._hex_to_ff("#ff0000") == "0xFF0000@1.00"
    assert video_overlay._hex_to_ff("00ff00", alpha=0.5) == "0x00FF00@0.50"
    # Garbage hex falls back to white
    assert video_overlay._hex_to_ff("not-a-hex") == "0xFFFFFF@1.00"


# ─── Font resolution ───
def test_resolve_font_finds_match(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fonts = tmp_path / "fonts"
    fonts.mkdir()
    (fonts / "ArchivoBlack-Regular.ttf").write_bytes(b"\x00\x01fake")
    monkeypatch.setattr(video_overlay, "FONTS_DIR", fonts)
    found = video_overlay._resolve_font_path("Archivo Black")
    assert found is not None
    assert found.name == "ArchivoBlack-Regular.ttf"


def test_resolve_font_returns_none_when_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(video_overlay, "FONTS_DIR", tmp_path / "does_not_exist")
    assert video_overlay._resolve_font_path("AnyFont") is None


# ─── Escape ───
def test_escape_drawtext_param():
    assert video_overlay._escape_drawtext_param("/tmp/a:b.txt") == r"/tmp/a\:b.txt"
    assert video_overlay._escape_drawtext_param("it's") == r"it\'s"
    assert video_overlay._escape_drawtext_param("a\\b") == r"a\\b"


# ─── add_hook_overlay no-op paths ───
def test_add_hook_overlay_disabled_returns_input(tmp_path: Path):
    cfg = _mkcfg(hook_overlay=cfg_mod.HookOverlay(enabled=False))
    vid = tmp_path / "v.mp4"
    vid.write_bytes(b"fake mp4")
    out = video_overlay.add_hook_overlay(vid, "Pull-ups", cfg)
    assert out == vid


def test_add_hook_overlay_empty_text_returns_input(tmp_path: Path):
    cfg = _mkcfg()
    vid = tmp_path / "v.mp4"
    vid.write_bytes(b"fake mp4")
    assert video_overlay.add_hook_overlay(vid, "   ", cfg) == vid
    assert video_overlay.add_hook_overlay(vid, "", cfg) == vid


def test_add_hook_overlay_no_font_returns_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # No fonts dir + no system fallback -> must degrade gracefully
    monkeypatch.setattr(video_overlay, "FONTS_DIR", tmp_path / "missing")
    monkeypatch.setattr(video_overlay, "_fallback_system_font", lambda: None)
    cfg = _mkcfg()
    vid = tmp_path / "v.mp4"
    vid.write_bytes(b"fake mp4")
    out = video_overlay.add_hook_overlay(vid, "Pull-ups change your life", cfg)
    assert out == vid


# ─── ffmpeg command construction (mocked subprocess) ───
def test_add_hook_overlay_builds_expected_drawtext(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Verify the drawtext filter we emit has every expected piece."""
    # Set up a font
    fonts = tmp_path / "fonts"
    fonts.mkdir()
    font = fonts / "ArchivoBlack.ttf"
    font.write_bytes(b"\x00\x01fake")
    monkeypatch.setattr(video_overlay, "FONTS_DIR", fonts)
    # Skip system probe path
    monkeypatch.setattr(video_overlay, "_fallback_system_font", lambda: None)
    # Probe stub
    monkeypatch.setattr(video_overlay, "_probe_video_size", lambda v: (1080, 1920))

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        out_path = Path(cmd[-1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"processed mp4")
        from subprocess import CompletedProcess
        return CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr(video_overlay.subprocess, "run", fake_run)

    cfg = _mkcfg()
    vid = tmp_path / "in.mp4"
    vid.write_bytes(b"fake mp4")
    out = video_overlay.add_hook_overlay(vid, "Stop scrolling, here's the truth about pull-ups.", cfg)

    assert out != vid
    assert out.exists()
    cmd = captured["cmd"]
    vf_idx = cmd.index("-vf")
    vf = cmd[vf_idx + 1]
    assert vf.startswith("drawtext=")
    for piece in (
        "textfile=",
        "fontfile=",
        "fontsize=",
        "fontcolor=0x",
        "x=(w-text_w)/2",
        "y=",
        "enable='between(t,0,",
        "alpha='if(lt(t,",
        "box=1",
        "boxcolor=0x",
        "borderw=",
        "bordercolor=0x",
    ):
        assert piece in vf, f"missing {piece!r} in drawtext filter"


def test_add_hook_overlay_handles_ffmpeg_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """If ffmpeg errors out, the input video must be returned — never blocks posting."""
    fonts = tmp_path / "fonts"
    fonts.mkdir()
    (fonts / "ArchivoBlack.ttf").write_bytes(b"\x00\x01fake")
    monkeypatch.setattr(video_overlay, "FONTS_DIR", fonts)
    monkeypatch.setattr(video_overlay, "_probe_video_size", lambda v: (1080, 1920))

    def fake_run(cmd, **kwargs):
        import subprocess as _sp
        raise _sp.CalledProcessError(1, cmd, stderr=b"boom")

    monkeypatch.setattr(video_overlay.subprocess, "run", fake_run)
    cfg = _mkcfg()
    vid = tmp_path / "v.mp4"
    vid.write_bytes(b"fake mp4")
    out = video_overlay.add_hook_overlay(vid, "Pull-ups change your life", cfg)
    assert out == vid  # graceful no-op


# ─── Hook text picker ───
def test_pick_hook_text_prefers_flagged_scene():
    scenes = [
        {"line": "Welcome to the show."},
        {"line": "Here's the real truth.", "hook": True},
        {"line": "Third scene."},
    ]
    assert video_overlay.pick_hook_text(scenes) == "Here's the real truth."


def test_pick_hook_text_falls_back_to_first_scene():
    scenes = [
        {"line": "First line only."},
        {"line": "Second."},
    ]
    assert video_overlay.pick_hook_text(scenes) == "First line only."


def test_pick_hook_text_handles_empty():
    assert video_overlay.pick_hook_text([]) == ""
    assert video_overlay.pick_hook_text([{"line": ""}]) == ""


# ─── Alpha envelope sanity ───
def test_alpha_envelope_ramp_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The alpha expression should go from 0 → 1 → 0 across the duration."""
    # Capture the generated filter text
    fonts = tmp_path / "fonts"; fonts.mkdir()
    (fonts / "ArchivoBlack.ttf").write_bytes(b"\x00\x01")
    monkeypatch.setattr(video_overlay, "FONTS_DIR", fonts)
    monkeypatch.setattr(video_overlay, "_probe_video_size", lambda v: (1080, 1920))

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        Path(cmd[-1]).parent.mkdir(parents=True, exist_ok=True)
        Path(cmd[-1]).write_bytes(b"x")
        from subprocess import CompletedProcess
        return CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr(video_overlay.subprocess, "run", fake_run)

    cfg = _mkcfg(hook_overlay=cfg_mod.HookOverlay(duration_s=2.0, fade_in_s=0.25, fade_out_s=0.3))
    vid = tmp_path / "v.mp4"; vid.write_bytes(b"x")
    video_overlay.add_hook_overlay(vid, "Hook line", cfg)
    vf = captured["cmd"][captured["cmd"].index("-vf") + 1]

    # Fade-in uses the configured fade_in_s
    m = re.search(r"if\(lt\(t,([0-9.]+)\),t/([0-9.]+),", vf)
    assert m is not None
    assert abs(float(m.group(1)) - 0.25) < 1e-3

    # Fade-out window ends at duration_s
    assert "lt(t,2.000)" in vf


# ─── Audit follow-ups ───
def test_missing_ffmpeg_binary_is_graceful(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Ffmpeg absent from PATH must not crash the pipeline."""
    fonts = tmp_path / "fonts"; fonts.mkdir()
    (fonts / "ArchivoBlack.ttf").write_bytes(b"\x00\x01")
    monkeypatch.setattr(video_overlay, "FONTS_DIR", fonts)
    monkeypatch.setattr(video_overlay, "_probe_video_size", lambda v: (1080, 1920))

    def fake_run(*_a, **_k):
        raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr(video_overlay.subprocess, "run", fake_run)
    cfg = _mkcfg()
    vid = tmp_path / "v.mp4"; vid.write_bytes(b"x")
    out = video_overlay.add_hook_overlay(vid, "Hook", cfg)
    assert out == vid


def test_hook_overlay_yaml_roundtrip():
    import yaml
    cfg = _mkcfg(hook_overlay=cfg_mod.HookOverlay(duration_s=1.6, y_ratio=0.1, max_words=6))
    dumped = cfg.model_dump(mode="json")
    serialised = yaml.safe_dump(dumped, sort_keys=False)
    loaded = cfg_mod.NicheConfig.model_validate(yaml.safe_load(serialised))
    assert loaded.hook_overlay.duration_s == 1.6
    assert loaded.hook_overlay.y_ratio == 0.1
    assert loaded.hook_overlay.max_words == 6


def test_wrap_hook_appends_ellipsis_on_word_cap():
    lines = video_overlay.wrap_hook(
        "one two three four five six seven eight nine ten",
        max_words=4, max_chars_per_line=40, max_lines=3,
    )
    assert lines
    assert lines[-1].endswith("…")


def test_wrap_hook_appends_ellipsis_on_line_cap():
    lines = video_overlay.wrap_hook(
        "alpha bravo charlie delta echo foxtrot golf hotel india",
        max_words=20, max_chars_per_line=8, max_lines=2,
    )
    assert lines[-1].endswith("…")
    for line in lines[:-1]:
        assert len(line) <= 8


def test_wrap_hook_no_ellipsis_when_all_words_fit():
    lines = video_overlay.wrap_hook(
        "short hook",
        max_words=8, max_chars_per_line=22, max_lines=3,
    )
    assert not any(line.endswith("…") for line in lines)
