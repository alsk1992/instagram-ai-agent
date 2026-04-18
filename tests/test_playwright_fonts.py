"""Tests for the font-embedding pipeline feeding Playwright."""
from __future__ import annotations

from pathlib import Path

import pytest

from instagram_ai_agent.content.generators import playwright_render
from instagram_ai_agent.core import config as cfg_mod


@pytest.fixture()
def tmp_fonts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fonts = tmp_path / "fonts"
    fonts.mkdir()
    monkeypatch.setattr(cfg_mod, "FONTS_DIR", fonts)
    monkeypatch.setattr(playwright_render, "FONTS_DIR", fonts)
    yield fonts


def _fake_ttf(path: Path) -> None:
    # Not a valid TTF, but enough to be read, base64'd, and referenced.
    path.write_bytes(b"\x00\x01FAKE_TTF_BYTES")


def test_finds_font_by_family_name(tmp_fonts: Path):
    _fake_ttf(tmp_fonts / "ArchivoBlack-Regular.ttf")
    found = playwright_render._find_font_file("Archivo Black")
    assert found is not None
    assert found.name == "ArchivoBlack-Regular.ttf"


def test_font_face_css_emits_blocks(tmp_fonts: Path):
    _fake_ttf(tmp_fonts / "ArchivoBlack.ttf")
    _fake_ttf(tmp_fonts / "Inter.ttf")
    css = playwright_render.font_face_css("Archivo Black", "Inter")
    assert "@font-face" in css
    assert css.count("@font-face") == 2
    assert "font-family: 'Archivo Black'" in css
    assert "font-family: 'Inter'" in css
    assert "base64," in css
    assert "format('truetype')" in css


def test_font_face_css_dedups(tmp_fonts: Path):
    _fake_ttf(tmp_fonts / "Inter.ttf")
    css = playwright_render.font_face_css("Inter", "Inter")
    assert css.count("@font-face") == 1


def test_font_face_css_missing_fonts_no_crash(tmp_fonts: Path):
    css = playwright_render.font_face_css("Nonexistent", "AlsoMissing")
    assert css == ""


def test_base_css_includes_font_faces(tmp_fonts: Path):
    _fake_ttf(tmp_fonts / "Inter.ttf")
    _fake_ttf(tmp_fonts / "ArchivoBlack.ttf")
    css = playwright_render.base_css(
        width=1080, height=1350, bg="#000", fg="#fff",
        body_font="Inter", heading_font="Archivo Black",
    )
    assert "@font-face" in css
    assert "body {" in css
    assert "1080px" in css
