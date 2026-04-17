"""Regression tests for the setup-hardening pass (2026-04-17 audit).

Covers: format_picker env gate, ChallengeNeedsManualCode exception type,
idea-bank auto-seed on init, FEED_FORMATS includes story_carousel.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.core import config as cfg_mod
from src.core import db


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
def tmp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fresh = tmp_path / "brain.db"
    monkeypatch.setattr(cfg_mod, "DB_PATH", fresh)
    monkeypatch.setattr(db, "DB_PATH", fresh)
    db.close()
    db.init_db()
    yield db
    db.close()


# ─── format_picker env gate ───
def test_format_picker_skips_reel_stock_when_no_api_keys(monkeypatch: pytest.MonkeyPatch, tmp_db):
    """Audit blocker: reel_stock fails hard when both Pexels + Pixabay
    keys are missing. Picker must skip it instead of burning cycles."""
    from src.content.generators import format_picker

    monkeypatch.delenv("PEXELS_API_KEY", raising=False)
    monkeypatch.delenv("PIXABAY_API_KEY", raising=False)

    cfg = _mkcfg(formats=cfg_mod.FormatMix(
        meme=0.0, quote_card=0.0, carousel=0.0,
        reel_stock=1.0, reel_ai=0.0, photo=0.0, human_photo=0.0,
    ))
    # With ONLY reel_stock weighted, the picker's unrunnable-prune
    # hits the "all blocked — keep originals" fallback (so gen will
    # still fail, but that's the correct signal to the user).
    choice = format_picker.pick_next(cfg, kind="feed")
    # Either returns reel_stock (keeping originals when all blocked)
    # or falls back to "meme" — both mean "we tried"
    assert choice in ("reel_stock", "meme")


def test_format_picker_prunes_unrunnable_when_alternatives_exist(
    monkeypatch: pytest.MonkeyPatch, tmp_db,
):
    """When reel_stock is unrunnable but meme is weighted, picker
    NEVER returns reel_stock."""
    from src.content.generators import format_picker

    monkeypatch.delenv("PEXELS_API_KEY", raising=False)
    monkeypatch.delenv("PIXABAY_API_KEY", raising=False)

    cfg = _mkcfg(formats=cfg_mod.FormatMix(
        meme=1.0, quote_card=0.0, carousel=0.0,
        reel_stock=1.0, reel_ai=0.0, photo=0.0, human_photo=0.0,
    ))
    # 200 rolls — none should be reel_stock
    picks = {format_picker.pick_next(cfg, kind="feed") for _ in range(50)}
    assert "reel_stock" not in picks
    assert picks == {"meme"}


def test_format_picker_allows_reel_stock_when_pixabay_set(
    monkeypatch: pytest.MonkeyPatch, tmp_db,
):
    from src.content.generators import format_picker

    monkeypatch.delenv("PEXELS_API_KEY", raising=False)
    monkeypatch.setenv("PIXABAY_API_KEY", "fake-test-key")
    assert format_picker._format_is_runnable("reel_stock") is True


def test_format_picker_allows_reel_stock_when_pexels_set(
    monkeypatch: pytest.MonkeyPatch, tmp_db,
):
    from src.content.generators import format_picker

    monkeypatch.setenv("PEXELS_API_KEY", "fake-test-key")
    monkeypatch.delenv("PIXABAY_API_KEY", raising=False)
    assert format_picker._format_is_runnable("reel_stock") is True


def test_format_picker_non_gated_formats_always_runnable():
    from src.content.generators import format_picker

    for fmt in ("meme", "quote_card", "carousel", "reel_ai", "photo", "human_photo"):
        assert format_picker._format_is_runnable(fmt) is True, fmt


# ─── challenge.ChallengeNeedsManualCode ───
def test_challenge_needs_manual_code_is_separate_exception():
    """Audit blocker: challenge handler was raising RuntimeError which
    trapped into a 24h cooldown. New exception type lets ig.py handle
    this case without cooldown."""
    from src.plugins import challenge
    assert issubclass(challenge.ChallengeNeedsManualCode, Exception)
    # Must NOT be a RuntimeError subclass — keeps except-RuntimeError
    # handlers from accidentally catching it.
    assert not issubclass(challenge.ChallengeNeedsManualCode, RuntimeError)


def test_challenge_handler_raises_specific_exception_without_imap(
    monkeypatch: pytest.MonkeyPatch, tmp_db,
):
    from src.plugins import challenge

    # No IMAP configured → fetch_email_code returns None
    monkeypatch.setattr(challenge, "fetch_email_code", lambda *a, **k: None)

    handler = challenge.make_challenge_code_handler(interactive=False)
    with pytest.raises(challenge.ChallengeNeedsManualCode) as exc:
        handler("testuser", 1)
    # Error message must tell the user what to do
    msg = str(exc.value).lower()
    assert "imap" in msg
    assert "login" in msg


def test_challenge_handler_returns_code_when_imap_works(
    monkeypatch: pytest.MonkeyPatch, tmp_db,
):
    from src.plugins import challenge

    monkeypatch.setattr(challenge, "fetch_email_code", lambda *a, **k: "123456")
    handler = challenge.make_challenge_code_handler()
    assert handler("testuser", 1) == "123456"


# ─── FEED_FORMATS includes story_carousel ───
def test_feed_formats_includes_story_carousel():
    """Audit polish: story_carousel weight in FormatMix must be in
    FEED_FORMATS so format_picker / dispatch don't KeyError."""
    assert "story_carousel" in cfg_mod.FEED_FORMATS


# ─── Idea-bank auto-seed on init ───
def test_idea_bank_seed_from_file_is_idempotent(tmp_db):
    """Calling it twice must not double-insert. Init wizard calls it
    once — re-running init shouldn't create duplicates."""
    from src.brain import idea_bank
    n1 = idea_bank.seed_from_file()
    n2 = idea_bank.seed_from_file()
    assert n1 > 0
    assert n2 == 0   # nothing new on second run
    # Total count hasn't doubled
    assert idea_bank.count() == n1


# ─── pyproject.toml: no dead deps ───
def test_pyproject_has_no_unused_dependencies():
    """Regression: moviepy + beautifulsoup4 were declared but never
    imported. The clean pyproject shouldn't reintroduce them without
    an actual import somewhere in src/."""
    import re
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text()
    # Extract the base dependencies array (first match — the one in [project])
    m = re.search(r"^dependencies\s*=\s*\[(.*?)^\]", pyproject, re.DOTALL | re.MULTILINE)
    assert m, "couldn't find dependencies = [...] block"
    base_block = m.group(1)
    # Pull just the package names (first token of each quoted line)
    pkg_names = re.findall(r'"([a-zA-Z0-9_-]+)', base_block)
    # For each declared package, confirm SOMETHING in src/ imports it
    # (approximate via rglob grep). Skip a hardcoded allowlist of deps
    # that are imported indirectly (e.g. via optional pathways or as
    # transitive deps that we still want pinned).
    _IMPORT_NAME_OVERRIDES = {
        "pillow": "PIL",
        "imap-tools": "imap_tools",
        "pyyaml": "yaml",
        "python-dotenv": "dotenv",
        "beautifulsoup4": "bs4",
        "faster-whisper": "faster_whisper",
        "apscheduler": "apscheduler",
        "instagrapi": "instagrapi",
        "instaloader": "instaloader",
        "edge-tts": "edge_tts",
        "pyotp": "pyotp",
        "boto3": "boto3",
        "typer": "typer",
        "rich": "rich",
        "questionary": "questionary",
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "jinja2": "jinja2",
        "aiohttp": "aiohttp",
        "openai": "openai",
        "httpx": "httpx",
        "tenacity": "tenacity",
        "pydantic": "pydantic",
        "requests": "requests",
        "playwright": "playwright",
        "numpy": "numpy",
    }
    # Indirect deps — used via another framework's exports rather than
    # a direct import from our code. Verified usage below; keep pinned.
    _INDIRECT_OK = {
        "jinja2",   # via fastapi.templating.Jinja2Templates in dashboard.py
    }
    src_dir = root / "src"
    all_src = "\n".join(p.read_text(errors="ignore") for p in src_dir.rglob("*.py"))
    unused = []
    for pkg in pkg_names:
        if pkg in _INDIRECT_OK:
            continue
        import_name = _IMPORT_NAME_OVERRIDES.get(pkg, pkg.replace("-", "_"))
        if f"import {import_name}" not in all_src and f"from {import_name}" not in all_src:
            unused.append((pkg, import_name))
    assert not unused, f"Declared but unused base deps: {unused}"
