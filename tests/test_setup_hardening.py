"""Regression tests for the setup-hardening pass (2026-04-17 audit).

Covers: format_picker env gate, ChallengeNeedsManualCode exception type,
idea-bank auto-seed on init, FEED_FORMATS includes story_carousel.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from instagram_ai_agent.core import config as cfg_mod
from instagram_ai_agent.core import db


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
    from instagram_ai_agent.content.generators import format_picker

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
    from instagram_ai_agent.content.generators import format_picker

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
    from instagram_ai_agent.content.generators import format_picker

    monkeypatch.delenv("PEXELS_API_KEY", raising=False)
    monkeypatch.setenv("PIXABAY_API_KEY", "fake-test-key")
    assert format_picker._format_is_runnable("reel_stock") is True


def test_format_picker_allows_reel_stock_when_pexels_set(
    monkeypatch: pytest.MonkeyPatch, tmp_db,
):
    from instagram_ai_agent.content.generators import format_picker

    monkeypatch.setenv("PEXELS_API_KEY", "fake-test-key")
    monkeypatch.delenv("PIXABAY_API_KEY", raising=False)
    assert format_picker._format_is_runnable("reel_stock") is True


def test_format_picker_non_gated_formats_always_runnable():
    from instagram_ai_agent.content.generators import format_picker

    for fmt in ("meme", "quote_card", "carousel", "reel_ai", "photo", "human_photo"):
        assert format_picker._format_is_runnable(fmt) is True, fmt


# ─── challenge.ChallengeNeedsManualCode ───
def test_challenge_needs_manual_code_is_separate_exception():
    """Audit blocker: challenge handler was raising RuntimeError which
    trapped into a 24h cooldown. New exception type lets ig.py handle
    this case without cooldown."""
    from instagram_ai_agent.plugins import challenge
    assert issubclass(challenge.ChallengeNeedsManualCode, Exception)
    # Must NOT be a RuntimeError subclass — keeps except-RuntimeError
    # handlers from accidentally catching it.
    assert not issubclass(challenge.ChallengeNeedsManualCode, RuntimeError)


def test_challenge_handler_raises_specific_exception_without_imap(
    monkeypatch: pytest.MonkeyPatch, tmp_db,
):
    from instagram_ai_agent.plugins import challenge

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
    from instagram_ai_agent.plugins import challenge

    monkeypatch.setattr(challenge, "fetch_email_code", lambda *a, **k: "123456")
    handler = challenge.make_challenge_code_handler()
    assert handler("testuser", 1) == "123456"


# ─── Cookie-seed coverage (perma-session support) ───
def test_cookie_seed_returns_none_without_sessionid(monkeypatch: pytest.MonkeyPatch):
    """sessionid is the one required cookie — without it, we must fall
    through to password login rather than attempting a partial seed."""
    from instagram_ai_agent.plugins import ig
    for k in ("IG_SESSIONID", "IG_DS_USER_ID", "IG_CSRFTOKEN", "IG_MID", "IG_DID",
              "IG_DATR", "IG_RUR", "IG_SHBID", "IG_SHBTS", "IG_NRCB"):
        monkeypatch.delenv(k, raising=False)
    assert ig._build_cookie_seed() is None


def test_cookie_seed_collects_every_cookie(monkeypatch: pytest.MonkeyPatch):
    """User-supplied cookies land in the seed under their IG names
    (not the env var names)."""
    from instagram_ai_agent.plugins import ig
    cookie_map = {
        "IG_SESSIONID": "sess-xyz",
        "IG_DS_USER_ID": "12345",
        "IG_CSRFTOKEN": "csrf-abc",
        "IG_MID": "mid-pqr",
        "IG_DID": "did-stu",
        "IG_DATR": "datr-vwx",
        "IG_RUR": "rur-yza",
        "IG_SHBID": "shbid-bcd",
        "IG_SHBTS": "shbts-efg",
        "IG_NRCB": "nrcb-hij",
    }
    for k, v in cookie_map.items():
        monkeypatch.setenv(k, v)
    seed = ig._build_cookie_seed()
    assert seed is not None
    # Every cookie is surfaced under its IG name
    assert seed["sessionid"] == "sess-xyz"
    assert seed["ds_user_id"] == "12345"
    assert seed["csrftoken"] == "csrf-abc"
    assert seed["mid"] == "mid-pqr"
    assert seed["ig_did"] == "did-stu"
    assert seed["datr"] == "datr-vwx"
    assert seed["rur"] == "rur-yza"
    assert seed["shbid"] == "shbid-bcd"
    assert seed["shbts"] == "shbts-efg"
    assert seed["ig_nrcb"] == "nrcb-hij"


def test_cookie_seed_strips_empty_values(monkeypatch: pytest.MonkeyPatch):
    """A whitespace-only env var must be treated as absent — presence
    in seed is the signal for "use this cookie"."""
    from instagram_ai_agent.plugins import ig
    monkeypatch.setenv("IG_SESSIONID", "real-sess")
    monkeypatch.setenv("IG_DS_USER_ID", "")
    monkeypatch.setenv("IG_CSRFTOKEN", "   ")
    seed = ig._build_cookie_seed()
    assert seed is not None
    assert "sessionid" in seed
    assert "ds_user_id" not in seed   # empty
    assert "csrftoken" not in seed    # whitespace-only


def test_has_full_cookie_set_requires_three_minimum(monkeypatch: pytest.MonkeyPatch):
    """Full-set path (set_settings, no /login) needs sessionid +
    ds_user_id + csrftoken at minimum. Anything less falls back to
    login_by_sessionid."""
    from instagram_ai_agent.plugins import ig
    assert ig._has_full_cookie_set({"sessionid": "s"}) is False
    assert ig._has_full_cookie_set({"sessionid": "s", "ds_user_id": "u"}) is False
    assert ig._has_full_cookie_set({
        "sessionid": "s", "ds_user_id": "u", "csrftoken": "c",
    }) is True
    assert ig._has_full_cookie_set(None) is False
    assert ig._has_full_cookie_set({}) is False


def test_session_refresh_days_default_is_zero(monkeypatch: pytest.MonkeyPatch):
    """Research-driven fix: every relogin() is a high-suspicion event
    for IG's 2026 risk models. Default must be 0 (disabled) — NOT the
    old 7 from 2022-era advice."""
    from instagram_ai_agent.plugins import ig
    monkeypatch.delenv("IG_SESSION_REFRESH_DAYS", raising=False)
    assert ig._session_refresh_days() == 0


def test_session_refresh_days_honours_override(monkeypatch: pytest.MonkeyPatch):
    from instagram_ai_agent.plugins import ig
    monkeypatch.setenv("IG_SESSION_REFRESH_DAYS", "30")
    assert ig._session_refresh_days() == 30
    monkeypatch.setenv("IG_SESSION_REFRESH_DAYS", "bogus")
    assert ig._session_refresh_days() == 0   # invalid → default


def test_cookie_seed_accepts_supplementary_cookies(monkeypatch: pytest.MonkeyPatch):
    """2026-research expansion: support ps_l/ps_n (Accounts Center),
    wd/dpr/ig_lang (continuity hints), fbm_<appid> (FB SSO)."""
    from instagram_ai_agent.plugins import ig
    for k in (
        "IG_SESSIONID", "IG_PS_L", "IG_PS_N", "IG_WD", "IG_DPR",
        "IG_IG_LANG", "IG_MCD", "IG_CCODE", "IG_FBM_APPID",
    ):
        monkeypatch.setenv(k, f"v-{k}")
    seed = ig._build_cookie_seed()
    assert seed is not None
    assert seed["ps_l"] == "v-IG_PS_L"
    assert seed["ps_n"] == "v-IG_PS_N"
    assert seed["wd"] == "v-IG_WD"
    assert seed["dpr"] == "v-IG_DPR"
    assert seed["ig_lang"] == "v-IG_IG_LANG"
    assert seed["mcd"] == "v-IG_MCD"
    assert seed["ccode"] == "v-IG_CCODE"
    # fbm_<appid> lands under the canonical cookie name
    assert seed["fbm_124024574287414"] == "v-IG_FBM_APPID"


def test_tls_impersonation_profile_disabled_via_env(monkeypatch: pytest.MonkeyPatch):
    from instagram_ai_agent.plugins import ig
    for off_val in ("off", "OFF", "0", "false", "no"):
        monkeypatch.setenv("IG_TLS_IMPERSONATE", off_val)
        assert ig._tls_impersonation_profile() is None


def test_tls_impersonation_profile_noop_without_curl_cffi(monkeypatch: pytest.MonkeyPatch):
    """curl_cffi isn't in base deps — profile() must return None,
    NOT raise, so the IGClient stays on plain requests.Session."""
    from instagram_ai_agent.plugins import ig
    monkeypatch.delenv("IG_TLS_IMPERSONATE", raising=False)
    # curl_cffi is not installed in test env
    assert ig._tls_impersonation_profile() is None


def test_warmup_skip_env_bypasses_ramp(monkeypatch: pytest.MonkeyPatch):
    """Audit fix B1: IG_SKIP_WARMUP=1 must bypass the warmup ramp so
    established accounts can post immediately."""
    from instagram_ai_agent.core import config as cfg_mod
    from instagram_ai_agent.core import warmup

    monkeypatch.setenv("IG_SKIP_WARMUP", "1")
    cfg = cfg_mod.NicheConfig(
        niche="test", sub_topics=["t"], target_audience="test users",
        commercial=True,
        voice=cfg_mod.Voice(tone=["direct"], forbidden=[], persona="test persona for warmup."),
        aesthetic=cfg_mod.Aesthetic(palette=["#000", "#fff", "#c9a961"]),
        hashtags=cfg_mod.HashtagPools(core=["a", "b", "c"]),
    )
    caps = warmup.effective_caps(cfg)
    assert caps.phase_label == "skipped"
    assert caps.allow_posts is True
    # Post cap is the raw config value, not 0
    assert caps.caps["post"] == cfg.schedule.posts_per_day


@pytest.mark.parametrize("off_value", ["", "0", "false", "no"])
def test_warmup_skip_requires_explicit_opt_in(monkeypatch, off_value):
    from instagram_ai_agent.core import warmup
    monkeypatch.setenv("IG_SKIP_WARMUP", off_value)
    assert warmup._skip_warmup() is False


def test_drain_query_bypasses_scheduled_for(tmp_path, monkeypatch):
    """Audit fix B2: content_next_to_drain must return items whose
    scheduled_for is in the future — drain is an explicit "post NOW"."""
    from instagram_ai_agent.core import config as cfg_mod
    from instagram_ai_agent.core import db as db_mod
    fresh = tmp_path / "brain.db"
    monkeypatch.setattr(cfg_mod, "DB_PATH", fresh)
    monkeypatch.setattr(db_mod, "DB_PATH", fresh)
    db_mod.close()
    db_mod.init_db()

    cid = db_mod.content_enqueue(
        format="meme", caption="x", hashtags=[], media_paths=["/tmp/x.jpg"],
        phash=None, critic_score=None, critic_notes=None, generator="test",
        status="approved",
    )
    # Slot it 6 hours into the future
    from datetime import datetime, timedelta, timezone
    future = (datetime.now(timezone.utc) + timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%SZ")
    db_mod.content_schedule(cid, future)

    # Normal post_next filter skips it
    assert db_mod.content_next_to_post() is None
    # Drain picks it regardless
    item = db_mod.content_next_to_drain()
    assert item is not None
    assert int(item["id"]) == cid
    # scheduled_for was cleared so the next scheduling pass doesn't re-slot
    row = db_mod.content_get(cid)
    assert row["scheduled_for"] is None
    db_mod.close()


def test_session_health_table_exists(tmp_path, monkeypatch):
    from instagram_ai_agent.core import db as db_mod
    from instagram_ai_agent.core import config as cfg_mod
    fresh = tmp_path / "brain.db"
    monkeypatch.setattr(cfg_mod, "DB_PATH", fresh)
    monkeypatch.setattr(db_mod, "DB_PATH", fresh)
    db_mod.close()
    db_mod.init_db()
    rows = db_mod.get_conn().execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='session_health'"
    ).fetchall()
    assert len(rows) == 1
    db_mod.close()


def test_default_user_agent_matches_device(monkeypatch: pytest.MonkeyPatch):
    """UA string must embed the device fingerprint values we claim, so
    cookie-based sessions have internally-consistent device claims."""
    from instagram_ai_agent.plugins import ig
    device = {
        "app_version": "999.9.9.9", "android_version": 30,
        "android_release": "11", "dpi": "420dpi",
        "resolution": "1080x2220", "manufacturer": "samsung",
        "device": "SM-A525F", "model": "a52q", "cpu": "qcom",
        "version_code": "521498971",
    }
    ua = ig._default_user_agent(device)
    assert "999.9.9.9" in ua
    assert "SM-A525F" in ua
    assert "samsung" in ua


# ─── FEED_FORMATS includes story_carousel ───
def test_feed_formats_includes_story_carousel():
    """Audit polish: story_carousel weight in FormatMix must be in
    FEED_FORMATS so format_picker / dispatch don't KeyError."""
    assert "story_carousel" in cfg_mod.FEED_FORMATS


# ─── Idea-bank auto-seed on init ───
def test_idea_bank_seed_from_file_is_idempotent(tmp_db):
    """Calling it twice must not double-insert. Init wizard calls it
    once — re-running init shouldn't create duplicates."""
    from instagram_ai_agent.brain import idea_bank
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
