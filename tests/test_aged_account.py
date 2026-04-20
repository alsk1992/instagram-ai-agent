"""Aged/bought-account safety gates — rur parser, device import,
rest + freeze gates, gentle_ping."""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from instagram_ai_agent.core import gates
from instagram_ai_agent.plugins import device as device_mod
from instagram_ai_agent.plugins import rur as rur_mod


# ─── rur parser ───
def test_rur_parse_cookie_editor_form():
    """Cookie-Editor exports \\054 sequences as literal backslash-054."""
    raw = r'RVA\05446297379238\0541735689600:01abc123signature'
    info = rur_mod.parse_rur(raw)
    assert info is not None
    assert info.region == "RVA"
    assert info.continent == "NA"
    assert info.user_id == "46297379238"
    assert info.issued_epoch == 1735689600


def test_rur_parse_live_cookie_form():
    """Live cookies replace \\054 with literal commas."""
    raw = "CLN,46297379238,1735689600:01xyz"
    info = rur_mod.parse_rur(raw)
    assert info is not None
    assert info.region == "CLN"
    assert info.continent == "EU"


def test_rur_parse_invalid_returns_none():
    assert rur_mod.parse_rur("") is None
    assert rur_mod.parse_rur("not a rur") is None
    assert rur_mod.parse_rur("XYZ\\054only\\054two") is not None  # format parses even if region unknown


def test_rur_unknown_region_is_question_mark():
    info = rur_mod.parse_rur(r"ZZZ\054123\0541735689600:01x")
    assert info is not None
    assert info.continent == "?"


def test_rur_is_stale_when_old():
    # Epoch 48h+1min ago
    old_epoch = int((datetime.now(timezone.utc) - timedelta(hours=48, minutes=1)).timestamp())
    info = rur_mod.parse_rur(f"RVA\\054123\\054{old_epoch}:01x")
    assert info is not None
    assert info.is_stale is True


def test_rur_fresh_not_stale():
    fresh_epoch = int(datetime.now(timezone.utc).timestamp())
    info = rur_mod.parse_rur(f"RVA\\054123\\054{fresh_epoch}:01x")
    assert info is not None
    assert info.is_stale is False


def test_continent_match_accepts_same_continent():
    rur_us = rur_mod.parse_rur(r"RVA\054123\0541735689600:01x")
    assert rur_mod.continent_matches(rur_us, "US") is True
    assert rur_mod.continent_matches(rur_us, "CA") is True
    assert rur_mod.continent_matches(rur_us, "GB") is False


def test_continent_match_fails_open_on_unknown():
    # Unknown region → always True (can't judge)
    rur_unk = rur_mod.parse_rur(r"ZZZ\054123\0541735689600:01x")
    assert rur_mod.continent_matches(rur_unk, "US") is True
    # Unknown country → always True (user hasn't declared, don't block)
    rur_us = rur_mod.parse_rur(r"RVA\054123\0541735689600:01x")
    assert rur_mod.continent_matches(rur_us, None) is True
    assert rur_mod.continent_matches(rur_us, "ZZ") is True  # unknown cc


# ─── Device bundle import ───
def test_device_import_flat_bundle(tmp_path, monkeypatch):
    """Flat device.json from a seller bundle — just UUIDs + device keys."""
    bundle = tmp_path / "seller.json"
    bundle.write_text(json.dumps({
        "phone_id": "seller-phone-uuid-aaaa",
        "uuid": "seller-client-uuid-bbbb",
        "client_session_id": "seller-session-uuid-cccc",
        "advertising_id": "seller-ad-uuid-dddd",
        "device_id": "android-selleraadeadbeef",
        "manufacturer": "xiaomi",
        "device": "Redmi-Note-11",
        "model": "spes",
    }))
    target = tmp_path / "device.json"
    monkeypatch.setenv("IG_DEVICE_IMPORT_PATH", str(bundle))

    settings = device_mod.load_or_create(target)
    assert settings["phone_id"] == "seller-phone-uuid-aaaa"
    assert settings["device_id"] == "android-selleraadeadbeef"
    assert settings["manufacturer"] == "xiaomi"
    assert target.exists()  # persisted to target


def test_device_import_nested_dump_settings_bundle(tmp_path, monkeypatch):
    """instagrapi's dump_settings() shape with nested uuids/device_settings."""
    bundle = tmp_path / "dump.json"
    bundle.write_text(json.dumps({
        "uuids": {
            "phone_id": "nested-phone-uuid",
            "uuid": "nested-client-uuid",
            "device_id": "android-nestedaabbccdd",
        },
        "device_settings": {
            "manufacturer": "oppo",
            "model": "CPH2505",
            "dpi": "480dpi",
        },
        "cookies": {"sessionid": "x"},  # ignored at import level
    }))
    target = tmp_path / "device.json"
    monkeypatch.setenv("IG_DEVICE_IMPORT_PATH", str(bundle))

    settings = device_mod.load_or_create(target)
    assert settings["phone_id"] == "nested-phone-uuid"
    assert settings["device_id"] == "android-nestedaabbccdd"
    assert settings["manufacturer"] == "oppo"


def test_device_import_partial_fills_missing_with_fresh(tmp_path, monkeypatch):
    """Bundle missing some UUID keys — missing ones fill from fresh gen.
    Should warn but not fail."""
    bundle = tmp_path / "partial.json"
    bundle.write_text(json.dumps({
        "phone_id": "only-phone-id",
        # missing uuid, client_session_id, advertising_id, device_id
    }))
    target = tmp_path / "device.json"
    monkeypatch.setenv("IG_DEVICE_IMPORT_PATH", str(bundle))

    settings = device_mod.load_or_create(target)
    assert settings["phone_id"] == "only-phone-id"
    # Filled values exist
    assert settings["uuid"]
    assert settings["device_id"].startswith("android-")


def test_device_import_bad_path_falls_back(tmp_path, monkeypatch):
    """Bundle path points to a non-existent file — fall back to fresh UUIDs,
    don't crash."""
    target = tmp_path / "device.json"
    monkeypatch.setenv("IG_DEVICE_IMPORT_PATH", str(tmp_path / "does_not_exist.json"))
    settings = device_mod.load_or_create(target)
    assert settings["phone_id"]
    assert settings["device_id"].startswith("android-")


def test_device_load_existing_unchanged_by_env(tmp_path, monkeypatch):
    """Once device.json exists, env-var imports DON'T overwrite — first-run only."""
    existing = {
        "phone_id": "existing-phone-id",
        "uuid": "existing-uuid",
        "client_session_id": "existing-session-id",
        "advertising_id": "existing-ad-id",
        "device_id": "android-existingxxxxxxxxxxxx",
        "manufacturer": "samsung",
        "device": "SM-A525F",
        "model": "a52q",
        "app_version": "302.0.0.23.114",
        "android_version": 30,
        "android_release": "11",
        "dpi": "420dpi",
        "resolution": "1080x2220",
        "cpu": "qcom",
        "version_code": "521498971",
    }
    target = tmp_path / "device.json"
    target.write_text(json.dumps(existing))

    bundle = tmp_path / "bundle.json"
    bundle.write_text(json.dumps({"phone_id": "new-bundle-phone"}))
    monkeypatch.setenv("IG_DEVICE_IMPORT_PATH", str(bundle))

    settings = device_mod.load_or_create(target)
    assert settings["phone_id"] == "existing-phone-id"  # existing wins


# ─── Rest + freeze gates ───
def test_rest_gate_inactive_when_unset(monkeypatch):
    monkeypatch.delenv("IG_REST_UNTIL", raising=False)
    s = gates.rest_status()
    assert s.active is False
    assert s.until is None


def test_rest_gate_active_when_future(monkeypatch):
    future = gates.suggest_rest_until(hours=24)
    monkeypatch.setenv("IG_REST_UNTIL", future)
    s = gates.rest_status()
    assert s.active is True
    assert s.remaining is not None
    assert 23 < (s.remaining_hours or 0) <= 24.1


def test_rest_gate_inactive_when_past(monkeypatch):
    past = (datetime.now(timezone.utc) - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    monkeypatch.setenv("IG_REST_UNTIL", past)
    s = gates.rest_status()
    assert s.active is False


def test_rest_gate_tolerates_malformed(monkeypatch):
    monkeypatch.setenv("IG_REST_UNTIL", "not-a-real-timestamp")
    s = gates.rest_status()
    assert s.active is False  # malformed → treated as no gate


def test_writes_blocked_matches_rest_status(monkeypatch):
    future = gates.suggest_rest_until(hours=2)
    monkeypatch.setenv("IG_REST_UNTIL", future)
    assert gates.writes_blocked() is True
    monkeypatch.delenv("IG_REST_UNTIL")
    assert gates.writes_blocked() is False


def test_freeze_gate_separate_from_rest(monkeypatch):
    """Freeze gate reads its own env var and is independent of rest."""
    monkeypatch.setenv("IG_FREEZE_PROFILE_UNTIL", gates.suggest_freeze_until(days=5))
    monkeypatch.delenv("IG_REST_UNTIL", raising=False)
    assert gates.profile_edits_blocked() is True
    assert gates.writes_blocked() is False


def test_suggest_timestamps_are_iso_utc():
    a = gates.suggest_rest_until(hours=48)
    b = gates.suggest_freeze_until(days=21)
    # Both must be parseable back
    assert datetime.fromisoformat(a.replace("Z", "+00:00")).tzinfo is not None
    assert datetime.fromisoformat(b.replace("Z", "+00:00")).tzinfo is not None
    # Suffix Z for display
    assert a.endswith("Z")
    assert b.endswith("Z")


# ─── TOTP secret validation ───
def test_totp_validates_valid_base32():
    """A real-shape TOTP secret produces a 6-digit code."""
    from instagram_ai_agent import cli
    # Known pyotp-compatible test secret (32 chars base32, no padding)
    code = cli._validate_totp_secret("JBSWY3DPEHPK3PXPJBSWY3DPEHPK3PXP")
    assert code is not None
    assert code.isdigit()
    assert len(code) == 6


def test_totp_rejects_non_base32():
    """Invalid characters → None (wizard shows error, skips writing)."""
    from instagram_ai_agent import cli
    assert cli._validate_totp_secret("NOT-BASE32-HAS-DASHES!") is None
    assert cli._validate_totp_secret("") is None
    assert cli._validate_totp_secret("123") is None  # too short


def test_totp_rejects_random_garbage():
    from instagram_ai_agent import cli
    # pyotp rejects lowercase base32 sometimes — our wizard uppers first,
    # so test the raw API rejects garbage
    assert cli._validate_totp_secret("!@#$%^&*()") is None


def test_totp_deterministic_same_window():
    """Calling _validate_totp_secret twice within 30s produces the same code."""
    from instagram_ai_agent import cli
    secret = "JBSWY3DPEHPK3PXPJBSWY3DPEHPK3PXP"
    a = cli._validate_totp_secret(secret)
    b = cli._validate_totp_secret(secret)
    assert a == b  # same 30s window = same code
