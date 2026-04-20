"""Web-origin cookie detection + UA/TLS auto-switch."""
from __future__ import annotations

import pytest

from instagram_ai_agent import cli
from instagram_ai_agent.plugins import ig as ig_mod


# ─── is_web_origin_cookies ───
def test_detects_wd_cookie():
    assert ig_mod.is_web_origin_cookies({"sessionid": "s", "wd": "1920x1080"}) is True


def test_detects_dpr_cookie():
    assert ig_mod.is_web_origin_cookies({"sessionid": "s", "dpr": "2"}) is True


def test_mobile_origin_cookies_false():
    """Cookies without wd/dpr (from mobile app or emulator-harvested) → mobile mode."""
    assert ig_mod.is_web_origin_cookies({
        "sessionid": "s", "ds_user_id": "1", "csrftoken": "c",
        "mid": "m", "ig_did": "d", "rur": "r",
    }) is False


def test_empty_seed_not_web():
    assert ig_mod.is_web_origin_cookies(None) is False
    assert ig_mod.is_web_origin_cookies({}) is False


# ─── TLS profile selection ───
def test_tls_profile_defaults_mobile(monkeypatch):
    monkeypatch.delenv("IG_TLS_IMPERSONATE", raising=False)
    try:
        import curl_cffi  # noqa: F401
    except Exception:
        pytest.skip("curl_cffi not installed")
    assert ig_mod._tls_impersonation_profile(web_mode=False) == "chrome131_android"


def test_tls_profile_web_mode_switches_to_desktop(monkeypatch):
    monkeypatch.delenv("IG_TLS_IMPERSONATE", raising=False)
    try:
        import curl_cffi  # noqa: F401
    except Exception:
        pytest.skip("curl_cffi not installed")
    assert ig_mod._tls_impersonation_profile(web_mode=True) == "chrome136"


def test_tls_profile_env_override_wins_over_web_mode(monkeypatch):
    monkeypatch.setenv("IG_TLS_IMPERSONATE", "safari18_ios")
    try:
        import curl_cffi  # noqa: F401
    except Exception:
        pytest.skip("curl_cffi not installed")
    assert ig_mod._tls_impersonation_profile(web_mode=True) == "safari18_ios"


def test_tls_profile_off_returns_none(monkeypatch):
    monkeypatch.setenv("IG_TLS_IMPERSONATE", "off")
    assert ig_mod._tls_impersonation_profile(web_mode=True) is None


# ─── _web_mode_headers ───
def test_web_mode_headers_contain_sec_ch_ua():
    h = ig_mod._web_mode_headers()
    assert "Sec-CH-UA" in h
    assert "Sec-CH-UA-Mobile" in h and h["Sec-CH-UA-Mobile"] == "?0"
    assert "Sec-CH-UA-Platform" in h and "Windows" in h["Sec-CH-UA-Platform"]
    assert h["X-IG-App-ID"] == "936619743392459"
    assert h["X-ASBD-ID"] == "198387"
    assert h["Referer"].startswith("https://www.instagram.com")


def test_desktop_chrome_ua_is_windows_chrome():
    ua = ig_mod.DESKTOP_CHROME_UA
    assert "Windows NT 10.0" in ua
    assert "Chrome/138" in ua
    assert "Safari/537.36" in ua
    # MUST NOT look like an Android Instagram app UA
    assert "Instagram " not in ua
    assert "Android" not in ua


# ─── _apply_web_identity ───
def test_apply_web_identity_pins_headers_on_both_sessions():
    class FakeSession:
        def __init__(self):
            self.headers = {}
    class FakeClient:
        def __init__(self):
            self.private = FakeSession()
            self.public = FakeSession()
            self.user_agent = "Instagram/Android/OldMobileUA"
    cl = FakeClient()
    ig_mod._apply_web_identity(cl)
    assert cl.private.headers["User-Agent"] == ig_mod.DESKTOP_CHROME_UA
    assert cl.public.headers["User-Agent"] == ig_mod.DESKTOP_CHROME_UA
    assert "Sec-CH-UA" in cl.private.headers
    assert cl.user_agent == ig_mod.DESKTOP_CHROME_UA


def test_apply_web_identity_tolerates_missing_sessions():
    class FakeClient:
        # No .private, no .public — shouldn't raise
        pass
    ig_mod._apply_web_identity(FakeClient())  # no AttributeError


# ─── _build_settings_from_cookies web UA selection ───
def test_build_settings_selects_web_ua_for_web_cookies(monkeypatch):
    """Web cookies → desktop Chrome UA. Mobile cookies → Android UA."""
    # Stub dev.load_or_create so we don't touch the real device.json
    fake_device = {
        "phone_id": "00000000-0000-0000-0000-000000000002",
        "uuid": "00000000-0000-0000-0000-000000000001",
        "client_session_id": "00000000-0000-0000-0000-000000000003",
        "advertising_id": "00000000-0000-0000-0000-000000000004",
        "device_id": "android-0000000000000001",
        "app_version": "302.0.0.23.114",
        "android_version": 30,
        "android_release": "11",
        "dpi": "420dpi",
        "resolution": "1080x2220",
        "manufacturer": "samsung",
        "device": "SM-A525F",
        "model": "a52q",
        "cpu": "qcom",
        "version_code": "521498971",
    }
    monkeypatch.setattr(ig_mod.dev, "load_or_create", lambda: fake_device)
    monkeypatch.delenv("IG_USER_AGENT", raising=False)

    client = ig_mod.IGClient.__new__(ig_mod.IGClient)  # bypass __init__
    client.username = "testuser"

    web_settings = client._build_settings_from_cookies({
        "sessionid": "s", "ds_user_id": "1", "csrftoken": "c", "wd": "1920x1080",
    })
    assert "Chrome/138" in web_settings["user_agent"]
    assert "Windows NT 10.0" in web_settings["user_agent"]

    mobile_settings = client._build_settings_from_cookies({
        "sessionid": "s", "ds_user_id": "1", "csrftoken": "c",
    })
    assert "Instagram" in mobile_settings["user_agent"]
    assert "Android" in mobile_settings["user_agent"]


# ─── _validate_cookie_jar picks endpoint + UA based on origin ───
def test_validate_web_mode_hits_www_instagram_with_desktop_ua(monkeypatch):
    import httpx

    captured = {}
    class FakeResponse:
        status_code = 200
        def json(self): return {"form_data": {"username": "testuser"}}

    def fake_get(url, cookies=None, headers=None, timeout=None, follow_redirects=None):
        captured["url"] = url
        captured["headers"] = headers or {}
        return FakeResponse()

    monkeypatch.setattr(httpx, "get", fake_get)
    ok, msg = cli._validate_cookie_jar({
        "IG_SESSIONID": "s", "IG_DS_USER_ID": "1", "IG_CSRFTOKEN": "c",
        "IG_WD": "1920x1080",
    })
    assert ok
    assert "testuser" in msg
    assert "web" in msg.lower()
    # Web-specific endpoint
    assert "www.instagram.com/api/v1/accounts/edit/web_form_data" in captured["url"]
    # Desktop UA
    assert "Chrome/138" in captured["headers"]["User-Agent"]
    assert "Sec-CH-UA" in captured["headers"]


def test_validate_mobile_mode_hits_i_instagram_with_mobile_ua(monkeypatch):
    import httpx

    captured = {}
    class FakeResponse:
        status_code = 200
        def json(self): return {"user": {"username": "mobileuser"}}

    def fake_get(url, cookies=None, headers=None, timeout=None, follow_redirects=None):
        captured["url"] = url
        captured["headers"] = headers or {}
        return FakeResponse()

    monkeypatch.setattr(httpx, "get", fake_get)
    ok, msg = cli._validate_cookie_jar({
        "IG_SESSIONID": "s", "IG_DS_USER_ID": "1", "IG_CSRFTOKEN": "c",
        # no IG_WD / IG_DPR → mobile mode
    })
    assert ok
    assert "mobile" in msg.lower()
    assert "i.instagram.com/api/v1/accounts/current_user" in captured["url"]
    assert "Instagram" in captured["headers"]["User-Agent"]
    assert "Android" in captured["headers"]["User-Agent"]


def test_validate_custom_ua_overrides_auto_detection(monkeypatch):
    """User-pinned IG_USER_AGENT wins over the auto-selected default."""
    import httpx

    captured = {}
    class FakeResponse:
        status_code = 200
        def json(self): return {"form_data": {"username": "x"}}

    def fake_get(url, cookies=None, headers=None, timeout=None, follow_redirects=None):
        captured["headers"] = headers or {}
        return FakeResponse()

    monkeypatch.setattr(httpx, "get", fake_get)
    cli._validate_cookie_jar({
        "IG_SESSIONID": "s", "IG_DS_USER_ID": "1", "IG_CSRFTOKEN": "c",
        "IG_WD": "1920x1080",
        "IG_USER_AGENT": "CustomPinnedUA/1.0",
    })
    assert captured["headers"]["User-Agent"] == "CustomPinnedUA/1.0"
