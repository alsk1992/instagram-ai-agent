"""Cookie-jar parser + VPS wizard branch tests."""
from __future__ import annotations

import json

import pytest

from instagram_ai_agent import cli


# ─── _parse_cookie_editor_json ───
def test_parse_full_jar():
    """Cookie-Editor JSON export with all 10 required cookies parses cleanly."""
    export = [
        {"name": "sessionid",  "value": "abc:123:def", "domain": ".instagram.com"},
        {"name": "ds_user_id", "value": "123456789",   "domain": "instagram.com"},
        {"name": "csrftoken",  "value": "csrf32charrandom",  "domain": ".instagram.com"},
        {"name": "mid",        "value": "midvaluexx",  "domain": "instagram.com"},
        {"name": "ig_did",     "value": "did-uuid",    "domain": ".instagram.com"},
        {"name": "datr",       "value": "datrval",     "domain": ".instagram.com"},
        {"name": "rur",        "value": "CLN\\054123", "domain": ".instagram.com"},
        {"name": "shbid",      "value": "shbid1",      "domain": "instagram.com"},
        {"name": "shbts",      "value": "shbts1",      "domain": "instagram.com"},
        {"name": "ig_nrcb",    "value": "1",           "domain": "instagram.com"},
    ]
    env = cli._parse_cookie_editor_json(json.dumps(export))
    assert env["IG_SESSIONID"] == "abc:123:def"
    assert env["IG_DS_USER_ID"] == "123456789"
    assert env["IG_RUR"] == "CLN\\054123"  # rur's literal backslashes preserved
    assert env["IG_NRCB"] == "1"


def test_parse_ignores_non_instagram_cookies():
    """Cookies for other domains (facebook, google) must be filtered out."""
    export = [
        {"name": "sessionid",  "value": "s1", "domain": ".instagram.com"},
        {"name": "ds_user_id", "value": "1",  "domain": "instagram.com"},
        {"name": "csrftoken",  "value": "c1", "domain": "instagram.com"},
        {"name": "sessionid",  "value": "FACEBOOK_COOKIE", "domain": "facebook.com"},
        {"name": "randomkey",  "value": "x",  "domain": ".instagram.com"},
    ]
    env = cli._parse_cookie_editor_json(json.dumps(export))
    # facebook sessionid must NOT have overwritten instagram's
    assert env["IG_SESSIONID"] == "s1"
    assert "randomkey" not in str(env).lower()


def test_parse_missing_required_raises():
    """Missing one of sessionid/ds_user_id/csrftoken is a hard fail."""
    export = [
        {"name": "sessionid", "value": "s1", "domain": ".instagram.com"},
        # ds_user_id missing
        {"name": "csrftoken", "value": "c1", "domain": "instagram.com"},
    ]
    with pytest.raises(ValueError, match="missing required"):
        cli._parse_cookie_editor_json(json.dumps(export))


def test_parse_rejects_netscape_format():
    """Cookie-Editor supports JSON and Netscape formats; we only accept JSON."""
    netscape = (
        "# Netscape HTTP Cookie File\n"
        ".instagram.com\tTRUE\t/\tTRUE\t1234567890\tsessionid\tabc\n"
    )
    with pytest.raises(ValueError, match="JSON"):
        cli._parse_cookie_editor_json(netscape)


def test_parse_rejects_wrapped_object():
    """Some exporters wrap in {"cookies": [...]} — accept the array only."""
    wrapped = json.dumps({"cookies": [
        {"name": "sessionid", "value": "s1", "domain": ".instagram.com"},
        {"name": "ds_user_id", "value": "1", "domain": "instagram.com"},
        {"name": "csrftoken",  "value": "c1", "domain": "instagram.com"},
    ]})
    with pytest.raises(ValueError, match="array"):
        cli._parse_cookie_editor_json(wrapped)


def test_parse_empty_raises_helpful_message():
    with pytest.raises(ValueError, match="empty"):
        cli._parse_cookie_editor_json("")


def test_parse_strips_fenced_blocks():
    """Users sometimes paste inside ```json fences; strip them."""
    export = [
        {"name": "sessionid",  "value": "s1", "domain": ".instagram.com"},
        {"name": "ds_user_id", "value": "1",  "domain": "instagram.com"},
        {"name": "csrftoken",  "value": "c1", "domain": "instagram.com"},
    ]
    wrapped = f"```json\n{json.dumps(export)}\n```"
    env = cli._parse_cookie_editor_json(wrapped)
    assert env["IG_SESSIONID"] == "s1"


def test_parse_honours_supplementary_cookies():
    """wd/dpr/ig_lang/mcd etc. map into IG_* env vars alongside the required ones."""
    export = [
        {"name": "sessionid",  "value": "s1",  "domain": ".instagram.com"},
        {"name": "ds_user_id", "value": "1",   "domain": "instagram.com"},
        {"name": "csrftoken",  "value": "c1",  "domain": "instagram.com"},
        {"name": "wd",         "value": "1920x1080", "domain": ".instagram.com"},
        {"name": "dpr",        "value": "2",   "domain": ".instagram.com"},
        {"name": "ig_lang",    "value": "en",  "domain": "instagram.com"},
        {"name": "mcd",        "value": "abc", "domain": "instagram.com"},
    ]
    env = cli._parse_cookie_editor_json(json.dumps(export))
    assert env["IG_WD"] == "1920x1080"
    assert env["IG_DPR"] == "2"
    assert env["IG_IG_LANG"] == "en"
    assert env["IG_MCD"] == "abc"


# ─── _validate_cookie_jar ───
def test_validate_cookie_jar_accepts_200(monkeypatch):
    import httpx

    class FakeResponse:
        status_code = 200
        def json(self): return {"user": {"username": "testuser"}}

    monkeypatch.setattr(httpx, "get", lambda *a, **k: FakeResponse())
    ok, msg = cli._validate_cookie_jar({"IG_SESSIONID": "s", "IG_DS_USER_ID": "1"})
    assert ok is True
    assert "testuser" in msg


def test_validate_cookie_jar_rejects_401(monkeypatch):
    import httpx

    class FakeResponse:
        status_code = 401
        def json(self): return {"message": "login_required"}

    monkeypatch.setattr(httpx, "get", lambda *a, **k: FakeResponse())
    ok, msg = cli._validate_cookie_jar({"IG_SESSIONID": "stale"})
    assert ok is False
    assert "stale" in msg.lower() or "401" in msg


def test_validate_cookie_jar_handles_network_error(monkeypatch):
    import httpx
    monkeypatch.setattr(
        httpx, "get",
        lambda *a, **k: (_ for _ in ()).throw(httpx.ConnectError("no network")),
    )
    ok, msg = cli._validate_cookie_jar({"IG_SESSIONID": "x"})
    assert ok is False
    assert "network" in msg.lower()


# ─── _COOKIE_NAME_TO_ENV mapping ───
def test_env_mapping_covers_all_required_cookies():
    """Required cookies (sessionid/ds_user_id/csrftoken) must all have env names."""
    for name in cli._COOKIE_REQUIRED:
        assert name in cli._COOKIE_NAME_TO_ENV


def test_env_mapping_round_trips_with_plugins_ig():
    """Every env var we map to must be an env var plugins/ig.py reads."""
    # Grep-equivalent: the cookie_seed loader in plugins/ig.py should read these
    import instagram_ai_agent.plugins.ig as ig_mod
    source = __import__("inspect").getsource(ig_mod)
    for env_var in cli._COOKIE_NAME_TO_ENV.values():
        assert env_var in source, (
            f"{env_var} is mapped in cli.py but not read anywhere in plugins/ig.py"
        )


# ─── single-line capture ───
def test_capture_single_line_happy_path(monkeypatch):
    """Providing values for all 3 required cookies (and skipping optional)
    returns a valid env dict."""
    import questionary
    # Feed answers in order matching _COOKIE_PROMPT_ORDER
    answers = iter([
        "sessid_value",     # sessionid (required)
        "123456789",         # ds_user_id (required)
        "csrftoken_val",     # csrftoken (required)
        "",                  # mid (skip)
        "",                  # ig_did (skip)
        "",                  # datr (skip)
        "",                  # rur (skip)
        "",                  # shbid (skip)
        "",                  # shbts (skip)
        "",                  # ig_nrcb (skip)
        "",                  # wd (skip)
    ])

    def fake_text(prompt, **kw):
        class _Q:
            def ask(self_inner):
                return next(answers)
        return _Q()

    monkeypatch.setattr(questionary, "text", fake_text)

    env = cli._capture_cookies_single_line()
    assert env["IG_SESSIONID"] == "sessid_value"
    assert env["IG_DS_USER_ID"] == "123456789"
    assert env["IG_CSRFTOKEN"] == "csrftoken_val"
    assert "IG_MID" not in env  # skipped


def test_capture_single_line_strips_quotes(monkeypatch):
    """Users sometimes copy values with surrounding double quotes from
    the Cookie-Editor UI — strip them silently."""
    import questionary
    answers = iter([
        '"sessid_quoted"', "123456789", "csrftoken_val",
        "", "", "", "", "", "", "", "",
    ])

    def fake_text(prompt, **kw):
        class _Q:
            def ask(self_inner):
                return next(answers)
        return _Q()

    monkeypatch.setattr(questionary, "text", fake_text)
    env = cli._capture_cookies_single_line()
    assert env["IG_SESSIONID"] == "sessid_quoted"


def test_capture_single_line_aborts_on_empty_required(monkeypatch):
    """Empty sessionid twice → return {} (caller treats as abort)."""
    import questionary
    # Empty for sessionid twice (prompt + retry), then loop continues
    answers = iter(["", ""])

    def fake_text(prompt, **kw):
        class _Q:
            def ask(self_inner):
                return next(answers)
        return _Q()

    monkeypatch.setattr(questionary, "text", fake_text)
    env = cli._capture_cookies_single_line()
    assert env == {}


def test_cookie_prompt_order_covers_required():
    """Required cookies must appear first (so partial paste still has
    the minimum for validation)."""
    required_names = cli._COOKIE_REQUIRED
    first_three_names = {entry[0] for entry in cli._COOKIE_PROMPT_ORDER[:3]}
    assert required_names == first_three_names
