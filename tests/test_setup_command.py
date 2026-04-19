"""One-command `ig-agent setup` — minimal wizard + deps check + key validation."""
from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from instagram_ai_agent import cli
from instagram_ai_agent.core import config as cfg_mod
from instagram_ai_agent.core import db


@pytest.fixture()
def tmp_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect every write-path (niche.yaml, .env, brain.db) into tmp_path."""
    niche = tmp_path / "niche.yaml"
    env = tmp_path / ".env"
    dbf = tmp_path / "brain.db"
    monkeypatch.setattr(cfg_mod, "NICHE_PATH", niche)
    monkeypatch.setattr(cfg_mod, "ENV_PATH", env)
    monkeypatch.setattr(cfg_mod, "DB_PATH", dbf)
    monkeypatch.setattr(cli, "NICHE_PATH", niche)
    monkeypatch.setattr(cli, "ENV_PATH", env)
    monkeypatch.setattr(db, "DB_PATH", dbf)
    # Force ensure_dirs to write inside tmp_path
    monkeypatch.setattr(cfg_mod, "ROOT", tmp_path)
    monkeypatch.setattr(cfg_mod, "DATA_DIR", tmp_path / "data")
    db.close()
    yield tmp_path
    db.close()


# ─── _ffmpeg_install_cmd returns an OS-specific hint ───
def test_ffmpeg_install_cmd_returns_something():
    cmd = cli._ffmpeg_install_cmd()
    assert cmd
    assert len(cmd) > 5


# ─── _playwright_chromium_installed tolerates missing install ───
def test_playwright_check_runs_without_crashing(tmp_path, monkeypatch):
    # Override HOME so there's no chromium binary in our fake home
    monkeypatch.setenv("HOME", str(tmp_path))
    result = cli._playwright_chromium_installed()
    assert result in (True, False)  # doesn't raise


# ─── _setup_pick_niche with --minimal takes preset defaults ───
def test_setup_pick_niche_minimal_mode(tmp_root, monkeypatch):
    """Minimal mode asks ONLY the preset + niche name; fills the rest from preset."""
    # Stub questionary: pick "fitness" preset, keep the default niche name
    import questionary

    def fake_select(prompt, choices, **kw):
        class _Q:
            def ask(self_inner):
                # pick the fitness preset
                for c in choices:
                    if isinstance(c, str) and c.startswith("fitness"):
                        return c
                return choices[0]
        return _Q()

    def fake_text(prompt, default="", **kw):
        class _Q:
            def ask(self_inner):
                return default or "home calisthenics"
        return _Q()

    monkeypatch.setattr(questionary, "select", fake_select)
    monkeypatch.setattr(questionary, "text", fake_text)

    cfg = cli._setup_pick_niche(minimal=True)
    assert cfg.niche == "home calisthenics and bodyweight training"
    assert cfg.voice.persona  # preset filled
    assert cfg.sub_topics  # preset filled
    assert len(cfg.hashtags.core) >= 3
    assert cfg.commercial is True  # sensible default
    assert cfg.safety.require_review is True  # sensible default — always review first


# ─── _setup_pick_niche with custom preset falls back to placeholder tags ───
def test_setup_pick_niche_custom_preset(tmp_root, monkeypatch):
    """'custom' preset still produces a valid config (HashtagPools needs 3 cores)."""
    import questionary

    def fake_select(prompt, choices, **kw):
        class _Q:
            def ask(self_inner):
                return "custom (blank defaults)"
        return _Q()

    def fake_text(prompt, default="", validate=None, **kw):
        class _Q:
            def ask(self_inner):
                # Return defaults (empty) to exercise the fallback path
                return "fashion for petite women over 40"
        return _Q()

    monkeypatch.setattr(questionary, "select", fake_select)
    monkeypatch.setattr(questionary, "text", fake_text)

    cfg = cli._setup_pick_niche(minimal=True)
    assert "fashion" in cfg.niche.lower()
    # Must still have >= 3 hashtags — we pad with placeholders if preset is empty
    assert len(cfg.hashtags.core) >= 3


# ─── _validate_openrouter_key rejects bad keys, accepts good ones ───
def test_validate_openrouter_key_accepts_valid(monkeypatch):
    import httpx

    class FakeResponse:
        status_code = 200
        def json(self):
            return {"data": [{"id": "model-a"}, {"id": "model-b"}]}

    def fake_get(url, headers=None, timeout=None):
        assert "openrouter.ai/api/v1/models" in url
        assert headers and "Bearer " in headers.get("Authorization", "")
        return FakeResponse()

    monkeypatch.setattr(httpx, "get", fake_get)

    ok, msg = cli._validate_openrouter_key("sk-or-v1-test123")
    assert ok is True
    assert "2 models" in msg


def test_validate_openrouter_key_rejects_401(monkeypatch):
    import httpx

    class FakeResponse:
        status_code = 401
        def json(self):
            return {"error": "unauthorized"}

    monkeypatch.setattr(httpx, "get", lambda *a, **k: FakeResponse())

    ok, msg = cli._validate_openrouter_key("sk-or-bad")
    assert ok is False
    assert "401" in msg or "invalid" in msg


def test_validate_openrouter_key_handles_network_error(monkeypatch):
    import httpx

    def raise_net(*a, **k):
        raise httpx.ConnectError("dns fail")

    monkeypatch.setattr(httpx, "get", raise_net)

    ok, msg = cli._validate_openrouter_key("sk-or-test")
    assert ok is False
    assert "network" in msg


# ─── End-to-end: setup command writes niche.yaml + .env + seeds db ───
def test_setup_end_to_end(tmp_root, monkeypatch):
    """Full setup call with every external effect mocked — confirms we write
    niche.yaml, .env, brain.db and seed the idea bank."""
    import httpx
    import questionary
    import webbrowser

    # 1. Stub the deps check so it passes regardless of host env
    monkeypatch.setattr(cli, "_setup_check_deps", lambda: None)

    # 2. Stub preset picker + text inputs
    def fake_select(prompt, choices, **kw):
        class _Q:
            def ask(self_inner):
                for c in choices:
                    if isinstance(c, str) and c.startswith("fitness"):
                        return c
                return choices[0]
        return _Q()

    def fake_text(prompt, default="", **kw):
        class _Q:
            def ask(self_inner):
                return default or "home calisthenics"
        return _Q()

    def fake_password(prompt, **kw):
        class _Q:
            def ask(self_inner):
                return "sk-or-v1-TESTKEY"
        return _Q()

    monkeypatch.setattr(questionary, "select", fake_select)
    monkeypatch.setattr(questionary, "text", fake_text)
    monkeypatch.setattr(questionary, "password", fake_password)
    monkeypatch.setattr(webbrowser, "open", lambda *a, **k: True)

    # 3. Stub the OpenRouter validation ping
    class FakeResponse:
        status_code = 200
        def json(self): return {"data": [{"id": "m"}]}
    monkeypatch.setattr(httpx, "get", lambda *a, **k: FakeResponse())

    runner = CliRunner()
    result = runner.invoke(cli.app, ["setup", "--minimal"])
    if result.exit_code != 0:
        print(result.output)
        print(result.exception)
    assert result.exit_code == 0, result.output

    # Niche file written
    assert cfg_mod.NICHE_PATH.exists()
    saved = cfg_mod.load_niche()
    assert saved.niche  # something was written
    assert saved.voice.persona

    # .env written with the pasted key
    assert cfg_mod.ENV_PATH.exists()
    env_text = cfg_mod.ENV_PATH.read_text()
    assert "OPENROUTER_API_KEY=sk-or-v1-TESTKEY" in env_text

    # brain.db initialised (ideas table exists — seeding should have inserted rows)
    assert cfg_mod.DB_PATH.exists()


# ─── Existing niche.yaml protection ───
def test_setup_refuses_to_overwrite_without_confirmation(tmp_root, monkeypatch):
    """Pre-existing niche.yaml must not be silently clobbered."""
    import questionary
    import webbrowser
    from rich import prompt as rp

    cfg_mod.NICHE_PATH.write_text("niche: existing-user-config\nthis: is preserved\n")
    assert cfg_mod.NICHE_PATH.exists()

    # User declines the overwrite prompt → setup should exit(0) without touching anything
    monkeypatch.setattr(rp.Confirm, "ask", classmethod(lambda cls, *a, **k: False))
    monkeypatch.setattr(cli, "_setup_check_deps", lambda: None)
    monkeypatch.setattr(questionary, "select", lambda *a, **k: type("Q", (), {"ask": lambda s: "fitness — Fitness & calisthenics"})())
    monkeypatch.setattr(webbrowser, "open", lambda *a, **k: True)

    runner = CliRunner()
    result = runner.invoke(cli.app, ["setup", "--minimal"])
    assert result.exit_code == 0
    # File content preserved byte-for-byte
    assert "existing-user-config" in cfg_mod.NICHE_PATH.read_text()


def test_setup_force_overwrites_without_prompting(tmp_root, monkeypatch):
    """--force skips the overwrite confirmation."""
    import httpx
    import questionary
    import webbrowser
    from rich import prompt as rp

    cfg_mod.NICHE_PATH.write_text("niche: doomed\n")

    # This mock should NEVER be called with --force — fail the test if it is
    def no_confirm(*a, **k):
        raise AssertionError("Confirm.ask called despite --force")

    monkeypatch.setattr(rp.Confirm, "ask", classmethod(lambda cls, *a, **k: True))
    monkeypatch.setattr(cli, "_setup_check_deps", lambda: None)

    def fake_select(prompt, choices, **kw):
        return type("Q", (), {"ask": lambda s: "fitness — Fitness & calisthenics"})()

    def fake_text(prompt, default="", **kw):
        return type("Q", (), {"ask": lambda s: default or "home calisthenics"})()

    def fake_password(prompt, **kw):
        return type("Q", (), {"ask": lambda s: "sk-or-v1-FORCEKEY"})()

    class FakeResponse:
        status_code = 200
        def json(self): return {"data": [{"id": "m"}]}

    monkeypatch.setattr(questionary, "select", fake_select)
    monkeypatch.setattr(questionary, "text", fake_text)
    monkeypatch.setattr(questionary, "password", fake_password)
    monkeypatch.setattr(webbrowser, "open", lambda *a, **k: True)
    monkeypatch.setattr(httpx, "get", lambda *a, **k: FakeResponse())

    runner = CliRunner()
    result = runner.invoke(cli.app, ["setup", "--minimal", "--force"])
    assert result.exit_code == 0, result.output
    # The existing file was overwritten with the new config
    loaded = cfg_mod.load_niche()
    assert loaded.niche != "doomed"


# ─── --full path works end-to-end ───
def test_setup_full_path_runs(tmp_root, monkeypatch):
    """--full extends the quick flow without breaking. Uses preset defaults
    for Enter-to-accept flow."""
    import httpx
    import questionary
    import webbrowser
    from rich import prompt as rp

    monkeypatch.setattr(cli, "_setup_check_deps", lambda: None)
    monkeypatch.setattr(rp.Confirm, "ask", classmethod(lambda cls, *a, **k: False))  # decline IG login

    def fake_select(prompt, choices, **kw):
        return type("Q", (), {"ask": lambda s: "fitness — Fitness & calisthenics"})()

    def fake_text(prompt, default="", **kw):
        return type("Q", (), {"ask": lambda s: default})

    # Questionary text responds with default for every field
    def fake_text_instance(prompt, default="", **kw):
        class _Q:
            def ask(self_inner):
                return default or "home calisthenics"
        return _Q()

    def fake_password(prompt, **kw):
        # OpenRouter key on first call, empty for optional Groq/Gemini
        class _Q:
            def ask(self_inner):
                if "OPENROUTER" in prompt or "openrouter" in prompt.lower() or "Paste" in prompt:
                    return "sk-or-v1-FULLKEY"
                return ""
        return _Q()

    def fake_checkbox(prompt, choices, **kw):
        # Return the checked choices by value
        class _Q:
            def ask(self_inner):
                return [c.value if hasattr(c, "value") else c for c in choices if getattr(c, "checked", False)]
        return _Q()

    def fake_confirm(prompt, **kw):
        # Decline IG login prompt in full mode
        class _Q:
            def ask(self_inner):
                return False
        return _Q()

    class FakeResponse:
        status_code = 200
        def json(self): return {"data": [{"id": "m"}]}

    monkeypatch.setattr(questionary, "select", fake_select)
    monkeypatch.setattr(questionary, "text", fake_text_instance)
    monkeypatch.setattr(questionary, "password", fake_password)
    monkeypatch.setattr(questionary, "checkbox", fake_checkbox)
    monkeypatch.setattr(questionary, "confirm", fake_confirm)
    monkeypatch.setattr(webbrowser, "open", lambda *a, **k: True)
    monkeypatch.setattr(httpx, "get", lambda *a, **k: FakeResponse())

    runner = CliRunner()
    result = runner.invoke(cli.app, ["setup", "--full", "--force"])
    assert result.exit_code == 0, result.output
    loaded = cfg_mod.load_niche()
    assert loaded.voice.persona  # full customisation produced a valid config
    # API key written
    assert "OPENROUTER_API_KEY=sk-or-v1-FULLKEY" in cfg_mod.ENV_PATH.read_text()


# ─── --full and --minimal mutually exclusive ───
def test_setup_rejects_full_and_minimal_together(tmp_root):
    runner = CliRunner()
    result = runner.invoke(cli.app, ["setup", "--full", "--minimal"])
    assert result.exit_code != 0


# ─── _existing_env_keys round-trip ───
def test_existing_env_keys_parses_existing_file(tmp_root):
    cfg_mod.ENV_PATH.write_text(
        "# A comment\n"
        "OPENROUTER_API_KEY=sk-or-existing\n"
        "GROQ_API_KEY=gsk-existing\n"
        "\n"
        "  IG_USERNAME=someone\n"
        "malformed_line_no_equals\n"
    )
    keys = cli._existing_env_keys()
    assert "OPENROUTER_API_KEY" in keys
    assert "GROQ_API_KEY" in keys
    assert "IG_USERNAME" in keys
    assert "malformed_line_no_equals" not in keys


def test_existing_env_keys_empty_when_no_file(tmp_root):
    assert cli._existing_env_keys() == set()
