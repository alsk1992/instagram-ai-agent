"""Friendly-error wrapper — replaces Python tracebacks with one-line
actionable messages for common first-run failures."""
from __future__ import annotations

import pytest

from instagram_ai_agent.core import friendly_errors as fe


def test_wrap_passes_through_normal_returns():
    @fe.wrap
    def ok():
        return 42
    assert ok() == 42


def test_wrap_preserves_keyboardinterrupt():
    """KeyboardInterrupt must never be swallowed — users need Ctrl+C."""
    @fe.wrap
    def cancel():
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        cancel()


def test_wrap_converts_exception_to_exit(monkeypatch: pytest.MonkeyPatch):
    @fe.wrap
    def bomb():
        raise RuntimeError("OPENROUTER_API_KEY not set")

    # IG_DEBUG off → prints friendly message + exits 1
    monkeypatch.delenv("IG_DEBUG", raising=False)
    with pytest.raises(SystemExit) as exc:
        bomb()
    assert exc.value.code == 1


def test_wrap_falls_through_when_debug_set(monkeypatch, capsys):
    @fe.wrap
    def bomb():
        raise RuntimeError("raw debug trace wanted")

    monkeypatch.setenv("IG_DEBUG", "1")
    with pytest.raises(SystemExit):
        bomb()
    out = capsys.readouterr()
    # With IG_DEBUG=1 the full traceback is printed to stderr
    assert "Traceback" in out.err or "RuntimeError" in out.err


@pytest.mark.parametrize("exc_class,msg,expected_substring", [
    (FileNotFoundError, "niche.yaml not found", "init"),
    (FileNotFoundError, "ffmpeg command not found", "ffmpeg"),
    (ModuleNotFoundError, "No module named 'playwright'", "playwright"),
    (RuntimeError, "OPENROUTER_API_KEY not set", "openrouter"),
    (RuntimeError, "IG_USERNAME missing", "init"),
])
def test_rules_match_known_failure_modes(exc_class, msg, expected_substring):
    try:
        raise exc_class(msg)
    except Exception as e:
        formatted = fe._format_error(e)
        assert expected_substring.lower() in formatted.lower()


def test_unknown_exception_falls_back_to_doctor_hint():
    try:
        raise ValueError("some very obscure internal condition")
    except Exception as e:
        formatted = fe._format_error(e)
        assert "doctor" in formatted.lower() or "IG_DEBUG" in formatted


def test_rules_respect_substring_filter():
    """A ``FileNotFoundError`` not mentioning niche.yaml / ffmpeg /
    playwright must NOT trip those specific rules — they fall through
    to the generic unknown-exception path."""
    try:
        raise FileNotFoundError("some/random/file.txt")
    except Exception as e:
        formatted = fe._format_error(e)
        # Should not be the niche.yaml / ffmpeg / playwright rule
        assert "niche.yaml" not in formatted.lower()
        assert "brew install ffmpeg" not in formatted.lower()
