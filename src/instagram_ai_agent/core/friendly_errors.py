"""Human-friendly error wrapper for CLI entrypoints.

Swaps Python tracebacks (useless to 80% of users) for one-line plain-
English explanations + actionable next steps. Falls back to the full
traceback when ``IG_DEBUG=1`` is set so power users can still see
everything. Catches the exceptions the CLI actually hits:

  * Missing API keys / env vars
  * Network / DNS / SSL errors
  * Playwright / ffmpeg binaries missing
  * Pydantic validation failures on niche.yaml
  * instagrapi challenge / login failures
  * SQLite brain.db locks
  * File-not-found for expected assets

Usage: wrap every Typer command body, or more cheaply, decorate main().
"""
from __future__ import annotations

import functools
import os
import sys
import traceback
from collections.abc import Callable
from typing import Any

# Mapping of (exception_type_name, substring in error message) → friendly handler.
# Ordered — first match wins.
_FRIENDLY_RULES: list[tuple[str, list[str], str, str]] = [
    # (type_name_regex, message-substrings, headline, action)
    (
        "FileNotFoundError",
        ["niche.yaml"],
        "niche.yaml isn't set up yet.",
        "Run: ig-agent init",
    ),
    (
        "FileNotFoundError",
        ["ffmpeg"],
        "ffmpeg isn't installed or isn't on PATH.",
        "macOS: brew install ffmpeg  ·  Ubuntu: sudo apt install ffmpeg  ·  Windows: winget install Gyan.FFmpeg",
    ),
    (
        "FileNotFoundError",
        ["playwright", "chromium"],
        "Playwright chromium isn't installed.",
        "Run: python -m playwright install chromium",
    ),
    (
        "ModuleNotFoundError",
        ["playwright"],
        "Playwright isn't installed.",
        "Run: pip install -e . (or `pipx inject instagram-ai-agent playwright`)",
    ),
    (
        "RuntimeError",
        ["OPENROUTER_API_KEY", "no provider", "No API key"],
        "No LLM provider API key is configured.",
        "Get a free key at https://openrouter.ai/keys and add OPENROUTER_API_KEY to .env",
    ),
    (
        "RuntimeError",
        ["IG_USERNAME", "IG_PASSWORD"],
        "Instagram credentials are missing.",
        "Run `ig-agent init` (or fill IG_USERNAME + IG_PASSWORD in .env)",
    ),
    (
        "ValidationError",
        [],
        "Your niche.yaml has a config error.",
        "Edit the file or rerun `ig-agent init`. Full details below.",
    ),
    (
        "ChallengeNeedsManualCode",
        [],
        "Instagram is asking for a verification code.",
        "Either set IMAP_HOST/USER/PASS in .env so the agent auto-reads it, OR run `ig-agent login` interactively and paste the code.",
    ),
    (
        "ChallengeRequired",
        [],
        "Instagram flagged this login as suspicious.",
        "Wait 24h before retrying. In the meantime: add a residential proxy (IG_PROXY) or paste browser cookies (IG_SESSIONID etc.) into .env. See README → Safety & Anti-Detection.",
    ),
    (
        "LoginRequired",
        [],
        "Your Instagram session is dead.",
        "Delete data/sessions/<username>.json and rerun `ig-agent login`.",
    ),
    (
        "BadPassword",
        [],
        "Instagram rejected the password.",
        "Check IG_PASSWORD in .env. If it's correct, IG may be blocking new-IP logins — add a residential proxy.",
    ),
    (
        "BackoffActive",
        [],
        "The agent is in a cooldown window.",
        "Run `ig-agent status` to see when it clears. Usually self-resolves within 24h.",
    ),
    (
        "OperationalError",
        ["database is locked"],
        "brain.db is locked by another process.",
        "Stop any other running `ig-agent` processes, or delete data/brain.db-wal if stuck.",
    ),
    (
        "ConnectionError",
        [],
        "Network request failed.",
        "Check your internet. If you're on a proxy, verify IG_PROXY / HTTP(S)_PROXY.",
    ),
    (
        "ReadTimeout",
        [],
        "A network request timed out.",
        "Usually transient — retry. If persistent, your proxy or IG's edge is slow from your network.",
    ),
]


def _match_rule(exc: BaseException) -> tuple[str, str] | None:
    name = type(exc).__name__
    msg = str(exc).lower()
    for type_pattern, substrings, headline, action in _FRIENDLY_RULES:
        if name != type_pattern and type_pattern.lower() not in name.lower():
            continue
        if substrings and not any(s.lower() in msg for s in substrings):
            continue
        return headline, action
    return None


def _format_error(exc: BaseException) -> str:
    match = _match_rule(exc)
    if match is None:
        headline = f"Unexpected error: {type(exc).__name__}"
        action = "Run `ig-agent doctor` to diagnose, or set IG_DEBUG=1 for the full traceback."
    else:
        headline, action = match
    out = [
        "",
        f"✗  {headline}",
        f"   {action}",
    ]
    # Show the raw message (one line, truncated) so users have a
    # fingerprint to search against.
    raw = str(exc).strip().replace("\n", " ")
    if raw and raw.lower() not in action.lower():
        out.append(f"   [{type(exc).__name__}: {raw[:180]}]")
    return "\n".join(out)


def wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator — swallows unexpected exceptions, prints a friendly
    summary, and exits 1. Set IG_DEBUG=1 to get the full traceback."""
    @functools.wraps(fn)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:
            if os.environ.get("IG_DEBUG", "").strip():
                traceback.print_exc()
            else:
                print(_format_error(exc), file=sys.stderr)
                print(
                    "\n   (set IG_DEBUG=1 for the full traceback)",
                    file=sys.stderr,
                )
            sys.exit(1)
    return _wrapper
