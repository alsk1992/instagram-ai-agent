"""Aged-account safety gates — rest period + profile-metadata freeze.

Two time-gated guards purpose-built for bought/aged-account operation:

* **Rest period** (``IG_REST_UNTIL``) — after importing a sold session,
  the session needs to "sit" on the buyer's IP for 24-72h before any
  write-path actions. During rest: reads + keep-alive pings only.
  Triggering posts/likes/follows/DMs from a new ASN on the first few
  hours post-sale is the #1 reported burn cause per 2026 operator
  threads (lolz.guru ColdPVA megathread, MP Social aged-account sub).

* **Profile-metadata freeze** (``IG_FREEZE_PROFILE_UNTIL``) — for 21
  days after setup, block any profile-editing endpoint (avatar, bio,
  password, 2FA toggle). IG's 2026 risk model treats these as
  ownership-change smoking-gun signals; frozen profile metadata lets
  the session "settle" as the prior owner's identity.

Both gates are configured via ISO-8601 UTC timestamps in ``.env``, read
once per call (so manual edits take effect without restart), and return
cleanly parseable ``GateStatus`` objects the CLI + orchestrator consume.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


@dataclass(frozen=True)
class GateStatus:
    """Returned by rest/freeze checks.

    ``active`` is True when the gate is currently blocking. ``remaining``
    is the timedelta until it lifts; None when no gate is configured.
    ``until`` is the configured ISO timestamp (for display).
    """
    active: bool
    remaining: timedelta | None
    until: str | None

    @property
    def remaining_hours(self) -> float | None:
        if self.remaining is None:
            return None
        return self.remaining.total_seconds() / 3600.0


def _parse_until(raw: str) -> datetime | None:
    """Accept ISO-8601 with or without 'Z' suffix, return UTC datetime."""
    s = raw.strip()
    if not s:
        return None
    # Normalise trailing Z → +00:00 so fromisoformat accepts it
    s_norm = s[:-1] + "+00:00" if s.endswith("Z") else s
    try:
        dt = datetime.fromisoformat(s_norm)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _gate_status(env_key: str) -> GateStatus:
    raw = os.environ.get(env_key, "").strip()
    until = _parse_until(raw)
    if until is None:
        return GateStatus(active=False, remaining=None, until=None)
    now = datetime.now(timezone.utc)
    remaining = until - now
    return GateStatus(
        active=remaining > timedelta(0),
        remaining=remaining if remaining > timedelta(0) else None,
        until=raw,
    )


def rest_status() -> GateStatus:
    """Current state of the post-purchase rest gate."""
    return _gate_status("IG_REST_UNTIL")


def freeze_status() -> GateStatus:
    """Current state of the profile-metadata freeze gate."""
    return _gate_status("IG_FREEZE_PROFILE_UNTIL")


def writes_blocked() -> bool:
    """True when the rest gate is active. Callers (poster, engager, DM
    worker) should no-op and log a single debug line when this returns
    True — don't burn budget pushing actions that get rejected downstream."""
    return rest_status().active


def profile_edits_blocked() -> bool:
    """True when the profile-freeze gate is active. Callers that edit
    avatar/bio/password/2FA should hard-fail loudly — these endpoints
    would succeed at IG but are ownership-change smoking-gun flags."""
    return freeze_status().active


def suggest_rest_until(hours: int = 48) -> str:
    """ISO-8601 UTC timestamp N hours from now. Helper for setup wizard
    to pre-fill a sensible default."""
    when = datetime.now(timezone.utc) + timedelta(hours=hours)
    return when.strftime("%Y-%m-%dT%H:%M:%SZ")


def suggest_freeze_until(days: int = 21) -> str:
    when = datetime.now(timezone.utc) + timedelta(days=days)
    return when.strftime("%Y-%m-%dT%H:%M:%SZ")
