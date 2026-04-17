"""Account warmup protocol — 14-day ramp on fresh accounts.

Days 1-3  : no posts, no DMs, ≤20% of daily budgets (mostly just scrolling).
Days 4-7  : no posts, no DMs, ≤35% of daily budgets.
Days 8-10 : 1 post, stories ok, ≤60% of daily budgets, still no DMs.
Days 11-14: 1 post, stories + DMs unlocked, ≤80% of daily budgets.
Day 15+   : full budgets.

The agent's first login writes ``warmup_start`` to the state K/V. From then on
every budget check consults :func:`effective_caps` which scales every action.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone

from src.core import db
from src.core.config import NicheConfig


@dataclass(frozen=True)
class WarmupPhase:
    day_range: tuple[int, int]
    multiplier: float
    allow_posts: bool
    allow_dms: bool
    label: str


_PHASES: tuple[WarmupPhase, ...] = (
    WarmupPhase((1, 3),   0.20, allow_posts=False, allow_dms=False, label="silent"),
    WarmupPhase((4, 7),   0.35, allow_posts=False, allow_dms=False, label="lurk"),
    WarmupPhase((8, 10),  0.60, allow_posts=True,  allow_dms=False, label="post-safe"),
    WarmupPhase((11, 14), 0.80, allow_posts=True,  allow_dms=True,  label="full-ramp"),
)

_WARMUP_KEY = "warmup_start"


def ensure_started() -> None:
    """Record today as the warmup start if not already set."""
    if db.state_get(_WARMUP_KEY) is None:
        db.state_set(_WARMUP_KEY, _today_iso())


def started_on() -> date | None:
    raw = db.state_get(_WARMUP_KEY)
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw).date()
    except ValueError:
        return None


def current_day() -> int | None:
    """Days since warmup started (day 1 = start date). Returns None if unstarted."""
    start = started_on()
    if start is None:
        return None
    delta = (date.today() - start).days
    return delta + 1  # day 1 is inception


def current_phase() -> WarmupPhase | None:
    d = current_day()
    if d is None:
        return None
    if d > 14:
        return None
    for phase in _PHASES:
        lo, hi = phase.day_range
        if lo <= d <= hi:
            return phase
    return _PHASES[-1]


@dataclass(frozen=True)
class EffectiveBudget:
    caps: dict[str, int]
    allow_posts: bool
    allow_dms: bool
    multiplier: float
    day: int | None
    phase_label: str


def effective_caps(cfg: NicheConfig) -> EffectiveBudget:
    """Return the current scaled caps. During warmup, budgets shrink & some actions disable."""
    raw_caps = {
        "like": cfg.budget.likes,
        "follow": cfg.budget.follows,
        "unfollow": cfg.budget.unfollows,
        "comment": cfg.budget.comments,
        "dm": cfg.budget.dms,
        "story_view": cfg.budget.story_views,
        "post": cfg.schedule.posts_per_day,
        "story_post": cfg.schedule.stories_per_day,
    }
    phase = current_phase()
    day = current_day()
    if phase is None:
        return EffectiveBudget(
            caps=raw_caps,
            allow_posts=True,
            allow_dms=True,
            multiplier=1.0,
            day=day,
            phase_label="complete" if day else "not_started",
        )
    scaled: dict[str, int] = {}
    for action, cap in raw_caps.items():
        if action == "post" and not phase.allow_posts:
            scaled[action] = 0
        elif action == "dm" and not phase.allow_dms:
            scaled[action] = 0
        elif action == "story_post" and not phase.allow_posts:
            # Treat stories as posts during silent/lurk phases
            scaled[action] = 0
        else:
            scaled[action] = max(0, int(cap * phase.multiplier))
    return EffectiveBudget(
        caps=scaled,
        allow_posts=phase.allow_posts,
        allow_dms=phase.allow_dms,
        multiplier=phase.multiplier,
        day=day,
        phase_label=phase.label,
    )


def _today_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
