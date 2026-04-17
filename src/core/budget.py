"""Daily action budgeter — caps actions per day below IG's unknown thresholds.

Budgets are scaled by the warmup phase (see :mod:`src.core.warmup`).
"""
from __future__ import annotations

from src.core import db
from src.core.config import NicheConfig
from src.core.warmup import effective_caps


def allowed(action: str, cfg: NicheConfig) -> tuple[bool, int, int]:
    """Return (allowed, used_today, cap) with warmup multipliers applied."""
    budget = effective_caps(cfg)
    cap = budget.caps.get(action)
    if cap is None:
        return True, 0, 10**9
    used = db.action_count_today(action)
    return used < cap, used, cap
