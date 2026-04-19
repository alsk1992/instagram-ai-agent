"""Event calendar — Nager.Date holidays + user-defined niche dates.

What it does
============
Every cycle this module collects:

  1. **Public holidays** for the configured ``holiday_country`` over the
     next ``events_lookahead_days`` from the Nager.Date REST API
     (https://date.nager.at — MIT-licensed, no key required, free).
  2. **User-defined dates** from ``niche.yaml.events_calendar`` —
     things like product launches, brand anniversaries, niche-specific
     observances.

For every event within the lookahead window we push a priority-4 entry
to ``context_feed`` so the generator sees it alongside trend signals.
Each event is debounced per-day via the ``state`` K/V so the same entry
doesn't flood the context every cycle.

Everything is graceful-no-op on network / parse / state failures.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

import httpx

from instagram_ai_agent.core import db
from instagram_ai_agent.core.config import NicheConfig
from instagram_ai_agent.core.logging_setup import get_logger

log = get_logger(__name__)


NAGER_URL = "https://date.nager.at/api/v3/PublicHolidays/{year}/{country}"

_SEEN_KEY = "events_seen"   # dict[event_id -> last_push_date (YYYY-MM-DD)]


# ─── Data ───
@dataclass(frozen=True)
class Event:
    date: date
    label: str
    source: str     # "holiday" | "user"
    note: str = ""

    @property
    def event_id(self) -> str:
        """Stable ID for dedup: YYYY-MM-DD:source:slugified-label."""
        slug = re.sub(r"[^a-z0-9]+", "-", self.label.lower()).strip("-")[:40] or "unknown"
        return f"{self.date.isoformat()}:{self.source}:{slug}"


# ─── Sources ───
async def fetch_holidays(country: str, year: int) -> list[Event]:
    """Call Nager.Date for a given year. Returns ``[]`` on any failure."""
    if not country:
        return []
    url = NAGER_URL.format(year=year, country=country.upper())
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
            r = await client.get(url)
            r.raise_for_status()
            payload = r.json()
    except Exception as e:
        log.warning("events: nager fetch failed (%s, %s): %s", country, year, e)
        return []

    out: list[Event] = []
    for item in payload or []:
        raw_date = item.get("date")
        label = item.get("localName") or item.get("name") or "holiday"
        if not raw_date:
            continue
        try:
            d = date.fromisoformat(raw_date)
        except ValueError:
            continue
        out.append(Event(date=d, label=str(label), source="holiday"))
    return out


def user_events(cfg: NicheConfig) -> list[Event]:
    """Parse ``cfg.events_calendar``. Invalid entries are skipped, not raised."""
    out: list[Event] = []
    for entry in cfg.events_calendar or []:
        raw_date = str(entry.get("date") or "").strip()
        label = str(entry.get("label") or "").strip()
        note = str(entry.get("note") or "").strip()
        if not (raw_date and label):
            continue
        try:
            d = date.fromisoformat(raw_date)
        except ValueError:
            log.debug("events: ignoring bad user date %r", raw_date)
            continue
        out.append(Event(date=d, label=label, source="user", note=note))
    return out


# ─── Window / dedup ───
def _today_utc() -> date:
    return datetime.now(UTC).date()


def in_window(events: list[Event], *, today: date, days: int) -> list[Event]:
    """Return events whose date is in [today, today+days). Sorted ascending."""
    end = today + timedelta(days=days)
    filtered = [e for e in events if today <= e.date < end]
    filtered.sort(key=lambda e: (e.date, e.label))
    return filtered


# Re-push policy: a user-added "Launch in 14 days" must not push 14 times
# across 14 days. We push once when the event first enters the lookahead
# window, then re-push only on the two days immediately before (T-1, T-0).
# Tweak here if niche research changes that pattern.
REPUSH_LEAD_DAYS = {0, 1}  # same-day + day-before re-push


def _seen() -> dict[str, dict]:
    """State store: {event_id: {"event_date": "YYYY-MM-DD", "pushes": [iso-dates]}}.

    Older schema (str value = last-push-date) is migrated transparently.
    """
    raw = db.state_get_json(_SEEN_KEY, default={}) or {}
    if not isinstance(raw, dict):
        return {}
    migrated: dict[str, dict] = {}
    for k, v in raw.items():
        if isinstance(v, dict):
            migrated[k] = v
        elif isinstance(v, str):
            # Old schema — infer event_date from the key prefix
            event_date = k.split(":", 1)[0] if ":" in k else v
            migrated[k] = {"event_date": event_date, "pushes": [v]}
    return migrated


def _record_seen(events: list[Event]) -> None:
    store = _seen()
    today = _today_utc()
    today_iso = today.isoformat()
    for e in events:
        rec = store.get(e.event_id) or {"event_date": e.date.isoformat(), "pushes": []}
        if today_iso not in rec["pushes"]:
            rec["pushes"] = (rec.get("pushes") or []) + [today_iso]
        rec["event_date"] = e.date.isoformat()
        store[e.event_id] = rec

    # Prune entries whose EVENT date has passed — by event_date, not push_date.
    # (The old prune used push_date, so events re-entered "fresh" every 3 days.)
    cutoff = (today - timedelta(days=3)).isoformat()
    store = {
        k: v for k, v in store.items()
        if str(v.get("event_date") or "") >= cutoff
    }
    db.state_set_json(_SEEN_KEY, store)


def _fresh(events: list[Event]) -> list[Event]:
    """Filter events to those worth pushing right now.

    Policy:
      * If we have never pushed this event → push (first sighting).
      * If today is within ``REPUSH_LEAD_DAYS`` of the event AND we haven't
        already pushed today → push.
      * Otherwise → skip.
    """
    store = _seen()
    today = _today_utc()
    today_iso = today.isoformat()
    fresh: list[Event] = []
    for e in events:
        rec = store.get(e.event_id)
        pushed = set((rec or {}).get("pushes") or [])
        if today_iso in pushed:
            continue   # already pushed today
        if not rec:
            fresh.append(e)      # first time we've seen it
            continue
        lead = (e.date - today).days
        if lead in REPUSH_LEAD_DAYS:
            fresh.append(e)      # countdown re-push (T-1 / T-0)
    return fresh


# ─── Orchestrator entry ───
async def upcoming(cfg: NicheConfig) -> list[Event]:
    """Combined + sorted list of all holidays + user events within the
    configured lookahead window. Does not touch the context feed."""
    today = _today_utc()
    days = cfg.events_lookahead_days

    events: list[Event] = list(user_events(cfg))

    if cfg.holidays_enabled and cfg.holiday_country:
        # Lookahead may cross a year boundary — query both years to be safe.
        years = {today.year, (today + timedelta(days=days)).year}
        for year in sorted(years):
            events.extend(await fetch_holidays(cfg.holiday_country, year))

    return in_window(events, today=today, days=days)


async def run_once(cfg: NicheConfig) -> int:
    """One cycle. Push fresh upcoming events to ``context_feed``.
    Returns number of context entries pushed."""
    upcoming_events = await upcoming(cfg)
    to_push = _fresh(upcoming_events)
    for e in to_push:
        lead = (e.date - _today_utc()).days
        when = (
            "today" if lead == 0
            else "tomorrow" if lead == 1
            else f"in {lead} days"
        )
        note = f" — {e.note}" if e.note else ""
        db.push_context(
            f"event.{e.source}",
            f"event {when}: {e.label}{note} ({e.date.isoformat()})",
            priority=4,
        )
    if to_push:
        _record_seen(to_push)
        log.info("events: %d upcoming pushed to context", len(to_push))
    return len(to_push)
