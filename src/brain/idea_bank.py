"""Idea bank — curated archetypes + hook formulas the pipeline draws from.

Instead of re-deriving "what kind of post is this?" per cycle, we pre-load
a library of post archetypes (hot take, listicle, myth bust, etc.) and
have the pipeline pick one at generation time. Each archetype carries:

  * ``archetype``     — internal label (contrarian_hot_take, listicle_rules…)
  * ``hook_formula``  — a parametric hook template (with `{placeholders}`)
  * ``format_hint``   — which post format best fits (meme | carousel | reel | any)
  * ``body_template`` — short prose instruction the LLM injects into its prompt
  * ``niche_tags``    — JSON list; empty ⇒ universal
  * ``license``       — CC0 / Apache-2.0 / CC-BY etc. (ingested from source)

The **picker** avoids archetypes used in the last N picks so the feed stays
varied, then weighted-random-picks by ``score`` (updated by retro performance
analysis in the future).

Sources:
  1. ``data/ideas/seed.json`` — curated CC0 library shipped with the repo.
  2. ``--fetch`` option on the CLI pulls from permissive public repos:
       - MaxsPrompts/Marketing-Prompts (Apache-2.0, CSV)
       - f/awesome-chatgpt-prompts (CC0, CSV)
     These are optional and sit behind the ``seed-idea-bank --fetch`` flag.
"""
from __future__ import annotations

import csv
import io
import json
import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import httpx

from src.core import db
from src.core.config import DATA_DIR, NicheConfig
from src.core.logging_setup import get_logger

log = get_logger(__name__)

SEED_PATH = DATA_DIR / "ideas" / "seed.json"

# External corpora — fetched only when user opts in.
# Marketing-Prompts was flagged dead (404) during audit, removed until a
# canonical URL is re-confirmed. Add entries here as they're verified.
EXTERNAL_SOURCES: dict[str, dict] = {
    "awesome-chatgpt": {
        "url": "https://raw.githubusercontent.com/f/awesome-chatgpt-prompts/main/prompts.csv",
        "license": "CC0",
        "columns": {"prompt": "prompt", "archetype": "act"},
    },
}


@dataclass(frozen=True)
class Idea:
    id: int | None
    archetype: str
    hook_formula: str
    format_hint: str
    niche_tags: list[str]
    body_template: str
    source: str
    license: str
    use_count: int = 0
    last_used_at: str | None = None
    score: float = 0.5


def is_commercial_license(license_str: str | None) -> bool:
    """Defence-in-depth license check.

    Return True only when the licence is a KNOWN commercial-safe flavour.
    Unknown / malformed / research-only licences return False.
    Accepts common phrasings: "CC0", "CC-BY", "CC BY 4.0", "Apache-2.0",
    "MIT", "BSD-3-Clause", "Unlicense", "public-domain", "pixabay" (custom
    content licence permits commercial use), "user-declared" (caller knows).
    Rejects: any string containing NC, non-commercial, research, S-Lab,
    OpenRAIL behavioural restrictions, Coqui Public Model Licence, Llama.
    """
    s = (license_str or "").strip()
    if not s:
        return False
    upper = s.upper()
    # Fast rejects — substring match on the full string
    REJECT_SUBSTRINGS = (
        "NC",                      # CC-BY-NC, CC BY-NC-SA, CC-BY-NC-ND
        "NON-COMMERCIAL",
        "NON COMMERCIAL",
        "NONCOMMERCIAL",
        "RESEARCH",                # research-only, research licence
        "S-LAB",                   # S-Lab Licence (CodeFormer)
        "COQUI PUBLIC",            # Coqui weights
        "LLAMA 2 COMMUNITY",
        "LLAMA COMMUNITY",
        "OPENRAIL-M",              # OpenRAIL-M has behavioural restrictions
        "CREATIVEML OPEN RAIL++-M",  # OK for commercial, but includes 'RAIL'
    )
    # Some OpenRAIL variants are actually commercial-OK (CreativeML Open
    # RAIL++-M allows commercial with behavioural restrictions). Whitelist
    # it after the blocklist sweep.
    OPENRAIL_COMMERCIAL_OK = ("CREATIVEML OPEN RAIL++-M",)
    is_openrail_ok = any(ok in upper for ok in OPENRAIL_COMMERCIAL_OK)
    if any(bad in upper for bad in REJECT_SUBSTRINGS) and not is_openrail_ok:
        return False

    # Accept allowlist — any of these markers in the string → commercial OK.
    ACCEPT_MARKERS = (
        "CC0",
        "CC-BY", "CC BY",          # CC-BY, CC-BY-SA, CC-BY 4.0
        "APACHE",
        "MIT",
        "BSD",
        "UNLICENSE",
        "PUBLIC DOMAIN", "PUBLIC-DOMAIN",
        "PIXABAY",
        "USER-DECLARED", "USER DECLARED",
        "OFL",                     # SIL Open Font License
        "ISC",
        "ZLIB",
    )
    # Reject CC-BY-NC etc. that also contain "CC-BY"
    if "BY-NC" in upper or "BY NC" in upper:
        return False
    return any(marker in upper for marker in ACCEPT_MARKERS) or is_openrail_ok


def _row_to_idea(row: sqlite3.Row) -> Idea:
    d = dict(row)
    return Idea(
        id=d.get("id"),
        archetype=d["archetype"],
        hook_formula=d["hook_formula"],
        format_hint=d.get("format_hint") or "any",
        niche_tags=json.loads(d.get("niche_tags") or "[]"),
        body_template=d.get("body_template") or "",
        source=d.get("source") or "curated",
        license=d.get("license") or "CC0",
        use_count=int(d.get("use_count") or 0),
        last_used_at=d.get("last_used_at"),
        score=float(d.get("score") or 0.5),
    )


# ─── Seeding ───
def seed_from_file(path: Path | None = None, *, source: str = "curated") -> int:
    """Populate the ``ideas`` table from the shipped JSON file. Idempotent."""
    p = path or SEED_PATH
    if not p.exists():
        log.warning("idea bank seed missing at %s", p)
        return 0
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    license_hint = str(data.get("_license") or "CC0").strip()
    rows = data.get("ideas") or []
    return _insert_many(rows, default_source=source, default_license=license_hint)


def seed_from_external(which: str) -> int:
    """Pull an external permissive corpus and insert. Safe to call repeatedly."""
    if which not in EXTERNAL_SOURCES:
        raise ValueError(f"unknown external source: {which!r}")
    meta = EXTERNAL_SOURCES[which]
    with httpx.Client(timeout=httpx.Timeout(60.0)) as c:
        r = c.get(meta["url"])
        r.raise_for_status()
    reader = csv.DictReader(io.StringIO(r.text))
    mapping = meta["columns"]
    rows: list[dict] = []
    for raw in reader:
        archetype = (raw.get(mapping["archetype"]) or "").strip()
        hook = (raw.get(mapping["prompt"]) or "").strip()
        if not (archetype and hook):
            continue
        rows.append({
            "archetype": archetype.lower().replace(" ", "_")[:80],
            "hook_formula": hook[:400],
            "format_hint": "any",
            "body_template": hook,
        })
    return _insert_many(rows, default_source=which, default_license=meta["license"])


def _insert_many(rows: Iterable[dict], *, default_source: str, default_license: str) -> int:
    """Insert only rows that don't already exist. Returns the real number of
    newly-inserted rows (NOT the number offered). Uses ``cursor.rowcount``
    which for ``INSERT OR IGNORE`` reports 1 on a real insert, 0 on a
    silent-skip — the only reliable counter here.
    """
    conn = db.get_conn()
    inserted = 0
    for r in rows:
        archetype = (r.get("archetype") or "").strip()
        hook = (r.get("hook_formula") or "").strip()
        if not (archetype and hook):
            continue
        format_hint = (r.get("format_hint") or "any").strip().lower()
        body = (r.get("body_template") or "").strip()
        tags = r.get("niche_tags") or []
        src = (r.get("source") or default_source).strip()
        lic = (r.get("license") or default_license).strip()
        cur = conn.execute(
            """
            INSERT OR IGNORE INTO ideas
              (archetype, hook_formula, format_hint, niche_tags,
               body_template, source, license)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (archetype, hook, format_hint, json.dumps(tags), body, src, lic),
        )
        # rowcount: 1 if the INSERT happened, 0 if the UNIQUE ignored it.
        if cur.rowcount == 1:
            inserted += 1
    return inserted


def count() -> int:
    row = db.get_conn().execute("SELECT COUNT(*) c FROM ideas").fetchone()
    return int(row["c"]) if row else 0


# ─── Picking ───
_RECENT_PICK_KEY = "idea_bank_recent_picks"
RECENT_WINDOW = 14


def _recent_picks() -> list[int]:
    return db.state_get_json(_RECENT_PICK_KEY, default=[]) or []


def _push_recent(idea_id: int) -> None:
    recent = _recent_picks()
    recent = [idea_id] + [i for i in recent if i != idea_id]
    recent = recent[:RECENT_WINDOW]
    db.state_set_json(_RECENT_PICK_KEY, recent)


def is_contrarian_archetype(archetype: str) -> bool:
    """True when an archetype name marks it as a contrarian / hot-take
    hook. Used by the pipeline to bias selection when contrarian mode
    fires. Covers the explicit ``contrarian_*`` family plus ``trend_contra``
    and ``myth_bust_*`` (myth-busting is a contrarian flavour)."""
    a = (archetype or "").lower()
    return a.startswith("contrarian_") or a == "trend_contra" or a.startswith("myth_bust_")


def pick_for(
    cfg: NicheConfig,
    *,
    format_name: str,
    allow_any: bool = True,
    commercial_only: bool = True,
    prefer_contrarian: bool = False,
) -> Idea | None:
    """Pick one idea appropriate to ``format_name``.

    * Filters by format compatibility (exact match or ``any``).
    * Excludes rows used in the last ``RECENT_WINDOW`` picks.
    * When ``commercial_only`` (default), refuses CC-BY-NC / research-only rows.
    * Weighted-random by ``score``.
    * When ``prefer_contrarian``, boosts contrarian archetypes' weights
      4× so they win the dice roll most of the time (with a small chance
      of falling back to a non-contrarian hook if no contrarian ideas
      survive the format + recency filters).
    """
    conn = db.get_conn()
    recent = _recent_picks()
    recent_sql = ""
    params: list = []
    if recent:
        qs = ",".join("?" * len(recent))
        recent_sql = f" AND id NOT IN ({qs})"
        params.extend(recent)

    format_clause = "format_hint = ?"
    params_fmt = [format_name]
    if allow_any:
        format_clause = "(format_hint = ? OR format_hint = 'any')"
    # Commercial filter: done in Python to avoid the shallow-blocklist SQL
    # trap (misses "CC BY-NC 4.0", "S-Lab License", OpenRAIL clauses, etc.).
    sql = f"SELECT * FROM ideas WHERE {format_clause}{recent_sql}"
    rows = conn.execute(sql, params_fmt + params).fetchall()
    if not rows:
        rows = conn.execute(
            f"SELECT * FROM ideas WHERE {format_clause}",
            params_fmt,
        ).fetchall()
    if commercial_only:
        rows = [r for r in rows if is_commercial_license(r["license"])]
    if not rows:
        return None

    ideas = [_row_to_idea(r) for r in rows]
    # Score-weighted random; ensure min weight > 0
    weights = [max(0.05, i.score) for i in ideas]
    if prefer_contrarian:
        # 4× boost to contrarian archetypes. If nothing matches, the
        # original distribution still holds — a random pick from the
        # non-contrarian pool is preferable to returning None because
        # the caller asked for contrarian framing in the prompts too.
        weights = [
            w * (4.0 if is_contrarian_archetype(i.archetype) else 1.0)
            for w, i in zip(weights, ideas, strict=True)
        ]
    chosen = random.choices(ideas, weights=weights, k=1)[0]
    return chosen


def mark_used(idea_id: int) -> None:
    db.get_conn().execute(
        """
        UPDATE ideas
        SET use_count = use_count + 1,
            last_used_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
        WHERE id = ?
        """,
        (idea_id,),
    )
    _push_recent(idea_id)


def adjust_score(idea_id: int, delta: float) -> None:
    """Retro feedback — nudges the score when a post performs (or doesn't)."""
    db.get_conn().execute(
        "UPDATE ideas SET score = max(0.0, min(1.5, score + ?)) WHERE id = ?",
        (float(delta), idea_id),
    )


# ─── Summaries (for dashboard) ───
def format_breakdown() -> list[dict]:
    rows = db.get_conn().execute(
        """
        SELECT format_hint, COUNT(*) AS n, AVG(score) AS avg_score, SUM(use_count) AS total_uses
        FROM ideas GROUP BY format_hint ORDER BY n DESC
        """
    ).fetchall()
    return [dict(r) for r in rows]


def license_breakdown() -> list[dict]:
    rows = db.get_conn().execute(
        "SELECT license, COUNT(*) c FROM ideas GROUP BY license ORDER BY c DESC"
    ).fetchall()
    return [dict(r) for r in rows]
