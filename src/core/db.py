"""SQLite brain.db — content queue, engagement queue, posts, intel, state.

Single-writer process is enforced by the orchestrator; WAL mode allows
concurrent readers (brain learner reads while orchestrator writes).
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from src.core.config import DB_PATH

SCHEMA = [
    # Content pipeline
    """
    CREATE TABLE IF NOT EXISTS content_queue (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        format        TEXT    NOT NULL,              -- meme|quote_card|carousel|reel_stock|reel_ai|photo
        status        TEXT    NOT NULL DEFAULT 'pending_review',
        caption       TEXT    NOT NULL DEFAULT '',
        hashtags      TEXT    NOT NULL DEFAULT '[]', -- JSON array
        media_paths   TEXT    NOT NULL DEFAULT '[]', -- JSON array of file paths
        phash         TEXT,                          -- perceptual hash of first media
        critic_score  REAL,
        critic_notes  TEXT,
        generator     TEXT,
        regens        INTEGER NOT NULL DEFAULT 0,
        scheduled_for TEXT,                          -- ISO-8601 UTC
        created_at    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
        posted_at     TEXT,
        ig_media_pk   TEXT,
        error         TEXT,
        meta          TEXT    NOT NULL DEFAULT '{}'  -- JSON extras
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_content_status ON content_queue(status)",
    "CREATE INDEX IF NOT EXISTS idx_content_scheduled ON content_queue(scheduled_for)",
    "CREATE INDEX IF NOT EXISTS idx_content_phash ON content_queue(phash)",
    # Posts (append-only log of what went out)
    """
    CREATE TABLE IF NOT EXISTS posts (
        ig_media_pk   TEXT PRIMARY KEY,
        content_id    INTEGER,
        format        TEXT,
        caption       TEXT,
        posted_at     TEXT NOT NULL,
        likes         INTEGER DEFAULT 0,
        comments      INTEGER DEFAULT 0,
        reach         INTEGER DEFAULT 0,
        last_checked  TEXT,
        shadowban     INTEGER DEFAULT 0,
        FOREIGN KEY (content_id) REFERENCES content_queue(id)
    )
    """,
    # Engagement queue — likes/follows/comments/story views to execute
    """
    CREATE TABLE IF NOT EXISTS engagement_queue (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        action        TEXT NOT NULL,            -- like|follow|unfollow|comment|story_view|dm
        target_user   TEXT,
        target_media  TEXT,
        payload       TEXT,                     -- JSON (e.g., comment text)
        status        TEXT NOT NULL DEFAULT 'pending',
        scheduled_for TEXT,
        created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
        attempted_at  TEXT,
        result        TEXT,
        error         TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_eng_status ON engagement_queue(status, scheduled_for)",
    # Action log (for budget enforcement)
    """
    CREATE TABLE IF NOT EXISTS action_log (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        action     TEXT NOT NULL,
        target     TEXT,
        result     TEXT NOT NULL,   -- ok|failed|skipped
        latency_ms INTEGER,
        at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_action_log_at ON action_log(action, at)",
    # Brain intel
    """
    CREATE TABLE IF NOT EXISTS competitor_posts (
        ig_pk        TEXT PRIMARY KEY,
        username     TEXT NOT NULL,
        format       TEXT,
        caption      TEXT,
        likes        INTEGER,
        comments     INTEGER,
        posted_at    TEXT,
        scraped_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
        engaged      INTEGER DEFAULT 0
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_comp_user ON competitor_posts(username, posted_at DESC)",
    """
    CREATE TABLE IF NOT EXISTS hashtag_top (
        hashtag    TEXT NOT NULL,
        ig_pk      TEXT NOT NULL,
        caption    TEXT,
        likes      INTEGER,
        posted_at  TEXT,
        scraped_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
        PRIMARY KEY (hashtag, ig_pk)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS narratives (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        topic       TEXT NOT NULL UNIQUE,
        mentions    INTEGER DEFAULT 1,
        first_seen  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
        last_seen   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
        sample_refs TEXT
    )
    """,
    # Target-account watcher (x-agent style)
    """
    CREATE TABLE IF NOT EXISTS target_feed (
        ig_pk      TEXT PRIMARY KEY,
        username   TEXT NOT NULL,
        kind       TEXT,                        -- post|reel|story
        caption    TEXT,
        likes      INTEGER,
        created_at TEXT,
        seen_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
        engaged    INTEGER DEFAULT 0
    )
    """,
    # Context feed (LLM-facing priority buffer — x-agent's context_feed)
    """
    CREATE TABLE IF NOT EXISTS context_feed (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        source     TEXT NOT NULL,
        text       TEXT NOT NULL,
        priority   INTEGER NOT NULL DEFAULT 1,  -- higher = more urgent
        consumed   INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_ctx_priority ON context_feed(consumed, priority DESC, created_at DESC)",
    # Challenge log
    """
    CREATE TABLE IF NOT EXISTS challenges (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        kind       TEXT NOT NULL,
        status     TEXT NOT NULL DEFAULT 'pending',
        payload    TEXT,
        created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
        solved_at  TEXT,
        method     TEXT
    )
    """,
    # K/V state (backoff timers, last cycle timestamps, personality)
    """
    CREATE TABLE IF NOT EXISTS state (
        key        TEXT PRIMARY KEY,
        value      TEXT NOT NULL,
        updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
    )
    """,
    # Health snapshots
    """
    CREATE TABLE IF NOT EXISTS health_snapshots (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        followers       INTEGER,
        following       INTEGER,
        media_count     INTEGER,
        engagement_rate REAL,
        shadowbanned    INTEGER DEFAULT 0,
        snapshot_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
    )
    """,
    # Session liveness log — every keep_alive / login probe result.
    # Lets the dashboard show a survival curve + lets alerting rules
    # fire on consecutive failures before a real write dies.
    """
    CREATE TABLE IF NOT EXISTS session_health (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
        status      TEXT NOT NULL,   -- alive|dead|challenge|throttled|error
        feed_items  INTEGER,
        latency_ms  INTEGER,
        note        TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_session_health_at ON session_health(at)",
    # DM funnel
    """
    CREATE TABLE IF NOT EXISTS dm_contacts (
        username      TEXT PRIMARY KEY,
        ig_user_id    TEXT,
        source        TEXT,                          -- how we found them (hashtag, competitor, reply, etc.)
        stage         TEXT NOT NULL DEFAULT 'discovered',
                                                    -- discovered → targeted → contacted → replied → converted / dropped
        priority      INTEGER NOT NULL DEFAULT 1,
        notes         TEXT,
        last_action_at TEXT,
        next_action_at TEXT,
        created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
        updated_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_dm_stage_next ON dm_contacts(stage, next_action_at)",
    """
    CREATE TABLE IF NOT EXISTS dm_messages (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        username    TEXT NOT NULL,
        direction   TEXT NOT NULL,                  -- out | in
        step        INTEGER,                        -- 0=intro, 1=follow-up, ...
        body        TEXT NOT NULL,
        sent_at     TEXT,
        ig_thread_id TEXT,
        FOREIGN KEY (username) REFERENCES dm_contacts(username)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_dm_msg_user ON dm_messages(username, sent_at DESC)",
    # Inbound comment tracking (on our own posts) + our replies
    """
    CREATE TABLE IF NOT EXISTS inbound_comments (
        comment_pk   TEXT PRIMARY KEY,
        media_pk     TEXT NOT NULL,
        username     TEXT,
        user_id      TEXT,
        text         TEXT,
        created_at   TEXT,
        scraped_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
        replied      INTEGER NOT NULL DEFAULT 0,
        reply_pk     TEXT,
        ignored      INTEGER NOT NULL DEFAULT 0,
        is_own       INTEGER NOT NULL DEFAULT 0
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_inbound_status ON inbound_comments(replied, ignored, scraped_at DESC)",
    # Follow-back tracking
    """
    CREATE TABLE IF NOT EXISTS inbound_followers (
        user_id     TEXT PRIMARY KEY,
        username    TEXT,
        full_name   TEXT,
        seen_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
        triage      TEXT DEFAULT 'pending',   -- pending | approved | rejected | followed_back
        triage_note TEXT
    )
    """,
    # Idea bank — archetypes + hook formulas the generator draws from
    """
    CREATE TABLE IF NOT EXISTS ideas (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        archetype     TEXT NOT NULL,
        hook_formula  TEXT NOT NULL,
        format_hint   TEXT NOT NULL DEFAULT 'any',
        niche_tags    TEXT NOT NULL DEFAULT '[]',
        body_template TEXT NOT NULL DEFAULT '',
        source        TEXT NOT NULL DEFAULT 'curated',
        license       TEXT NOT NULL DEFAULT 'CC0',
        use_count     INTEGER NOT NULL DEFAULT 0,
        last_used_at  TEXT,
        score         REAL NOT NULL DEFAULT 0.5,
        created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
        UNIQUE (archetype, hook_formula)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_ideas_format ON ideas(format_hint, last_used_at)",
    # RAG knowledge chunks — drop a file into data/knowledge/, get retrieval
    """
    CREATE TABLE IF NOT EXISTS knowledge_chunks (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        source          TEXT NOT NULL,
        source_hash     TEXT NOT NULL,
        chunk_index     INTEGER NOT NULL,
        text            TEXT NOT NULL,
        embedding       BLOB NOT NULL,
        embedding_model TEXT NOT NULL,
        added_at        TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_kc_source ON knowledge_chunks(source)",
    "CREATE INDEX IF NOT EXISTS idx_kc_hash ON knowledge_chunks(source_hash)",
    "CREATE INDEX IF NOT EXISTS idx_kc_model ON knowledge_chunks(embedding_model)",
]

_local = threading.local()


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=30.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def get_conn() -> sqlite3.Connection:
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = _connect()
        _local.conn = conn
    return conn


def close() -> None:
    conn = getattr(_local, "conn", None)
    if conn is not None:
        conn.close()
        _local.conn = None


@contextmanager
def tx() -> Iterator[sqlite3.Connection]:
    """Explicit transaction. Use for multi-statement consistency."""
    conn = get_conn()
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def init_db(path: Path | None = None) -> None:
    if path is not None:
        global DB_PATH  # noqa: PLW0603
        # Re-binding is only for tests; production uses config constant.
    conn = get_conn()
    for stmt in SCHEMA:
        conn.execute(stmt)


# ───── State K/V ─────
def state_get(key: str, default: str | None = None) -> str | None:
    row = get_conn().execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default


def state_set(key: str, value: str) -> None:
    get_conn().execute(
        """
        INSERT INTO state (key, value, updated_at) VALUES (?, ?, strftime('%Y-%m-%dT%H:%M:%SZ','now'))
        ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
        """,
        (key, value),
    )


def state_get_json(key: str, default: Any = None) -> Any:
    raw = state_get(key)
    if raw is None:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return default


def state_set_json(key: str, value: Any) -> None:
    state_set(key, json.dumps(value, separators=(",", ":")))


# ───── Content queue ─────
def content_enqueue(
    *,
    format: str,
    caption: str,
    hashtags: list[str],
    media_paths: list[str],
    phash: str | None,
    critic_score: float | None,
    critic_notes: str | None,
    generator: str,
    status: str = "pending_review",
    scheduled_for: str | None = None,
    meta: dict[str, Any] | None = None,
) -> int:
    cur = get_conn().execute(
        """
        INSERT INTO content_queue
          (format, status, caption, hashtags, media_paths, phash,
           critic_score, critic_notes, generator, scheduled_for, meta)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            format,
            status,
            caption,
            json.dumps(hashtags),
            json.dumps(media_paths),
            phash,
            critic_score,
            critic_notes,
            generator,
            scheduled_for,
            json.dumps(meta or {}),
        ),
    )
    return int(cur.lastrowid)


def content_next_to_post() -> dict[str, Any] | None:
    row = get_conn().execute(
        """
        SELECT * FROM content_queue
        WHERE status='approved'
          AND (scheduled_for IS NULL OR scheduled_for <= strftime('%Y-%m-%dT%H:%M:%SZ','now'))
        ORDER BY COALESCE(scheduled_for, created_at) ASC
        LIMIT 1
        """
    ).fetchone()
    return _row_to_content(row) if row else None


def content_next_to_drain() -> dict[str, Any] | None:
    """Drain picks ANY approved item regardless of scheduled_for — the
    user explicitly asked for an immediate post, so bypass the best-
    hours slot filter.

    Also clears scheduled_for on the picked row so the poster doesn't
    re-enqueue it in the next scheduling pass."""
    row = get_conn().execute(
        """
        SELECT * FROM content_queue
        WHERE status='approved'
        ORDER BY COALESCE(scheduled_for, created_at) ASC
        LIMIT 1
        """
    ).fetchone()
    if row is None:
        return None
    # Strip the scheduled_for on the in-flight item so subsequent
    # schedule_approved_items doesn't re-slot it after posting.
    get_conn().execute(
        "UPDATE content_queue SET scheduled_for=NULL WHERE id=?",
        (int(row["id"]),),
    )
    return _row_to_content(row)


def content_get(cid: int) -> dict[str, Any] | None:
    row = get_conn().execute("SELECT * FROM content_queue WHERE id=?", (cid,)).fetchone()
    return _row_to_content(row) if row else None


def content_list(status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    if status:
        rows = get_conn().execute(
            "SELECT * FROM content_queue WHERE status=? ORDER BY created_at DESC LIMIT ?",
            (status, limit),
        ).fetchall()
    else:
        rows = get_conn().execute(
            "SELECT * FROM content_queue ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [_row_to_content(r) for r in rows]


def content_update_status(cid: int, status: str, error: str | None = None) -> None:
    get_conn().execute(
        "UPDATE content_queue SET status=?, error=? WHERE id=?", (status, error, cid)
    )


def content_mark_posted(cid: int, ig_media_pk: str) -> None:
    get_conn().execute(
        """
        UPDATE content_queue
        SET status='posted',
            ig_media_pk=?,
            posted_at=strftime('%Y-%m-%dT%H:%M:%SZ','now')
        WHERE id=?
        """,
        (ig_media_pk, cid),
    )


def content_schedule(cid: int, scheduled_for_iso: str) -> None:
    get_conn().execute(
        "UPDATE content_queue SET scheduled_for=? WHERE id=?", (scheduled_for_iso, cid)
    )


def existing_phashes(lookback: int = 60) -> list[str]:
    rows = get_conn().execute(
        """
        SELECT phash FROM content_queue
        WHERE phash IS NOT NULL AND status IN ('posted','approved','pending_review')
        ORDER BY created_at DESC LIMIT ?
        """,
        (lookback,),
    ).fetchall()
    return [r["phash"] for r in rows if r["phash"]]


def _row_to_content(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    d["hashtags"] = json.loads(d.get("hashtags") or "[]")
    d["media_paths"] = json.loads(d.get("media_paths") or "[]")
    d["meta"] = json.loads(d.get("meta") or "{}")
    return d


# ───── Posts ─────
def post_record(
    ig_media_pk: str, content_id: int | None, format: str, caption: str
) -> None:
    get_conn().execute(
        """
        INSERT OR REPLACE INTO posts (ig_media_pk, content_id, format, caption, posted_at)
        VALUES (?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%SZ','now'))
        """,
        (ig_media_pk, content_id, format, caption),
    )


def post_update_metrics(
    ig_media_pk: str, likes: int, comments: int, reach: int = 0
) -> None:
    get_conn().execute(
        """
        UPDATE posts
        SET likes=?, comments=?, reach=?,
            last_checked=strftime('%Y-%m-%dT%H:%M:%SZ','now')
        WHERE ig_media_pk=?
        """,
        (likes, comments, reach, ig_media_pk),
    )


# ───── Engagement queue ─────
def engagement_enqueue(
    action: str,
    *,
    target_user: str | None = None,
    target_media: str | None = None,
    payload: dict[str, Any] | None = None,
    scheduled_for: str | None = None,
) -> int:
    cur = get_conn().execute(
        """
        INSERT INTO engagement_queue (action, target_user, target_media, payload, scheduled_for)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            action,
            target_user,
            target_media,
            json.dumps(payload or {}),
            scheduled_for,
        ),
    )
    return int(cur.lastrowid)


def engagement_next(limit: int = 1) -> list[dict[str, Any]]:
    rows = get_conn().execute(
        """
        SELECT * FROM engagement_queue
        WHERE status='pending'
          AND (scheduled_for IS NULL OR scheduled_for <= strftime('%Y-%m-%dT%H:%M:%SZ','now'))
        ORDER BY created_at ASC LIMIT ?
        """,
        (limit,),
    ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["payload"] = json.loads(d.get("payload") or "{}")
        out.append(d)
    return out


def engagement_mark(eid: int, result: str, error: str | None = None) -> None:
    status = "done" if result == "ok" else "failed"
    get_conn().execute(
        """
        UPDATE engagement_queue
        SET status=?, result=?, error=?,
            attempted_at=strftime('%Y-%m-%dT%H:%M:%SZ','now')
        WHERE id=?
        """,
        (status, result, error, eid),
    )


# ───── Action log / budget ─────
def action_log(action: str, target: str | None, result: str, latency_ms: int) -> None:
    get_conn().execute(
        "INSERT INTO action_log (action, target, result, latency_ms) VALUES (?, ?, ?, ?)",
        (action, target, result, latency_ms),
    )


def action_count_today(action: str) -> int:
    row = get_conn().execute(
        """
        SELECT COUNT(*) AS c FROM action_log
        WHERE action=?
          AND result='ok'
          AND at >= strftime('%Y-%m-%dT00:00:00Z','now')
        """,
        (action,),
    ).fetchone()
    return int(row["c"]) if row else 0


# ───── Competitor + hashtag intel ─────
def competitor_upsert(
    ig_pk: str, username: str, caption: str, likes: int, comments: int, posted_at: str
) -> None:
    get_conn().execute(
        """
        INSERT OR REPLACE INTO competitor_posts
          (ig_pk, username, caption, likes, comments, posted_at, scraped_at)
        VALUES (?, ?, ?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%SZ','now'))
        """,
        (ig_pk, username, caption, likes, comments, posted_at),
    )


def competitor_top_recent(username: str, limit: int = 5) -> list[dict[str, Any]]:
    rows = get_conn().execute(
        """
        SELECT * FROM competitor_posts
        WHERE username=?
        ORDER BY likes DESC LIMIT ?
        """,
        (username, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def hashtag_upsert(hashtag: str, ig_pk: str, caption: str, likes: int, posted_at: str) -> None:
    get_conn().execute(
        """
        INSERT OR REPLACE INTO hashtag_top (hashtag, ig_pk, caption, likes, posted_at, scraped_at)
        VALUES (?, ?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%SZ','now'))
        """,
        (hashtag, ig_pk, caption, likes, posted_at),
    )


# ───── Context feed ─────
def push_context(source: str, text: str, priority: int = 1) -> None:
    get_conn().execute(
        "INSERT INTO context_feed (source, text, priority) VALUES (?, ?, ?)",
        (source, text, priority),
    )


def pop_context(limit: int = 10) -> list[dict[str, Any]]:
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT id, source, text, priority FROM context_feed
        WHERE consumed=0
        ORDER BY priority DESC, created_at DESC LIMIT ?
        """,
        (limit,),
    ).fetchall()
    ids = [r["id"] for r in rows]
    if ids:
        qs = ",".join("?" * len(ids))
        conn.execute(f"UPDATE context_feed SET consumed=1 WHERE id IN ({qs})", ids)
    return [dict(r) for r in rows]


def narrative_bump(topic: str, sample_ref: str | None = None) -> None:
    conn = get_conn()
    row = conn.execute("SELECT id, sample_refs FROM narratives WHERE topic=?", (topic,)).fetchone()
    if row:
        refs = json.loads(row["sample_refs"] or "[]")
        if sample_ref and sample_ref not in refs:
            refs = ([sample_ref] + refs)[:10]
        conn.execute(
            """
            UPDATE narratives
            SET mentions = mentions + 1,
                last_seen = strftime('%Y-%m-%dT%H:%M:%SZ','now'),
                sample_refs = ?
            WHERE id=?
            """,
            (json.dumps(refs), row["id"]),
        )
    else:
        conn.execute(
            "INSERT INTO narratives (topic, sample_refs) VALUES (?, ?)",
            (topic, json.dumps([sample_ref] if sample_ref else [])),
        )


# ───── Target-account feed (x-agent watcher equivalent) ─────
def target_feed_upsert(
    ig_pk: str, username: str, kind: str, caption: str, likes: int, created_at: str
) -> bool:
    """Returns True if this was newly inserted (worth reacting to)."""
    conn = get_conn()
    existed = conn.execute("SELECT 1 FROM target_feed WHERE ig_pk=?", (ig_pk,)).fetchone()
    if existed:
        return False
    conn.execute(
        """
        INSERT INTO target_feed (ig_pk, username, kind, caption, likes, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (ig_pk, username, kind, caption, likes, created_at),
    )
    return True


# ───── Challenges ─────
def challenge_log(kind: str, payload: dict[str, Any] | None = None) -> int:
    cur = get_conn().execute(
        "INSERT INTO challenges (kind, payload) VALUES (?, ?)",
        (kind, json.dumps(payload or {})),
    )
    return int(cur.lastrowid)


def challenge_resolve(cid: int, method: str) -> None:
    get_conn().execute(
        """
        UPDATE challenges
        SET status='solved',
            solved_at=strftime('%Y-%m-%dT%H:%M:%SZ','now'),
            method=?
        WHERE id=?
        """,
        (method, cid),
    )


# ───── Health ─────
def health_record(
    followers: int,
    following: int,
    media_count: int,
    engagement_rate: float,
    shadowbanned: bool,
) -> None:
    get_conn().execute(
        """
        INSERT INTO health_snapshots
          (followers, following, media_count, engagement_rate, shadowbanned)
        VALUES (?, ?, ?, ?, ?)
        """,
        (followers, following, media_count, engagement_rate, 1 if shadowbanned else 0),
    )


def health_latest() -> dict[str, Any] | None:
    row = get_conn().execute(
        "SELECT * FROM health_snapshots ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return dict(row) if row else None


# ───── DM funnel ─────
def dm_upsert_contact(
    username: str,
    *,
    ig_user_id: str | None = None,
    source: str | None = None,
    stage: str = "discovered",
    priority: int = 1,
    notes: str | None = None,
) -> None:
    conn = get_conn()
    existing = conn.execute(
        "SELECT stage, priority FROM dm_contacts WHERE username=?", (username,)
    ).fetchone()
    if existing:
        # Only bump metadata; never regress stage backwards via seeder
        new_priority = max(int(existing["priority"] or 1), int(priority))
        conn.execute(
            """
            UPDATE dm_contacts
            SET ig_user_id = COALESCE(?, ig_user_id),
                source     = COALESCE(?, source),
                priority   = ?,
                notes      = COALESCE(?, notes),
                updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now')
            WHERE username = ?
            """,
            (ig_user_id, source, new_priority, notes, username),
        )
    else:
        conn.execute(
            """
            INSERT INTO dm_contacts
              (username, ig_user_id, source, stage, priority, notes)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (username, ig_user_id, source, stage, priority, notes),
        )


def dm_advance(username: str, new_stage: str, *, next_after_iso: str | None = None) -> None:
    get_conn().execute(
        """
        UPDATE dm_contacts
        SET stage=?,
            last_action_at=strftime('%Y-%m-%dT%H:%M:%SZ','now'),
            next_action_at=?,
            updated_at=strftime('%Y-%m-%dT%H:%M:%SZ','now')
        WHERE username=?
        """,
        (new_stage, next_after_iso, username),
    )


def dm_contacts_due(stage: str, limit: int = 5) -> list[dict[str, Any]]:
    rows = get_conn().execute(
        """
        SELECT * FROM dm_contacts
        WHERE stage=?
          AND (next_action_at IS NULL OR next_action_at <= strftime('%Y-%m-%dT%H:%M:%SZ','now'))
        ORDER BY priority DESC, created_at ASC
        LIMIT ?
        """,
        (stage, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def dm_step_count(username: str, direction: str = "out") -> int:
    row = get_conn().execute(
        "SELECT COUNT(*) c FROM dm_messages WHERE username=? AND direction=?",
        (username, direction),
    ).fetchone()
    return int(row["c"]) if row else 0


def dm_record_message(
    username: str,
    direction: str,
    body: str,
    *,
    step: int | None = None,
    sent_at_iso: str | None = None,
    ig_thread_id: str | None = None,
) -> int:
    cur = get_conn().execute(
        """
        INSERT INTO dm_messages (username, direction, step, body, sent_at, ig_thread_id)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (username, direction, step, body, sent_at_iso or now_iso(), ig_thread_id),
    )
    return int(cur.lastrowid)


# ───── Inbound comments ─────
def inbound_comment_upsert(
    comment_pk: str,
    *,
    media_pk: str,
    username: str | None,
    user_id: str | None,
    text: str,
    created_at: str | None,
    is_own: bool = False,
) -> bool:
    """Upsert an inbound comment row. Returns True if this is new."""
    conn = get_conn()
    existed = conn.execute(
        "SELECT 1 FROM inbound_comments WHERE comment_pk=?", (comment_pk,)
    ).fetchone()
    if existed:
        return False
    conn.execute(
        """
        INSERT INTO inbound_comments
          (comment_pk, media_pk, username, user_id, text, created_at, is_own)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (comment_pk, media_pk, username, user_id, text, created_at, 1 if is_own else 0),
    )
    return True


def inbound_comments_to_reply(limit: int = 5) -> list[dict[str, Any]]:
    rows = get_conn().execute(
        """
        SELECT * FROM inbound_comments
        WHERE replied=0 AND ignored=0 AND is_own=0
        ORDER BY scraped_at DESC LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def inbound_comment_mark_replied(comment_pk: str, reply_pk: str) -> None:
    get_conn().execute(
        "UPDATE inbound_comments SET replied=1, reply_pk=? WHERE comment_pk=?",
        (reply_pk, comment_pk),
    )


def inbound_comment_ignore(comment_pk: str) -> None:
    get_conn().execute(
        "UPDATE inbound_comments SET ignored=1 WHERE comment_pk=?", (comment_pk,)
    )


# ───── Inbound followers ─────
def follower_upsert(user_id: str, username: str, full_name: str = "") -> bool:
    conn = get_conn()
    existed = conn.execute(
        "SELECT 1 FROM inbound_followers WHERE user_id=?", (user_id,)
    ).fetchone()
    if existed:
        return False
    conn.execute(
        """
        INSERT INTO inbound_followers (user_id, username, full_name)
        VALUES (?, ?, ?)
        """,
        (user_id, username, full_name),
    )
    return True


def followers_pending(limit: int = 20) -> list[dict[str, Any]]:
    rows = get_conn().execute(
        "SELECT * FROM inbound_followers WHERE triage='pending' ORDER BY seen_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def follower_triage(user_id: str, decision: str, note: str | None = None) -> None:
    get_conn().execute(
        "UPDATE inbound_followers SET triage=?, triage_note=? WHERE user_id=?",
        (decision, note, user_id),
    )


def dm_last_out(username: str) -> dict[str, Any] | None:
    row = get_conn().execute(
        """
        SELECT * FROM dm_messages
        WHERE username=? AND direction='out'
        ORDER BY sent_at DESC, id DESC LIMIT 1
        """,
        (username,),
    ).fetchone()
    return dict(row) if row else None


# ───── Utilities ─────
def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
