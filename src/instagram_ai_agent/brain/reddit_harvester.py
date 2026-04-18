"""Reddit question harvester.

Pulls recent question-like posts from niche subreddits and pushes them
into ``context_feed`` so the pipeline has real audience language to
riff on. Uses PRAW (BSD-2) — free Reddit API read tier, no user login
required (client-credentials only).

Safety:
  * Dedup by (sub, post_id) in ``state`` K/V with a rolling cutoff so
    the same question never appears twice.
  * NSFW / over_18 subs and posts are filtered out unconditionally.
  * Every import is lazy and graceful — PRAW absent or missing creds
    means the harvester is a no-op.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from instagram_ai_agent.core import db
from instagram_ai_agent.core.config import NicheConfig
from instagram_ai_agent.core.logging_setup import get_logger

log = get_logger(__name__)


_SEEN_KEY = "reddit_questions_seen"

# Subreddits to never touch even if a user lists them. This is a first-line
# token-match; we additionally block anything Reddit flags as ``over18`` and
# any post with ``over_18`` set. An NSFW-themed sub that is NOT flagged
# ``over18`` AND whose name doesn't match this set will still bypass —
# users are responsible for what they put in ``reddit_subs``.
_HARD_BLOCK = {"gonewild", "nsfw", "porn", "amateur"}

# Question-word heuristics (lower-cased)
_QUESTION_WORDS = (
    "how", "why", "what", "when", "where", "which", "who",
    "should", "is", "are", "does", "do", "can", "could",
    "would", "will", "am", "has", "have",
)
# Only strip DOUBLE/curly quotes — apostrophes are kept so contracted
# question words like "what's" still match the question-word list.
_STRIP_PUNCT = re.compile(r"[\"“”]")


@dataclass(frozen=True)
class RedditQuestion:
    post_id: str
    subreddit: str
    title: str
    score: int
    num_comments: int
    url: str
    created_utc: float
    author: str

    @property
    def key(self) -> str:
        return f"{self.subreddit.lower()}:{self.post_id}"


# ─── PRAW availability ───
def _praw_available() -> bool:
    try:
        import praw  # noqa: F401
        return True
    except Exception:
        return False


def _creds_configured() -> bool:
    return all(
        os.environ.get(k) for k in ("REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USER_AGENT")
    )


def _get_reddit() -> Any | None:
    if not _praw_available() or not _creds_configured():
        return None
    import praw

    return praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent=os.environ["REDDIT_USER_AGENT"],
        check_for_async=False,
    )


# ─── Filters ───
_FIRST_WORD = re.compile(r"[A-Za-z]+")


def is_question_title(title: str) -> bool:
    """Heuristic: ends with ``?`` OR starts with a question word.

    ``What's``, ``How're``, etc. are normalised to their alphabetic root
    so contracted question words are recognised.
    """
    if not title:
        return False
    t = _STRIP_PUNCT.sub("", title).strip()
    if not t:
        return False
    if t.endswith("?"):
        return True
    m = _FIRST_WORD.match(t)
    if not m:
        return False
    return m.group(0).lower() in _QUESTION_WORDS


def is_sensitive(sub_name: str, *, over_18: bool, nsfw_post: bool) -> bool:
    """Block NSFW subs + posts unconditionally."""
    if over_18 or nsfw_post:
        return True
    return sub_name.lower() in _HARD_BLOCK


# ─── Dedup state ───
def _seen() -> dict[str, float]:
    """State: {key: unix_ts_of_post}."""
    raw = db.state_get_json(_SEEN_KEY, default={}) or {}
    return raw if isinstance(raw, dict) else {}


def _record_seen(items: list[RedditQuestion], *, lookback_hours: int) -> None:
    store = _seen()
    for q in items:
        store[q.key] = q.created_utc
    # Prune: keep entries newer than 2× lookback_hours so we can't
    # re-push them, but don't let the store grow unbounded.
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=lookback_hours * 2)).timestamp()
    store = {k: v for k, v in store.items() if v >= cutoff}
    db.state_set_json(_SEEN_KEY, store)


def _fresh(items: list[RedditQuestion]) -> list[RedditQuestion]:
    store = _seen()
    return [q for q in items if q.key not in store]


# ─── Fetch ───
def fetch_questions(cfg: NicheConfig) -> list[RedditQuestion]:
    """Collect question-like posts across the configured subs. All network
    work happens here; filters are pure. Returns empty list on any error
    including missing PRAW / creds / blocked-sub.
    """
    reddit = _get_reddit()
    if reddit is None:
        return []

    cutoff_ts = (
        datetime.now(timezone.utc) - timedelta(hours=cfg.reddit_lookback_hours)
    ).timestamp()

    out: list[RedditQuestion] = []
    for sub in cfg.reddit_subs:
        sub_name = str(sub).lstrip("/").removeprefix("r/").strip()
        if not sub_name:
            continue
        if sub_name.lower() in _HARD_BLOCK:
            log.debug("reddit: skipping blocked sub %r", sub_name)
            continue

        try:
            sr = reddit.subreddit(sub_name)
            # Over-18 subs: hard-block regardless of user config
            try:
                over_18 = bool(getattr(sr, "over18", False))
            except Exception:
                over_18 = False
            if over_18:
                log.debug("reddit: skipping over-18 sub %r", sub_name)
                continue

            for post in sr.top(time_filter="day", limit=cfg.reddit_posts_per_sub):
                if getattr(post, "over_18", False):
                    continue
                if float(getattr(post, "created_utc", 0) or 0) < cutoff_ts:
                    continue
                if int(getattr(post, "score", 0) or 0) < cfg.reddit_min_score:
                    continue
                title = str(getattr(post, "title", "") or "").strip()
                if not is_question_title(title):
                    continue
                out.append(RedditQuestion(
                    post_id=str(post.id),
                    subreddit=sub_name,
                    title=title,
                    score=int(post.score or 0),
                    num_comments=int(getattr(post, "num_comments", 0) or 0),
                    url=f"https://reddit.com{getattr(post, 'permalink', '')}",
                    created_utc=float(post.created_utc or 0),
                    author=str(getattr(post.author, "name", "")) if getattr(post, "author", None) else "",
                ))
        except Exception as e:
            log.warning("reddit harvest failed for r/%s: %s", sub_name, e)
            continue

    # Best first by score
    out.sort(key=lambda q: q.score, reverse=True)
    return out


async def run_once(cfg: NicheConfig) -> int:
    """Full cycle. Returns number of NEW questions pushed to context_feed."""
    if not cfg.reddit_enabled or not cfg.reddit_subs:
        return 0

    # PRAW is sync; run the network part off the event loop
    import asyncio

    all_qs = await asyncio.to_thread(fetch_questions, cfg)
    fresh = _fresh(all_qs)
    if not fresh:
        return 0

    for q in fresh:
        db.push_context(
            f"reddit.{q.subreddit}",
            f"question ({q.score}👍 {q.num_comments}💬) r/{q.subreddit}: {q.title}",
            priority=3,
        )

    _record_seen(fresh, lookback_hours=cfg.reddit_lookback_hours)
    log.info("reddit: pushed %d fresh questions to context", len(fresh))
    return len(fresh)
