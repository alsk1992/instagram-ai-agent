"""Instagram scraping via the already-logged-in instagrapi client.

Prior design used Instaloader for "public" scraping. IG closed the
unauthenticated `/api/v1/tags/web_info/` and profile endpoints in
late 2024 — Instaloader now 403s on every call. And even with our
session cookies injected, Instaloader hits `i.instagram.com` mobile
endpoints that require Bearer-token auth, which cookies don't
satisfy.

2026 design: reuse the SAME instagrapi ``Client`` that the rest of
the agent logs in with (session file at ``data/sessions/<user>.json``).
One authenticated session, one rate-limit pool, zero architecture mismatch.

Shape: ``PublicScraper`` still exposes ``profile_posts()`` and
``hashtag_posts()`` so every caller (competitor_intel, trend_miner,
watcher) keeps working without edits. The methods now delegate to
``instagrapi.Client`` methods (``user_medias``, ``hashtag_medias_top_v1``)
under the hood.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from instagram_ai_agent.core.logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class ScrapedPost:
    ig_pk: str
    username: str
    caption: str
    likes: int
    comments: int
    posted_at: str  # ISO
    url: str


class PublicScraper:
    """instagrapi-backed scraper. Lazy-loads the session file on first call;
    returns empty lists when no session is available yet (e.g. first boot
    before ``ig-agent login`` has succeeded)."""

    def __init__(self) -> None:
        self._cl = None
        self._tried_load = False

    def _client(self):
        """Load the instagrapi Client from the persisted session file.

        Uses the SAME session file the main orchestrator logs in with.
        Cached after first successful load. Returns None when no session
        exists — callers degrade gracefully to empty results.
        """
        if self._cl is not None:
            return self._cl
        if self._tried_load:
            return None
        self._tried_load = True
        try:
            import os as _os
            username = _os.environ.get("IG_USERNAME", "").strip()
            if not username:
                log.debug("PublicScraper: IG_USERNAME not set — no scraping this run")
                return None
            from instagram_ai_agent.core.config import DATA_DIR
            sp = DATA_DIR / "sessions" / f"{username}.json"
            if not sp.exists():
                log.debug(
                    "PublicScraper: no session at %s — ig-agent login hasn't run yet",
                    sp,
                )
                return None
            from instagrapi import Client
            cl = Client()
            cl.load_settings(str(sp))
            self._cl = cl
            return cl
        except Exception as e:
            log.warning("PublicScraper: failed to load instagrapi session — %s", e)
            return None

    def profile_posts(self, username: str, limit: int = 12) -> list[ScrapedPost]:
        cl = self._client()
        if cl is None:
            return []
        try:
            uid = cl.user_id_from_username(username)
            medias = cl.user_medias(uid, amount=limit)
        except Exception as e:
            log.warning("scraper: failed profile %s — %s", username, e)
            return []
        return [self._media_to_scraped(m, default_username=username) for m in medias]

    def hashtag_posts(self, hashtag: str, limit: int = 20) -> list[ScrapedPost]:
        cl = self._client()
        if cl is None:
            return []
        tag = hashtag.lstrip("#")
        try:
            medias = cl.hashtag_medias_top_v1(tag, amount=limit)
        except Exception as e:
            log.warning("scraper: failed hashtag %s — %s", tag, e)
            return []
        return [self._media_to_scraped(m) for m in medias]

    @staticmethod
    def _media_to_scraped(m: Any, default_username: str = "") -> ScrapedPost:
        """Normalise an instagrapi Media object into our ScrapedPost shape."""
        try:
            posted_at = m.taken_at.isoformat() + "Z" if m.taken_at else ""
        except Exception:
            posted_at = ""
        return ScrapedPost(
            ig_pk=str(getattr(m, "pk", "") or ""),
            username=(getattr(getattr(m, "user", None), "username", None) or default_username or ""),
            caption=(getattr(m, "caption_text", "") or "")[:1000],
            likes=int(getattr(m, "like_count", 0) or 0),
            comments=int(getattr(m, "comment_count", 0) or 0),
            posted_at=posted_at,
            url=f"https://www.instagram.com/p/{getattr(m, 'code', '')}/",
        )


def posts_to_dicts(posts: list[ScrapedPost]) -> list[dict[str, Any]]:
    return [p.__dict__ for p in posts]
