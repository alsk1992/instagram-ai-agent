"""Public Instagram scraping via Instaloader — no login when the target is public.

This is the read-only surface the brain uses so we don't spend our main
account's rate-limit on intel gathering.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import instaloader

from src.core.logging_setup import get_logger

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
    """Thin, reusable Instaloader facade."""

    def __init__(self) -> None:
        self.L = instaloader.Instaloader(
            download_pictures=False,
            download_videos=False,
            download_video_thumbnails=False,
            download_geotags=False,
            download_comments=False,
            save_metadata=False,
            compress_json=False,
            quiet=True,
            request_timeout=30,
        )

    def profile_posts(self, username: str, limit: int = 12) -> list[ScrapedPost]:
        try:
            profile = instaloader.Profile.from_username(self.L.context, username)
        except Exception as e:
            log.warning("Instaloader: failed profile %s — %s", username, e)
            return []
        out: list[ScrapedPost] = []
        for i, p in enumerate(profile.get_posts()):
            if i >= limit:
                break
            try:
                out.append(
                    ScrapedPost(
                        ig_pk=str(p.mediaid),
                        username=username,
                        caption=(p.caption or "")[:1000],
                        likes=int(getattr(p, "likes", 0) or 0),
                        comments=int(getattr(p, "comments", 0) or 0),
                        posted_at=p.date_utc.isoformat() + "Z" if p.date_utc else "",
                        url=f"https://www.instagram.com/p/{p.shortcode}/",
                    )
                )
            except Exception as e:
                log.debug("skipping post on %s: %s", username, e)
                continue
        return out

    def hashtag_posts(self, hashtag: str, limit: int = 20) -> list[ScrapedPost]:
        try:
            tag = instaloader.Hashtag.from_name(self.L.context, hashtag.lstrip("#"))
        except Exception as e:
            log.warning("Instaloader: failed hashtag %s — %s", hashtag, e)
            return []
        out: list[ScrapedPost] = []
        for i, p in enumerate(tag.get_top_posts()):
            if i >= limit:
                break
            try:
                out.append(
                    ScrapedPost(
                        ig_pk=str(p.mediaid),
                        username=p.owner_username or "",
                        caption=(p.caption or "")[:1000],
                        likes=int(getattr(p, "likes", 0) or 0),
                        comments=int(getattr(p, "comments", 0) or 0),
                        posted_at=p.date_utc.isoformat() + "Z" if p.date_utc else "",
                        url=f"https://www.instagram.com/p/{p.shortcode}/",
                    )
                )
            except Exception as e:
                log.debug("skipping tag post on #%s: %s", hashtag, e)
                continue
        return out


def posts_to_dicts(posts: list[ScrapedPost]) -> list[dict[str, Any]]:
    return [p.__dict__ for p in posts]
