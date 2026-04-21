"""Reddit top-posts scraper via the public JSON endpoint. No PRAW, no API
key, no registered app. Just appends ``.json`` to any subreddit URL.

Reddit's public JSON endpoint has served this use case since 2008 and
doesn't require auth for read access. Rate limit is generous (~60 req/min
per IP without signed requests). We send a polite User-Agent + accept
application/json and tolerate the occasional 429 by backing off.

For the transform pipeline we care specifically about VIDEO posts — those
with ``is_video: true`` and a ``media.reddit_video.fallback_url``
resolvable as mp4. The fallback_url works without auth.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import httpx

from instagram_ai_agent.core.logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class RedditVideoPost:
    post_id: str
    title: str
    author: str  # without /u/ prefix
    subreddit: str
    permalink: str  # https://reddit.com/r/X/comments/Y/...
    video_url: str  # direct mp4 URL (no audio — Reddit hosts audio separately)
    score: int
    num_comments: int
    duration_s: float
    width: int
    height: int
    nsfw: bool


_UA = "ig-agent (anonymous public-read, contact: issues on github.com/alsk1992/instagram-ai-agent)"


async def top_videos(
    subreddit: str,
    *,
    timeframe: str = "week",  # hour|day|week|month|year|all
    limit: int = 25,
    min_score: int = 100,
    max_duration_s: float = 60.0,
    include_nsfw: bool = False,
) -> list[RedditVideoPost]:
    """Fetch top posts from a subreddit and filter to video posts matching
    our criteria. Returns ordered by score descending.

    Reddit's JSON returns up to ~25 posts per page; we take one page which
    is plenty for harvesting — score filter handles quality."""
    sub = subreddit.lstrip("r/").lstrip("/")
    url = f"https://old.reddit.com/r/{sub}/top.json"
    params = {"t": timeframe, "limit": str(limit)}
    headers = {"User-Agent": _UA, "Accept": "application/json"}

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(20.0)) as client:
            r = await client.get(url, params=params, headers=headers)
            if r.status_code == 429:
                # Reddit rate-limited us — brief back-off then one retry
                await asyncio.sleep(5)
                r = await client.get(url, params=params, headers=headers)
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        log.warning("reddit_public: fetch r/%s failed — %s", sub, e)
        return []

    children = (data.get("data") or {}).get("children") or []
    out: list[RedditVideoPost] = []
    for c in children:
        post = (c or {}).get("data") or {}
        if not post.get("is_video"):
            continue
        if not include_nsfw and post.get("over_18"):
            continue
        score = int(post.get("score") or 0)
        if score < min_score:
            continue
        media = post.get("media") or {}
        rv = media.get("reddit_video") or {}
        fallback = rv.get("fallback_url") or ""
        if not fallback:
            continue
        duration = float(rv.get("duration") or 0)
        if duration <= 0 or duration > max_duration_s:
            continue
        out.append(RedditVideoPost(
            post_id=str(post.get("id") or ""),
            title=(post.get("title") or "")[:300],
            author=str(post.get("author") or ""),
            subreddit=sub,
            permalink="https://www.reddit.com" + (post.get("permalink") or ""),
            video_url=fallback,
            score=score,
            num_comments=int(post.get("num_comments") or 0),
            duration_s=duration,
            width=int(rv.get("width") or 0),
            height=int(rv.get("height") or 0),
            nsfw=bool(post.get("over_18")),
        ))
    return sorted(out, key=lambda p: p.score, reverse=True)


async def top_videos_across_subs(
    subs: list[str],
    *,
    timeframe: str = "week",
    min_score: int = 100,
) -> list[RedditVideoPost]:
    """Gather + merge top videos across several subs. Dedupes by post_id."""
    seen: set[str] = set()
    results: list[RedditVideoPost] = []
    for sub in subs:
        posts = await top_videos(sub, timeframe=timeframe, min_score=min_score)
        for p in posts:
            if p.post_id in seen:
                continue
            seen.add(p.post_id)
            results.append(p)
        # polite spacing between subs to stay under Reddit's unsigned RPM
        await asyncio.sleep(1.5)
    return sorted(results, key=lambda p: p.score, reverse=True)
