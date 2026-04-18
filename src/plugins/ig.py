"""Instagram client wrapper around instagrapi.

Owns session/device persistence, proxy, challenge resolution, and the
posting/engagement primitives the orchestrator consumes.
"""
from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Any

from instagrapi import Client
from instagrapi.exceptions import (
    BadPassword,
    ChallengeRequired,
    ClientError,
    LoginRequired,
    PleaseWaitFewMinutes,
    RateLimitError,
    TwoFactorRequired,
)

from src.core import db
from src.core.config import SESSIONS_DIR
from src.core.logging_setup import get_logger
from src.plugins import challenge as ch
from src.plugins import device as dev

log = get_logger(__name__)


class BackoffActive(RuntimeError):
    """Raised when the agent is in a cooldown and must not make requests."""


def _build_cookie_seed() -> dict[str, str] | None:
    """Collect every IG web cookie we can consume from env.

    Priority order for auth quality (best → worst):
      1. FULL set — sessionid + ds_user_id + csrftoken + mid + ig_did +
         datr + rur. Loaded via ``cl.set_settings`` so instagrapi never
         needs to hit ``/login`` — zero challenge surface.
      2. MINIMAL set — just sessionid. Loaded via ``login_by_sessionid``,
         which fetches the remaining cookies from IG (can still trigger
         a suspicious-login check on a fresh IP).
      3. None — fall through to username/password login.

    Returns a dict of every cookie provided (may be sparse) or None if
    sessionid is absent.
    """
    cookies = {
        "sessionid":  os.environ.get("IG_SESSIONID", "").strip(),
        "ds_user_id": os.environ.get("IG_DS_USER_ID", "").strip(),
        "csrftoken":  os.environ.get("IG_CSRFTOKEN", "").strip(),
        "mid":        os.environ.get("IG_MID", "").strip(),
        "ig_did":     os.environ.get("IG_DID", "").strip(),
        "datr":       os.environ.get("IG_DATR", "").strip(),
        "rur":        os.environ.get("IG_RUR", "").strip(),
        "shbid":      os.environ.get("IG_SHBID", "").strip(),
        "shbts":      os.environ.get("IG_SHBTS", "").strip(),
        "ig_nrcb":    os.environ.get("IG_NRCB", "").strip(),
    }
    if not cookies["sessionid"]:
        return None
    # Strip empties so we can detect "full set" via key presence.
    return {k: v for k, v in cookies.items() if v}


def _default_user_agent(device: dict) -> str:
    """Instagram Android user agent string — mirrors instagrapi's own
    builder so the cookies we paste match a plausible device claim."""
    return (
        f"Instagram {device.get('app_version', '302.0.0.23.114')} "
        f"Android ({device.get('android_version', 30)}/"
        f"{device.get('android_release', '11')}; "
        f"{device.get('dpi', '420dpi')}; "
        f"{device.get('resolution', '1080x2220')}; "
        f"{device.get('manufacturer', 'samsung')}; "
        f"{device.get('device', 'SM-A525F')}; "
        f"{device.get('model', 'a52q')}; "
        f"{device.get('cpu', 'qcom')}; en_US; "
        f"{device.get('version_code', '521498971')})"
    )


def _session_refresh_days() -> int:
    """Days after which we force a fresh password login. Default 7 per
    2026 community consensus — cookies survive longer but server-side
    session TTL decays and triggers silent rate-limit cascades."""
    raw = os.environ.get("IG_SESSION_REFRESH_DAYS", "").strip()
    if not raw:
        return 7
    try:
        return max(0, int(raw))
    except ValueError:
        return 7


def _has_full_cookie_set(seed: dict[str, str]) -> bool:
    """True when we have enough cookies to skip instagrapi's /login call
    entirely. ``sessionid``, ``ds_user_id``, and ``csrftoken`` are the
    minimum for ``cl.set_settings`` to reconstruct a valid session."""
    return bool(seed and seed.get("sessionid") and seed.get("ds_user_id") and seed.get("csrftoken"))


class IGClient:
    def __init__(self, username: str | None = None, password: str | None = None):
        import time as _time
        self.username = username or os.environ.get("IG_USERNAME", "")
        self._password = password or os.environ.get("IG_PASSWORD", "")
        self.cl = Client()
        self.session_path: Path = SESSIONS_DIR / f"{self.username}.json"
        self._logged_in = False
        self._client_created_at = _time.time()

        # Proxy (sticky per account)
        proxy = os.environ.get("IG_PROXY")
        if proxy:
            self.cl.set_proxy(proxy)

        # Persistent device fingerprint
        dev.apply_to(self.cl)

        # Geographic coherence — timezone / locale / country must match
        # the account's history, otherwise IG flags the session as
        # suspicious and triggers an email challenge. We let the user
        # override via env; defaults are instagrapi's own sensible values.
        _country = os.environ.get("IG_COUNTRY_CODE", "").strip()
        _tz_offset = os.environ.get("IG_TIMEZONE_OFFSET", "").strip()
        _locale = os.environ.get("IG_LOCALE", "").strip()
        if _country:
            self.cl.set_country(_country)
        if _tz_offset:
            try:
                self.cl.set_timezone_offset(int(_tz_offset))
            except ValueError:
                log.warning("IG_TIMEZONE_OFFSET=%r isn't an integer (seconds)", _tz_offset)
        if _locale:
            self.cl.set_locale(_locale)

        # User-agent override — rare, but some users run behind CDN
        # rewriters or need to spoof a specific build.
        _ua = os.environ.get("IG_USER_AGENT", "").strip()
        if _ua:
            self.cl.set_user_agent(_ua)

        # Direct-cookie escape hatch: paste a pre-logged-in session
        # from a browser to skip the first-login challenge entirely.
        # Triggered when all four cookies are provided AND no session
        # file exists yet (so we don't clobber a working session).
        self._cookie_seed = _build_cookie_seed()

        # Wire challenge handlers. When stdin is a TTY (user ran
        # `ig-agent login` directly) we enable interactive code entry
        # so a first-time setup doesn't dead-end in a 24h cooldown.
        import sys as _sys
        _interactive = _sys.stdin.isatty() and _sys.stdout.isatty()
        self.cl.challenge_code_handler = ch.make_challenge_code_handler(interactive=_interactive)
        self.cl.totp_code_handler = ch.make_totp_handler()

        # Random but human-ish request delays
        self.cl.delay_range = [2, 6]

    # ───── Auth ─────
    def login(self) -> None:
        self._ensure_backoff_ok()
        if not (self.username and self._password):
            raise RuntimeError("IG_USERNAME / IG_PASSWORD must be set in env.")

        self.session_path.parent.mkdir(parents=True, exist_ok=True)
        settings_loaded = False

        # Session-age refresh: if the persisted session is older than
        # IG_SESSION_REFRESH_DAYS (default 7) we force a fresh password
        # login. Rationale: cookies remain valid for weeks but IG's
        # server-side session TTL is shorter; refreshing before it
        # silently decays avoids the rate-limit / challenge cascade.
        refresh_days = _session_refresh_days()
        if refresh_days > 0 and self.session_path.exists():
            import time
            age_days = (time.time() - self.session_path.stat().st_mtime) / 86400.0
            if age_days >= refresh_days:
                log.info(
                    "Session is %.1fd old (≥ %d) — forcing fresh login",
                    age_days, refresh_days,
                )
                try:
                    self.session_path.unlink()
                except OSError:
                    pass

        # If the user pointed us at an existing instagrapi session JSON
        # (exported from another tool, or transplanted from a different
        # machine) load that directly. Bypasses the fresh-login challenge
        # loop entirely.
        external_session = os.environ.get("IG_SESSION_FILE", "").strip()
        if external_session and Path(external_session).exists() and not self.session_path.exists():
            try:
                self.cl.load_settings(external_session)
                self.cl.dump_settings(str(self.session_path))  # copy into our canonical slot
                settings_loaded = True
                log.info("Loaded external session from %s → %s", external_session, self.session_path)
            except Exception as e:
                log.warning("IG_SESSION_FILE load failed: %s — continuing with normal login", e)

        if not settings_loaded and self.session_path.exists():
            try:
                self.cl.load_settings(str(self.session_path))
                settings_loaded = True
            except Exception as e:
                log.warning("Failed to load session, will re-login: %s", e)

        # Cookie-seed paths — two routes depending on how much the user
        # pasted into .env:
        #
        #   FULL set (sessionid + ds_user_id + csrftoken + more):
        #     reconstruct a full instagrapi settings dict and load via
        #     cl.set_settings(). Zero network calls to /login → zero
        #     challenge risk. This is the gold standard for first-boot
        #     on a fresh VPS — identical to moving a warmed-up account
        #     from one box to another.
        #
        #   MINIMAL set (just sessionid):
        #     fall back to login_by_sessionid() which fetches the
        #     remaining cookies from IG. Still cleaner than password
        #     login, but IG DOES observe this call.
        if not settings_loaded and self._cookie_seed:
            if _has_full_cookie_set(self._cookie_seed):
                try:
                    settings = self._build_settings_from_cookies(self._cookie_seed)
                    self.cl.set_settings(settings)
                    # Probe to confirm the session is live
                    self.cl.get_timeline_feed()
                    self._logged_in = True
                    self.cl.dump_settings(str(self.session_path))
                    log.info(
                        "Loaded full cookie set (%d cookies) for %s — no /login call needed",
                        len(self._cookie_seed), self.username,
                    )
                    from src.core.warmup import ensure_started
                    ensure_started()
                    return
                except Exception as e:
                    log.warning(
                        "Full-cookie-set load failed: %s — trying login_by_sessionid", e,
                    )
            try:
                self.cl.login_by_sessionid(self._cookie_seed["sessionid"])
                self._logged_in = True
                self.cl.dump_settings(str(self.session_path))
                log.info("Logged in via IG_SESSIONID for %s", self.username)
                from src.core.warmup import ensure_started
                ensure_started()
                return
            except Exception as e:
                log.warning("IG_SESSIONID login failed: %s — falling back to password login", e)

        try:
            self.cl.login(self.username, self._password)
            # Probe the session — load_settings alone doesn't validate it.
            self.cl.get_timeline_feed()
            self._logged_in = True
            self.cl.dump_settings(str(self.session_path))
            # Record warmup start if this is the first successful login
            from src.core.warmup import ensure_started
            ensure_started()
            log.info("IG login ok for %s (reused session: %s)", self.username, settings_loaded)
        except LoginRequired:
            # Stale session — wipe and try clean login
            log.warning("Session invalid; relogging in clean")
            self.cl = Client()
            dev.apply_to(self.cl)
            self.cl.challenge_code_handler = ch.make_challenge_code_handler()
            self.cl.totp_code_handler = ch.make_totp_handler()
            if os.environ.get("IG_PROXY"):
                self.cl.set_proxy(os.environ["IG_PROXY"])
            self.cl.login(self.username, self._password)
            self.cl.dump_settings(str(self.session_path))
            self._logged_in = True
        except TwoFactorRequired:
            code = ch.totp_code()
            if not code:
                raise RuntimeError("2FA required but IG_TOTP_SECRET not set.")
            self.cl.login(self.username, self._password, verification_code=code)
            self.cl.dump_settings(str(self.session_path))
            self._logged_in = True
        except ChallengeRequired as e:
            # instagrapi attempts to call challenge_code_handler internally; if
            # we land here the handler already raised. Distinguish between:
            #   (a) needs-manual-code — user just hasn't entered it yet, so
            #       DON'T enter cooldown. Re-raise with a clear message so
            #       they can re-run `ig-agent login` and paste the code.
            #   (b) genuine challenge refusal (IG says no) — cooldown 24h.
            inner = getattr(e, "__cause__", None) or getattr(e, "__context__", None)
            if isinstance(inner, ch.ChallengeNeedsManualCode):
                log.error(
                    "IG sent a code but we couldn't read it. Either set "
                    "IMAP_HOST/IMAP_USER/IMAP_PASS in .env, OR run `ig-agent "
                    "login` directly in a terminal to paste the code by hand."
                )
                raise inner from e
            _enter_cooldown("challenge_required", hours=24)
            raise
        except BadPassword:
            log.error("Bad password for %s — aborting.", self.username)
            raise

    def _build_settings_from_cookies(self, cookies: dict[str, str]) -> dict:
        """Assemble an instagrapi settings dict from pasted cookies +
        our persisted device fingerprint. Produces the same shape as
        ``cl.dump_settings()`` so set_settings() accepts it natively.
        """
        import time
        device_settings = dev.load_or_create()
        # Pull UUIDs out of the persisted device file so the cookie
        # jar is consistent with the fingerprint.
        uuids = {
            k: device_settings[k]
            for k in ("phone_id", "uuid", "client_session_id", "advertising_id", "device_id")
            if k in device_settings
        }
        # Slice device keys (instagrapi's expected shape)
        device_keys = (
            "app_version", "android_version", "android_release",
            "dpi", "resolution", "manufacturer", "device", "model",
            "cpu", "version_code",
        )
        device = {k: device_settings[k] for k in device_keys if k in device_settings}

        return {
            "cookies": dict(cookies),
            "last_login": int(time.time()),
            "device_settings": device,
            "user_agent": (
                os.environ.get("IG_USER_AGENT", "").strip()
                or _default_user_agent(device)
            ),
            "authorization_data": {
                "ds_user_id": cookies.get("ds_user_id", ""),
                "sessionid": cookies.get("sessionid", ""),
                "should_use_header_over_cookies": True,
            },
            "uuids": uuids,
            "mid": cookies.get("mid", ""),
            "ig_u_rur": cookies.get("rur", ""),
            "ig_www_claim": "",
            # country / timezone / locale echoed so instagrapi's
            # header builders use them consistently with our settings.
            "country": os.environ.get("IG_COUNTRY_CODE", "").strip() or "US",
            "country_code": 1,
            "locale": os.environ.get("IG_LOCALE", "").strip() or "en_US",
            "timezone_offset": (
                int(os.environ.get("IG_TIMEZONE_OFFSET", "0").strip() or "0")
            ),
        }

    def _ensure_backoff_ok(self) -> None:
        until = db.state_get("backoff_until")
        if until and until > db.now_iso():
            raise BackoffActive(f"Cooldown active until {until}")

    # ───── Uploads ─────
    def upload_photo(self, path: str, caption: str) -> str:
        self._ensure_backoff_ok()
        self._ensure_logged_in()
        media = self._retry(lambda: self.cl.photo_upload(path, caption))
        return str(media.pk)

    def upload_album(self, paths: list[str], caption: str) -> str:
        self._ensure_backoff_ok()
        self._ensure_logged_in()
        media = self._retry(lambda: self.cl.album_upload(paths, caption))
        return str(media.pk)

    def upload_reel(
        self,
        video_path: str,
        caption: str,
        thumbnail: str | None = None,
    ) -> str:
        self._ensure_backoff_ok()
        self._ensure_logged_in()
        thumb_path = Path(thumbnail) if thumbnail else None
        media = self._retry(
            lambda: self.cl.clip_upload(
                Path(video_path),
                caption,
                thumbnail=thumb_path,
            )
        )
        return str(media.pk)

    def upload_story_image(
        self,
        path: str,
        caption: str = "",
        *,
        mention: str | None = None,
        hashtag: str | None = None,
        link: str | None = None,
    ) -> str:
        self._ensure_backoff_ok()
        self._ensure_logged_in()
        mentions, hashtags, links = self._story_stickers(mention, hashtag, link)
        media = self._retry(
            lambda: self.cl.photo_upload_to_story(
                Path(path),
                caption,
                mentions=mentions,
                hashtags=hashtags,
                links=links,
            )
        )
        return str(media.pk)

    def upload_story_video(
        self,
        path: str,
        caption: str = "",
        *,
        mention: str | None = None,
        hashtag: str | None = None,
        link: str | None = None,
    ) -> str:
        self._ensure_backoff_ok()
        self._ensure_logged_in()
        mentions, hashtags, links = self._story_stickers(mention, hashtag, link)
        media = self._retry(
            lambda: self.cl.video_upload_to_story(
                Path(path),
                caption,
                mentions=mentions,
                hashtags=hashtags,
                links=links,
            )
        )
        return str(media.pk)

    def _story_stickers(
        self,
        mention: str | None,
        hashtag: str | None,
        link: str | None,
    ):
        from instagrapi.types import StoryHashtag, StoryLink, StoryMention

        mentions: list = []
        if mention:
            try:
                user = self.cl.user_info_by_username(mention.lstrip("@"))
                mentions.append(
                    StoryMention(
                        user=user,
                        x=0.5, y=0.92,
                        width=0.5, height=0.05,
                    )
                )
            except Exception as e:
                log.debug("Skipping mention sticker (%s): %s", mention, e)

        hashtags: list = []
        if hashtag:
            try:
                tag_info = self.cl.hashtag_info(hashtag.lstrip("#"))
                hashtags.append(
                    StoryHashtag(
                        hashtag=tag_info,
                        x=0.5, y=0.1,
                        width=0.4, height=0.05,
                    )
                )
            except Exception as e:
                log.debug("Skipping hashtag sticker (%s): %s", hashtag, e)

        links: list = []
        if link:
            try:
                links.append(StoryLink(webUri=link))
            except Exception as e:
                log.debug("Skipping story link (%s): %s", link, e)

        return mentions, hashtags, links

    # ───── Engagement ─────
    def _ensure_post_cooldown_clear(self) -> None:
        """Enforce the 30-90min silence after a post. Skipping write
        actions inside that window stops the "posted + engaged within
        seconds" bot-script fingerprint. Reads always allowed."""
        from src.plugins import human_mimic as _hm
        remaining = _hm.post_cooldown_remaining_s()
        if remaining > 0:
            raise BackoffActive(
                f"post-cooldown active — {int(remaining / 60)}min of silence remaining "
                "after the most recent post"
            )

    def like(self, media_pk: str) -> bool:
        self._ensure_backoff_ok()
        self._ensure_post_cooldown_clear()
        self._ensure_logged_in()
        return bool(self._retry(lambda: self.cl.media_like(media_pk)))

    def follow(self, user_id: str) -> bool:
        self._ensure_backoff_ok()
        self._ensure_post_cooldown_clear()
        self._ensure_logged_in()
        return bool(self._retry(lambda: self.cl.user_follow(user_id)))

    def unfollow(self, user_id: str) -> bool:
        self._ensure_backoff_ok()
        self._ensure_post_cooldown_clear()
        self._ensure_logged_in()
        return bool(self._retry(lambda: self.cl.user_unfollow(user_id)))

    def comment(self, media_pk: str, text: str) -> str:
        self._ensure_backoff_ok()
        # Note: no post-cooldown gate here — the first-comment-hashtag
        # drop in poster.py IS the exception that justifies posting a
        # comment immediately after an upload. Engager-driven comments
        # on OTHER users' posts go through _ensure_post_cooldown_clear
        # via the worker layer.
        self._ensure_logged_in()

        # Typing delay for a human-shaped submit time. Uses cfg from
        # the env-facing helper — avoids threading NicheConfig through
        # every call site. Only adds latency; no failure mode.
        import os as _os
        if _os.environ.get("IG_DISABLE_TYPING_DELAYS", "") != "1":
            from src.plugins import human_mimic as _hm
            _hm.sleep_typing(text)

        com = self._retry(lambda: self.cl.media_comment(media_pk, text))
        return str(getattr(com, "pk", com))

    def media_comments(self, media_pk: str, limit: int = 30) -> list[dict]:
        self._ensure_logged_in()
        raw = self._retry(lambda: self.cl.media_comments(media_pk, amount=limit))
        out: list[dict] = []
        for c in raw or []:
            try:
                out.append({
                    "pk": str(c.pk),
                    "text": c.text or "",
                    "user_id": str(getattr(c.user, "pk", "") or ""),
                    "username": getattr(c.user, "username", "") or "",
                    "created_at": c.created_at_utc.isoformat() if getattr(c, "created_at_utc", None) else "",
                    "reply_to": getattr(c, "replied_to_comment_id", None),
                })
            except Exception:
                continue
        return out

    def reply_to_comment(self, media_pk: str, comment_pk: str, text: str) -> str:
        """Reply to a comment in-thread (instagrapi wraps the same endpoint)."""
        self._ensure_backoff_ok()
        self._ensure_logged_in()
        com = self._retry(
            lambda: self.cl.media_comment(media_pk, text, replied_to_comment_id=int(comment_pk))
        )
        return str(getattr(com, "pk", com))

    def pending_followers(self, amount: int = 50) -> list[dict]:
        """Return recent followers we don't follow back yet."""
        self._ensure_logged_in()
        me = self.cl.user_id
        followers = self._retry(lambda: self.cl.user_followers(me, amount=amount))
        following = self._retry(lambda: self.cl.user_following(me, amount=amount * 2))
        following_ids = set(following.keys())
        out: list[dict] = []
        for uid, user in followers.items():
            if uid in following_ids:
                continue
            out.append({
                "user_id": str(uid),
                "username": getattr(user, "username", "") or "",
                "full_name": getattr(user, "full_name", "") or "",
                "is_private": bool(getattr(user, "is_private", False)),
                "is_verified": bool(getattr(user, "is_verified", False)),
            })
        return out

    def send_dm(self, user_id: str, text: str) -> str:
        self._ensure_backoff_ok()
        self._ensure_logged_in()
        thread = self._retry(
            lambda: self.cl.direct_send(text, user_ids=[int(user_id)])
        )
        return str(getattr(thread, "id", "") or "")

    def view_stories(self, user_id: str) -> int:
        self._ensure_backoff_ok()
        self._ensure_logged_in()
        stories = self._retry(lambda: self.cl.user_stories(user_id))
        if not stories:
            return 0
        self._retry(lambda: self.cl.story_seen(stories))
        return len(stories)

    # ───── Scrapes (self) ─────
    def self_info(self) -> dict[str, Any]:
        self._ensure_logged_in()
        info = self.cl.account_info()
        return info.model_dump() if hasattr(info, "model_dump") else dict(info)

    def media_metrics(self, media_pk: str) -> dict[str, Any]:
        self._ensure_logged_in()
        info = self.cl.media_info(media_pk)
        return {
            "likes": getattr(info, "like_count", 0) or 0,
            "comments": getattr(info, "comment_count", 0) or 0,
            "play_count": getattr(info, "play_count", 0) or 0,
            "view_count": getattr(info, "view_count", 0) or 0,
        }

    def user_id_from_username(self, username: str) -> str:
        self._ensure_logged_in()
        return str(self.cl.user_id_from_username(username))

    def user_medias(self, user_id: str, amount: int = 10) -> list[dict[str, Any]]:
        self._ensure_logged_in()
        medias = self.cl.user_medias(user_id, amount)
        return [
            {
                "pk": str(m.pk),
                "caption": m.caption_text or "",
                "likes": m.like_count or 0,
                "comments": m.comment_count or 0,
                "taken_at": m.taken_at.isoformat() if m.taken_at else None,
                "media_type": int(m.media_type) if m.media_type is not None else 0,
                "url": f"https://www.instagram.com/p/{m.code}/" if m.code else "",
            }
            for m in medias
        ]

    def hashtag_top(self, hashtag: str, amount: int = 20) -> list[dict[str, Any]]:
        self._ensure_logged_in()
        medias = self.cl.hashtag_medias_top(hashtag, amount=amount)
        return [
            {
                "pk": str(m.pk),
                "caption": m.caption_text or "",
                "likes": m.like_count or 0,
                "comments": m.comment_count or 0,
                "taken_at": m.taken_at.isoformat() if m.taken_at else None,
                "username": m.user.username if getattr(m, "user", None) else "",
            }
            for m in medias
        ]

    # ───── Internal ─────
    def _ensure_logged_in(self) -> None:
        # Client rotation — every 2-4h, tear down the underlying
        # instagrapi Client and load a fresh one from the persisted
        # session so we reset the TCP pool + HTTP/2 stream IDs. Real
        # users close the app; bot scripts hold one connection for
        # days. Mimic the former.
        if self._should_rotate_client():
            self._rotate_client_now()

        if not self._logged_in:
            self.login()

    def _should_rotate_client(self) -> bool:
        import time as _time
        from src.plugins import human_mimic as _hm
        age = _time.time() - self._client_created_at
        try:
            return _hm.should_rotate_client(age, seed_ts=self._client_created_at)
        except Exception:
            return False

    def _rotate_client_now(self) -> None:
        """Rebuild the underlying instagrapi Client with the persisted
        session + device. On failure, keep the old client — this is a
        polish pass, not a correctness step."""
        import time as _time
        try:
            new_cl = Client()
            dev.apply_to(new_cl)
            if os.environ.get("IG_PROXY"):
                new_cl.set_proxy(os.environ["IG_PROXY"])
            new_cl.challenge_code_handler = ch.make_challenge_code_handler()
            new_cl.totp_code_handler = ch.make_totp_handler()
            new_cl.delay_range = [2, 6]
            if self.session_path.exists():
                new_cl.load_settings(str(self.session_path))
            self.cl = new_cl
            self._client_created_at = _time.time()
            log.info("client rotation: new instagrapi Client instantiated")
        except Exception as e:
            log.warning("client rotation failed (keeping old client): %s", e)

    def _retry(self, fn, attempts: int = 3):
        last: Exception | None = None
        for i in range(attempts):
            start = time.monotonic()
            try:
                out = fn()
                db.action_log(
                    fn.__name__ if hasattr(fn, "__name__") else "call",
                    None,
                    "ok",
                    int((time.monotonic() - start) * 1000),
                )
                # Human-ish spacing between IG calls
                time.sleep(random.uniform(2.0, 5.0))
                return out
            except PleaseWaitFewMinutes as e:
                last = e
                wait = 300 * (i + 1)
                log.warning("PleaseWaitFewMinutes — backing off %ds", wait)
                time.sleep(wait)
            except RateLimitError as e:
                last = e
                _enter_cooldown("rate_limit", hours=1 + i)
                raise
            except ChallengeRequired as e:
                _enter_cooldown("challenge", hours=24)
                raise e
            except (LoginRequired,) as e:
                last = e
                log.warning("LoginRequired mid-call — relogging")
                try:
                    self.login()
                except Exception:
                    raise
                continue
            except ClientError as e:
                last = e
                log.warning("IG ClientError: %s — retrying", e)
                time.sleep(5 * (i + 1))
        db.action_log("retry_exhausted", None, "failed", 0)
        if last:
            raise last
        raise RuntimeError("retry exhausted without exception")


def _enter_cooldown(reason: str, hours: int) -> None:
    """Record a global backoff state. Orchestrator checks this before acting."""
    from datetime import datetime, timedelta, timezone

    until = (datetime.now(timezone.utc) + timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
    db.state_set("backoff_until", until)
    db.state_set("backoff_reason", reason)
    log.error("Entering cooldown: %s (until %s)", reason, until)
