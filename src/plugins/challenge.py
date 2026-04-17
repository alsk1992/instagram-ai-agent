"""Challenge resolver — email IMAP + TOTP.

Plugged into instagrapi via `cl.challenge_code_handler = ...`. When IG sends
a security code, we poll the inbox for it (Gmail/Outlook/etc.), extract the
6-digit code, and hand it back. For 2FA, we generate via TOTP secret.
"""
from __future__ import annotations

import os
import re
import time
from typing import Any

import pyotp
from imap_tools import AND, MailBox

from src.core import db
from src.core.logging_setup import get_logger

log = get_logger(__name__)

_CODE_RE = re.compile(r"\b(\d{6})\b")
# Subjects IG actually sends. Keep regex loose — they iterate on wording.
_SUBJECT_HINTS = (
    "instagram",
    "security code",
    "verification",
    "log in",
    "confirm your account",
)


def _imap_settings() -> tuple[str, int, str, str] | None:
    host = os.environ.get("IMAP_HOST")
    user = os.environ.get("IMAP_USER")
    pw = os.environ.get("IMAP_PASS")
    if not (host and user and pw):
        return None
    port = int(os.environ.get("IMAP_PORT", "993"))
    return host, port, user, pw


def fetch_email_code(timeout_s: int = 180, poll_s: int = 10) -> str | None:
    """Poll the configured IMAP inbox until an Instagram code arrives."""
    settings = _imap_settings()
    if settings is None:
        log.warning("IMAP not configured; cannot auto-resolve email challenge.")
        return None
    host, port, user, pw = settings

    start = time.monotonic()
    seen: set[str] = set()
    while time.monotonic() - start < timeout_s:
        try:
            with MailBox(host, port=port).login(user, pw, initial_folder="INBOX") as mb:
                # Recent messages first. Only look at the last 15 minutes to
                # avoid picking up stale codes from a previous flow.
                cutoff = time.strftime("%d-%b-%Y", time.gmtime(time.time() - 900))
                messages = list(mb.fetch(AND(date_gte=cutoff), reverse=True, limit=25))
                for m in messages:
                    if m.uid in seen:
                        continue
                    seen.add(m.uid)
                    subj = (m.subject or "").lower()
                    body = (m.text or m.html or "").lower()
                    if not any(h in subj or h in body for h in _SUBJECT_HINTS):
                        continue
                    # Prefer codes found in subject, then body.
                    for src in (m.subject or "", m.text or m.html or ""):
                        match = _CODE_RE.search(src)
                        if match:
                            code = match.group(1)
                            log.info("Email challenge code fetched from IMAP: %s", code)
                            return code
        except Exception as e:
            log.warning("IMAP poll error: %s", e)

        time.sleep(poll_s)

    log.error("No challenge code found in inbox within %ds", timeout_s)
    return None


def totp_code(secret: str | None = None) -> str | None:
    s = secret or os.environ.get("IG_TOTP_SECRET")
    if not s:
        return None
    # Normalize: uppercase, strip spaces (pyotp accepts base32)
    normalized = s.replace(" ", "").upper()
    return pyotp.TOTP(normalized).now()


def make_challenge_code_handler():
    """Return a callable compatible with instagrapi's challenge_code_handler."""

    def handler(username: str, choice: Any) -> str:
        cid = db.challenge_log("code_required", {"username": username, "choice": str(choice)})
        # instagrapi passes an enum/int indicating email (1) vs sms (0). We try
        # the inbox first since SMS auto-resolution requires paid services.
        code = fetch_email_code()
        if code:
            db.challenge_resolve(cid, "imap")
            return code
        raise RuntimeError(
            f"Challenge for {username!r} needs manual code entry "
            "(IMAP not configured or email not received)."
        )

    return handler


def make_totp_handler():
    def handler(username: str) -> str:
        code = totp_code()
        if not code:
            raise RuntimeError(f"TOTP requested for {username} but IG_TOTP_SECRET not set.")
        return code

    return handler
