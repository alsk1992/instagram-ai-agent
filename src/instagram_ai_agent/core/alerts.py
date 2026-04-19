"""Telegram alerts — fire-and-forget notifications on key events."""
from __future__ import annotations

import asyncio
import os
from typing import Any

import aiohttp

from instagram_ai_agent.core.logging_setup import get_logger

log = get_logger(__name__)


async def send(text: str, *, level: str = "info") -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not (token and chat_id):
        return False

    emoji = {"info": "ℹ️", "ok": "✅", "warn": "⚠️", "err": "🚨"}.get(level, "")
    body = f"{emoji} {text}" if emoji else text
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload: dict[str, Any] = {
        "chat_id": chat_id,
        "text": body[:4000],
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as sess:
            async with sess.post(url, json=payload) as r:
                if r.status != 200:
                    log.warning("Telegram send failed: %s", await r.text())
                    return False
        return True
    except Exception as e:
        log.warning("Telegram send error: %s", e)
        return False


def send_sync(text: str, *, level: str = "info") -> bool:
    """Sync wrapper for convenience from non-async code paths."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Fire-and-forget from within an async context
            asyncio.ensure_future(send(text, level=level))
            return True
    except RuntimeError as e:
        # No running event loop — fall through to synchronous asyncio.run.
        log.debug("alerts: no running loop (%s) — using asyncio.run fallback", e)
    return asyncio.run(send(text, level=level))
