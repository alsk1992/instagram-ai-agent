#!/usr/bin/env python
"""Live smoke-test every external integration we ship.

Unit tests use mocks — they prove the parser + wiring are correct but
can't catch silent API breakages (endpoint moved, auth tightened, schema
drifted). This script actually hits the real endpoints.

Run:
    python scripts/smoke_external.py

For each integration reports OK / FAIL with the exact reason. Exit code
non-zero when any required integration fails.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

# Make imports work from a source checkout
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import httpx


GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
DIM = "\033[2m"
RESET = "\033[0m"


results: list[tuple[str, str, str]] = []  # (name, status, detail); status ∈ {ok, fail, skip}


def record(name: str, status: str, detail: str) -> None:
    results.append((name, status, detail))
    marker = {
        "ok": f"{GREEN}✓{RESET}",
        "skip": f"{YELLOW}∼{RESET}",
        "fail": f"{RED}✗{RESET}",
    }.get(status, "?")
    print(f"  {marker} {name:<28}  {detail}")


# ─── Keyless endpoints ────────────────────────────────────────────
async def smoke_hackernews() -> None:
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(15.0),
            headers={"User-Agent": "ig-agent-smoke/0.2"},
        ) as client:
            r = await client.get(
                "https://hn.algolia.com/api/v1/search",
                params={"tags": "front_page", "hitsPerPage": 3},
            )
            r.raise_for_status()
            hits = (r.json() or {}).get("hits") or []
            if hits:
                record("HackerNews Algolia", "ok", f"{len(hits)} hits, e.g. {hits[0].get('title', '')[:50]!r}")
            else:
                record("HackerNews Algolia", "fail", "returned zero hits")
    except Exception as e:
        record("HackerNews Algolia", "fail", f"{type(e).__name__}: {e}")


async def smoke_devto() -> None:
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(15.0),
            headers={"User-Agent": "ig-agent-smoke/0.2"},
        ) as client:
            r = await client.get(
                "https://dev.to/api/articles",
                params={"tag": "python", "top": 7, "per_page": 3},
            )
            r.raise_for_status()
            items = r.json() or []
            if items:
                record("Dev.to API", "ok", f"{len(items)} items, e.g. {items[0].get('title', '')[:50]!r}")
            else:
                record("Dev.to API", "fail", "returned empty list")
    except Exception as e:
        record("Dev.to API", "fail", f"{type(e).__name__}: {e}")


async def smoke_wiki_otd() -> None:
    try:
        today = time.gmtime()
        url = f"https://en.wikipedia.org/api/rest_v1/feed/onthisday/events/{today.tm_mon:02d}/{today.tm_mday:02d}"
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(15.0),
            headers={"User-Agent": "ig-agent-smoke/0.2"},
        ) as client:
            r = await client.get(url)
            r.raise_for_status()
            events = (r.json() or {}).get("events") or []
            if events:
                record("Wikipedia OTD", "ok", f"{len(events)} events for today, e.g. {events[0].get('year')}: {events[0].get('text', '')[:40]!r}")
            else:
                record("Wikipedia OTD", "fail", "no events returned")
    except Exception as e:
        record("Wikipedia OTD", "fail", f"{type(e).__name__}: {e}")


async def smoke_openverse() -> None:
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(20.0),
            headers={"User-Agent": "ig-agent-smoke/0.2"},
        ) as client:
            r = await client.get(
                "https://api.openverse.org/v1/images/",
                params={"q": "sunset", "page_size": 3, "license_type": "commercial"},
            )
            r.raise_for_status()
            hits = (r.json() or {}).get("results") or []
            if hits:
                record("Openverse", "ok", f"{len(hits)} CC/PD hits, first licence={hits[0].get('license')}")
            else:
                record("Openverse", "fail", "returned zero results")
    except Exception as e:
        record("Openverse", "fail", f"{type(e).__name__}: {e}")


async def smoke_pollinations() -> None:
    """Real image generation is slow (30s+) — just verify the endpoint
    responds with a 2xx + a non-zero body on HEAD/short GET."""
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(90.0),
            headers={"User-Agent": "ig-agent-smoke/0.2"},
        ) as client:
            # Tiny image to prove the pipeline works end-to-end
            r = await client.get(
                "https://image.pollinations.ai/prompt/smoketest",
                params={"width": 128, "height": 128, "nologo": "true", "seed": 1},
                follow_redirects=True,
            )
            r.raise_for_status()
            size = len(r.content)
            if size > 1000:
                record("Pollinations", "ok", f"returned {size / 1024:.1f} KB image")
            else:
                record("Pollinations", "fail", f"tiny response: {size}B — endpoint may be down")
    except Exception as e:
        record("Pollinations", "fail", f"{type(e).__name__}: {e}")


async def smoke_nager_date() -> None:
    """Holidays API — used by brain/events.py for themed-day seeds."""
    try:
        year = time.gmtime().tm_year
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(10.0),
            headers={"User-Agent": "ig-agent-smoke/0.2"},
        ) as client:
            r = await client.get(f"https://date.nager.at/api/v3/PublicHolidays/{year}/US")
            r.raise_for_status()
            holidays = r.json() or []
            if holidays:
                record("Nager.Date holidays", "ok", f"{len(holidays)} US holidays for {year}")
            else:
                record("Nager.Date holidays", "fail", "empty list")
    except Exception as e:
        record("Nager.Date holidays", "fail", f"{type(e).__name__}: {e}")


# ─── Key-gated endpoints ──────────────────────────────────────────
async def smoke_openrouter() -> None:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        record("OpenRouter", "skip", f"{YELLOW}skipped — no OPENROUTER_API_KEY in env{RESET}")
        return
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            r = await client.get(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {key}"},
            )
            if r.status_code == 200:
                n = len((r.json() or {}).get("data") or [])
                record("OpenRouter", "ok", f"{n} models available")
            else:
                record("OpenRouter", "fail", f"HTTP {r.status_code}")
    except Exception as e:
        record("OpenRouter", "fail", f"{type(e).__name__}: {e}")


async def smoke_pexels() -> None:
    key = os.environ.get("PEXELS_API_KEY")
    if not key:
        record("Pexels", "skip", f"{YELLOW}skipped — no PEXELS_API_KEY in env{RESET}")
        return
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            r = await client.get(
                "https://api.pexels.com/videos/search",
                params={"query": "sunset", "per_page": 1},
                headers={"Authorization": key},
            )
            r.raise_for_status()
            n = len((r.json() or {}).get("videos") or [])
            record("Pexels", "ok" if n > 0 else "fail", f"{n} results")
    except Exception as e:
        record("Pexels", "fail", f"{type(e).__name__}: {e}")


async def smoke_pixabay() -> None:
    key = os.environ.get("PIXABAY_API_KEY")
    if not key:
        record("Pixabay", "skip", f"{YELLOW}skipped — no PIXABAY_API_KEY in env{RESET}")
        return
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            r = await client.get(
                "https://pixabay.com/api/",
                params={"key": key, "q": "sunset", "per_page": 3},
            )
            r.raise_for_status()
            total = (r.json() or {}).get("total", 0)
            record("Pixabay", "ok" if total > 0 else "fail", f"{total} total hits")
    except Exception as e:
        record("Pixabay", "fail", f"{type(e).__name__}: {e}")


# ─── Main ─────────────────────────────────────────────────────────
async def main() -> int:
    print(f"\n{DIM}Live smoke test — hits every external endpoint with a real request.{RESET}\n")

    print(f"{DIM}— keyless (anyone can run these):{RESET}")
    await asyncio.gather(
        smoke_hackernews(),
        smoke_devto(),
        smoke_wiki_otd(),
        smoke_openverse(),
        smoke_nager_date(),
    )
    await smoke_pollinations()  # slow — not parallelised

    print(f"\n{DIM}— key-gated (skipped unless env var is set):{RESET}")
    await asyncio.gather(
        smoke_openrouter(),
        smoke_pexels(),
        smoke_pixabay(),
    )

    n_ok = sum(1 for _, s, _ in results if s == "ok")
    n_skip = sum(1 for _, s, _ in results if s == "skip")
    n_fail = sum(1 for _, s, _ in results if s == "fail")

    print(f"\n{DIM}— summary:{RESET}")
    print(f"  {n_ok} ok · {n_skip} skipped (no key) · {n_fail} failed")
    if n_fail:
        print(f"\n{RED}failures:{RESET}")
        for name, s, detail in results:
            if s == "fail":
                print(f"  {RED}•{RESET} {name}: {detail}")
        return 1
    if n_ok > 0 and n_fail == 0:
        print(f"\n{GREEN}all green{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
