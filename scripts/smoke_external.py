#!/usr/bin/env python
"""Live smoke-test every external + local integration we ship.

Unit tests use mocks — they prove the parser + wiring are correct but
can't catch: silent API breakage (endpoint moved, auth tightened, schema
drift) OR broken local tooling (ffmpeg missing, Playwright half-installed,
a library that pip-installed but crashes on import). This script actually
exercises the real thing.

Run:
    python scripts/smoke_external.py

Report per integration: OK / SKIP (missing key or optional dep) / FAIL.
Exit 0 when all checks are ok-or-skip, exit 1 on any real failure.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
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


async def smoke_groq() -> None:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        record("Groq", "skip", f"{YELLOW}skipped — no GROQ_API_KEY in env{RESET}")
        return
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            r = await client.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {key}"},
            )
            r.raise_for_status()
            n = len((r.json() or {}).get("data") or [])
            record("Groq", "ok" if n > 0 else "fail", f"{n} models available")
    except Exception as e:
        record("Groq", "fail", f"{type(e).__name__}: {e}")


async def smoke_gemini() -> None:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        record("Gemini", "skip", f"{YELLOW}skipped — no GEMINI_API_KEY in env{RESET}")
        return
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            r = await client.get(
                f"https://generativelanguage.googleapis.com/v1beta/models?key={key}",
            )
            r.raise_for_status()
            n = len((r.json() or {}).get("models") or [])
            record("Gemini", "ok" if n > 0 else "fail", f"{n} models available")
    except Exception as e:
        record("Gemini", "fail", f"{type(e).__name__}: {e}")


async def smoke_cerebras() -> None:
    key = os.environ.get("CEREBRAS_API_KEY")
    if not key:
        record("Cerebras", "skip", f"{YELLOW}skipped — no CEREBRAS_API_KEY in env{RESET}")
        return
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            r = await client.get(
                "https://api.cerebras.ai/v1/models",
                headers={"Authorization": f"Bearer {key}"},
            )
            r.raise_for_status()
            n = len((r.json() or {}).get("data") or [])
            record("Cerebras", "ok" if n > 0 else "fail", f"{n} models available")
    except Exception as e:
        record("Cerebras", "fail", f"{type(e).__name__}: {e}")


async def smoke_freesound() -> None:
    key = os.environ.get("FREESOUND_API_KEY")
    if not key:
        record("Freesound", "skip", f"{YELLOW}skipped — no FREESOUND_API_KEY in env{RESET}")
        return
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            r = await client.get(
                "https://freesound.org/apiv2/search/text/",
                params={"query": "ambient", "page_size": 1, "token": key},
            )
            r.raise_for_status()
            n = (r.json() or {}).get("count", 0)
            record("Freesound", "ok" if n > 0 else "fail", f"{n} matches")
    except Exception as e:
        record("Freesound", "fail", f"{type(e).__name__}: {e}")


async def smoke_telegram() -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not (token and chat):
        record("Telegram", "skip", f"{YELLOW}skipped — TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set{RESET}")
        return
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            # getMe doesn't send a message — just proves the token is valid
            r = await client.get(f"https://api.telegram.org/bot{token}/getMe")
            r.raise_for_status()
            me = (r.json() or {}).get("result") or {}
            record("Telegram", "ok", f"bot {me.get('username', '?')!r} reachable")
    except Exception as e:
        record("Telegram", "fail", f"{type(e).__name__}: {e}")


async def smoke_reddit() -> None:
    cid = os.environ.get("REDDIT_CLIENT_ID")
    secret = os.environ.get("REDDIT_CLIENT_SECRET")
    ua = os.environ.get("REDDIT_USER_AGENT") or "ig-agent-smoke/0.2"
    if not (cid and secret):
        record("Reddit (PRAW)", "skip", f"{YELLOW}skipped — REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET not set{RESET}")
        return
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            r = await client.post(
                "https://www.reddit.com/api/v1/access_token",
                auth=(cid, secret),
                headers={"User-Agent": ua},
                data={"grant_type": "client_credentials"},
            )
            if r.status_code == 200 and (r.json() or {}).get("access_token"):
                record("Reddit (PRAW)", "ok", "client_credentials token granted")
            else:
                record("Reddit (PRAW)", "fail", f"HTTP {r.status_code}")
    except Exception as e:
        record("Reddit (PRAW)", "fail", f"{type(e).__name__}: {e}")


async def smoke_comfyui() -> None:
    url = os.environ.get("COMFYUI_URL")
    if not url:
        record("ComfyUI (local)", "skip", f"{YELLOW}skipped — COMFYUI_URL not set{RESET}")
        return
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
            r = await client.get(url.rstrip("/") + "/system_stats")
            r.raise_for_status()
            record("ComfyUI (local)", "ok", f"reachable at {url}")
    except Exception as e:
        record("ComfyUI (local)", "fail", f"{type(e).__name__}: {e}")


# ─── Local binaries + libraries ───────────────────────────────────
def smoke_ffmpeg() -> None:
    # ffmpeg is used everywhere (video + LUT + subtitle burn-in) — hard-fail
    # if missing. ffprobe is only used for MP3 duration probing; we fall back
    # to mutagen (pure-Python) at runtime, so a missing ffprobe is a skip.
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        try:
            import imageio_ffmpeg
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg_path = None
    if ffmpeg_path:
        try:
            out = subprocess.run(
                [ffmpeg_path, "-version"], capture_output=True, text=True, timeout=8,
            )
            ver = (out.stdout or "").splitlines()[0][:60]
            record("ffmpeg", "ok", ver)
        except Exception as e:
            record("ffmpeg", "fail", f"{type(e).__name__}: {e}")
    else:
        record("ffmpeg", "fail", "not on PATH and no imageio-ffmpeg fallback")

    ffprobe_path = shutil.which("ffprobe")
    if ffprobe_path:
        try:
            out = subprocess.run(
                [ffprobe_path, "-version"], capture_output=True, text=True, timeout=8,
            )
            ver = (out.stdout or "").splitlines()[0][:60]
            record("ffprobe", "ok", ver)
        except Exception as e:
            record("ffprobe", "fail", f"{type(e).__name__}: {e}")
    else:
        # mutagen provides the pure-Python MP3 duration probe we actually need
        try:
            import mutagen  # noqa: F401
            record("ffprobe", "skip",
                   f"{YELLOW}not on PATH — using mutagen fallback at runtime{RESET}")
        except Exception:
            record("ffprobe", "fail", "not on PATH and mutagen missing")


async def smoke_playwright_chromium() -> None:
    try:
        from playwright.async_api import async_playwright
    except Exception as e:
        record("Playwright (lib)", "fail", f"import failed: {e}")
        return
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(args=["--no-sandbox"])
            ctx = await browser.new_context()
            page = await ctx.new_page()
            await page.set_content("<h1>ok</h1>")
            title = await page.evaluate("() => document.querySelector('h1').textContent")
            await browser.close()
            record("Playwright chromium", "ok" if title == "ok" else "fail",
                   "headless launch + set_content roundtrip" if title == "ok" else "launch ok, evaluate failed")
    except Exception as e:
        record("Playwright chromium", "fail", f"{type(e).__name__}: {e}")


def smoke_pillow() -> None:
    try:
        from PIL import Image, ImageDraw
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            tmp = Path(f.name)
        img = Image.new("RGB", (64, 64), (10, 10, 10))
        draw = ImageDraw.Draw(img)
        draw.rectangle([4, 4, 60, 60], outline=(201, 169, 97), width=2)
        img.save(tmp, "JPEG", quality=90)
        size = tmp.stat().st_size
        tmp.unlink(missing_ok=True)
        record("Pillow", "ok" if size > 200 else "fail", f"round-trip JPEG {size}B")
    except Exception as e:
        record("Pillow", "fail", f"{type(e).__name__}: {e}")


def smoke_film_emulation() -> None:
    """Prove the module we just wrote actually runs end-to-end on Pillow."""
    try:
        from PIL import Image
        import tempfile
        from instagram_ai_agent.plugins import film_emulation
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            tmp = Path(f.name)
        Image.new("RGB", (128, 128), (120, 120, 120)).save(tmp, "JPEG", quality=95)
        before = tmp.read_bytes()
        film_emulation.apply_film_look(tmp, strength="medium", seed=1)
        after = tmp.read_bytes()
        tmp.unlink(missing_ok=True)
        record("film_emulation", "ok" if before != after else "fail",
               "grain+vignette+cast applied" if before != after else "no-op")
    except Exception as e:
        record("film_emulation", "fail", f"{type(e).__name__}: {e}")


async def smoke_edge_tts() -> None:
    """edge-tts: Microsoft Edge WSS endpoint, keyless, ships a real voice."""
    try:
        import edge_tts
    except Exception as e:
        record("edge-tts (MS Edge)", "fail", f"import failed: {e}")
        return
    try:
        import tempfile
        comm = edge_tts.Communicate(text="smoke test", voice="en-US-AriaNeural")
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            out = Path(f.name)
        await comm.save(str(out))
        size = out.stat().st_size
        out.unlink(missing_ok=True)
        record("edge-tts (MS Edge)", "ok" if size > 1000 else "fail",
               f"{size / 1024:.1f} KB mp3 generated")
    except Exception as e:
        record("edge-tts (MS Edge)", "fail", f"{type(e).__name__}: {e}")


def smoke_kokoro() -> None:
    """Kokoro: Apache-2.0 local TTS. Optional — only probe when installed.

    Forces CPU device (via IG_KOKORO_DEVICE env or explicit torch.device('cpu'))
    so the probe doesn't crash on older GPUs with cuDNN-SM mismatch — a
    common dev-machine failure mode. CI usually runs on CPU anyway.
    """
    try:
        import kokoro  # noqa: F401
    except Exception:
        record("Kokoro TTS (local)", "skip", f"{YELLOW}skipped — kokoro not installed{RESET}")
        return
    try:
        import tempfile
        import numpy as np
        import soundfile as sf
        import torch
        from kokoro import KPipeline
        device = torch.device(os.environ.get("IG_KOKORO_DEVICE", "cpu"))
        pipe = KPipeline(lang_code="a", device=device)
        chunks = [audio.numpy() for _, _, audio in pipe("smoke test", voice="af_bella")]
        full = np.concatenate(chunks) if chunks else np.zeros(24000)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            out = Path(f.name)
        sf.write(str(out), full, 24000)
        size = out.stat().st_size
        out.unlink(missing_ok=True)
        record("Kokoro TTS (local)", "ok" if size > 10_000 else "fail",
               f"{size / 1024:.1f} KB wav generated on {device}")
    except Exception as e:
        record("Kokoro TTS (local)", "fail", f"{type(e).__name__}: {e}")


def smoke_sqlite_integrity() -> None:
    try:
        # Use a tmp DB so we don't touch the user's brain.db
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            tmp = Path(f.name)
        import sqlite3
        con = sqlite3.connect(str(tmp))
        con.execute("CREATE TABLE x (a INT)")
        con.execute("INSERT INTO x VALUES (1)")
        con.commit()
        row = con.execute("PRAGMA integrity_check").fetchone()
        con.close()
        tmp.unlink(missing_ok=True)
        record("SQLite integrity", "ok" if row and row[0] == "ok" else "fail",
               f"pragma returned {row[0] if row else '?'}")
    except Exception as e:
        record("SQLite integrity", "fail", f"{type(e).__name__}: {e}")


def smoke_optional_libs() -> None:
    """Soft-probe each optional extra so the report shows what's enabled."""
    for label, module in (
        ("curl_cffi (TLS)", "curl_cffi"),
        ("praw (Reddit)", "praw"),
        ("sentence-transformers (RAG)", "sentence_transformers"),
        ("librosa (beat-sync)", "librosa"),
        ("whisperx (kinetic caps)", "whisperx"),
        ("realesrgan (finish pass)", "realesrgan"),
        ("gfpgan (face restore)", "gfpgan"),
        ("pyiqa (quality critic)", "pyiqa"),
        ("imageio-ffmpeg", "imageio_ffmpeg"),
        ("mutagen (MP3 probe)", "mutagen"),
    ):
        try:
            __import__(module)
            record(label, "ok", "importable")
        except Exception as e:
            record(label, "skip", f"{YELLOW}not installed ({type(e).__name__}){RESET}")


# ─── Main ─────────────────────────────────────────────────────────
async def main() -> int:
    print(f"\n{DIM}Live smoke test — hits every external endpoint with a real request.{RESET}\n")

    print(f"{DIM}— keyless HTTP endpoints (anyone can run these):{RESET}")
    await asyncio.gather(
        smoke_hackernews(),
        smoke_devto(),
        smoke_wiki_otd(),
        smoke_openverse(),
        smoke_nager_date(),
    )
    await smoke_pollinations()  # slow — not parallelised

    print(f"\n{DIM}— local binaries + libraries:{RESET}")
    smoke_ffmpeg()
    smoke_pillow()
    smoke_sqlite_integrity()
    smoke_film_emulation()
    await smoke_edge_tts()
    await smoke_playwright_chromium()
    smoke_kokoro()

    print(f"\n{DIM}— optional extras (probe install state):{RESET}")
    smoke_optional_libs()

    print(f"\n{DIM}— key-gated HTTP (skipped unless env var is set):{RESET}")
    await asyncio.gather(
        smoke_openrouter(),
        smoke_groq(),
        smoke_gemini(),
        smoke_cerebras(),
        smoke_pexels(),
        smoke_pixabay(),
        smoke_freesound(),
        smoke_telegram(),
        smoke_reddit(),
        smoke_comfyui(),
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
