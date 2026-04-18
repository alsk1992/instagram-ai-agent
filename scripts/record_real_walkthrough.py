#!/usr/bin/env python
"""Record REAL executions — not SVG panels of faked output.

Captures:
  * docs/media/real-dashboard.webm  — Playwright video of the dashboard
    loading, scrolling, showing actual HTML/CSS from the repo.
  * docs/media/real-cli-*.cast      — asciinema recordings of real
    `ig-agent doctor`, `ig-agent status`, etc.
  * docs/media/real-cli-*.mp4       — agg-rendered MP4 per cast
    (when `agg` is installed).

The dashboard data is seeded demo rows so the screenshot isn't empty
(the sandbox has no IG account). The CODE is 100% real — same
FastAPI app + templates users see when they run `ig-agent dashboard`.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "docs" / "media"


def record_dashboard_video() -> Path | None:
    """Boot the real dashboard, browse it via Playwright with video
    recording ON. Produces a WebM (Chromium native), then converts
    to MP4 via the imageio-ffmpeg binary."""
    from playwright.sync_api import sync_playwright

    work = Path(tempfile.mkdtemp(prefix="ig_dash_rec_"))
    port = 18081
    server = _start_demo_dashboard(work, port)
    try:
        time.sleep(3.0)
        url = f"http://127.0.0.1:{port}"
        video_dir = work / "video"
        video_dir.mkdir()
        with sync_playwright() as p:
            browser = p.chromium.launch(args=["--no-sandbox"])
            ctx = browser.new_context(
                viewport={"width": 1280, "height": 720},
                device_scale_factor=1,
                record_video_dir=str(video_dir),
                record_video_size={"width": 1280, "height": 720},
            )
            page = ctx.new_page()
            page.goto(url, wait_until="networkidle", timeout=15_000)
            page.wait_for_timeout(2000)

            # Simulate real browsing — hold at top so the user sees the
            # header + niche card, slow-scroll through posts + queue,
            # bounce back.
            for _ in range(6):
                page.mouse.wheel(0, 280)
                page.wait_for_timeout(900)
            page.wait_for_timeout(1500)
            for _ in range(3):
                page.mouse.wheel(0, -350)
                page.wait_for_timeout(700)
            page.evaluate("window.scrollTo({top: 0, behavior: 'smooth'})")
            page.wait_for_timeout(2500)

            video_path = page.video.path() if page.video else None
            ctx.close()
            browser.close()

        if not video_path or not Path(video_path).exists():
            print("  ✗ Playwright didn't emit a video file")
            return None

        MEDIA.mkdir(parents=True, exist_ok=True)
        webm_out = MEDIA / "real-dashboard.webm"
        shutil.copyfile(video_path, webm_out)

        # Convert WebM → MP4 via bundled ffmpeg
        mp4_out = MEDIA / "real-dashboard.mp4"
        ffmpeg = _ffmpeg_path()
        if ffmpeg:
            subprocess.run(
                [ffmpeg, "-y", "-i", str(webm_out),
                 "-c:v", "libx264", "-preset", "medium", "-crf", "23",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                 str(mp4_out)],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )
            print(f"  → {webm_out.relative_to(ROOT)}  ({webm_out.stat().st_size / 1024 / 1024:.1f} MB)")
            print(f"  → {mp4_out.relative_to(ROOT)}  ({mp4_out.stat().st_size / 1024 / 1024:.1f} MB)")
            return mp4_out
        print(f"  → {webm_out.relative_to(ROOT)} (no ffmpeg — WebM only)")
        return webm_out

    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except Exception:
            server.kill()
        shutil.rmtree(work, ignore_errors=True)


def record_cli_casts() -> list[Path]:
    """Run real CLI commands through asciinema, save .cast files per.
    Uses a dedicated tmp workdir so the commands operate on a clean
    demo-seeded state."""
    casts_made: list[Path] = []

    work = Path(tempfile.mkdtemp(prefix="ig_cli_rec_"))
    MEDIA.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["IG_SKIP_WARMUP"] = "1"
    env["OPENROUTER_API_KEY"] = env.get("OPENROUTER_API_KEY", "demo-key-sk-fake")
    env["IG_USERNAME"] = env.get("IG_USERNAME", "yourbrand")
    env["IG_PASSWORD"] = env.get("IG_PASSWORD", "demo-password")
    env["COLUMNS"] = "110"
    env["LINES"] = "34"
    env["TERM"] = "xterm-256color"
    env["NO_COLOR"] = ""
    env["FORCE_COLOR"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    # Seed a minimal, self-consistent niche.yaml + brain.db under work/
    _seed_demo_workdir(work, env)

    # Each recording: (output_cast, shell command, idle_time_limit)
    demos: list[tuple[str, str, float]] = [
        ("real-cli-doctor.cast", "ig-agent doctor", 1.5),
        ("real-cli-status.cast", "ig-agent status", 1.5),
        ("real-cli-warmup.cast", "ig-agent warmup-status", 1.5),
    ]
    for cast_name, cmd, idle in demos:
        cast_path = MEDIA / cast_name
        subprocess.run(
            [
                "asciinema", "rec",
                "-q",
                "--overwrite",
                "--idle-time-limit", str(idle),
                "--cols", "110",
                "--rows", "34",
                "-c", f"bash -c 'cd {work} && source {ROOT}/.venv/bin/activate && {cmd}'",
                str(cast_path),
            ],
            env=env, check=False,
        )
        if cast_path.exists() and cast_path.stat().st_size > 100:
            casts_made.append(cast_path)
            print(f"  → {cast_path.relative_to(ROOT)}  ({cast_path.stat().st_size / 1024:.0f} KB)")

    shutil.rmtree(work, ignore_errors=True)
    return casts_made


def render_casts_to_mp4(casts: list[Path]) -> list[Path]:
    """Convert each .cast → .mp4 via agg (if installed)."""
    agg = shutil.which("agg") or str(Path.home() / ".cargo" / "bin" / "agg")
    if not Path(agg).exists():
        print("  ⚠  agg not installed — skipping MP4 render (casts are still playable on asciinema.org)")
        return []

    mp4s: list[Path] = []
    ffmpeg = _ffmpeg_path()
    for cast in casts:
        gif_tmp = cast.with_suffix(".gif")
        subprocess.run(
            [agg, "--font-size", "18", "--speed", "1.3", str(cast), str(gif_tmp)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        mp4 = cast.with_suffix(".mp4")
        if ffmpeg:
            subprocess.run(
                [ffmpeg, "-y", "-i", str(gif_tmp),
                 "-movflags", "+faststart", "-pix_fmt", "yuv420p",
                 "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                 "-c:v", "libx264", "-preset", "medium", "-crf", "22",
                 str(mp4)],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )
            mp4s.append(mp4)
            print(f"  → {mp4.relative_to(ROOT)}  ({mp4.stat().st_size / 1024:.0f} KB)")
        gif_tmp.unlink(missing_ok=True)
    return mp4s


# ─── Helpers ───────────────────────────────────────────────────
def _ffmpeg_path() -> str:
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return ""


def _start_demo_dashboard(workdir: Path, port: int) -> subprocess.Popen:
    env = os.environ.copy()
    env["IG_SKIP_WARMUP"] = "1"
    env["IG_DEMO_WORKDIR"] = str(workdir)

    # Seed niche.yaml so the dashboard's load_niche() succeeds
    (workdir / "niche.yaml").write_text(
        "niche: home calisthenics for dads 35+\n"
        "sub_topics: [pullups, mobility, recovery]\n"
        "target_audience: office workers rebuilding fitness at home\n"
        "commercial: true\n"
        "voice:\n"
        "  tone: [direct, dry humour]\n"
        "  persona: ex-office worker, rebuilt body at home\n"
        "  forbidden: []\n"
        "aesthetic:\n"
        "  palette: ['#0a0a0a', '#f5f5f0', '#c9a961']\n"
        "hashtags:\n"
        "  core: [calisthenics, homeworkout, bodyweighttraining]\n",
        encoding="utf-8",
    )

    # Seed demo DB before boot
    seed = (
        "from instagram_ai_agent.core import config, db; import os; "
        "os.chdir(os.environ['IG_DEMO_WORKDIR']); "
        "config.ensure_dirs(); db.init_db(); "
        "c = db.get_conn(); "
        "c.execute(\"INSERT OR IGNORE INTO posts "
        "(ig_media_pk, format, caption, posted_at, likes, comments, reach) VALUES "
        "('demo1', 'carousel', 'Most guys quit pullups too early.', "
        "strftime('%Y-%m-%dT%H:%M:%SZ','now','-3 days'), 347, 28, 2100), "
        "('demo2', 'reel_stock', '3 mobility drills that fixed my shoulders.', "
        "strftime('%Y-%m-%dT%H:%M:%SZ','now','-1 days'), 892, 43, 5800), "
        "('demo3', 'meme', 'Bro-science vs actual programming.', "
        "strftime('%Y-%m-%dT%H:%M:%SZ','now','-5 hours'), 156, 12, 980)\"); "
        "c.execute(\"INSERT OR IGNORE INTO content_queue "
        "(format, status, caption, hashtags, media_paths, generator) VALUES "
        "('carousel', 'approved', 'Stop ignoring your mobility.', '[]', '[]', 'carousel'), "
        "('meme', 'pending_review', 'Monday vs Friday gym vibes.', '[]', '[]', 'meme'), "
        "('reel_stock', 'pending_review', '5 pullup mistakes.', '[]', '[]', 'reel_stock')\"); "
        "c.commit()"
    )
    subprocess.check_call([sys.executable, "-c", seed], env=env, cwd=str(ROOT))

    cmd = [
        sys.executable, "-c",
        f"import os; os.chdir({str(workdir)!r}); "
        "from instagram_ai_agent.dashboard import create_app; "
        "import uvicorn; "
        f"uvicorn.run(create_app(), host='127.0.0.1', port={port}, log_level='warning')",
    ]
    return subprocess.Popen(cmd, env=env, cwd=str(ROOT),
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def _seed_demo_workdir(work: Path, env: dict) -> None:
    """Minimal niche.yaml + seeded brain.db so ig-agent commands show
    real output in the cast, not 'niche.yaml not found'."""
    niche_yaml = work / "niche.yaml"
    niche_yaml.write_text(
        "niche: home calisthenics for dads 35+\n"
        "sub_topics: [pullups, mobility, recovery]\n"
        "target_audience: office workers rebuilding fitness at home\n"
        "commercial: true\n"
        "voice:\n"
        "  tone: [direct, dry humour]\n"
        "  persona: ex-office worker, rebuilt body at home\n"
        "  forbidden: []\n"
        "aesthetic:\n"
        "  palette: ['#0a0a0a', '#f5f5f0', '#c9a961']\n"
        "hashtags:\n"
        "  core: [calisthenics, homeworkout, bodyweighttraining]\n",
        encoding="utf-8",
    )
    env["IG_DEMO_WORKDIR"] = str(work)
    seed = (
        "from instagram_ai_agent.core import config, db; "
        "from instagram_ai_agent.brain import idea_bank; import os; "
        "os.chdir(os.environ['IG_DEMO_WORKDIR']); "
        "config.ensure_dirs(); db.init_db(); idea_bank.seed_from_file(); "
        "c = db.get_conn(); "
        "c.execute(\"INSERT OR IGNORE INTO posts "
        "(ig_media_pk, format, caption, posted_at, likes, comments, reach) VALUES "
        "('d1', 'carousel', 'x', strftime('%Y-%m-%dT%H:%M:%SZ','now','-3 days'), 347, 28, 2100), "
        "('d2', 'reel_stock', 'y', strftime('%Y-%m-%dT%H:%M:%SZ','now','-1 days'), 892, 43, 5800)\"); "
        "c.execute(\"INSERT OR IGNORE INTO content_queue (format, status, caption, hashtags, media_paths, generator) "
        "VALUES ('carousel', 'approved', 'Stop ignoring mobility.', '[]', '[]', 'carousel'), "
        "('meme', 'pending_review', 'Monday vs Friday gym vibes.', '[]', '[]', 'meme')\"); "
        "c.commit()"
    )
    subprocess.check_call([sys.executable, "-c", seed], env=env, cwd=str(ROOT))


# ─── Main ──────────────────────────────────────────────────────
def main() -> int:
    print("▶ Recording real dashboard (Playwright video)")
    dash = record_dashboard_video()

    print("\n▶ Recording real CLI sessions (asciinema)")
    casts = record_cli_casts()

    print("\n▶ Rendering CLI casts to MP4 (agg)")
    cli_mp4s = render_casts_to_mp4(casts)

    print("\n✓ Done.")
    if dash:
        print(f"   Dashboard video: {dash.relative_to(ROOT)}")
    for c in casts:
        print(f"   CLI cast:         {c.relative_to(ROOT)}")
    for m in cli_mp4s:
        print(f"   CLI MP4:          {m.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
