#!/usr/bin/env python
"""Regenerate every demo asset shown in the README.

Produces:
  * docs/media/*.svg  — CLI output renderings (doctor, status, warmup,
    init wizard preview, review panel). Rich's Console.export_svg()
    gives us pixel-perfect renderings that GitHub renders natively.
  * docs/screenshots/*.png — dashboard screenshots via Playwright
    against the local read-only web UI.

Run after meaningful UI changes:
    python scripts/gen_demo_media.py

Safe to run idempotently — every file is overwritten.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS_MEDIA = ROOT / "docs" / "media"
DOCS_SCREENSHOTS = ROOT / "docs" / "screenshots"


def main() -> int:
    DOCS_MEDIA.mkdir(parents=True, exist_ok=True)
    DOCS_SCREENSHOTS.mkdir(parents=True, exist_ok=True)

    print("▶ Rich SVG exports (CLI output)")
    _render_doctor()
    _render_warmup_status()
    _render_status()
    _render_wizard_preview()
    _render_review_panel()
    _render_presets_list()

    print("▶ Playwright dashboard screenshots")
    try:
        _render_dashboard_screenshots()
    except Exception as e:
        print(f"  ⚠  dashboard screenshots failed: {e}")
        print("  (hint: pip install -e . && python -m playwright install chromium)")

    print("\n✓ Done.")
    print(f"   SVGs: {DOCS_MEDIA}")
    print(f"   PNGs: {DOCS_SCREENSHOTS}")
    return 0


# ─── Rich SVG renders ─────────────────────────────────────────
def _svg(path: Path, title: str, render_fn) -> None:
    """Render a Rich Console block into an SVG with a given terminal title."""
    from rich.console import Console
    buf = Console(record=True, width=100, force_terminal=True, color_system="truecolor")
    render_fn(buf)
    svg = buf.export_svg(title=title, theme=None)
    path.write_text(svg, encoding="utf-8")
    print(f"  → {path.relative_to(ROOT)}")


def _render_doctor() -> None:
    from rich.table import Table
    def render(c):
        c.rule("[bold]ig-agent doctor")
        t = Table(show_lines=False)
        t.add_column("", no_wrap=True)
        t.add_column("check")
        t.add_column("hint", style="dim")
        t.add_row("✅", "Python 3.12", "")
        t.add_row("✅", "ffmpeg + ffprobe on PATH", "")
        t.add_row("✅", "Playwright chromium installed", "")
        t.add_row("✅", "Fonts: 2 TTF file(s) in data/fonts", "")
        t.add_row("✅", "LLM provider(s): openrouter", "")
        t.add_row("✅", "IG creds set for user yourbrand", "")
        t.add_row("✅", "niche.yaml valid (niche: 'home calisthenics')", "")
        t.add_row("✅", "Idea bank seeded (90 archetypes)", "")
        t.add_row("✅", "TLS impersonation (curl_cffi) installed", "")
        t.add_row("⚠️", "Reddit harvester (praw) not installed",
                  "Optional — install via `pip install '.[reddit]'`")
        t.add_row("✅", "Device fingerprint: data/device.json", "")
        t.add_row("⚠️", "COMFYUI_URL not set (optional)",
                  "Falls back to Pollinations Flux cloud generation.")
        c.print(t)
        c.print("\n[yellow]1 warning(s)[/yellow] — optional, won't stop core posting.")
    _svg(DOCS_MEDIA / "doctor.svg", "ig-agent doctor", render)


def _render_warmup_status() -> None:
    from rich.table import Table
    def render(c):
        t = Table(title="warmup — day 6 · phase lurk")
        t.add_column("action", style="cyan")
        t.add_column("effective cap", justify="right")
        t.add_column("raw cap", justify="right")
        for action, eff, raw in [
            ("like", 35, 100), ("follow", 10, 30), ("comment", 7, 20),
            ("post", 0, 1), ("story_post", 0, 3), ("story_view", 70, 200),
            ("dm", 0, 5), ("unfollow", 10, 30),
        ]:
            t.add_row(action, str(eff), str(raw))
        c.print(t)
        c.print("[dim]posts allowed:[/dim] [red]False[/red]   "
                "[dim]DMs allowed:[/dim] [red]False[/red]   "
                "[dim]multiplier:[/dim] 0.35")
        c.print("[yellow]Fresh account — posts unlock on day 8.[/yellow]")
    _svg(DOCS_MEDIA / "warmup-status.svg", "ig-agent warmup-status", render)


def _render_status() -> None:
    from rich.table import Table
    def render(c):
        t = Table(title="content queue", show_lines=False)
        t.add_column("status", style="cyan")
        t.add_column("count", justify="right")
        for s, n in [("approved", 4), ("pending_review", 12),
                     ("posted", 47), ("failed", 1), ("rejected", 3)]:
            t.add_row(s, str(n))
        c.print(t)
        h = Table(title="latest health snapshot")
        for k in ("followers", "following", "media_count",
                  "engagement_rate", "shadowbanned"):
            h.add_column(k, justify="right")
        h.add_row("1847", "423", "47", "0.0614", "no")
        c.print(h)
        c.print("[dim]niche:[/dim] home calisthenics")
        c.print("[dim]formats:[/dim] {'meme': 0.25, 'carousel': 0.25, "
                "'reel_stock': 0.2, 'quote_card': 0.15, 'photo': 0.1, 'reel_ai': 0.05}")
        c.print("[dim]providers:[/dim] ['openrouter', 'groq', 'gemini']")
    _svg(DOCS_MEDIA / "status.svg", "ig-agent status", render)


def _render_wizard_preview() -> None:
    def render(c):
        c.rule("[bold]ig-agent setup")
        c.print("[dim]Pick a starter preset. Every field is editable below — "
                "this just saves you from inventing defaults.[/dim]")
        c.print("[bold]?[/bold] Starter preset:")
        c.print("  ❯ [cyan]fitness[/cyan] — Fitness & calisthenics")
        c.print("    [dim]food — Food / recipes[/dim]")
        c.print("    [dim]travel — Travel / adventure[/dim]")
        c.print("    [dim]finance — Personal finance[/dim]")
        c.print("    [dim]mindfulness — Mindfulness / mental health[/dim]")
        c.print("    [dim]productivity — Productivity / learning[/dim]")
        c.print("    [dim]fashion — Fashion / style[/dim]")
        c.print("    [dim]pets — Pets (dogs / cats)[/dim]")
        c.print("    [dim]custom (blank defaults)[/dim]")
        c.print()
        c.print("[bold]?[/bold] Niche [dim](home calisthenics and bodyweight training)[/dim]: "
                "home calisthenics for dads 35+")
        c.print("[bold]?[/bold] Sub-topics [dim](pullups, mobility, progressions, recovery)[/dim]: "
                "[dim]↵ (accepted)[/dim]")
        c.print("[bold]?[/bold] Target audience [dim](office workers rebuilding strength at home)[/dim]: "
                "[dim]↵[/dim]")
        c.print("[bold]?[/bold] Are you monetising this page? [cyan](Y/n)[/cyan]: Y")
    _svg(DOCS_MEDIA / "wizard-preview.svg", "ig-agent init", render)


def _render_review_panel() -> None:
    def render(c):
        c.rule("[bold]id=42 · carousel · score=0.83")
        c.print("[cyan]caption:[/cyan] Most guys quit pullups too early. "
                "Here's the actual progression that built mine.")
        c.print("[cyan]hashtags:[/cyan] #calisthenics #homeworkout "
                "#bodyweighttraining #pullupprogression #dadfit")
        c.print("[dim]media:[/dim] data/media/staged/slide01_default_abc123.jpg")
        c.print("[dim]media:[/dim] data/media/staged/slide02_default_def456.jpg")
        c.print("[dim]media:[/dim] data/media/staged/slide03_default_ghi789.jpg")
        c.print("[dim]notes:[/dim] strong on-niche hook, defensible claim, clean voice. "
                "hashtag mix balanced between core + growth.")
        c.print()
        c.print("[bold]?[/bold] Action:")
        c.print("  ❯ [green]approve[/green]")
        c.print("    [red]reject[/red]")
        c.print("    [dim]open media[/dim]")
        c.print("    [dim]skip[/dim]")
        c.print("    [dim]quit[/dim]")
    _svg(DOCS_MEDIA / "review.svg", "ig-agent review", render)


def _render_presets_list() -> None:
    from rich.table import Table
    def render(c):
        c.rule("[bold]Starter presets")
        t = Table(show_lines=False)
        t.add_column("preset", style="cyan")
        t.add_column("niche")
        t.add_column("audience", style="dim")
        presets = [
            ("fitness", "home calisthenics", "office workers rebuilding strength"),
            ("food", "quick healthy weeknight recipes", "time-pressed parents"),
            ("travel", "budget solo travel in SE asia", "first-time solo travelers"),
            ("finance", "personal finance without spreadsheets", "mid-career earners"),
            ("mindfulness", "practical mindfulness, no jargon",
             "overstimulated professionals"),
            ("productivity", "deep-work for knowledge workers",
             "remote workers + students"),
            ("fashion", "budget capsule wardrobe for men 25-40",
             "men who want put-together"),
            ("pets", "positive-reinforcement dog training",
             "first-time rescue adopters"),
        ]
        for k, niche, aud in presets:
            t.add_row(k, niche, aud)
        c.print(t)
    _svg(DOCS_MEDIA / "presets.svg", "Starter presets", render)


# ─── Playwright dashboard screenshots ─────────────────────────
def _render_dashboard_screenshots() -> None:
    """Boot the dashboard with seeded demo data, screenshot each page."""
    from playwright.sync_api import sync_playwright

    workdir = Path(tempfile.mkdtemp(prefix="ig_agent_demo_"))
    port = 18080
    server = _start_demo_dashboard(workdir, port)
    try:
        time.sleep(2.5)   # let uvicorn boot
        url = f"http://127.0.0.1:{port}"
        with sync_playwright() as p:
            browser = p.chromium.launch(args=["--no-sandbox"])
            ctx = browser.new_context(
                viewport={"width": 1440, "height": 900},
                device_scale_factor=2,
            )
            page = ctx.new_page()
            try:
                page.goto(url, wait_until="networkidle", timeout=10_000)
                page.screenshot(path=str(DOCS_SCREENSHOTS / "dashboard-home.png"))
                print(f"  → docs/screenshots/dashboard-home.png")
            except Exception as e:
                print(f"  ⚠  dashboard-home screenshot failed: {e}")
            ctx.close()
            browser.close()
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except Exception:
            server.kill()


def _start_demo_dashboard(workdir: Path, port: int) -> subprocess.Popen:
    """Populate a tmp brain.db with demo rows + boot uvicorn."""
    # Seed a fake brain.db with plausible rows
    env = os.environ.copy()
    env["IG_SKIP_WARMUP"] = "1"   # so post-status rows look populated
    env["OPENROUTER_API_KEY"] = env.get("OPENROUTER_API_KEY", "demo-key")
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    # Bootstrap the tmp workdir's data/
    script = (
        "from instagram_ai_agent.core import config, db; "
        "from pathlib import Path; "
        "import os; "
        "os.chdir(os.environ['IG_DEMO_WORKDIR']); "
        "config.ensure_dirs(); "
        "db.init_db(); "
        "c = db.get_conn(); "
        "c.execute(\"INSERT OR IGNORE INTO posts (ig_media_pk, format, caption, posted_at, likes, comments, reach) "
        "VALUES ('demo1', 'carousel', 'Most guys quit pullups too early.', strftime('%Y-%m-%dT%H:%M:%SZ','now','-3 days'), 347, 28, 2100), "
        "('demo2', 'reel_stock', '3 mobility drills that fixed my shoulders.', strftime('%Y-%m-%dT%H:%M:%SZ','now','-1 days'), 892, 43, 5800), "
        "('demo3', 'meme', 'Bro-science vs actual programming.', strftime('%Y-%m-%dT%H:%M:%SZ','now','-5 hours'), 156, 12, 980)\"); "
        "c.execute(\"INSERT OR IGNORE INTO content_queue (format, status, caption, hashtags, media_paths, generator) "
        "VALUES ('carousel', 'approved', 'Stop ignoring your mobility.', '[]', '[]', 'carousel'), "
        "('meme', 'pending_review', 'Monday vs Friday gym vibes.', '[]', '[]', 'meme'), "
        "('reel_stock', 'pending_review', '5 pullup mistakes.', '[]', '[]', 'reel_stock')\"); "
        "c.commit(); "
    )

    env["IG_DEMO_WORKDIR"] = str(workdir)
    subprocess.check_call(
        [sys.executable, "-c", script],
        env=env,
        cwd=str(ROOT),
    )

    # Start uvicorn in the workdir so DATA_DIR resolves to workdir/data
    cmd = [
        sys.executable, "-c",
        f"import os; os.chdir({str(workdir)!r}); "
        "from instagram_ai_agent.dashboard import create_app; "
        "import uvicorn; "
        f"uvicorn.run(create_app(), host='127.0.0.1', port={port}, log_level='warning')",
    ]
    return subprocess.Popen(
        cmd,
        env=env,
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


if __name__ == "__main__":
    sys.exit(main())
