"""Command-line interface — setup wizard + ops commands."""
from __future__ import annotations

import asyncio
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

import questionary
import typer
import yaml
from rich.console import Console
from rich.prompt import Confirm
from rich.table import Table

from instagram_ai_agent.content import pipeline as content_pipeline
from instagram_ai_agent.core import db
from instagram_ai_agent.core.config import (
    ENV_PATH,
    NICHE_PATH,
    ROOT,
    Aesthetic,
    Budget,
    FormatMix,
    HashtagPools,
    BrandCharacter,
    HumanPhoto,
    NicheConfig,
    Safety,
    Schedule,
    StoryMix,
    Voice,
    ensure_dirs,
    load_env,
    load_niche,
    save_niche,
)
from instagram_ai_agent.core.llm import providers_configured
from instagram_ai_agent.core.logging_setup import setup_logging
from instagram_ai_agent.workers import poster

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()
log = setup_logging()


# ───────── helpers ─────────
def _split(text: str) -> list[str]:
    return [t.strip() for t in re.split(r"[,\n]", text or "") if t.strip()]


def _write_env(new_values: dict[str, str]) -> None:
    """Merge new_values into .env, preserving any existing lines we didn't touch."""
    existing: dict[str, str] = {}
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            existing[k.strip()] = v.strip()
    existing.update({k: v for k, v in new_values.items() if v})

    with open(ENV_PATH, "w", encoding="utf-8") as f:
        for k, v in existing.items():
            f.write(f"{k}={v}\n")


def _require_niche() -> NicheConfig:
    try:
        return load_niche()
    except FileNotFoundError:
        console.print("[red]niche.yaml not found.[/red] Run `ig-agent init` first.")
        raise typer.Exit(2)


# ───────── init wizard ─────────
@app.command()
def init(
    force: bool = typer.Option(False, "--force", help="Overwrite existing niche.yaml"),
) -> None:
    """Interactive setup wizard — generates niche.yaml + .env."""
    ensure_dirs()

    if NICHE_PATH.exists() and not force:
        if not Confirm.ask(f"[yellow]{NICHE_PATH.name} already exists. Overwrite?[/yellow]"):
            raise typer.Exit(0)

    console.rule("[bold]ig-agent setup")

    # Preset picker — saves users from inventing palette/voice/hashtags
    # from scratch. Pick a preset → wizard fills reasonable defaults
    # → they only need to edit what's specific to their page.
    from instagram_ai_agent.niche_presets import PRESETS, by_key

    console.print(
        "[dim]Pick a starter preset. Every field is editable below — "
        "this just saves you from inventing defaults.[/dim]"
    )
    preset_choices = [f"{p.key} — {p.label}" for p in PRESETS] + ["custom (blank defaults)"]
    preset_label = questionary.select("Starter preset:", choices=preset_choices).ask()
    if preset_label is None:
        raise typer.Exit(1)
    preset_key = preset_label.split(" — ")[0] if "—" in preset_label else None
    preset = by_key(preset_key) if preset_key else None

    default_niche = preset.niche if preset else ""
    default_subs = ", ".join(preset.sub_topics) if preset else ""
    default_audience = preset.target_audience if preset else ""
    default_persona = preset.persona if preset else ""
    default_tone = ", ".join(preset.voice_tone) if preset else "direct"
    default_forbidden = ", ".join(preset.voice_forbidden) if preset else ""
    default_palette = ", ".join(preset.palette) if preset else "#0a0a0a, #f5f5f0, #c9a961"
    default_core_tags = ", ".join(preset.core_hashtags) if preset else ""
    default_growth_tags = ", ".join(preset.growth_hashtags) if preset else ""
    default_hours = ", ".join(str(h) for h in (preset.best_hours_utc if preset else [14, 18, 21]))

    niche = questionary.text(
        "Niche (what this page is about):",
        default=default_niche,
        validate=lambda x: len(x.strip()) >= 3 or "Minimum 3 chars",
    ).ask()
    if niche is None:
        raise typer.Exit(1)
    sub_topics = _split(
        questionary.text("Sub-topics (comma-separated):", default=default_subs).ask() or ""
    )
    if not sub_topics:
        sub_topics = [niche.strip()]
    audience = questionary.text(
        "Target audience (who it's for):",
        default=default_audience,
        validate=lambda x: len(x.strip()) >= 5 or "Minimum 5 chars",
    ).ask()
    commercial = questionary.confirm(
        "Are you monetising this page? (affects licence gates)",
        default=True,
    ).ask()

    console.rule("[bold]voice")
    console.print("[dim]How should the AI sound? You can always tweak niche.yaml later.[/dim]")
    tone = _split(
        questionary.text(
            "Voice tone (comma-separated, e.g. direct, dry humour):",
            default=default_tone,
        ).ask() or "direct"
    ) or ["direct"]
    forbidden = _split(
        questionary.text(
            "Words/phrases to NEVER use (comma-separated, optional):",
            default=default_forbidden,
        ).ask() or ""
    )
    persona = questionary.text(
        "One-sentence persona (who's writing these posts?):",
        default=default_persona,
        validate=lambda x: len(x.strip()) >= 10 or "Give a proper sentence",
    ).ask()
    cta_styles = _split(
        questionary.text(
            "Preferred CTAs (comma-separated):",
            default="save for later, tag a mate, follow for more",
        ).ask() or ""
    ) or ["save for later"]

    console.rule("[bold]aesthetic")
    palette_input = questionary.text(
        "Palette (3–5 hex colours, comma-separated):",
        default=default_palette,
    ).ask() or ""
    palette = _split(palette_input) or ["#0a0a0a", "#f5f5f0", "#c9a961"]
    heading_font = questionary.text("Heading font family:", default="Archivo Black").ask() or "Archivo Black"
    body_font = questionary.text("Body font family:", default="Inter").ask() or "Inter"
    watermark = questionary.text(
        "Watermark (usually your @handle, optional):", default=""
    ).ask() or None
    lut = questionary.text(
        "LUT filename in data/luts/ (optional, e.g. warm-contrast.cube):", default=""
    ).ask() or None

    console.rule("[bold]hashtags")
    core = _split(
        questionary.text(
            "Core hashtags (comma-separated, min 3):",
            default=default_core_tags,
        ).ask() or ""
    )
    growth = _split(
        questionary.text(
            "Growth hashtags (comma-separated, optional):",
            default=default_growth_tags,
        ).ask() or ""
    )
    long_tail = _split(questionary.text("Long-tail hashtags (comma-separated, optional):").ask() or "")
    per_post = int(
        questionary.text("Hashtags per post (3–30):", default="15", validate=_is_int_in(3, 30)).ask()
    )

    console.rule("[bold]feed formats")
    format_choices = questionary.checkbox(
        "Feed post formats allowed:",
        choices=[
            questionary.Choice("meme", checked=True),
            questionary.Choice("quote_card", checked=True),
            questionary.Choice("carousel", checked=True),
            questionary.Choice("reel_stock", checked=True),
            questionary.Choice("photo", checked=False),
            questionary.Choice("reel_ai", checked=False),
        ],
    ).ask() or ["meme"]
    formats = _even_mix(format_choices)

    console.rule("[bold]human photos (optional)")
    human_enabled = questionary.confirm(
        "Enable photorealistic human-subject posts (free via Pollinations Flux)?",
        default=False,
    ).ask()
    human_photo = HumanPhoto(enabled=bool(human_enabled))
    if human_enabled:
        console.print(
            "[dim]Pick a style:\n"
            "  • unique  — every post a different person (community vibe)\n"
            "  • brand   — same persona across all posts (a recurring character)[/dim]"
        )
        style = questionary.select(
            "Human-photo style:",
            choices=["unique", "brand"],
        ).ask() or "unique"
        if style == "brand":
            console.print("[dim]Define the recurring persona — Flux will anchor to the seed.[/dim]")
            ch = BrandCharacter(
                enabled=True,
                name=questionary.text("Internal character name (optional):").ask() or None,
                age_range=questionary.text("Age range:", default="30s").ask() or "30s",
                gender=questionary.select(
                    "Gender presentation:",
                    choices=["man", "woman", "androgynous", "non-binary"],
                ).ask() or "androgynous",
                ethnicity=questionary.text(
                    "Ethnicity hint (or leave 'unspecified'):", default="unspecified",
                ).ask() or "unspecified",
                hair=questionary.text("Hair (e.g. 'short brown hair, stubble'):").ask() or "",
                build=questionary.text("Build (e.g. 'lean athletic'):").ask() or "",
                wardrobe_style=questionary.text(
                    "Wardrobe style (e.g. 'dark training kit'):",
                ).ask() or "",
                vibe=questionary.text("Vibe (e.g. 'tired but determined'):").ask() or "",
                seed=random.randint(1_000_000, 9_999_999),
            )
            human_photo = HumanPhoto(enabled=True, character=ch)
        human_weight = float(
            questionary.text(
                "Weight of human_photo in feed mix (0.0–0.5):",
                default="0.15",
                validate=_is_float_in(0.0, 0.5),
            ).ask()
        )
        story_human_weight = float(
            questionary.text(
                "Weight of story_human in story mix (0.0–0.5):",
                default="0.10",
                validate=_is_float_in(0.0, 0.5),
            ).ask()
        )
    else:
        human_weight = 0.0
        story_human_weight = 0.0

    console.rule("[bold]story formats")
    story_choices = questionary.checkbox(
        "Story formats allowed:",
        choices=[
            questionary.Choice("story_quote", checked=True),
            questionary.Choice("story_announcement", checked=True),
            questionary.Choice("story_photo", checked=False),
            questionary.Choice("story_video", checked=True),
            questionary.Choice("story_human", checked=bool(human_enabled and story_human_weight > 0)),
        ],
    ).ask() or ["story_quote"]
    stories = _even_story_mix(story_choices, story_human_weight=story_human_weight)

    console.rule("[bold]schedule + budget")
    posts_per_day = int(
        questionary.text("Posts per day (0–5):", default="1", validate=_is_int_in(0, 5)).ask()
    )
    stories_per_day = int(
        questionary.text("Stories per day (0–20):", default="3", validate=_is_int_in(0, 20)).ask()
    )
    hours_raw = questionary.text(
        "Best hours UTC (comma-separated, e.g. 14, 18, 21):",
        default=default_hours,
    ).ask() or default_hours
    best_hours = sorted({int(h) for h in _split(hours_raw) if h.isdigit() and 0 <= int(h) < 24})

    console.rule("[bold]brain")
    competitors = _split(questionary.text("Competitors (@handle, comma-separated):").ask() or "")
    reference_accounts = _split(
        questionary.text("Reference/aesthetic accounts (optional):").ask() or ""
    )
    watch_target = (
        questionary.text("Primary account to watch for reactions (optional):").ask() or ""
    ).strip() or None

    require_review = questionary.confirm(
        "Human-review gate before posting? (recommended for first 2-4 weeks)", default=True
    ).ask()

    has_gpu = questionary.confirm("Do you have a local NVIDIA GPU?", default=False).ask()

    cfg = NicheConfig(
        niche=niche.strip(),
        sub_topics=sub_topics,
        target_audience=audience.strip(),
        commercial=commercial,
        voice=Voice(tone=tone, forbidden=forbidden, persona=persona.strip(), cta_styles=cta_styles),
        aesthetic=Aesthetic(
            palette=palette,
            heading_font=heading_font.strip(),
            body_font=body_font.strip(),
            watermark=watermark.strip() if watermark else None,
            lut=lut.strip() if lut else None,
        ),
        hashtags=HashtagPools(core=core or ["niche"], growth=growth, long_tail=long_tail, per_post=per_post),
        formats=_apply_human_weight(formats, human_weight),
        stories=stories,
        human_photo=human_photo,
        schedule=Schedule(posts_per_day=posts_per_day, stories_per_day=stories_per_day, best_hours_utc=best_hours),
        budget=Budget(),
        safety=Safety(require_review=require_review),
        competitors=[c.lstrip("@") for c in competitors],
        reference_accounts=[r.lstrip("@") for r in reference_accounts],
        watch_target=watch_target.lstrip("@") if watch_target else None,
        has_gpu=has_gpu,
    )
    save_niche(cfg)
    console.print(f"[green]Wrote[/green] {NICHE_PATH}")

    # .env
    console.rule("[bold]api keys")
    console.print(
        "[dim]All keys are optional individually — you need at least one LLM provider.[/dim]"
    )
    env_updates: dict[str, str] = {}
    env_updates["IG_USERNAME"] = (
        questionary.text("IG_USERNAME (your Instagram handle):").ask() or ""
    ).lstrip("@")
    env_updates["IG_PASSWORD"] = questionary.password("IG_PASSWORD:").ask() or ""
    env_updates["OPENROUTER_API_KEY"] = (
        questionary.password("OPENROUTER_API_KEY (https://openrouter.ai/keys):").ask() or ""
    )
    env_updates["GROQ_API_KEY"] = questionary.password("GROQ_API_KEY (optional):").ask() or ""
    env_updates["GEMINI_API_KEY"] = questionary.password("GEMINI_API_KEY (optional):").ask() or ""
    env_updates["CEREBRAS_API_KEY"] = questionary.password("CEREBRAS_API_KEY (optional):").ask() or ""
    env_updates["PEXELS_API_KEY"] = questionary.password("PEXELS_API_KEY (for reels):").ask() or ""
    env_updates["PIXABAY_API_KEY"] = questionary.password("PIXABAY_API_KEY (for reels):").ask() or ""
    env_updates["TELEGRAM_BOT_TOKEN"] = (
        questionary.password("TELEGRAM_BOT_TOKEN (alerts, optional):").ask() or ""
    )
    env_updates["TELEGRAM_CHAT_ID"] = (
        questionary.text("TELEGRAM_CHAT_ID (optional):").ask() or ""
    )

    console.print(
        "\n[dim]IG email-code challenges: without these the agent falls back to "
        "manual code entry. Recommended for long-running accounts so the orchestrator "
        "can auto-resolve.[/dim]"
    )
    env_updates["IMAP_HOST"] = (
        questionary.text("IMAP_HOST (e.g. imap.gmail.com, optional):").ask() or ""
    )
    env_updates["IMAP_USER"] = (
        questionary.text("IMAP_USER (email, optional):").ask() or ""
    )
    env_updates["IMAP_PASS"] = (
        questionary.password("IMAP_PASS (app password, optional):").ask() or ""
    )

    _write_env(env_updates)
    console.print(f"[green]Wrote[/green] {ENV_PATH}")

    db.init_db()
    console.print(f"[green]Initialised[/green] brain.db")

    # Auto-seed the idea bank — every generation cycle picks an archetype,
    # so a fresh install with zero ideas strips a quality layer. The seed
    # corpus ships in data/ideas/seed.json (CC0, 90 archetypes).
    try:
        from instagram_ai_agent.brain import idea_bank
        n_seeded = idea_bank.seed_from_file()
        if n_seeded > 0:
            console.print(f"[green]Seeded[/green] idea bank with {n_seeded} archetypes")
    except Exception as e:
        console.print(f"[yellow]Warning:[/yellow] idea-bank seeding skipped: {e}")

    console.print("\n[bold]Next steps:[/bold]")
    console.print("  1. [bold]ig-agent login[/bold]                 # verify Instagram auth")
    console.print("  2. [bold]ig-agent generate -n 3[/bold]         # make 3 posts")
    console.print("  3. [bold]ig-agent review[/bold]                # approve them")
    console.print("  4. [bold]ig-agent drain[/bold]                 # post NOW")
    console.print("  5. [bold]ig-agent run[/bold]                   # start the full agent")


def _is_int_in(lo: int, hi: int):
    def _check(x: str) -> bool | str:
        try:
            v = int(x)
        except Exception:
            return "Must be an integer"
        if not lo <= v <= hi:
            return f"Must be between {lo} and {hi}"
        return True
    return _check


def _even_mix(choices: list[str]) -> FormatMix:
    all_fields = {"meme", "quote_card", "carousel", "reel_stock", "reel_ai", "photo"}
    weights: dict[str, float] = {f: 0.0 for f in all_fields}
    if not choices:
        choices = ["meme"]
    # Default-leaning weights biased toward things that perform well
    preset = {
        "meme": 0.30,
        "quote_card": 0.15,
        "carousel": 0.25,
        "reel_stock": 0.20,
        "photo": 0.05,
        "reel_ai": 0.05,
    }
    total = sum(preset[c] for c in choices)
    for c in choices:
        weights[c] = preset[c] / total
    return FormatMix(**weights)


def _even_story_mix(choices: list[str], *, story_human_weight: float = 0.0) -> StoryMix:
    all_fields = {
        "story_quote", "story_announcement", "story_photo", "story_video", "story_human",
    }
    weights: dict[str, float] = {f: 0.0 for f in all_fields}
    if not choices:
        choices = ["story_quote"]
    preset = {
        "story_quote": 0.35,
        "story_announcement": 0.25,
        "story_photo": 0.20,
        "story_video": 0.20,
        "story_human": 0.20,
    }
    # Reserve story_human_weight first, then distribute remainder across the rest
    remainder = max(0.0, 1.0 - story_human_weight)
    non_human = [c for c in choices if c != "story_human"]
    if non_human:
        total = sum(preset[c] for c in non_human)
        for c in non_human:
            weights[c] = preset[c] / total * remainder
    if "story_human" in choices:
        weights["story_human"] = story_human_weight
    return StoryMix(**weights)


def _apply_human_weight(fm: FormatMix, weight: float) -> FormatMix:
    """Return a new FormatMix with ``human_photo`` weight set, remainder re-scaled."""
    if weight <= 0:
        return fm
    current = fm.model_dump()
    current["human_photo"] = 0.0
    total_others = sum(v for k, v in current.items() if k != "human_photo")
    if total_others <= 0:
        return FormatMix(human_photo=weight, photo=1.0 - weight)
    scale = (1.0 - weight) / total_others
    for k in list(current):
        if k != "human_photo":
            current[k] = current[k] * scale
    current["human_photo"] = weight
    return FormatMix(**current)


def _is_float_in(lo: float, hi: float):
    def _check(x: str) -> bool | str:
        try:
            v = float(x)
        except Exception:
            return "Must be a number"
        if not lo <= v <= hi:
            return f"Must be between {lo} and {hi}"
        return True
    return _check


# ───────── login ─────────
@app.command()
def login() -> None:
    """Verify IG credentials, persist session + device fingerprint."""
    load_env()
    ensure_dirs()
    db.init_db()
    from instagram_ai_agent.plugins.ig import IGClient
    cl = IGClient()
    try:
        cl.login()
    except Exception as e:
        console.print(f"[red]Login failed:[/red] {e}")
        raise typer.Exit(1)
    console.print(f"[green]Session persisted[/green] → {cl.session_path}")


# ───────── run / orchestrator ─────────
@app.command()
def run() -> None:
    """Start the full orchestrator (generator + brain + poster + engager)."""
    load_env()
    ensure_dirs()
    db.init_db()
    _require_niche()
    if not providers_configured():
        console.print("[red]No LLM providers set.[/red] Configure OPENROUTER_API_KEY at minimum.")
        raise typer.Exit(2)
    from instagram_ai_agent.orchestrator import main as orch_main
    orch_main()


# ───────── generate (one cycle) ─────────
@app.command()
def generate(
    format: str = typer.Option(None, help="Force a specific format: meme|quote_card|carousel|reel_stock|photo"),
    count: int = typer.Option(1, min=1, max=20),
    contrarian: bool = typer.Option(
        None, "--contrarian/--no-contrarian",
        help="Force contrarian hot-take mode on/off for this batch. Default follows the dice.",
    ),
) -> None:
    """Generate N items into the content queue and exit."""
    load_env()
    ensure_dirs()
    db.init_db()
    cfg = _require_niche()
    if not providers_configured():
        console.print("[red]No LLM providers set.[/red]")
        raise typer.Exit(2)

    async def _go():
        made: list[int] = []
        for i in range(count):
            cid = await content_pipeline.generate_one(
                cfg, format_override=format, contrarian_override=contrarian,
            )
            if cid is not None:
                made.append(cid)
                console.print(f"[green]✓[/green] Enqueued content id={cid} ({format or 'auto'})")
            else:
                console.print("[yellow]· skipped or rejected[/yellow]")
        console.print(f"\nDone: {len(made)} / {count}")
    asyncio.run(_go())


# ───────── review ─────────
@app.command()
def review() -> None:
    """Walk through pending_review items and approve/reject/regen."""
    load_env()
    db.init_db()
    cfg = _require_niche()

    # Non-TTY guard — questionary hangs forever when stdin isn't a
    # terminal (e.g. `ig-agent review` over SSH without -t). Fall
    # back to a status summary + pointer at the dashboard.
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        console.print("[yellow]ig-agent review needs an interactive terminal.[/yellow]")
        console.print("  Run locally, or:  [bold]ssh -t <host> ig-agent review[/bold]")
        console.print("  Or use the dashboard: [bold]ig-agent dashboard[/bold]")
        raise typer.Exit(1)

    items = db.content_list(status="pending_review", limit=200)
    if not items:
        console.print("[green]No items pending review.[/green]")
        return
    console.print(
        f"[dim]{len(items)} pending items. "
        "Select 'quit' any time to stop — approved ones stay approved.[/dim]"
    )

    for item in items:
        _print_item(item)
        choice = questionary.select(
            "Action:",
            choices=["approve", "reject", "open media", "skip", "quit"],
        ).ask()
        if choice == "approve":
            db.content_update_status(int(item["id"]), "approved")
            poster.schedule_approved_items(cfg)
            console.print(f"[green]approved[/green] id={item['id']}")
        elif choice == "reject":
            db.content_update_status(int(item["id"]), "rejected")
            console.print(f"[red]rejected[/red] id={item['id']}")
        elif choice == "open media":
            _open_paths(item["media_paths"])
        elif choice == "quit":
            return


def _print_item(item: dict[str, Any]) -> None:
    console.rule(f"[bold]id={item['id']} · {item['format']} · score={item.get('critic_score')}")
    console.print(f"[cyan]caption:[/cyan] {item['caption'][:400]}")
    tags = item.get("hashtags") or []
    if tags:
        console.print(f"[cyan]hashtags:[/cyan] {' '.join('#' + t for t in tags)}")
    for p in item["media_paths"]:
        console.print(f"[dim]media:[/dim] {p}")
    if item.get("critic_notes"):
        console.print(f"[dim]notes:[/dim] {item['critic_notes']}")


def _open_paths(paths: list[str]) -> None:
    """Best-effort open with the OS default (xdg-open / open / start)."""
    import subprocess

    for p in paths:
        if not Path(p).exists():
            continue
        try:
            if sys.platform.startswith("darwin"):
                subprocess.Popen(["open", p])
            elif os.name == "nt":
                os.startfile(p)  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", p], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            console.print(f"[yellow]couldn't open {p}: {e}[/yellow]")


# ───────── post (one) ─────────
@app.command()
def post() -> None:
    """Post the next approved item (manual trigger)."""
    load_env()
    db.init_db()
    cfg = _require_niche()

    async def _go():
        cid = await poster.post_next(cfg)
        console.print(f"[green]posted[/green] id={cid}" if cid else "[yellow]nothing to post[/yellow]")
    asyncio.run(_go())


# ───────── status ─────────
@app.command()
def status() -> None:
    """Print queue depth, last post, health, backoff."""
    load_env()
    db.init_db()
    cfg = _require_niche()

    by_status = {}
    for row in db.content_list(status=None, limit=500):
        by_status[row["status"]] = by_status.get(row["status"], 0) + 1

    table = Table(title="content queue", show_lines=False)
    table.add_column("status", style="cyan")
    table.add_column("count", justify="right")
    for k, v in sorted(by_status.items()):
        table.add_row(k, str(v))
    console.print(table)

    latest = db.health_latest()
    if latest:
        h = Table(title="latest health snapshot")
        for k in ("followers", "following", "media_count", "engagement_rate", "shadowbanned"):
            h.add_column(k, justify="right")
        h.add_row(
            str(latest.get("followers")),
            str(latest.get("following")),
            str(latest.get("media_count")),
            f"{latest.get('engagement_rate', 0):.4f}",
            "yes" if latest.get("shadowbanned") else "no",
        )
        console.print(h)

    until = db.state_get("backoff_until")
    if until:
        console.print(f"[yellow]backoff_until:[/yellow] {until} ({db.state_get('backoff_reason') or ''})")

    console.print(f"[dim]niche:[/dim] {cfg.niche}")
    console.print(f"[dim]formats:[/dim] {cfg.formats.normalized()}")
    console.print(f"[dim]providers:[/dim] {providers_configured()}")


# ───────── add-content (manual upload) ─────────
@app.command("add-content")
def add_content(
    format: str = typer.Argument(..., help="meme|quote_card|carousel|reel_stock|photo"),
    media: list[Path] = typer.Argument(..., help="One or more media paths"),
    caption: str = typer.Option("", help="Caption text"),
    approve: bool = typer.Option(False, help="Mark approved immediately (skip review)"),
) -> None:
    """Drop external media straight into the content queue."""
    load_env()
    db.init_db()
    _require_niche()
    for p in media:
        if not p.exists():
            console.print(f"[red]missing:[/red] {p}")
            raise typer.Exit(1)
    from instagram_ai_agent.content import dedup, hashtags as hashtags_mod
    cfg = _require_niche()

    phash = dedup.compute_phash(media[0])
    tags = hashtags_mod.build_hashtags(cfg)
    full_caption = (caption.strip() + "\n\n" + hashtags_mod.format_hashtags(tags)).strip()
    cid = db.content_enqueue(
        format=format,
        caption=full_caption,
        hashtags=tags,
        media_paths=[str(m.resolve()) for m in media],
        phash=phash,
        critic_score=None,
        critic_notes="manual",
        generator="manual",
        status="approved" if approve else "pending_review",
    )
    if approve:
        poster.schedule_approved_items(cfg)
    console.print(f"[green]enqueued[/green] id={cid} status={'approved' if approve else 'pending_review'}")


# ───────── drain (flush approved queue synchronously) ─────────
@app.command()
def drain(limit: int = typer.Option(3, min=1, max=10)) -> None:
    """Post up to `limit` approved items right now, spaced out."""
    load_env()
    db.init_db()
    cfg = _require_niche()

    async def _go():
        n = 0
        for _ in range(limit):
            # drain=True → bypass scheduled_for best-hours filter so the
            # user gets an immediate first-post-proof without waiting
            # until their configured posting window.
            cid = await poster.post_next(cfg, drain=True)
            if cid is None:
                break
            n += 1
        console.print(f"drained {n}/{limit}")
    asyncio.run(_go())


# ───────── niche tools ─────────
@app.command("dashboard")
def dashboard_cmd(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8080),
) -> None:
    """Start the local read-only web dashboard."""
    load_env()
    ensure_dirs()
    db.init_db()
    _require_niche()
    import uvicorn
    from instagram_ai_agent.dashboard import create_app
    application = create_app()
    console.print(f"[green]Dashboard[/green] → http://{host}:{port}")
    uvicorn.run(application, host=host, port=port, log_level="warning")


@app.command("warmup-status")
def warmup_status() -> None:
    """Show current warmup day + scaled per-action caps."""
    load_env()
    db.init_db()
    cfg = _require_niche()
    from instagram_ai_agent.core.warmup import current_day, effective_caps

    budget = effective_caps(cfg)
    t = Table(title=f"warmup — day {budget.day or 'not started'} · phase {budget.phase_label}")
    t.add_column("action", style="cyan")
    t.add_column("effective cap", justify="right")
    t.add_column("raw cap", justify="right")
    raw_map = {
        "like": cfg.budget.likes, "follow": cfg.budget.follows, "unfollow": cfg.budget.unfollows,
        "comment": cfg.budget.comments, "dm": cfg.budget.dms, "story_view": cfg.budget.story_views,
        "post": cfg.schedule.posts_per_day, "story_post": cfg.schedule.stories_per_day,
    }
    for action, cap in budget.caps.items():
        t.add_row(action, str(cap), str(raw_map.get(action, 0)))
    console.print(t)
    console.print(
        f"[dim]posts allowed:[/dim] {budget.allow_posts}   "
        f"[dim]DMs allowed:[/dim] {budget.allow_dms}   "
        f"[dim]multiplier:[/dim] {budget.multiplier:.2f}"
    )
    if current_day() is None:
        console.print("[yellow]Warmup not yet started — runs automatically on first login.[/yellow]")


@app.command("reddit-questions")
def reddit_questions_cmd(
    push: bool = typer.Option(False, "--push", help="Also push fresh questions to context_feed"),
    limit: int = typer.Option(20, min=1, max=100),
) -> None:
    """Harvest question-like posts from configured niche subreddits."""
    load_env()
    db.init_db()
    cfg = _require_niche()
    from instagram_ai_agent.brain import reddit_harvester

    if not cfg.reddit_subs:
        console.print("[yellow]No reddit_subs configured in niche.yaml.[/yellow]")
        raise typer.Exit(0)
    if not reddit_harvester._praw_available():
        console.print("[red]PRAW not installed.[/red] Try: pip install '.[reddit]'")
        raise typer.Exit(1)
    if not reddit_harvester._creds_configured():
        console.print("[red]Missing REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET / REDDIT_USER_AGENT.[/red]")
        raise typer.Exit(1)

    async def _go():
        all_qs = await asyncio.to_thread(reddit_harvester.fetch_questions, cfg)
        pushed = await reddit_harvester.run_once(cfg) if push else 0
        return all_qs, pushed

    questions, pushed = asyncio.run(_go())
    if not questions:
        console.print("[dim]No question-like posts found in the configured subs.[/dim]")
        return

    t = Table(title=f"reddit questions ({len(questions)})")
    t.add_column("sub", style="cyan")
    t.add_column("score", justify="right")
    t.add_column("comments", justify="right")
    t.add_column("title")
    for q in questions[:limit]:
        t.add_row(f"r/{q.subreddit}", str(q.score), str(q.num_comments), q.title[:80])
    console.print(t)
    if push:
        console.print(f"[green]pushed[/green] {pushed} fresh question(s) to context_feed")


@app.command("events")
def events_cmd(
    push: bool = typer.Option(False, "--push", help="Also push fresh events to context_feed"),
) -> None:
    """Show upcoming holidays + user-defined niche dates."""
    load_env()
    db.init_db()
    cfg = _require_niche()
    from instagram_ai_agent.brain import events as events_mod

    async def _go():
        up = await events_mod.upcoming(cfg)
        pushed = 0
        if push:
            pushed = await events_mod.run_once(cfg)
        return up, pushed

    upcoming, pushed = asyncio.run(_go())

    if not upcoming:
        console.print("[dim]No upcoming events in the configured lookahead window.[/dim]")
        return

    t = Table(title=f"upcoming events ({len(upcoming)})")
    t.add_column("date", style="cyan")
    t.add_column("label")
    t.add_column("source", style="dim")
    t.add_column("note", style="dim")
    for e in upcoming:
        t.add_row(e.date.isoformat(), e.label, e.source, e.note or "")
    console.print(t)
    if push:
        console.print(f"[green]pushed[/green] {pushed} fresh event(s) to context_feed")


@app.command("index-knowledge")
def index_knowledge_cmd(
    clear: bool = typer.Option(False, "--clear", help="Drop the existing index before re-indexing"),
) -> None:
    """Index files dropped into data/knowledge/ for niche RAG retrieval."""
    load_env()
    ensure_dirs()
    db.init_db()
    cfg = _require_niche()
    from instagram_ai_agent.brain import rag

    if clear:
        removed = rag.clear_index()
        console.print(f"[yellow]cleared[/yellow] {removed} chunks")

    async def _go():
        return await rag.index_dir(cfg=cfg.rag)

    stats = asyncio.run(_go())
    console.print(
        f"[green]files seen:[/green] {stats.files_seen}   "
        f"[green]indexed:[/green] {stats.files_indexed}   "
        f"[dim]skipped:[/dim] {stats.files_skipped}   "
        f"[green]chunks added:[/green] {stats.chunks_added}"
    )
    if stats.embed_failures:
        console.print(
            f"[red]embed failures:[/red] {stats.embed_failures}   "
            f"[red]last error:[/red] {stats.last_error or '(unknown)'}"
        )
        console.print(
            "[yellow]Tip: set GEMINI_API_KEY or install `.[rag]` for local embeddings.[/yellow]"
        )
        raise typer.Exit(1)
    s = rag.stats()
    console.print(f"[dim]library:[/dim] {s['chunks']} chunks across {s['sources']} sources")


@app.command("seed-idea-bank")
def seed_idea_bank_cmd(
    fetch: list[str] = typer.Option(
        None,
        "--fetch",
        help="Also pull an external corpus: marketing-prompts or awesome-chatgpt",
    ),
) -> None:
    """Populate the idea bank from the shipped seed.json and optionally external corpora."""
    load_env()
    db.init_db()
    _require_niche()
    from instagram_ai_agent.brain import idea_bank

    console.rule("[bold]idea bank")
    n_seed = idea_bank.seed_from_file()
    console.print(f"[green]curated seed:[/green] +{n_seed} rows")

    for source in fetch or []:
        try:
            n_ext = idea_bank.seed_from_external(source)
            console.print(f"[green]{source}:[/green] +{n_ext} rows")
        except Exception as e:
            console.print(f"[red]{source} failed:[/red] {e}")

    total = idea_bank.count()
    console.print(f"\n[bold]total ideas:[/bold] {total}")

    # License breakdown for transparency
    lic_table = Table(title="license breakdown")
    lic_table.add_column("license", style="cyan")
    lic_table.add_column("count", justify="right")
    for row in idea_bank.license_breakdown():
        lic_table.add_row(row["license"], str(row["c"]))
    console.print(lic_table)


@app.command("hashtag-review")
def hashtag_review() -> None:
    """Review and approve dynamically discovered hashtags into the growth pool."""
    load_env()
    db.init_db()
    cfg = _require_niche()
    from instagram_ai_agent.brain import hashtag_discovery

    # Refresh suggestions every time we review, so stale data doesn't linger
    hashtag_discovery.persist_suggestions(cfg)
    suggestions = db.state_get_json("hashtag_suggestions", default=[]) or []
    if not suggestions:
        console.print("[green]No pending hashtag suggestions.[/green]")
        return

    approved: list[str] = []
    for s in suggestions[:30]:
        console.rule(f"[bold]#{s['tag']}  ·  score={s['score']}")
        console.print(f"[dim]seen on:[/dim] {', '.join(('@' + u) for u in s.get('users', [])[:6])}")
        console.print(
            f"[dim]count:[/dim] {s['count']}   "
            f"[dim]likes_sum:[/dim] {s['likes_sum']}"
        )
        choice = questionary.select(
            "Action:", choices=["approve → growth", "skip", "reject permanently", "quit"],
        ).ask()
        if choice == "approve → growth":
            approved.append(s["tag"])
        elif choice == "quit":
            break

    if approved:
        n = hashtag_discovery.approve_into_growth(cfg, approved)
        console.print(f"[green]Merged {n} tags into growth pool (niche.yaml updated).[/green]")


@app.command("doctor")
def doctor() -> None:
    """Diagnostic self-check — walks through every dependency + config
    step a new user might have skipped or misconfigured. Prints a
    checklist with concrete next-actions for anything failing.

    Run this BEFORE asking for help. 80% of 'why isn't it working'
    questions surface a yellow/red row here."""
    import importlib
    import shutil as _shutil

    load_env()
    results: list[tuple[str, str, str]] = []   # (status, label, hint)

    def ok(label: str, hint: str = "") -> None:
        results.append(("✅", label, hint))

    def warn(label: str, hint: str) -> None:
        results.append(("⚠️", label, hint))

    def fail(label: str, hint: str) -> None:
        results.append(("❌", label, hint))

    # Python version
    import sys as _sys
    v = _sys.version_info
    if v >= (3, 11):
        ok(f"Python {v.major}.{v.minor}")
    else:
        fail(f"Python {v.major}.{v.minor}", "3.11+ required — upgrade Python.")

    # ffmpeg
    if _shutil.which("ffmpeg") and _shutil.which("ffprobe"):
        ok("ffmpeg + ffprobe on PATH")
    else:
        fail("ffmpeg/ffprobe missing",
             "brew install ffmpeg  /  sudo apt install ffmpeg")

    # Playwright chromium
    try:
        from playwright.sync_api import sync_playwright as _spw
        with _spw() as p:
            try:
                br = p.chromium.launch(args=["--no-sandbox"])
                br.close()
                ok("Playwright chromium installed")
            except Exception as e:
                fail("Playwright chromium missing/broken",
                     f"Run: python -m playwright install chromium  ({e})")
    except Exception as e:
        fail("Playwright lib missing", f"pip install -e . ({e})")

    # Fonts present
    from instagram_ai_agent.core.config import FONTS_DIR
    tt = list(FONTS_DIR.glob("*.ttf")) if FONTS_DIR.exists() else []
    if tt:
        ok(f"Fonts: {len(tt)} TTF file(s) in {FONTS_DIR}")
    else:
        warn("No fonts in data/fonts/",
             "Re-run ./install.sh or download Archivo Black + Inter manually.")

    # LLM provider
    providers = providers_configured()
    if providers:
        ok(f"LLM provider(s): {', '.join(providers)}")
    else:
        fail("No LLM provider key set",
             "Add OPENROUTER_API_KEY to .env (free at https://openrouter.ai/keys)")

    # IG creds
    if os.environ.get("IG_USERNAME") and os.environ.get("IG_PASSWORD"):
        ok(f"IG creds set for user {os.environ['IG_USERNAME']}")
    else:
        fail("IG_USERNAME / IG_PASSWORD missing",
             "Fill them in .env (or run `ig-agent init` to use the wizard)")

    # niche.yaml
    try:
        _cfg = load_niche()
        ok(f"niche.yaml valid (niche: {_cfg.niche!r})")
    except FileNotFoundError:
        fail("niche.yaml not found", "Run `ig-agent init`.")
    except Exception as e:
        fail("niche.yaml invalid", f"{e}")

    # Idea bank
    try:
        db.init_db()
        from instagram_ai_agent.brain import idea_bank as _ib
        n_ideas = _ib.count()
        if n_ideas > 0:
            ok(f"Idea bank seeded ({n_ideas} archetypes)")
        else:
            warn("Idea bank empty", "Run `ig-agent seed-idea-bank`")
    except Exception as e:
        warn("Couldn't count ideas", str(e))

    # Optional extras
    for label, mod, extra in (
        ("TLS impersonation (curl_cffi)", "curl_cffi", "[tls]"),
        ("Reddit harvester (praw)", "praw", "[reddit]"),
        ("Niche RAG embeddings", "sentence_transformers", "[rag]"),
        ("Finish pass (Real-ESRGAN)", "realesrgan", "[finish]"),
        ("Beat-sync reels (librosa)", "librosa", "[beat]"),
    ):
        try:
            importlib.import_module(mod)
            ok(f"{label} installed")
        except Exception:
            warn(f"{label} not installed", f"Optional — install via `pip install '.{extra}'`")

    # Device fingerprint
    from instagram_ai_agent.core.config import DEVICE_PATH
    if DEVICE_PATH.exists():
        ok(f"Device fingerprint: {DEVICE_PATH}")
    else:
        warn("Device fingerprint not yet generated",
             "Created automatically on first `ig-agent login`.")

    # ComfyUI
    if os.environ.get("COMFYUI_URL"):
        ok(f"ComfyUI URL set: {os.environ['COMFYUI_URL']}")
    else:
        warn("COMFYUI_URL not set (optional)",
             "Falls back to Pollinations Flux cloud generation.")

    # Render
    console.rule("[bold]ig-agent doctor")
    t = Table(show_lines=False)
    t.add_column("", no_wrap=True)
    t.add_column("check")
    t.add_column("hint", style="dim")
    for status, label, hint in results:
        t.add_row(status, label, hint)
    console.print(t)

    fails = sum(1 for r in results if r[0] == "❌")
    warns = sum(1 for r in results if r[0] == "⚠️")
    if fails:
        console.print(f"\n[red]{fails} blocker(s)[/red] — fix these before `ig-agent run`.")
    elif warns:
        console.print(f"\n[yellow]{warns} warning(s)[/yellow] — optional, won't stop core posting.")
    else:
        console.print("\n[green]All checks passed — you're ready to run.[/green]")


@app.command("show-niche")
def show_niche() -> None:
    load_env()
    cfg = _require_niche()
    console.print(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False, allow_unicode=True))


# ─────────  lora subcommands (brand LoRA training prep + import)  ─────────
lora_app = typer.Typer(no_args_is_help=True, help="Train + wire a brand LoRA into the image pipeline.")
app.add_typer(lora_app, name="lora")


@lora_app.command("prepare")
def lora_prepare(
    src: Path = typer.Argument(..., exists=True, help="Folder of source images (≥10 images, ≥512 px each)."),
    name: str = typer.Option(..., "--name", help="Short slug for the LoRA (letters/digits/_/-)."),
    trigger: str = typer.Option(..., "--trigger", help="Trigger word the LoRA will attach to (e.g. 'mascxyz')."),
    min_images: int = typer.Option(10, min=4, max=200, help="Minimum images required."),
    no_auto_caption: bool = typer.Option(False, "--no-auto-caption", help="Skip vision-LLM captioning."),
) -> None:
    """Build a FluxGym / kohya-ss ready dataset folder."""
    load_env()
    ensure_dirs()
    _require_niche()
    from instagram_ai_agent.plugins import lora as lora_mod
    try:
        summary = asyncio.run(
            lora_mod.prepare_dataset(
                src, name=name, trigger=trigger,
                min_images=min_images, auto_caption=not no_auto_caption,
            )
        )
    except (ValueError, FileNotFoundError, NotADirectoryError) as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    console.print(
        f"[green]Dataset ready[/green] → {summary.dataset_dir}\n"
        f"  images: {summary.image_count}, captions: {summary.captions_written}\n\n"
        f"[dim]Next:[/dim] train with FluxGym or kohya-ss, then import the "
        f".safetensors with\n"
        f"  [bold]ig-agent lora import <file> --name {name} --trigger {trigger}[/bold]"
    )


@lora_app.command("import")
def lora_import(
    source: Path = typer.Argument(..., exists=True, help=".safetensors file produced by training."),
    name: str = typer.Option(..., "--name", help="Short slug (must match the prepare name for consistency)."),
    trigger: str = typer.Option(..., "--trigger", help="Trigger word used at training time."),
    base_model: str = typer.Option(
        "flux-schnell", "--base-model",
        help="flux-schnell (Apache-2.0) | flux-dev (NON-COMMERCIAL) | sdxl (OpenRAIL++)",
    ),
    strength: float = typer.Option(0.85, "--strength", min=-2.0, max=2.0),
    overwrite: bool = typer.Option(False, "--overwrite"),
    activate: bool = typer.Option(True, "--activate/--no-activate", help="Set active in niche.yaml."),
) -> None:
    """Copy a trained LoRA into the agent + optionally activate it."""
    load_env()
    ensure_dirs()
    cfg = _require_niche()
    from instagram_ai_agent.plugins import lora as lora_mod
    try:
        info = lora_mod.import_lora(source, name=name, overwrite=overwrite)
    except (ValueError, FileNotFoundError, FileExistsError) as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    console.print(
        f"[green]Imported[/green] {info.path.name} ({info.size_mb:.1f} MB)"
        + (f"  [dim]base model hint:[/dim] {info.base_model_hint}" if info.base_model_hint else "")
    )
    if info.base_model_hint and info.base_model_hint != base_model:
        console.print(
            f"[yellow]Note:[/yellow] file metadata says {info.base_model_hint!r} "
            f"but --base-model was {base_model!r}. Double-check."
        )
    if not activate:
        return
    try:
        lora_mod.activate_in_niche(
            cfg, name=name, trigger=trigger,
            base_model=base_model, strength_model=strength, strength_clip=strength,
        )
    except (ValueError, FileNotFoundError) as e:
        console.print(f"[red]Activation blocked:[/red] {e}")
        raise typer.Exit(2)
    console.print(f"[green]Activated[/green] lora={name!r} trigger={trigger!r} in niche.yaml")


@lora_app.command("list")
def lora_list() -> None:
    """List every LoRA file under data/loras/."""
    load_env()
    ensure_dirs()
    cfg = _require_niche()
    from instagram_ai_agent.plugins import lora as lora_mod
    items = lora_mod.list_loras()
    if not items:
        console.print("[dim]No LoRAs imported yet. Run `ig-agent lora prepare` then train + import.[/dim]")
        return
    t = Table(title="LoRAs")
    t.add_column("name", style="cyan")
    t.add_column("size", justify="right")
    t.add_column("base hint", style="dim")
    t.add_column("active?", justify="center")
    for i in items:
        active = "yes" if (cfg.lora.enabled and cfg.lora.name == i.name) else ""
        t.add_row(i.name, f"{i.size_mb:.1f} MB", i.base_model_hint or "-", active)
    console.print(t)
    if cfg.lora.enabled:
        console.print(
            f"[dim]active trigger:[/dim] {cfg.lora.trigger_word!r}   "
            f"[dim]strength:[/dim] model={cfg.lora.strength_model} clip={cfg.lora.strength_clip}"
        )


@lora_app.command("activate")
def lora_activate(
    name: str = typer.Argument(...),
    trigger: str = typer.Option(..., "--trigger"),
    base_model: str = typer.Option("flux-schnell", "--base-model"),
    strength: float = typer.Option(0.85, "--strength"),
) -> None:
    """Set an already-imported LoRA as the active brand LoRA in niche.yaml."""
    load_env()
    cfg = _require_niche()
    from instagram_ai_agent.plugins import lora as lora_mod
    try:
        lora_mod.activate_in_niche(
            cfg, name=name, trigger=trigger,
            base_model=base_model, strength_model=strength, strength_clip=strength,
        )
    except (ValueError, FileNotFoundError) as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]Activated[/green] lora={name!r}")


@lora_app.command("deactivate")
def lora_deactivate() -> None:
    """Disable the active LoRA (image gen reverts to plain checkpoint)."""
    load_env()
    cfg = _require_niche()
    from instagram_ai_agent.plugins import lora as lora_mod
    lora_mod.deactivate_in_niche(cfg)
    console.print("[green]Deactivated.[/green]")


@lora_app.command("remove")
def lora_remove(
    name: str = typer.Argument(...),
    yes: bool = typer.Option(False, "--yes", help="Skip confirmation."),
) -> None:
    """Delete a LoRA file from data/loras/."""
    load_env()
    cfg = _require_niche()
    from instagram_ai_agent.plugins import lora as lora_mod
    if cfg.lora.enabled and cfg.lora.name == name:
        console.print(
            f"[yellow]{name!r} is currently active in niche.yaml — run "
            f"`ig-agent lora deactivate` first.[/yellow]"
        )
        raise typer.Exit(1)
    if not yes and not Confirm.ask(f"Delete LoRA {name!r} from disk?"):
        return
    if lora_mod.remove_lora(name):
        console.print(f"[green]Removed[/green] {name}.safetensors")
    else:
        console.print(f"[yellow]Not found:[/yellow] {name}")
        raise typer.Exit(1)


# ─────────  controlnet subcommands (reference-image conditioning)  ─────────
controlnet_app = typer.Typer(
    no_args_is_help=True,
    help="Condition every AI image on a reference (pose / depth / canny).",
)
app.add_typer(controlnet_app, name="controlnet")


@controlnet_app.command("set")
def controlnet_set(
    reference: Path = typer.Argument(..., exists=True, help="Reference image (JPG/PNG/WebP)."),
    mode: str = typer.Option("pose", "--mode", help="pose | depth | canny"),
) -> None:
    """Point ControlNet at a reference image for the given mode."""
    load_env()
    ensure_dirs()
    cfg = _require_niche()
    from instagram_ai_agent.plugins import controlnet as cn_mod
    try:
        dest = cn_mod.set_reference(reference, mode=mode, cfg=cfg)
    except (ValueError, FileNotFoundError) as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    resolved = cn_mod.mode_for(cfg.model_copy(update={"controlnet": cfg.controlnet.model_copy(update={"mode": mode, "enabled": True, "reference_image": dest.name})}))
    console.print(
        f"[green]ControlNet reference set[/green] → {dest}\n"
        f"  mode: {mode}   preprocessor: {resolved.preprocessor} ({resolved.license})\n"
        f"  [dim]Next:[/dim] import the matching .safetensors with "
        f"[bold]ig-agent controlnet import-model <file>[/bold]"
    )


@controlnet_app.command("import-model")
def controlnet_import_model(
    source: Path = typer.Argument(..., exists=True, help="ControlNet .safetensors file."),
    overwrite: bool = typer.Option(False, "--overwrite"),
) -> None:
    """Copy a ControlNet .safetensors into data/controlnet_models/."""
    load_env()
    ensure_dirs()
    cfg = _require_niche()
    from instagram_ai_agent.plugins import controlnet as cn_mod
    try:
        dest = cn_mod.set_model(source, cfg=cfg, overwrite=overwrite)
    except (ValueError, FileNotFoundError, FileExistsError) as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]Imported[/green] {dest.name} → model_name set in niche.yaml")


@controlnet_app.command("clear")
def controlnet_clear(
    yes: bool = typer.Option(False, "--yes"),
) -> None:
    """Disable ControlNet + remove all stored reference images."""
    load_env()
    cfg = _require_niche()
    if not yes and not Confirm.ask("Clear ControlNet config and delete all reference images?"):
        return
    from instagram_ai_agent.plugins import controlnet as cn_mod
    cn_mod.clear_reference(cfg)
    console.print("[green]ControlNet cleared.[/green]")


@controlnet_app.command("show")
def controlnet_show() -> None:
    """Print the current ControlNet config + readiness."""
    load_env()
    cfg = _require_niche()
    from instagram_ai_agent.plugins import controlnet as cn_mod
    t = Table(title="ControlNet")
    t.add_column("field", style="cyan")
    t.add_column("value")
    t.add_row("enabled", str(cfg.controlnet.enabled))
    t.add_row("mode", cfg.controlnet.mode)
    t.add_row("reference_image", cfg.controlnet.reference_image or "(none)")
    t.add_row("model_name", cfg.controlnet.model_name or "(none)")
    t.add_row("strength", f"{cfg.controlnet.strength:.2f}")
    t.add_row("start..end", f"{cfg.controlnet.start_percent}..{cfg.controlnet.end_percent}")
    t.add_row("active?", "yes" if cn_mod.is_active(cfg) else "no")
    resolved = cn_mod.mode_for(cfg)
    t.add_row("preprocessor", f"{resolved.preprocessor} ({resolved.license})")
    console.print(t)

    models = cn_mod.list_models()
    if models:
        mt = Table(title="imported ControlNet models")
        mt.add_column("name", style="cyan")
        mt.add_column("size", justify="right")
        for p in models:
            mt.add_row(p.name, f"{p.stat().st_size / (1024 * 1024):.1f} MB")
        console.print(mt)


def _friendly_main() -> None:
    """Entry-point wrapper — converts uncaught exceptions into one-line
    actionable messages. Set ``IG_DEBUG=1`` to bypass and see the
    original Python traceback. This is what the ``ig-agent`` script
    points at (see pyproject.toml::project.scripts)."""
    from instagram_ai_agent.core.friendly_errors import wrap as _wrap
    _wrap(app)()


if __name__ == "__main__":
    _friendly_main()
