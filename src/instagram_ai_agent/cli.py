"""Command-line interface — setup wizard + ops commands."""
from __future__ import annotations

import asyncio
import os
import random
import re
import shutil
import subprocess
import sys
import webbrowser
from datetime import UTC
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
    Aesthetic,
    BrandCharacter,
    Budget,
    FormatMix,
    HashtagPools,
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
    console.print("[green]Initialised[/green] brain.db")

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


# ───────── one-command setup ─────────
@app.command()
def setup(
    full: bool = typer.Option(False, "--full", help="Customise voice / palette / hashtags / format mix / schedule / optional providers"),
    minimal: bool = typer.Option(False, "--minimal", help="Take preset defaults for everything except niche name"),
    with_login: bool = typer.Option(False, "--with-login", help="Also prompt for Instagram credentials inline + verify the session"),
    force: bool = typer.Option(False, "--force", help="Overwrite existing niche.yaml without asking"),
    review: bool = typer.Option(False, "--review", help="Opt-in guardrail: queue each post for one-click approval at /review before it goes live. Default is fully autonomous."),
    run_after: bool = typer.Option(False, "--run", help="After setup + login succeed, start the orchestrator daemon — one command, walk away"),
) -> None:
    """One-command setup. Two paths:

    * default (quick): 4 questions, everything else preset defaults. ~2 minutes.
    * ``--full``: extends quick with voice, palette, hashtags, format mix,
      schedule, optional providers, inline IG login. ~5 minutes.

    Auto-installs Playwright chromium if missing. Detects ffmpeg and prints
    the exact install command if it's not on PATH. Validates the OpenRouter
    key live before exiting. Refuses to overwrite an existing ``niche.yaml``
    without ``--force`` or explicit confirmation.
    """
    if full and minimal:
        console.print("[red]✗[/red] --full and --minimal are mutually exclusive.")
        raise typer.Exit(2)

    ensure_dirs()

    mode_label = "full" if full else ("minimal" if minimal else "quick")
    n_steps = 5 if full else 4

    console.rule(f"[bold cyan]ig-agent setup[/bold cyan]  [dim]({mode_label})[/dim]")
    if full:
        console.print(
            "[dim]Deep customisation path — voice, palette, hashtags, formats, "
            "schedule, optional providers. ~5 minutes.[/dim]\n"
        )
    else:
        console.print(
            "[dim]Quick path — 4 questions, preset defaults for the rest. "
            "~2 minutes. Run with [bold]--full[/bold] for deep customisation.[/dim]\n"
        )

    # ─── Existing niche.yaml guard ─────────────────────────────
    if NICHE_PATH.exists() and not force:
        console.print(
            f"[yellow]⚠[/yellow] {NICHE_PATH.name} already exists at {NICHE_PATH}."
        )
        if not Confirm.ask("  Overwrite? (existing values will be lost)", default=False):
            console.print(
                "  [dim]Aborted. Use [bold]--force[/bold] to skip this prompt, "
                "or edit [italic]niche.yaml[/italic] directly.[/dim]"
            )
            raise typer.Exit(0)

    # ─── Existing .env warning ─────────────────────────────────
    existing_keys = _existing_env_keys()
    if existing_keys:
        console.print(
            f"[dim]Keeping existing .env keys (not overwriting): "
            f"{', '.join(sorted(existing_keys))}[/dim]\n"
        )

    # ─── Step 1 — system deps ─────────────────────────────────
    console.print(f"[bold]Step 1/{n_steps}[/bold] — checking system dependencies")
    _setup_check_deps()

    # ─── Step 2 — niche ───────────────────────────────────────
    console.print(f"\n[bold]Step 2/{n_steps}[/bold] — pick your niche")
    cfg = _setup_pick_niche(minimal=minimal)

    # Autonomy — the default. Posts go live when they pass the critic, no
    # human gate. --review opts into the training-wheels guardrail (queue +
    # one-click approval at /review) for users who want to see every post
    # before it goes live.
    cfg = cfg.model_copy(update={"safety": Safety(require_review=review)})
    if review:
        console.print(
            "  [dim]🛟 review guardrail ON — posts queue at /review for approval.[/dim]"
        )
    else:
        console.print(
            "  [dim]⚡ fully autonomous — posts go live when they pass the critic.[/dim]"
        )

    # ─── Step 3 (full only) — customisation ───────────────────
    step_num = 2
    if full:
        step_num += 1
        console.print(f"\n[bold]Step {step_num}/{n_steps}[/bold] — customise voice, palette, hashtags, format mix, schedule")
        cfg = _setup_full_customise(cfg)

    # ─── Next step — free AI provider key ─────────────────────
    step_num += 1
    console.print(f"\n[bold]Step {step_num}/{n_steps}[/bold] — free AI provider (1 key, 30s)")
    api_key = _setup_get_openrouter_key()

    # Optional extra providers (full mode only)
    extra_keys: dict[str, str] = {}
    if full:
        extra_keys = _setup_optional_providers()

    # ─── IG login (optional) ─────────────────────────────────
    ig_user, ig_pass = "", ""
    cookie_env: dict[str, str] = {}
    if with_login or full:
        want_ig = with_login or questionary.confirm(
            "Configure Instagram login now? (optional — can defer to `ig-agent login`)",
            default=False,
        ).ask()
        if want_ig:
            # Two Instagram auth paths. Framing is by METHOD (what you paste),
            # not by INFRASTRUCTURE (where the agent runs) — both methods work
            # on both laptop and VPS. The wizard recommendation that VPS needs
            # cookies is documented in the help text but the choice is yours.
            method = questionary.select(
                "How do you want to authenticate to Instagram?",
                choices=[
                    questionary.Choice(
                        "Username + password — simplest, works fine on home WiFi / trusted IP",
                        value="userpass",
                    ),
                    questionary.Choice(
                        "Browser cookie jar — strongest trust signal, required for VPS / server / new IP",
                        value="cookies",
                    ),
                ],
            ).ask()
            if method == "cookies":
                ig_user, cookie_env = _setup_capture_cookies()
            else:
                ig_user = questionary.text("Instagram username:").ask() or ""
                ig_pass = questionary.password("Instagram password:").ask() or ""

    # ─── Final step — save + seed + verify ────────────────────
    step_num += 1
    console.print(f"\n[bold]Step {step_num}/{n_steps}[/bold] — saving + seeding + verifying")
    _setup_save_and_verify(
        cfg,
        api_key=api_key,
        extra_keys={**extra_keys, **cookie_env},
        ig_user=ig_user,
        ig_pass=ig_pass,
    )

    # ─── Auto-login when credentials OR cookies were captured ────
    # Chain setup → login → (optional) run so `ig-agent setup --with-login --run`
    # is TRULY one-command: the user walks away after this single invocation.
    # Cookie-jar path skips /login entirely (set_settings); u/p path logs in normally.
    logged_in = False
    if ig_user and (ig_pass or cookie_env):
        console.print("\n[bold]Verifying Instagram session…[/bold]")
        # Reload env so the new IG creds / cookies are visible to IGClient
        load_env()
        os.environ["IG_USERNAME"] = ig_user
        if ig_pass:
            os.environ["IG_PASSWORD"] = ig_pass
        for k, v in cookie_env.items():
            os.environ[k] = v
        try:
            from instagram_ai_agent.plugins.ig import IGClient
            cl = IGClient()
            cl.login()
            console.print(f"  [green]✓[/green] IG session persisted → {cl.session_path}")
            logged_in = True
        except Exception as e:
            console.print(f"  [yellow]⚠[/yellow] Auto-login failed ({type(e).__name__}): {e}")
            console.print(
                "  [dim]Finish manually with [bold]ig-agent login[/bold] — "
                "common causes: email-code challenge (paste the code when prompted), "
                "bad_password-false-positive on new IP (add IG_PROXY or IG_SESSIONID "
                "to .env), stale cookies (re-extract from your browser).[/dim]"
            )

    # ─── Summary + next steps ────────────────────────────────
    console.rule("[bold green]you're set")
    if not ig_user:
        console.print(
            "[dim]Instagram login deferred — run [bold]ig-agent login[/bold] when "
            "you're ready. You can generate + review posts without it.[/dim]\n"
        )
    elif not logged_in:
        console.print(
            "[dim]IG creds saved but session not yet created. Run "
            "[bold]ig-agent login[/bold] to finish.[/dim]\n"
        )

    # ─── Auto-start the orchestrator if requested ─────────────
    if run_after:
        if not logged_in and ig_user:
            console.print(
                "[yellow]⚠[/yellow] --run requested but login didn't succeed. "
                "Fix the login issue above and re-run: [bold cyan]ig-agent run[/bold cyan]"
            )
            raise typer.Exit(1)
        if not providers_configured():
            console.print("[yellow]⚠[/yellow] No AI provider key — can't start the daemon.")
            raise typer.Exit(1)
        console.print("\n[bold green]starting orchestrator…[/bold green]\n")
        from instagram_ai_agent.orchestrator import main as orch_main
        orch_main()  # blocks — this is the long-running daemon
        return

    console.print("[bold]Next:[/bold]")
    if logged_in:
        console.print("  [bold cyan]ig-agent run[/bold cyan]                 start the autonomous daemon — generates + posts forever")
    console.print("  [bold cyan]ig-agent generate -n 3[/bold cyan]    make your first 3 posts (~2 min)")
    console.print("  [bold cyan]ig-agent review[/bold cyan]           walk + approve each")
    console.print("  [bold cyan]ig-agent dashboard[/bold cyan]        browse everything in a web UI")
    if not full:
        console.print(
            "\n[dim]Want to customise voice/palette/hashtags/schedule? "
            "Re-run [bold]ig-agent setup --full[/bold] or edit [italic]niche.yaml[/italic].[/dim]"
        )


def _existing_env_keys() -> set[str]:
    """Return the set of keys currently present in .env (if any)."""
    if not ENV_PATH.exists():
        return set()
    keys: set[str] = set()
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k = line.partition("=")[0].strip()
        if k:
            keys.add(k)
    return keys


def _setup_full_customise(cfg: NicheConfig) -> NicheConfig:
    """Full-mode extras: voice tone, forbidden words, palette, hashtags,
    format mix, schedule, posts-per-day. Each question shows the preset-
    derived default so Enter accepts it."""
    # Voice
    tone = _split(
        questionary.text(
            "Voice tone (comma-separated):",
            default=", ".join(cfg.voice.tone),
        ).ask() or ", ".join(cfg.voice.tone)
    ) or list(cfg.voice.tone)
    forbidden = _split(
        questionary.text(
            "Words/phrases to NEVER use (comma-separated, optional):",
            default=", ".join(cfg.voice.forbidden),
        ).ask() or ", ".join(cfg.voice.forbidden)
    )
    # CTA styles
    cta_default = ", ".join(cfg.voice.cta_styles) if cfg.voice.cta_styles else "save for later, tag a mate, follow for more"
    cta_styles = _split(
        questionary.text("Preferred CTAs (comma-separated):", default=cta_default).ask() or cta_default
    ) or ["save for later"]

    # Aesthetic
    palette = _split(
        questionary.text(
            "Palette (3–5 hex colours, comma-separated):",
            default=", ".join(cfg.aesthetic.palette),
        ).ask() or ", ".join(cfg.aesthetic.palette)
    ) or list(cfg.aesthetic.palette)
    watermark = questionary.text(
        "Watermark (usually your @handle, optional):",
        default=cfg.aesthetic.watermark or "",
    ).ask() or None

    # Hashtags
    core_default = ", ".join(cfg.hashtags.core)
    core = _split(
        questionary.text("Core hashtags (comma-separated, min 3):", default=core_default).ask()
        or core_default
    )
    while len(core) < 3:
        core.append(f"niche{len(core)}")
    growth_default = ", ".join(cfg.hashtags.growth)
    growth = _split(
        questionary.text("Growth hashtags (comma-separated, optional):", default=growth_default).ask()
        or growth_default
    )

    # Format mix — checkbox
    current_weights = cfg.formats.model_dump()
    active_formats = [name for name, w in current_weights.items() if w > 0 and name != "story_carousel"]
    format_choices = questionary.checkbox(
        "Feed post formats allowed:",
        choices=[
            questionary.Choice(name, checked=(name in active_formats))
            for name in ("meme", "quote_card", "carousel", "reel_stock", "reel_ai", "photo")
        ],
    ).ask() or active_formats
    formats = _even_mix(format_choices)

    # Schedule
    posts_per_day = int(
        questionary.text(
            "Posts per day (0–5, 1 is safest for fresh accounts):",
            default=str(cfg.schedule.posts_per_day),
            validate=_is_int_in(0, 5),
        ).ask()
    )
    hours_default = ", ".join(str(h) for h in cfg.schedule.best_hours_utc)
    hours_str = questionary.text(
        "Best hours UTC (comma-separated, when your audience is active):",
        default=hours_default,
    ).ask() or hours_default
    best_hours = [int(h) for h in re.findall(r"\d+", hours_str) if 0 <= int(h) <= 23] or list(cfg.schedule.best_hours_utc)

    return cfg.model_copy(update={
        "voice": Voice(
            tone=tone,
            forbidden=forbidden,
            persona=cfg.voice.persona,
            cta_styles=cta_styles,
        ),
        "aesthetic": Aesthetic(
            palette=palette,
            heading_font=cfg.aesthetic.heading_font,
            body_font=cfg.aesthetic.body_font,
            watermark=watermark,
        ),
        "hashtags": HashtagPools(
            core=core,
            growth=growth,
            long_tail=cfg.hashtags.long_tail,
            per_post=cfg.hashtags.per_post,
        ),
        "formats": formats,
        "schedule": Schedule(
            posts_per_day=posts_per_day,
            stories_per_day=cfg.schedule.stories_per_day,
            best_hours_utc=best_hours,
        ),
    })


def _setup_optional_providers() -> dict[str, str]:
    """Prompt for additional free-tier provider keys. All are optional —
    OpenRouter alone is enough to run the agent."""
    console.print(
        "  [dim]OpenRouter alone works. These are fallbacks for when OpenRouter's "
        "free tier quotas hit — agent auto-routes across providers.[/dim]"
    )
    keys: dict[str, str] = {}
    groq = questionary.password("GROQ_API_KEY (optional, https://console.groq.com):").ask() or ""
    if groq.strip():
        keys["GROQ_API_KEY"] = groq.strip()
    gemini = questionary.password("GEMINI_API_KEY (optional, https://aistudio.google.com):").ask() or ""
    if gemini.strip():
        keys["GEMINI_API_KEY"] = gemini.strip()
    return keys


def _setup_save_and_verify(
    cfg: NicheConfig,
    *,
    api_key: str,
    extra_keys: dict[str, str],
    ig_user: str,
    ig_pass: str,
) -> None:
    """Persist config + verify the write succeeded by reloading from disk."""
    save_niche(cfg)
    console.print(f"  [green]✓[/green] niche.yaml  → {NICHE_PATH}")

    env_updates: dict[str, str] = {}
    if api_key:
        env_updates["OPENROUTER_API_KEY"] = api_key
    env_updates.update(extra_keys)
    if ig_user:
        env_updates["IG_USERNAME"] = ig_user
    if ig_pass:
        env_updates["IG_PASSWORD"] = ig_pass
    _write_env(env_updates)
    written_keys = sorted(k for k, v in env_updates.items() if v)
    console.print(f"  [green]✓[/green] .env        → {ENV_PATH}"
                  + (f"  [dim](wrote: {', '.join(written_keys)})[/dim]" if written_keys else ""))

    try:
        db.init_db()
        console.print("  [green]✓[/green] brain.db    → initialised")
    except Exception as e:
        console.print(f"  [red]✗[/red] brain.db init FAILED: {e}")
        raise typer.Exit(1)

    try:
        from instagram_ai_agent.brain import idea_bank
        n_seeded = idea_bank.seed_from_file()
        if n_seeded > 0:
            console.print(f"  [green]✓[/green] idea bank  → {n_seeded} archetypes seeded")
        else:
            console.print("  [dim]idea bank already seeded (no-op)[/dim]")
    except Exception as e:
        console.print(f"  [yellow]⚠[/yellow] idea-bank seed skipped: {e}")

    # ─── Post-save verification ────────────────────────────────
    # Reload niche.yaml from disk to confirm the write is parseable.
    try:
        loaded = load_niche()
        assert loaded.niche == cfg.niche
        console.print("  [green]✓[/green] verified    → niche.yaml round-trips cleanly")
    except Exception as e:
        console.print(f"  [red]✗[/red] niche.yaml reload FAILED: {e}")
        raise typer.Exit(1)


def _setup_check_deps() -> None:
    """Verify system deps; auto-install what we can, print the copy-paste
    command for anything that needs elevated privileges."""
    # Python version — already enforced by pyproject, but double-check
    py = f"{sys.version_info.major}.{sys.version_info.minor}"
    if sys.version_info >= (3, 11):
        console.print(f"  [green]✓[/green] Python {py}")
    else:
        console.print(f"  [red]✗[/red] Python {py} — need 3.11+. Upgrade and re-run setup.")
        raise typer.Exit(1)

    # ffmpeg — auto-install via winget (Windows) or brew (macOS). Linux
    # needs sudo which we can't prompt for cleanly, so we print the command.
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        console.print("  [green]✓[/green] ffmpeg + ffprobe")
    else:
        if _try_auto_install_ffmpeg():
            console.print("  [green]✓[/green] ffmpeg installed")
        else:
            cmd = _ffmpeg_install_cmd()
            console.print(f"  [red]✗[/red] ffmpeg missing — run: [bold]{cmd}[/bold]")
            if not Confirm.ask(
                "  Continue setup anyway? (you'll need ffmpeg before `generate`)",
                default=True,
            ):
                raise typer.Exit(1)

    # Playwright chromium — auto-install when missing (300MB, ~90s)
    if _playwright_chromium_installed():
        console.print("  [green]✓[/green] Playwright chromium")
    else:
        console.print("  [yellow]⚠[/yellow] Playwright chromium missing — installing (~90s)…")
        try:
            subprocess.run(
                [sys.executable, "-m", "playwright", "install", "chromium"],
                check=True,
            )
            console.print("  [green]✓[/green] Playwright chromium installed")
        except subprocess.CalledProcessError as e:
            console.print(f"  [red]✗[/red] chromium install failed: {e}")
            console.print("     Re-run [bold]python -m playwright install chromium[/bold] manually.")
            raise typer.Exit(1)


def _ffmpeg_install_cmd() -> str:
    if sys.platform == "darwin":
        return "brew install ffmpeg"
    if sys.platform.startswith("linux"):
        # Debian/Ubuntu is the common path; Arch/Fedora users can adapt.
        return "sudo apt install ffmpeg  (or: brew install ffmpeg)"
    if sys.platform == "win32":
        return "winget install Gyan.FFmpeg  (or: choco install ffmpeg)"
    return "install ffmpeg via your package manager"


def _try_auto_install_ffmpeg() -> bool:
    """Attempt silent install of ffmpeg via the OS's package manager.

    Returns True when ffmpeg + ffprobe are both on PATH after the attempt.
    Does NOT raise — callers fall back to printing the manual install
    command on failure.

    * Windows: winget install Gyan.FFmpeg (no admin prompt usually)
    * macOS: brew install ffmpeg (no sudo required)
    * Linux: requires sudo for apt/dnf/pacman — returns False, user must
      run the printed command themselves.
    """
    # Linux needs sudo, which we can't prompt for silently — fall through.
    if sys.platform.startswith("linux"):
        return False

    commands: list[list[str]] = []
    if sys.platform == "win32":
        if shutil.which("winget"):
            commands.append(["winget", "install", "--silent", "--accept-package-agreements",
                             "--accept-source-agreements", "Gyan.FFmpeg"])
        elif shutil.which("choco"):
            commands.append(["choco", "install", "-y", "ffmpeg"])
    elif sys.platform == "darwin":
        if shutil.which("brew"):
            commands.append(["brew", "install", "ffmpeg"])

    if not commands:
        return False

    for cmd in commands:
        pretty = " ".join(cmd)
        console.print(f"  [yellow]⚠[/yellow] ffmpeg missing — auto-installing via [bold]{pretty}[/bold] (~30s)…")
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except Exception as e:
            console.print(f"     [dim]{type(e).__name__}: {e}[/dim]")
            continue
        if r.returncode != 0:
            tail = (r.stderr or r.stdout or "")[-300:].strip()
            console.print(f"     [dim]exit {r.returncode}: {tail}[/dim]")
            continue
        # On Windows, winget may have installed to a location that isn't on
        # PATH until the shell restarts. Probe common install locations so
        # the current process can still find ffmpeg.
        if not shutil.which("ffmpeg"):
            _prepend_windows_ffmpeg_to_path()
        if shutil.which("ffmpeg") and shutil.which("ffprobe"):
            return True
        console.print(
            "     [dim]installed but not yet on PATH — close + reopen PowerShell, "
            "then re-run `ig-agent setup`.[/dim]"
        )
        return False
    return False


def _prepend_windows_ffmpeg_to_path() -> None:
    """On Windows, winget installs ffmpeg under
    %LOCALAPPDATA%\\Microsoft\\WinGet\\Packages\\... and the new PATH entry
    isn't visible to the current process. Probe known locations and prepend
    whatever we find to the in-process PATH so subsequent subprocess calls
    can invoke ffmpeg without a shell restart."""
    if sys.platform != "win32":
        return
    candidates = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages",
        Path("C:/Program Files/ffmpeg"),
        Path("C:/ffmpeg"),
    ]
    for root in candidates:
        if not root.is_dir():
            continue
        for ff in root.rglob("ffmpeg.exe"):
            bin_dir = str(ff.parent)
            if bin_dir not in os.environ.get("PATH", ""):
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
            return


def _playwright_chromium_installed() -> bool:
    """True when the chromium browser binary is ACTUALLY present and launchable.

    We check both that the browser directory exists AND that the headless
    shell binary inside it is a real file. Playwright leaves the dir behind
    if an install is interrupted (Ctrl-C or disk-full) — just checking the
    dir would incorrectly report success after a broken install.
    """
    try:
        from playwright._impl._driver import compute_driver_executable  # noqa: F401
    except Exception:
        return False
    # Playwright drops browsers under ~/.cache/ms-playwright on Linux/macOS,
    # %USERPROFILE%\AppData\Local\ms-playwright on Windows.
    candidates = [
        Path.home() / ".cache" / "ms-playwright",
        Path.home() / "AppData" / "Local" / "ms-playwright",
        Path.home() / "Library" / "Caches" / "ms-playwright",
    ]
    browser_bins = [
        "chrome-linux/headless_shell",
        "chrome-linux/chrome",
        "chrome-mac/Chromium.app/Contents/MacOS/Chromium",
        "chrome-mac/Chromium.app/Contents/MacOS/Chromium Headless Shell",
        "chrome-win/chrome.exe",
        "chrome-win/headless_shell.exe",
    ]
    for root in candidates:
        if not root.is_dir():
            continue
        for browser_dir in root.glob("chromium-*"):
            for rel in browser_bins:
                if (browser_dir / rel).exists():
                    return True
    return False


def _setup_pick_niche(*, minimal: bool) -> NicheConfig:
    from instagram_ai_agent.niche_presets import PRESETS, by_key

    preset_choices = [f"{p.key} — {p.label}" for p in PRESETS] + ["custom (blank defaults)"]
    preset_label = questionary.select("Starter preset:", choices=preset_choices).ask()
    if preset_label is None:
        raise typer.Exit(1)
    preset_key = preset_label.split(" — ")[0] if "—" in preset_label else None
    preset = by_key(preset_key) if preset_key else None

    niche = questionary.text(
        "Niche (what this page is about):",
        default=preset.niche if preset else "",
        validate=lambda x: len(x.strip()) >= 3 or "Minimum 3 chars",
    ).ask()
    if niche is None:
        raise typer.Exit(1)

    if minimal:
        sub_topics = list(preset.sub_topics) if preset else [niche.strip()]
        audience = preset.target_audience if preset else f"people interested in {niche}"
        persona = preset.persona if preset else f"writer covering {niche}, direct no-nonsense voice"
    else:
        sub_topics = _split(
            questionary.text(
                "Sub-topics (comma-separated):",
                default=", ".join(preset.sub_topics) if preset else "",
            ).ask() or ""
        )
        if not sub_topics:
            sub_topics = [niche.strip()]
        audience = questionary.text(
            "Target audience (who it's for):",
            default=preset.target_audience if preset else "",
            validate=lambda x: len(x.strip()) >= 5 or "Minimum 5 chars",
        ).ask() or (preset.target_audience if preset else niche)
        persona = questionary.text(
            "One-sentence persona (who's writing these posts?):",
            default=preset.persona if preset else "",
            validate=lambda x: len(x.strip()) >= 10 or "Give a proper sentence",
        ).ask() or (preset.persona if preset else f"writer covering {niche}")

    # Everything else uses preset or sensible defaults — editable in niche.yaml
    tone = list(preset.voice_tone) if preset else ["direct"]
    forbidden = list(preset.voice_forbidden) if preset else []
    palette = list(preset.palette) if preset else ["#0a0a0a", "#f5f5f0", "#c9a961"]
    core_tags = list(preset.core_hashtags) if preset else [
        niche.replace(" ", "").lower()[:20], "instagram", "content",
    ]
    # HashtagPools requires >=3 core items
    while len(core_tags) < 3:
        core_tags.append(f"niche{len(core_tags)}")
    growth_tags = list(preset.growth_hashtags) if preset else []
    best_hours = list(preset.best_hours_utc) if preset else [14, 18, 21]

    if preset and preset.format_weights:
        formats = FormatMix(**preset.format_weights)
    else:
        formats = FormatMix()

    cfg = NicheConfig(
        niche=niche.strip(),
        sub_topics=sub_topics,
        target_audience=audience.strip(),
        commercial=True,
        voice=Voice(
            tone=tone,
            forbidden=forbidden,
            persona=persona.strip(),
            cta_styles=["save for later", "tag a mate", "follow for more"],
        ),
        aesthetic=Aesthetic(
            palette=palette,
            heading_font="Archivo Black",
            body_font="Inter",
            watermark=None,
        ),
        hashtags=HashtagPools(core=core_tags, growth=growth_tags, long_tail=[], per_post=15),
        formats=formats,
        stories=StoryMix(),
        schedule=Schedule(posts_per_day=1, stories_per_day=3, best_hours_utc=best_hours),
        safety=Safety(require_review=False),  # overridden in setup() based on --review flag
    )
    return cfg


def _setup_get_openrouter_key() -> str:
    """Browser-open flow for OpenRouter free-tier key + live validation."""
    url = "https://openrouter.ai/keys"
    console.print(f"  Opening [bold]{url}[/bold] — sign in, click [bold]Create key[/bold].")
    console.print(
        "  [dim](OpenRouter ships free-tier models that work out of the box. "
        "Sign-in is Google/GitHub — no credit card.)[/dim]"
    )

    try:
        webbrowser.open(url, new=2)
    except Exception:
        console.print(f"  [yellow]⚠[/yellow] Couldn't auto-open a browser — visit {url} manually.")

    for attempt in range(3):
        key = questionary.password("  Paste your OpenRouter key:").ask() or ""
        key = key.strip()
        if not key:
            if attempt < 2 and Confirm.ask("  Empty key — try again?", default=True):
                continue
            console.print("  [yellow]⚠[/yellow] Continuing without a key — add it to .env before `generate`.")
            return ""

        if not (key.startswith("sk-or-") or key.startswith("sk-")):
            console.print("  [yellow]⚠[/yellow] That doesn't look like an OpenRouter key (expected sk-or-…).")
            if not Confirm.ask("  Save it anyway?", default=False):
                continue

        # Live ping — ensures the key works before the user exits setup
        ok, msg = _validate_openrouter_key(key)
        if ok:
            console.print(f"  [green]✓[/green] Key validated — {msg}")
            return key
        console.print(f"  [red]✗[/red] Key rejected — {msg}")
        if attempt < 2 and Confirm.ask("  Try a different key?", default=True):
            continue
        break

    console.print("  [yellow]⚠[/yellow] Proceeding without a validated key — you can edit .env later.")
    return ""


# ─── Cookie-jar capture (VPS path) ─────────────────────────────
# Maps Cookie-Editor exported names (lowercase, same as DevTools rows) to
# the IG_* env vars the agent reads. Kept exhaustive to match plugins/ig.py.
_COOKIE_NAME_TO_ENV: dict[str, str] = {
    "sessionid":   "IG_SESSIONID",
    "ds_user_id":  "IG_DS_USER_ID",
    "csrftoken":   "IG_CSRFTOKEN",
    "mid":         "IG_MID",
    "ig_did":      "IG_DID",
    "datr":        "IG_DATR",
    "rur":         "IG_RUR",
    "shbid":       "IG_SHBID",
    "shbts":       "IG_SHBTS",
    "ig_nrcb":     "IG_NRCB",
    "wd":          "IG_WD",
    "dpr":         "IG_DPR",
    "ig_lang":     "IG_IG_LANG",
    "ps_l":        "IG_PS_L",
    "ps_n":        "IG_PS_N",
    "mcd":         "IG_MCD",
    "ccode":       "IG_CCODE",
}

# Minimum set required for the agent to skip the /login call entirely
# (full-jar path in plugins/ig.py:_has_full_cookie_set).
_COOKIE_REQUIRED = {"sessionid", "ds_user_id", "csrftoken"}


def _parse_cookie_editor_json(raw: str) -> dict[str, str]:
    """Parse a Cookie-Editor / EditThisCookie export blob.

    Accepts the JSON array shape the extension exports: list of dicts with
    ``name``, ``value``, ``domain``, plus optional ``httpOnly``/``secure``/
    ``sameSite``/``expirationDate`` fields. Filters to instagram.com domains,
    maps known names to IG_* env var names, returns dict suitable for .env.

    Raises ValueError on malformed input or when required cookies are missing.
    """
    import json as _json

    s = raw.strip()
    if not s:
        raise ValueError("empty input — paste the Cookie-Editor JSON export")

    # Users sometimes paste with surrounding fences / labels — strip common variants
    if s.startswith("```"):
        s = s.strip("`").partition("\n")[2].rstrip("`").strip()

    try:
        data = _json.loads(s)
    except _json.JSONDecodeError as e:
        raise ValueError(
            f"not valid JSON: {e}. Make sure you clicked Cookie-Editor → "
            "Export → JSON (not Netscape) and pasted the whole blob."
        ) from e

    if not isinstance(data, list):
        raise ValueError(
            "expected a JSON array (Cookie-Editor's JSON format). "
            "Try exporting again — click the extension → Export dropdown → JSON."
        )

    env: dict[str, str] = {}
    seen: set[str] = set()
    for row in data:
        if not isinstance(row, dict):
            continue
        domain = str(row.get("domain") or "").lstrip(".").lower()
        if "instagram.com" not in domain:
            continue
        name = str(row.get("name") or "").strip()
        value = str(row.get("value") or "")
        if not name or not value:
            continue
        env_name = _COOKIE_NAME_TO_ENV.get(name.lower())
        if env_name:
            env[env_name] = value
            seen.add(name.lower())

    missing = _COOKIE_REQUIRED - seen
    if missing:
        raise ValueError(
            f"missing required cookie(s): {', '.join(sorted(missing))}. "
            "Log into instagram.com in the same browser first (you need to be "
            "actively logged in), then re-export."
        )
    return env


def _validate_cookie_jar(env: dict[str, str]) -> tuple[bool, str]:
    """Live ping Instagram with the pasted cookies. 200 = alive. 401/403 =
    dead cookies, user must re-extract.

    Auto-selects the endpoint + UA based on cookie origin:
      * Web-origin (``wd``/``dpr`` present) → www.instagram.com/accounts/edit
        with a desktop Chrome UA. This is the "loud fail" endpoint per 2026
        research — returns 403 "useragent mismatch" if UA-cookie pair wrong.
      * Mobile-origin (default) → i.instagram.com/accounts/current_user with
        instagrapi's canonical mobile Android UA.
    """
    web_mode = bool(env.get("IG_WD") or env.get("IG_DPR"))
    try:
        import httpx
        cookies = {
            "sessionid":  env.get("IG_SESSIONID", ""),
            "ds_user_id": env.get("IG_DS_USER_ID", ""),
            "csrftoken":  env.get("IG_CSRFTOKEN", ""),
            "mid":        env.get("IG_MID", ""),
            "ig_did":     env.get("IG_DID", ""),
            "rur":        env.get("IG_RUR", ""),
        }
        custom_ua = env.get("IG_USER_AGENT", "").strip()
        if web_mode:
            url = "https://www.instagram.com/api/v1/accounts/edit/web_form_data/"
            headers = {
                "User-Agent": custom_ua or (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/138.0.0.0 Safari/537.36"
                ),
                "X-IG-App-ID":       "936619743392459",
                "X-ASBD-ID":         "198387",
                "X-Requested-With":  "XMLHttpRequest",
                "Sec-CH-UA":         '"Chromium";v="138", "Google Chrome";v="138", "Not/A)Brand";v="24"',
                "Sec-CH-UA-Mobile":  "?0",
                "Sec-CH-UA-Platform": '"Windows"',
                "Sec-Fetch-Dest":    "empty",
                "Sec-Fetch-Mode":    "cors",
                "Sec-Fetch-Site":    "same-origin",
                "Referer":           "https://www.instagram.com/accounts/edit/",
                "Accept":            "*/*",
                "Accept-Language":   "en-US,en;q=0.9",
            }
        else:
            url = "https://i.instagram.com/api/v1/accounts/current_user/?edit=true"
            headers = {
                "User-Agent": custom_ua or (
                    "Instagram 381.0.0.48.119 Android (34/14; 420dpi; 1080x2340; "
                    "samsung; SM-S918B; dm3q; qcom; en_GB; 697519287)"
                ),
                "X-IG-App-ID": "936619743392459",
            }
        r = httpx.get(
            url, cookies=cookies, headers=headers,
            timeout=10.0, follow_redirects=False,
        )
    except Exception as e:
        return False, f"network error: {e}"
    if r.status_code == 200:
        try:
            # Both endpoints return JSON with the username somewhere
            payload = r.json() or {}
            username = (
                payload.get("user", {}).get("username")
                or payload.get("form_data", {}).get("username")
                or "?"
            )
        except Exception:
            username = "?"
        mode = "web" if web_mode else "mobile"
        return True, f"logged in as @{username} [{mode} mode]"
    if r.status_code in (401, 403):
        return False, f"cookies rejected (HTTP {r.status_code}) — likely stale or UA/TLS mismatch"
    return False, f"unexpected HTTP {r.status_code}"


def _aged_account_extras(env: dict[str, str]) -> dict[str, str]:
    """Follow-up prompts when cookies were captured — for aged/bought accounts.

    Adds to env (all optional, skip-friendly):
      * IG_DEVICE_IMPORT_PATH — path to seller's device.json bundle
      * IG_REST_UNTIL — 48h default, blocks write actions during rest
      * IG_FREEZE_PROFILE_UNTIL — 21d default, blocks profile-edit endpoints

    Also decodes the rur cookie from the captured jar and warns loudly
    when the edge continent mismatches the user's declared country.
    """
    from instagram_ai_agent.core import gates
    from instagram_ai_agent.plugins import rur as rur_mod

    # Aged-account pre-flight on the captured rur cookie
    rur_raw = env.get("IG_RUR", "")
    rur_info = rur_mod.parse_rur(rur_raw) if rur_raw else None
    if rur_info:
        age_h = rur_info.age_hours
        if age_h is None:
            console.print(f"  [dim]rur region: {rur_info.region} (timestamp unparseable)[/dim]")
        else:
            age_label = f"{age_h:.1f}h old"
            if rur_info.is_stale:
                console.print(
                    f"  [yellow]⚠[/yellow] rur region {rur_info.region} ({age_label}) — "
                    "session is nearing rotation (>48h). First API call will likely "
                    "force a /login which burns the aged-cookie value. Re-extract "
                    "cookies from the live browser session before proceeding."
                )
            else:
                console.print(
                    f"  [green]✓[/green] rur region {rur_info.region}, continent {rur_info.continent} "
                    f"({age_label} — fresh)"
                )

    # Aged-account setup confirmation — gate the extras on this
    is_aged = questionary.confirm(
        "Is this an aged/bought account? (preserves seller session with rest + freeze periods)",
        default=False,
    ).ask()
    if not is_aged:
        return env

    # Device bundle import
    console.print(
        "  [dim]If the seller provided a device.json / settings.json bundle with "
        "phone_id / device_id / ig_did / advertising_id, import it now to preserve "
        "the account's device-trust lineage. Fresh UUIDs trigger 'device reset' "
        "review within 24-72h on aged accounts.[/dim]"
    )
    bundle_path = questionary.text(
        "Path to seller's device bundle (leave blank if you don't have one):",
        default="",
    ).ask() or ""
    if bundle_path.strip():
        p = Path(bundle_path.strip()).expanduser()
        if p.exists():
            env["IG_DEVICE_IMPORT_PATH"] = str(p.resolve())
            console.print(f"  [green]✓[/green] device bundle queued for import: {p}")
        else:
            console.print(f"  [yellow]⚠[/yellow] {p} not found — skipping device import")

    # Rest period (48h default)
    rest_until = gates.suggest_rest_until(hours=48)
    if questionary.confirm(
        f"Enable 48h rest period? (blocks posts/follows/likes until {rest_until[:16]}Z — "
        "2026 operator consensus for sold sessions)",
        default=True,
    ).ask():
        env["IG_REST_UNTIL"] = rest_until
        console.print(f"  [green]✓[/green] rest period active until {rest_until}")

    # Profile freeze (21d default)
    freeze_until = gates.suggest_freeze_until(days=21)
    if questionary.confirm(
        f"Enable 21-day profile-metadata freeze? (blocks avatar/bio/password/2FA edits — "
        "ownership-change flags in IG's risk model)",
        default=True,
    ).ask():
        env["IG_FREEZE_PROFILE_UNTIL"] = freeze_until
        console.print(f"  [green]✓[/green] profile freeze active until {freeze_until}")

    return env


def _setup_capture_cookies() -> tuple[str, dict[str, str]]:
    """Guided cookie-jar capture — the 2026 VPS-safe auth path.

    Returns (username, env_var_dict). Username is needed so the agent can
    name the session file correctly even though /login is never called.
    """
    console.print(
        "[bold]Cookie-jar capture[/bold] — 2026 best practice for VPS / server deployments."
    )
    console.print(
        "  [dim]Web-based logins from a fresh server IP almost always trigger "
        "challenges. Pasting a valid cookie jar from a browser where you're "
        "already logged in skips the login call entirely — zero challenge surface.[/dim]\n"
    )
    console.print("  [bold]Steps:[/bold]")
    console.print("    1. Install [bold]Cookie-Editor[/bold] → [cyan]https://cookie-editor.com[/cyan]")
    console.print("    2. Log into [cyan]https://instagram.com[/cyan] in that browser")
    console.print("    3. Click the Cookie-Editor icon → [bold]Export[/bold] dropdown → [bold]JSON[/bold]")
    console.print("    4. Paste the full export below when prompted")
    console.print()

    ig_user = questionary.text(
        "Instagram username (for session-file naming):",
        validate=lambda x: len(x.strip()) >= 1 or "required",
    ).ask() or ""
    ig_user = ig_user.strip().lstrip("@")

    for attempt in range(3):
        blob = questionary.text(
            "Paste the Cookie-Editor JSON export (then press Enter):",
            multiline=True,
        ).ask() or ""

        try:
            env = _parse_cookie_editor_json(blob)
        except ValueError as e:
            console.print(f"  [red]✗[/red] {e}")
            if attempt < 2 and Confirm.ask("  Try pasting again?", default=True):
                continue
            return ig_user, {}

        found = [k for k in _COOKIE_NAME_TO_ENV.values() if k in env]
        console.print(f"  [green]✓[/green] Parsed {len(found)} cookies: {', '.join(found)}")

        # Optional UA pin — matters massively for cookie-jar path survival
        ua = questionary.text(
            "User agent to pin (leave blank to auto-derive from device.json):",
            default="",
        ).ask() or ""
        if ua.strip():
            env["IG_USER_AGENT"] = ua.strip()

        ok, msg = _validate_cookie_jar(env)
        if ok:
            console.print(f"  [green]✓[/green] Live validation passed — {msg}")
            # Aged-account extras — rur sanity, device bundle import,
            # rest/freeze period defaults. All optional, skip-friendly.
            env = _aged_account_extras(env)
            return ig_user, env
        console.print(f"  [red]✗[/red] Live validation failed — {msg}")
        if attempt < 2 and Confirm.ask("  Re-export cookies and try again?", default=True):
            continue
        if Confirm.ask("  Save the cookies anyway and continue?", default=False):
            env = _aged_account_extras(env)
            return ig_user, env
        return ig_user, {}

    return ig_user, {}


def _validate_openrouter_key(key: str) -> tuple[bool, str]:
    try:
        import httpx
        r = httpx.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=10.0,
        )
    except Exception as e:
        return False, f"network error: {e}"
    if r.status_code == 200:
        try:
            n_models = len(r.json().get("data", []))
        except Exception:
            n_models = 0
        return True, f"{n_models} models available"
    if r.status_code in (401, 403):
        return False, "invalid key (401/403)"
    return False, f"HTTP {r.status_code}"


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

    if not os.environ.get("IG_USERNAME") or not os.environ.get("IG_PASSWORD"):
        console.print("[red]✗[/red] IG_USERNAME or IG_PASSWORD missing in .env.")
        console.print(
            "  Fix: edit [italic].env[/italic] and fill both, OR run "
            "[bold cyan]ig-agent setup --with-login[/bold cyan]."
        )
        raise typer.Exit(2)

    from instagrapi.exceptions import (
        BadPassword as _BadPassword,
    )
    from instagrapi.exceptions import (
        ChallengeRequired as _ChallengeRequired,
    )
    from instagrapi.exceptions import (
        LoginRequired as _LoginRequired,
    )
    from instagrapi.exceptions import (
        TwoFactorRequired as _TwoFactorRequired,
    )

    from instagram_ai_agent.plugins.ig import ChallengeNeedsManualCode, IGClient

    cl = IGClient()
    try:
        cl.login()
    except _TwoFactorRequired:
        console.print("[red]✗[/red] 2FA is enabled on this account.")
        console.print(
            "  Add [bold]IG_TOTP_SECRET[/bold] (base32) to .env — find it in "
            "Instagram → Settings → Security → Two-Factor Authentication → "
            "Authentication App."
        )
        raise typer.Exit(1)
    except _BadPassword:
        console.print("[red]✗[/red] Instagram rejected the password.")
        console.print(
            "  Note: on a fresh VPS, IG sometimes returns bad_password on a "
            "correct password — it really means 'suspicious IP'. Fixes:\n"
            "    • add a residential proxy: [bold]IG_PROXY=http://…[/bold] in .env\n"
            "    • paste [bold]IG_SESSIONID[/bold] from your browser cookies (skips login entirely)\n"
            "    • otherwise, double-check the password"
        )
        raise typer.Exit(1)
    except ChallengeNeedsManualCode:
        console.print(
            "[yellow]⚠[/yellow] Instagram sent an email-code challenge. "
            "The CLI prompted for it — if you're seeing this, you didn't enter it.\n"
            "  To automate this: set [bold]IMAP_HOST/USER/PASS[/bold] in .env "
            "(Gmail users: use an app password)."
        )
        raise typer.Exit(1)
    except _ChallengeRequired as e:
        console.print(f"[red]✗[/red] Instagram challenge requires manual resolution: {e}")
        console.print(
            "  Open the IG app → confirm the login in the 'Suspicious login' "
            "notification → re-run [bold cyan]ig-agent login[/bold cyan]."
        )
        raise typer.Exit(1)
    except _LoginRequired as e:
        console.print(f"[red]✗[/red] LoginRequired: {e}")
        console.print(
            "  Your session cookie is dead. Delete [italic]data/sessions/*.json[/italic] "
            "and re-run [bold cyan]ig-agent login[/bold cyan]."
        )
        raise typer.Exit(1)
    except Exception as e:
        # Fall-through — capture unexpected errors with exception type for clarity
        console.print(f"[red]✗[/red] Login failed ({type(e).__name__}): {e}")
        console.print(
            "  Try [bold cyan]ig-agent doctor[/bold cyan] to diagnose, or "
            "[bold]tail logs/orchestrator.log[/bold] for the full stack."
        )
        raise typer.Exit(1)
    console.print(f"[green]✓[/green] Session persisted → {cl.session_path}")


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

    # Pre-flight: confirm there's a usable IG session file BEFORE kicking off
    # the orchestrator loop. Otherwise the first post job fails with a
    # cryptic instagrapi stack that's buried in logs.
    from instagram_ai_agent.core.config import DATA_DIR
    sessions_dir = DATA_DIR / "sessions"
    has_session = sessions_dir.is_dir() and any(sessions_dir.glob("*.json"))
    if not has_session:
        username = os.environ.get("IG_USERNAME") or ""
        if not username or not os.environ.get("IG_PASSWORD"):
            console.print(
                "[red]✗[/red] No Instagram session and IG_USERNAME/IG_PASSWORD are missing."
            )
        else:
            console.print(
                "[red]✗[/red] No Instagram session yet — "
                f"data/sessions/{username}.json does not exist."
            )
        console.print(
            "  Run [bold cyan]ig-agent login[/bold cyan] first to create the "
            "session, then [bold cyan]ig-agent run[/bold cyan].\n"
            "  (Or generate + review offline: [bold cyan]ig-agent generate -n 3[/bold cyan] "
            "doesn't need a session.)"
        )
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

    configured = providers_configured()
    if not configured:
        # Actionable diagnostic — tell the user exactly which keys are
        # missing, don't just say "no providers".
        expected = {
            "OPENROUTER_API_KEY": "https://openrouter.ai/keys",
            "GROQ_API_KEY":       "https://console.groq.com",
            "GEMINI_API_KEY":     "https://aistudio.google.com",
            "CEREBRAS_API_KEY":   "https://cloud.cerebras.ai",
        }
        missing = [k for k in expected if not os.environ.get(k)]
        console.print("[red]✗[/red] No working AI provider — need at least one of:")
        for key in missing:
            console.print(f"    [dim]•[/dim] [bold]{key}[/bold]  → {expected[key]}")
        console.print(
            "\n  Fastest fix: run [bold cyan]ig-agent setup[/bold cyan] "
            "(opens OpenRouter in your browser + validates the key live).\n"
            "  Or edit [italic].env[/italic] and rerun."
        )
        raise typer.Exit(2)

    console.print(
        f"[dim]Providers: {', '.join(configured)} · generating {count} "
        f"{'post' if count == 1 else 'posts'}. Each takes ~30–120s "
        "(LLM calls + image render).[/dim]\n"
    )

    async def _go():
        made: list[int] = []
        failures: list[str] = []
        from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Generating", total=count)
            for i in range(count):
                progress.update(task, description=f"Generating post {i + 1}/{count} ({format or 'auto-pick'})")
                try:
                    cid = await content_pipeline.generate_one(
                        cfg, format_override=format, contrarian_override=contrarian,
                    )
                except Exception as e:
                    failures.append(f"#{i + 1}: {type(e).__name__}: {e}")
                    cid = None
                if cid is not None:
                    made.append(cid)
                    progress.log(f"[green]✓[/green] post {i + 1}: id={cid} ({format or 'auto'})")
                else:
                    progress.log(f"[yellow]·[/yellow] post {i + 1}: skipped or rejected (hit regen cap)")
                progress.advance(task)

        console.print(f"\n[bold]Done:[/bold] {len(made)} / {count} enqueued")
        if failures:
            console.print("\n[yellow]Failures:[/yellow]")
            for f in failures[:3]:
                console.print(f"  {f}")
            if len(failures) > 3:
                console.print(f"  … and {len(failures) - 3} more.")
            console.print(
                "\n[dim]Full traceback: [bold]tail -f logs/orchestrator.log[/bold][/dim]"
            )
        if made:
            console.print(
                "\n[dim]Next:[/dim] [bold cyan]ig-agent dashboard[/bold cyan]  "
                "[dim](approve visually)[/dim]   or   [bold cyan]ig-agent review[/bold cyan]  [dim](in terminal)[/dim]"
            )

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
        console.print("[green]Nothing to review.[/green]")
        console.print(
            "  [dim]Run [bold cyan]ig-agent generate -n 3[/bold cyan] to make "
            "some posts, then re-run review. Prefer a visual web UI? "
            "[bold cyan]ig-agent dashboard[/bold cyan].[/dim]"
        )
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
    """Print queue depth, next scheduled posts, orchestrator heartbeat, backoff, pause state."""
    load_env()
    db.init_db()
    cfg = _require_niche()

    # ─── Agent pulse ─────────────────────────────────────────
    from datetime import datetime
    paused = (db.state_get("paused") or "").lower() in ("1", "true", "yes")
    last_beat = db.state_get("last_heartbeat")
    beat_line = "[dim]never[/dim]"
    if last_beat:
        try:
            beat_dt = datetime.fromisoformat(last_beat.replace("Z", "+00:00"))
            age_min = (datetime.now(UTC) - beat_dt).total_seconds() / 60
            if age_min < 5:
                beat_line = f"[green]{age_min:.1f} min ago[/green]"
            elif age_min < 60:
                beat_line = f"[yellow]{age_min:.0f} min ago[/yellow]"
            else:
                beat_line = f"[red]{age_min / 60:.1f} h ago — orchestrator may be down[/red]"
        except Exception:
            beat_line = last_beat

    header = Table.grid(padding=(0, 2))
    header.add_column()
    header.add_column()
    header.add_row("[bold]agent:[/bold]",
                   "[red]PAUSED[/red]" if paused else "[green]active[/green]")
    header.add_row("[bold]heartbeat:[/bold]", beat_line)
    until = db.state_get("backoff_until")
    if until:
        reason = db.state_get("backoff_reason") or ""
        header.add_row("[bold]backoff:[/bold]",
                       f"[yellow]until {until}[/yellow] [dim]({reason})[/dim]")
    header.add_row("[bold]niche:[/bold]", cfg.niche)
    header.add_row("[bold]providers:[/bold]", ", ".join(providers_configured()) or "[red]none[/red]")
    console.print(header)
    console.print()

    # ─── Queue ─────────────────────────────────────────
    by_status = {}
    for row in db.content_list(status=None, limit=500):
        by_status[row["status"]] = by_status.get(row["status"], 0) + 1

    table = Table(title="content queue", show_lines=False)
    table.add_column("status", style="cyan")
    table.add_column("count", justify="right")
    for k, v in sorted(by_status.items()):
        table.add_row(k, str(v))
    console.print(table)

    # ─── Next scheduled posts ─────────────────────────
    approved = db.content_list(status="approved", limit=50)
    upcoming = sorted(
        [r for r in approved if r.get("scheduled_for")],
        key=lambda r: r["scheduled_for"] or "",
    )[:3]
    if upcoming:
        nxt = Table(title="next 3 scheduled posts")
        nxt.add_column("scheduled_for", style="cyan")
        nxt.add_column("format")
        nxt.add_column("caption preview")
        for r in upcoming:
            preview = (r.get("caption") or "").split("\n")[0][:50]
            nxt.add_row(r["scheduled_for"] or "?", r["format"], preview)
        console.print(nxt)
    elif by_status.get("approved", 0):
        console.print(
            "[dim]Approved items present but not yet scheduled. "
            "They'll slot in on the next orchestrator scheduling tick.[/dim]"
        )

    # ─── Recent actions (last 35 min) ─────────────────
    try:
        recent = dict(
            db.get_conn().execute(
                "SELECT action, COUNT(*) c FROM action_log "
                "WHERE at >= datetime('now', '-35 minutes') "
                "GROUP BY action ORDER BY c DESC LIMIT 8"
            ).fetchall()
        )
    except Exception:
        recent = {}
    if recent:
        r_tbl = Table(title="actions in the last 35 min")
        r_tbl.add_column("action", style="cyan")
        r_tbl.add_column("count", justify="right")
        for k, v in recent.items():
            r_tbl.add_row(str(k), str(v))
        console.print(r_tbl)

    # ─── Health ─────────────────────────────────────────
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

    if paused:
        console.print(
            "\n[dim]Agent is paused — run [bold cyan]ig-agent resume[/bold cyan] to continue.[/dim]"
        )


# ───────── pause / resume ─────────
@app.command("pause")
def pause_cmd() -> None:
    """Halt all IG writes + content generation. Brain + monitoring keep running.

    Used for: weekend pauses, debugging, temporary stop without killing the
    orchestrator process. Use ``ig-agent resume`` to re-enable."""
    load_env()
    db.init_db()
    db.state_set("paused", "1")
    console.print("[yellow]⏸[/yellow]  Agent paused — all IG writes + generation halted.")
    console.print(
        "  [dim]Brain modules (trend miner, RAG index, etc.) keep running "
        "so the queue is ready when you resume.[/dim]\n"
        "  Resume: [bold cyan]ig-agent resume[/bold cyan]"
    )


@app.command("resume")
def resume_cmd() -> None:
    """Clear the pause state — orchestrator resumes on the next tick."""
    load_env()
    db.init_db()
    was_paused = (db.state_get("paused") or "").lower() in ("1", "true", "yes")
    db.state_set("paused", "0")
    if was_paused:
        console.print("[green]▶[/green]  Agent resumed — IG writes + generation re-enabled.")
    else:
        console.print("[dim]Already running — no change.[/dim]")


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
    from instagram_ai_agent.content import dedup
    from instagram_ai_agent.content import hashtags as hashtags_mod
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

    # brain.db integrity
    try:
        db_ok, detail = db.integrity_check()
        if db_ok:
            ok("brain.db integrity: ok")
        else:
            fail("brain.db integrity failed",
                 f"{detail}. Back up data/brain.db and "
                 "run `rm data/brain.db*` to start fresh (loses queue + post history).")
    except Exception as e:
        warn("Couldn't run integrity check", str(e))

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
