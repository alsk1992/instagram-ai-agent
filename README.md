# instagram-ai-agent

> **Autonomous AI agent for Instagram.** Generates reels, carousels, memes, quote cards and engages with your audience on autopilot. Single-process, free-tier, commercial-safe.

`instagram-ai-agent` is a niche-targeted content pipeline for Instagram — the AI does the mining, ideating, generating, captioning, scheduling, posting, replying, and self-health-monitoring. Describe your niche once in `niche.yaml`, give it an Instagram account, and it runs.

Built on a free-tier stack: `instagrapi` transport, `OpenRouter`+`Groq`+`Gemini`+`Cerebras` LLM router, `Pollinations`/`Pexels`/`Pixabay` media, `Playwright` HTML-rendered visuals, `edge-tts`+`WhisperX`+`ffmpeg` for reels, optional `ComfyUI`+`FLUX`+`LoRA`+`ControlNet` for brand-consistent generation.

## 🚀 One-line install

```bash
curl -fsSL https://raw.githubusercontent.com/alsk1992/instagram-ai-agent/main/install.sh | bash
```

Checks Python 3.11+/ffmpeg, clones the repo, creates `.venv`, installs deps + Playwright chromium + fonts. Then:

```bash
cd instagram-ai-agent && source .venv/bin/activate
ig-agent init       # interactive wizard → writes niche.yaml + .env
ig-agent run        # start the full agent
```

That's it. First post lands in your queue within ~10 minutes of `ig-agent run`.

**Prefer Make?** `make install && make init && make run`.

## What it does

1. **Mines the niche** — scrapes top posts of your hashtags + competitors via Instaloader, pushes trend signals to the brain.
2. **Generates content** — picks a format (meme / quote card / carousel / stock reel / AI photo) from your configured mix, produces the asset, writes a niche-voice caption, runs an LLM critic, dedups via perceptual hash.
3. **Schedules + posts** — approved items get assigned to your configured "best hours UTC", posted one at a time with exponential backoff on errors and a 24h cooldown on challenge/ban signals.
4. **Engages** — drains an engagement queue of likes/follows/comments/story-views, capped by daily budget.
5. **Watches** — polls a target account, pushes new posts to the context feed so the next generation can react.
6. **Probes health** — tracks follower/ER drift + shadow-ban detection.

All in one `python -m src.orchestrator` process.

## Stack (every piece free)

- **LLM:** OpenRouter (29 free models, one key), Groq (fastest), Gemini Flash (highest daily quota), Cerebras (1M tok/day). First provider to respond wins; others are automatic fallbacks.
- **Images:** Pillow (memes), Playwright → HTML/CSS (quote cards, carousel slides), Pollinations Flux (photo generation).
- **Reels:** Pexels + Pixabay (stock footage) → ffmpeg (concat + 9:16 crop + LUT + caption burn-in) → edge-tts (voiceover) → faster-whisper (word-level timestamps for captions).
- **Transport:** instagrapi with persistent device fingerprint, session, proxy, IMAP+TOTP challenge resolver.
- **Scraping:** Instaloader for public profiles/hashtags.
- **State:** SQLite brain.db (WAL mode), no external services.
- **Scheduling:** APScheduler inside the async event loop.
- **Alerts:** Telegram webhook.

## Setup (the long version)

If you'd rather do it manually instead of using the one-liner installer above:

```bash
git clone https://github.com/alsk1992/instagram-ai-agent.git
cd instagram-ai-agent
./scripts/bootstrap.sh          # venv, pip install, playwright, fonts, default meme bg
source .venv/bin/activate
ig-agent init                   # interactive wizard → writes niche.yaml + .env
ig-agent login                  # verify IG credentials + persist session
ig-agent generate --count 3     # make 3 pieces of content (optional preview)
ig-agent review                 # approve/reject queue items
ig-agent run                    # start the full orchestrator
```

Or use the Makefile shortcuts:

```bash
make install    # runs install.sh
make init       # the wizard
make login      # verify IG
make run        # start orchestrator
make test       # run the full pytest suite
make status     # queue depth + health snapshot
make dashboard  # local read-only web dashboard on :8080
```

### Minimum env you actually need

You can run with just these two:

```
OPENROUTER_API_KEY=...          # pick up at https://openrouter.ai/keys
IG_USERNAME=...
IG_PASSWORD=...
```

Add `PEXELS_API_KEY` / `PIXABAY_API_KEY` when you turn on reel_stock. Add `IMAP_HOST` / `IMAP_USER` / `IMAP_PASS` to auto-resolve email challenges. Add `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` for alerts.

## The `niche.yaml`

This is the spine. Every generator reads from it. Example:

```yaml
niche: "home calisthenics for dads 35+"
sub_topics: [bodyweight, mobility, recovery, progressions]
target_audience: "office workers rebuilding fitness at home"
commercial: true

voice:
  tone: [direct, no-nonsense, dry humour]
  forbidden: ["hustle", "grind culture"]
  persona: "ex-office worker, 40, rebuilt body at home — talks like your mate at the pub."
  cta_styles: ["save for later", "tag a mate", "follow for more"]

aesthetic:
  palette: ["#0a0a0a", "#f5f5f0", "#c9a961"]
  heading_font: "Archivo Black"
  body_font: "Inter"
  watermark: "@dadpilled"
  lut: null

hashtags:
  core: [calisthenics, homeworkout, bodyweighttraining]
  growth: [fitnessmotivation, fittips, fathersover40]
  long_tail: [pullupprogression, mobilitytraining, dadbodtransformation]
  per_post: 15

formats:
  meme: 0.30
  quote_card: 0.15
  carousel: 0.25
  reel_stock: 0.20
  reel_ai: 0.0
  photo: 0.10

schedule:
  posts_per_day: 1
  stories_per_day: 3
  best_hours_utc: [14, 18, 21]

safety:
  require_review: true
  critic_min_score: 0.65
  dedup_hamming_threshold: 8

competitors: [hybrid.athlete.x, calimove]
watch_target: null
```

## Commands

```
ig-agent init                   # wizard
ig-agent login                  # verify + persist session
ig-agent generate [--format F] [--count N]
ig-agent review                 # approve/reject pending
ig-agent post                   # post the next approved item once
ig-agent drain --limit 3        # burst-post up to N approved
ig-agent status                 # queue depth, health, backoff
ig-agent add-content FORMAT PATH [PATH...] --caption "..." [--approve]
ig-agent show-niche             # dump the full config
ig-agent run                    # start the full orchestrator
```

## Directory layout

```
ig-agent/
├── niche.yaml                           # the spine
├── .env                                 # secrets
├── pyproject.toml
├── start.sh / watchdog.sh
├── scripts/
│   ├── bootstrap.sh
│   └── gen_default_assets.py
├── src/
│   ├── cli.py                           # typer entry
│   ├── orchestrator.py                  # APScheduler wiring
│   ├── core/
│   │   ├── config.py / db.py / llm.py / alerts.py / budget.py / logging_setup.py
│   ├── plugins/
│   │   ├── ig.py                        # instagrapi wrapper
│   │   ├── device.py                    # persistent fingerprint
│   │   └── challenge.py                 # IMAP + TOTP resolver
│   ├── content/
│   │   ├── pipeline.py                  # format-pick → gen → critic → dedup → enqueue
│   │   ├── captions.py / hashtags.py / critic.py / dedup.py / style.py
│   │   └── generators/
│   │       ├── meme.py                  # Pillow + template JSON
│   │       ├── quote_card.py            # Playwright HTML
│   │       ├── carousel.py              # Playwright HTML, multi-slide
│   │       ├── photo.py                 # Pollinations Flux
│   │       └── reel_stock.py            # Pexels + ffmpeg + edge-tts + whisper
│   ├── brain/
│   │   ├── scraper.py                   # Instaloader
│   │   ├── competitor_intel.py
│   │   ├── trend_miner.py
│   │   └── watcher.py
│   └── workers/
│       ├── poster.py / engager.py / health.py
└── data/
    ├── brain.db                         # SQLite (WAL)
    ├── device.json                      # never rotate
    ├── sessions/{username}.json
    ├── media/{staged,posted}/
    ├── fonts/
    └── luts/
```

## Safety defaults (intentionally conservative)

- `require_review = true` — every item waits in `pending_review` until you `ig-agent review`. Flip to false once the critic is calibrated to your voice.
- `posts_per_day = 1` — start slow. New accounts should warm up for 14 days before posting at all.
- Daily caps: 150 likes, 25 follows/unfollows, 15 comments, 0 DMs, 400 story views.
- `backoff_until` state entry blocks **all** IG calls when hit; 24h on challenges, 1–3h on rate-limits.
- PHash dedup against the last 60 posts at Hamming threshold 8.
- Sticky proxy + device fingerprint (never rotate mid-life).

## Free-tier licensing — what the agent deliberately avoids

Because `niche.yaml` is set to `commercial: true` by default, generators route around non-commercial tooling:

- Uses **Flux.1 schnell** via Pollinations (Apache), not Flux-dev (non-commercial).
- Uses **edge-tts** + optional CosyVoice 2 (Apache), not XTTS-v2 / F5 / ElevenLabs free tier.
- Uses **Pexels / Pixabay** APIs (commercial OK, no attribution), not Videvo (mixed).
- Uses **BiRefNet via rembg** (MIT), not BRIA-RMBG-2.0 (non-commercial).

## What's not here yet

Things the single-agent scope doesn't try to solve — they belong in a later swarm phase:

- Multi-account orchestration (account pool, per-account proxies, warmup subsystem).
- Dashboard UI (the `status` command is a CLI stand-in).
- Paid/AI video generation (`reel_ai`) — enable by wiring a `fal.ai` / `Runway` / local `Wan2.2` caller in `content/generators/`.
- DM funnel / engagement CRM.
- Cloudflare R2 upload of finished media (currently lives on local disk).

Each is a bounded addition on top of this base.
