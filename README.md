<div align="center">

# instagram-ai-agent

**An AI that runs your Instagram page for you.**

Tell it what your page is about. It writes the posts, designs the images, writes the captions, posts on schedule, replies to comments, and follows the best practices that keep accounts from getting banned. You review posts before they go live.

Free to run on free-tier AI models. Runs on your own laptop or a $5/month server. No subscription.



<a href="https://github.com/alsk1992/instagram-ai-agent/releases/download/walkthrough-v1/walkthrough.mp4">
  <img src="docs/media/walkthrough-poster.jpg" alt="Click to play the 43-second walkthrough (MP4 with audio)" width="720">
</a>

_43-second walkthrough — real `ig-agent doctor`, `status`, `warmup-status` runs + real dashboard navigation, narrated by [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (Apache-2.0, CPU)._

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-748%20passing-brightgreen)](tests/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Status](https://img.shields.io/badge/status-active-brightgreen)](https://github.com/alsk1992/instagram-ai-agent)

[Quickstart](#-quickstart) · [Features](#-features) · [Architecture](#%EF%B8%8F-architecture) · [Safety](#%EF%B8%8F-safety--anti-detection) · [Configuration](#-configuration) · [FAQ](#-faq)

</div>

---

## 🚀 Quickstart

**Two commands. ~2 minutes to your first generated post.**

```bash
pipx install git+https://github.com/alsk1992/instagram-ai-agent.git
ig-agent setup
```

That's the whole thing. `setup` auto-installs Playwright chromium, picks a niche from 8 starter presets (fitness / food / travel / finance / mindfulness / productivity / fashion / pets), opens your browser to grab a free OpenRouter API key, validates it live, seeds the idea-bank with 90 proven hook archetypes, and drops you at:

```bash
ig-agent generate -n 3    # make your first 3 posts (~2 min)
ig-agent dashboard        # open http://127.0.0.1:8080 — approve posts visually
ig-agent drain            # post the approved ones right now (once IG is connected)
```

Prefer the terminal? `ig-agent review` walks you through each pending post. Something off? `ig-agent doctor` diagnoses your setup in 3 seconds.

`ig-agent login` connects your Instagram account when you're ready to actually post.

<details>
<summary><strong>Prerequisites</strong> — what you need on the machine before `pipx install`</summary>

- **Python 3.11+** — macOS: `brew install python@3.12` · Ubuntu: `sudo apt install python3.12` · Windows: [python.org](https://www.python.org/downloads/)
- **ffmpeg** — macOS: `brew install ffmpeg` · Ubuntu: `sudo apt install ffmpeg` · Windows: `winget install Gyan.FFmpeg`
- **pipx** — `python -m pip install --user pipx && pipx ensurepath`

`ig-agent setup` will check all three and print the exact install command for anything missing.

</details>

<details>
<summary><strong>Alternative installs</strong> — source clone, bash one-liner, Windows PowerShell, Makefile</summary>

One-line POSIX installer (clones the repo + installs everything):
```bash
curl -fsSL https://raw.githubusercontent.com/alsk1992/instagram-ai-agent/main/install.sh | bash
```

Windows PowerShell:
```powershell
irm https://raw.githubusercontent.com/alsk1992/instagram-ai-agent/main/install.ps1 | iex
```

From source: `git clone … && cd instagram-ai-agent && make install && make setup`

For deep configuration (story mix, hashtag pools, anti-detection toggles), use `ig-agent init` — the 40-question advanced wizard.

</details>

<details>
<summary><strong>First-time Instagram setup</strong> — how it keeps your account safe</summary>

- **Brand-new account?** Leave the warmup setting alone. The agent waits ~7 days before posting, then slowly increases activity over 14 days — this is the single biggest thing that prevents Instagram from banning automated accounts. Use this time to run `generate` + `review` so you have content queued and ready.
- **Account that already posts regularly?** Set `IG_SKIP_WARMUP=1` in `.env` — it'll post immediately.
- **Posting from a VPS or a new IP?** Instagram will often send an email-code challenge the first time. You can either let the agent read your Gmail (set `IMAP_HOST/USER/PASS`), paste a browser cookie to skip the challenge, or run `ig-agent login` in a terminal and type the code manually. Details in the [Safety section](#%EF%B8%8F-safety--anti-detection).

</details>

---

## ✨ Features

<table>
<tr>
<td width="50%" valign="top">

### What it posts
- **Memes** — image with bold overlay text, the classic fast-scroll stopper
- **Quote cards** — designed single-slide cards with a memorable line
- **Carousels** — multi-slide posts that tell a story (hook → point → point → CTA)
- **Reels from stock** — short videos cut from free stock footage, voiced over
- **Reels from AI** — fully AI-generated short videos
- **Photos + AI portraits** — image posts, optionally with a consistent "face" across every post
- **Stories** — quote, announcement, photo, video (the ephemeral 24h ones)
- **Narrative carousels** — same character across all slides
- **Hot takes / contrarian posts** — contrarian angle for engagement spikes
- **Auto-beat-synced cuts** — reels cut on the music's beat
- **Karaoke subtitles** — word-level captions appear on every reel

</td>
<td width="50%" valign="top">

### Intelligence layer
- **LLM router**: OpenRouter + Groq + Gemini + Cerebras with automatic fallback
- **90-archetype idea bank**: every post rides a proven hook formula, not an LLM guess
- **Niche RAG**: drops PDFs/markdown into `data/knowledge/`, agent cites it
- **Trend miner**: scrapes competitor + hashtag feeds, feeds the generator
- **Reddit question harvester**: turns "what the community is asking" into content
- **Event calendar**: Nager.Date holidays + user dates → seasonal posts
- **Critic 3.0**: 8-dimension rubric with save_potential weighted 3× (matches Instagram's 2026 algorithm weighting — saves drive reach, not likes)
- **Genius-tier pre-stages**: every post passes through angle brainstorm (15 candidates → winning hook), slide-1 hook optimiser (8 scroll-stop variants), specificity rewrite (strips "pro tips / game-changer" filler), story-arc enforcer (prescriptive → lived-experience), voice fingerprint (shipped-caption few-shot), comment-bait CTA engineer (binary-pick / number-drop / emoji-react patterns)

</td>
</tr>
<tr>
<td width="50%" valign="top">

### Anti-detection layer ([details](#%EF%B8%8F-safety--anti-detection))
- **Pre-post scroll**: mimics a human opening the app before posting
- **Post cooldown**: 30–90min silence after each post before writes
- **Typing delays**: length-proportional sleeps on comment/DM send
- **First-comment hashtags** (opt-in): hashtags out of caption → self-reply
- **Caption entropy guard**: near-duplicate detection vs your last 10 posts
- **Aspect-ratio pre-flight**: rejects off-spec media before upload
- **Client rotation**: recycles the TCP session every 2–4h
- **Session refresh**: weekly forced re-login prevents silent decay
- **Persistent device fingerprint**: pinned once, never rotated
- **10-cookie seed**: full browser-cookie paste to skip first-login challenge

</td>
<td width="50%" valign="top">

### Engagement & ops
- **Reply-to-own-comments**: LLM triages + drafts on-brand replies
- **Follow-back**: reciprocate when niche-aligned accounts follow you
- **Story viewer**: views back the stories of recent engagers
- **DM pipeline**: seed → curate → send with per-day caps
- **Shadowban probe**: twice-daily reach drift + follower gain tracking
- **Budget enforcement**: per-action daily caps that ramp over a 14-day warmup
- **Auto-scheduling**: best-hours-UTC windows with randomised jitter
- **Human-review gate**: toggle on/off; approve/reject every piece via CLI

</td>
</tr>
<tr>
<td width="50%" valign="top">

### Brand consistency (optional stack)
- **FluxGym brand LoRA CLI**: prep → train externally → import → every image carries your brand identity
- **ControlNet pose/depth/canny**: point at a reference image, every generation respects its composition
- **Stable Audio Open Small**: commercial-safe generative music beds
- **Real-ESRGAN + GFPGAN finish pass**: 2x upscale + face restoration

</td>
<td width="50%" valign="top">

### Dev experience
- **One-line installer** (POSIX, idempotent)
- **Typer CLI**: `ig-agent` with `init / login / run / review / generate / status / dashboard / lora / controlnet`
- **Makefile shortcuts**: `make install / init / run / test / dashboard`
- **Local read-only dashboard**: `make dashboard` → `http://127.0.0.1:8080`
- **748 tests**, 0 import failures across 90+ modules
- **Commercial-licence gates baked in**: FLUX.1-dev, OpenPose, StoryDiffusion code all blocked under `commercial=True`

</td>
</tr>
</table>

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        ig-agent (one process)                        │
│                                                                      │
│   BRAIN                      CONTENT                   WORKERS       │
│   ┌──────────────┐          ┌────────────────┐        ┌────────────┐ │
│   │ trend_miner  │─────▶    │ format_picker  │──┐     │ poster     │ │
│   │ competitor   │   ctx    │ idea_bank      │  │     │ engager    │ │
│   │ watcher      │──▶ feed  │ generator x10  │──┤──▶──│ comment_r. │ │
│   │ reddit       │          │ critic 2.0     │  │     │ follow_b.  │ │
│   │ events       │          │ dedup (phash)  │  │     │ dm_worker  │ │
│   │ rag          │          │ caption gen    │──┘     │ story_view │ │
│   └──────────────┘          └────────────────┘        │ health     │ │
│                                     │                 └────────────┘ │
│                                     ▼                        ▲       │
│                              ┌────────────────┐              │       │
│                              │  content_queue │──────────────┘       │
│                              │  engagement_q  │                      │
│                              │  (SQLite WAL)  │                      │
│                              └────────────────┘                      │
│                                     │                                │
└─────────────────────────────────────┼────────────────────────────────┘
                                      ▼
                              ┌────────────────┐
                              │   instagrapi   │
                              │   (transport)  │
                              └────────────────┘
                                      │
                                 Instagram
```

Everything is scheduled by **APScheduler** inside a single async event loop. State lives in **SQLite WAL mode** — no external services, no Redis, no separate workers. Scales vertically: one account per process, one process per VPS.

---

## 🛡️ Safety & Anti-Detection

Instagram's 2026 ML detector flags bot-script patterns on top of plain request inspection. The agent ships with an 8-feature behavioural layer, all on by default (toggles under `human_mimic` in niche.yaml):

| Feature | What it does | Gap it closes |
|---|---|---|
| `pre_post_scroll` | Opens the feed + marks 3–5 posts as seen before uploading | Cold-post bot-script fingerprint |
| `post_cooldown` | 30–90min mandatory silence on likes/follows/unfollows after a post | "Posted + engaged within seconds" pattern |
| `comment_reply_delay` | 5–60min random stagger between replies in a batch | Burst-reply signal |
| `first_comment_hashtags` (opt-in) | Moves hashtag block out of caption into a self-reply | 2026 ML hashtag-heavy caption downrank |
| `typing_delays` | Length-proportional sleep before comment/DM submit | Sub-second submission timing |
| `caption_entropy_check` | Rejects captions ≥85% similar to your last 10 | LLM-template drift that image-phash misses |
| `aspect_ratio_check` | Pre-flight checks for 4:5 / 1:1 / 9:16 before upload | Silent server-side re-compression + downrank |
| `rotate_client` | Recycles the TCP/HTTP-2 session every 2–4h | Long-lived connection botnet signal |

Plus the authentication layer:

- **Pin device fingerprint forever** — `data/device.json` is generated once and never rotated.
- **Paste browser cookies into `.env`** to skip first-login challenges entirely. Supports all 10 IG web cookies (`sessionid`, `ds_user_id`, `csrftoken`, `mid`, `ig_did`, `datr`, `rur`, `shbid`, `shbts`, `ig_nrcb`). Full set triggers `cl.set_settings()` path with zero `/login` call.
- **Geographic coherence** — set `IG_COUNTRY_CODE` / `IG_TIMEZONE_OFFSET` / `IG_LOCALE` to match the account's origin.
- **Weekly session refresh** — `IG_SESSION_REFRESH_DAYS=7` forces a fresh login to prevent silent server-side TTL decay.
- **14-day warmup ramp** — action budgets auto-scale from ~10% on day 1 to 100% by day 14.

### One account = one VPS = one IP

Don't share any of: the process, the working directory, the IP, the device fingerprint. Residential proxies (Decodo / NetNut / IPBurger / Smartproxy) with **sticky 24–48h sessions over IPv4** are the 2026 consensus.

### 2026 gotchas

- **Same Reel across multiple accounts** triggers network-level flagging within ~48h. Every account needs unique content.
- **Bio links on day 1** = instant shadowban. Wait day 7+.
- **Meta Graph API** only does Reels on Business/Creator accounts (2–12 week approval). For personal accounts, instagrapi-style private API remains the only path.

Full research notes + sources in the [safety deep-dive](#%EF%B8%8F-safety--anti-detection).

---

## 🔧 Configuration

Everything lives in two files: `niche.yaml` (the spine) and `.env` (credentials).

The setup wizard picks a starter preset so you don't invent defaults:

<p align="center"><img src="docs/media/presets.svg" alt="starter niche presets" width="900"></p>
<p align="center"><img src="docs/media/wizard-preview.svg" alt="init wizard" width="900"></p>

### `niche.yaml` — the spine

```yaml
niche: "home calisthenics for dads 35+"
sub_topics: [pullups, mobility, recovery]
target_audience: "office workers rebuilding fitness at home"
commercial: true                       # toggles licence gates

voice:
  tone: [direct, dry humour, no-nonsense]
  forbidden: ["hustle", "grind culture"]
  persona: "ex-office worker, 40, rebuilt body at home"
  cta_styles: [save for later, tag a mate, follow for more]

aesthetic:
  palette: ["#0a0a0a", "#f5f5f0", "#c9a961"]
  heading_font: "Archivo Black"
  body_font: "Inter"
  watermark: "@dadpilled"

hashtags:
  core: [calisthenics, homeworkout, bodyweighttraining]
  growth: [fittips, fathersover40]
  per_post: 15

formats:                               # distribution weights
  meme: 0.30
  quote_card: 0.15
  carousel: 0.25
  reel_stock: 0.20
  reel_ai: 0.05
  photo: 0.05

schedule:
  posts_per_day: 1
  stories_per_day: 3
  best_hours_utc: [14, 18, 21]

safety:
  require_review: true                 # toggle to false for autonomous mode
  critic_min_score: 0.65
```

Full schema: [`src/core/config.py::NicheConfig`](src/core/config.py). The wizard (`ig-agent init`) walks you through every field.

### Minimum `.env`

```
OPENROUTER_API_KEY=sk-or-v1-...        # at least one LLM provider
IG_USERNAME=your_handle
IG_PASSWORD=your_password
```

Optional but recommended for production:

```
IG_PROXY=http://user:pass@proxy:port   # sticky residential
IG_SESSIONID=<paste from browser>      # skips first-login challenge
IG_DATR=<paste from browser>           # persistent device marker
IMAP_HOST=imap.gmail.com               # auto-resolve email challenges
IMAP_USER=you@gmail.com
IMAP_PASS=app-password
PEXELS_API_KEY=...                     # for reel_stock format
```

See [`.env.example`](.env.example) for the full list with inline explanations.

---

## 📚 CLI reference

```
# Onboarding
ig-agent setup                         # one-command quick setup (4 questions, ~2 min)
ig-agent setup --full                  # extended (~15 questions: voice, palette, hashtags, formats, schedule)
ig-agent setup --minimal               # take preset defaults for everything except niche name
ig-agent setup --with-login            # also prompt for IG credentials inline
ig-agent setup --force                 # overwrite existing niche.yaml without asking
ig-agent init                          # legacy 40-question deep wizard

# Daily ops
ig-agent login                         # verify IG credentials, handle challenges
ig-agent run                           # start the full orchestrator (long-running)
ig-agent pause                         # halt all IG writes + generation (brain keeps running)
ig-agent resume                        # clear pause state
ig-agent status                        # agent pulse (heartbeat, backoff, queue, next posts, recent actions)
ig-agent doctor                        # diagnostic self-check with DB integrity

# Content
ig-agent generate [-n N]               # generate N posts into the queue (with live progress)
ig-agent generate --format carousel    # force a specific format
ig-agent generate --contrarian         # force hot-take mode for this batch
ig-agent review                        # terminal walk-through: approve/reject pending items
ig-agent drain [--limit N]             # post up to N approved items immediately
ig-agent dashboard                     # local web dashboard on :8080 — visual approve/reject at /review
ig-agent add-content <format> <media>  # drop external media straight into the queue

# Introspection
ig-agent warmup-status                 # current day + scaled caps
ig-agent show-niche                    # dump resolved niche.yaml
ig-agent events [--push]               # upcoming holidays + user dates
ig-agent reddit-questions [--push]     # niche-subreddit question harvester
ig-agent index-knowledge [--clear]     # rebuild RAG index from data/knowledge/
ig-agent seed-idea-bank [--fetch ...]  # reload archetype library
ig-agent hashtag-review                # approve discovered hashtags

# Brand LoRA / ControlNet (optional, advanced)
ig-agent lora prepare <dir> --name X --trigger W
ig-agent lora import <file.safetensors> --name X --trigger W
ig-agent lora list / activate / deactivate / remove

ig-agent controlnet set <ref.jpg> --mode pose|depth|canny
ig-agent controlnet import-model <file.safetensors>
ig-agent controlnet show / clear
```

---

## 🧪 Testing

```
make test                              # pytest full suite (~12s)
make lint                              # ruff
```

Every feature has tests. The suite is the contract.

---

## 🆘 Troubleshooting — "it's not working"

If you hit a wall, run:

```bash
ig-agent doctor
```

<p align="center"><img src="docs/media/doctor.svg" alt="ig-agent doctor output" width="900"></p>

It walks every dependency + config step with clear pass/warn/fail rows and a concrete action for anything broken. Cover 80% of "why isn't this working" questions in 3 seconds.

### Common first-run issues

| Symptom | Usually means | Fix |
|---|---|---|
| `ig-agent drain` returns "drained 0/1" | Warmup is blocking posts for 7 days on a fresh account | Set `IG_SKIP_WARMUP=1` in `.env` if your account already has post history; otherwise wait until day 8. |
| `Login failed: bad_password` | Actually, IG usually returns this on first login from a new IP even with correct credentials | Add a residential proxy (`IG_PROXY=http://...`); or paste `IG_SESSIONID` cookie from browser — see [Safety](#%EF%B8%8F-safety--anti-detection) |
| `No module named 'playwright'` | install.sh was interrupted or ran outside the venv | `source .venv/bin/activate && pip install -e .` |
| `ffmpeg: not found` | System package missing | macOS: `brew install ffmpeg` · Ubuntu: `sudo apt install ffmpeg` · Arch: `pacman -S ffmpeg` |
| Terminal looks frozen during `generate` | LLM calls take 10–30s each | Watch the log — `tail -f logs/orchestrator.log` — genuinely working, just slow |
| Dashboard shows "connection refused" | It binds to `127.0.0.1:8080` on the VPS | SSH-tunnel: `ssh -L 8080:127.0.0.1:8080 you@vps` then open `http://127.0.0.1:8080` locally |
| Asks for "email-code" challenge at login | IG security on new IPs | Either set `IMAP_HOST/USER/PASS` so the agent auto-reads the code, OR run `ig-agent login` in an interactive terminal and paste it manually |

### Starting from zero

Never used a terminal / Python / GitHub? Honest advice: **pair with a technical friend for the first 30 minutes.** The install is one line but the concepts (API keys, proxies, sessions, cookies) are a lot to absorb solo. Once setup is done the agent runs itself — you just need to get it over the initial hump.

## ❓ FAQ

**Q: How much does this cost to run?**
A: $0 to a few dollars/month. OpenRouter has 29 free models. Pexels/Pixabay/Pollinations/Freesound are free. A $5/mo VPS (Hetzner, Vultr, OVH, Contabo) is plenty. Residential proxy is optional but recommended — $10–30/mo if you take one. Image/reel generation costs nothing unless you opt into ComfyUI with paid GPU time.

**Q: Will this get my account banned?**
A: Any Instagram automation carries risk. The agent's defaults are conservative: 1 post/day, 14-day warmup, 30–90min post-cooldown, human-review gate. The anti-detection layer closes the biggest 2026 gaps but is not magic. **Use a fresh account you don't mind losing for the first few weeks** while you dial in the cadence.

**Q: Does it work for multi-account?**
A: One agent per account per VPS per IP. The architecture is single-tenant by design — multi-account sharing IPs or fingerprints is the #1 way to get the whole pool banned.

**Q: Can I use it commercially (monetised page)?**
A: Yes. `commercial: true` in `niche.yaml` activates licence gates that refuse non-commercial-only components (FLUX.1-dev, OpenPose, CodeFormer, pyiqa, StoryDiffusion code). Every default component is under a commercial-safe licence (MIT / Apache-2.0 / BSD / CC0 / CreativeML-Open-RAIL++).

**Q: Does it support Threads / TikTok / X?**
A: No. Instagram only. (The architecture is portable — fork it for any platform with an unofficial Python client.)

**Q: What LLM models are used?**
A: Whatever your LLM router has keys for. The default chain is OpenRouter (free tier) → Groq → Gemini Flash → Cerebras. Each route (captions, critic, scripts, vision) picks the best-suited free-tier model and falls back on failure.

**Q: How do I add new content formats?**
A: Write a generator under `src/content/generators/`, register it in `pipeline._dispatch`, add a weight field to `FormatMix`. See `story_carousel.py` for a recent example.

---

## 🤝 Contributing

Contributions welcome. The repo is small enough that a casual read of `src/orchestrator.py` + `src/content/pipeline.py` orients you in ~30 minutes.

- **Bug reports**: open an issue with the log tail + your niche.yaml (redact secrets). Reproduce steps help a lot.
- **PRs**: run `make test && make lint` first. Keep new features behind a config flag — user configs should never break on upgrade.
- **Discussion**: for design-level questions (new format, new platform, licence concerns), open a GitHub Discussion rather than diving into a PR.

Security issues: see [SECURITY.md](SECURITY.md).

---

## 📝 License

[MIT](LICENSE). Do what you want, just don't blame us when you get rate-limited.

Third-party components are listed in [NOTICE.md](NOTICE.md) with their individual licences. The agent's `commercial=true` mode refuses any non-commercial-only dependency at load time.

---

<div align="center">

Built with Claude Code 4.7 (1M context) · [GitHub](https://github.com/alsk1992/instagram-ai-agent)

</div>
