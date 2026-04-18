# Security Policy

## Reporting a vulnerability

If you find a security issue in `instagram-ai-agent` itself (code execution, credential leakage, path traversal, auth bypass on the local dashboard, etc.):

1. **Do not open a public issue.**
2. Email the details to the repository owner via GitHub (profile → direct message, or open a private Security Advisory: https://github.com/alsk1992/instagram-ai-agent/security/advisories/new).
3. Include: affected commit/version, reproduction steps, blast radius, any PoC you have.

You'll get acknowledgment within 5 business days. Confirmed issues get a coordinated disclosure timeline.

## Scope

**In scope:**
- Code execution via crafted `niche.yaml` / `.env` / template files
- Path traversal via user-supplied filenames (LoRA names, ControlNet model names, reference images)
- Credential leakage in logs, state files, or error messages
- Unauthenticated access to the local dashboard when `DASH_USER`/`DASH_PASS` are set
- Commercial-licence-gate bypass paths (shipping NC-licensed outputs under `commercial=true`)
- Contrarian-safety blocklist bypass (posting content the regex layer should have refused)

**Out of scope:**
- Instagram banning / shadowbanning your account (this is a behavioural risk, documented in [README.md](README.md))
- Third-party service outages (OpenRouter, Pollinations, Pexels)
- Issues in upstream dependencies (report those upstream)
- "Anyone with shell access can read `.env`" — that's by design, it's a local-only config file

## Credential hygiene

`.env` is in `.gitignore`. Do not commit it. If you ever commit one by accident:

1. Rotate every key in it immediately (IG password, LLM API keys, IMAP password, proxy credentials).
2. Log out of Instagram from every device (forces `sessionid` rotation).
3. Force-push a history rewrite to drop the file from git history (e.g. `git filter-repo --path .env --invert-paths`).

## Operating-mode safety

The agent ships with three independent safety layers activated by default:

1. **Commercial-licence gates** — block non-commercial code paths under `commercial=true` at config load.
2. **Contrarian-safety blocklist** (`src/content/contrarian_safety.py`) — 16-pattern regex against medical misinformation, conspiracy tropes, self-harm content, and group-disparagement framings. Content matching any pattern is refused pre-enqueue.
3. **Critic 2.0** — LLM-rubric score on every draft. Items below `safety.critic_min_score` (default 0.65) go to `regen`; below 0.40 to `reject`.

If you're operating without `require_review: true` (full autonomous mode), treat the blocklist + critic as your only backstop. Spot-check outputs for the first two weeks.
