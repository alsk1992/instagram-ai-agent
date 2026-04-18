# Contributing

Thanks for considering a contribution. The repo is small and the architecture is single-process — reading `src/orchestrator.py` + `src/content/pipeline.py` gets you oriented in ~30 minutes.

## Quickstart (dev)

```bash
git clone https://github.com/alsk1992/instagram-ai-agent.git
cd instagram-ai-agent
./install.sh
source .venv/bin/activate
make test          # 671 tests, ~12s
make lint          # ruff
```

Run the full suite before opening a PR. If a test depends on a real API key, skip-mark it — the CI environment has none.

## Architecture in one picture

```
NicheConfig (src/core/config.py)
     │
     ▼
Orchestrator (src/orchestrator.py) ─── APScheduler jobs
     │
     ├─ brain/*        pull signals → context_feed (SQLite)
     ├─ content/       format_picker → generators → critic → content_queue
     └─ workers/       poster, engager, comment_replier, etc. → instagrapi
```

- **Brain** mines (trend_miner, watcher, reddit, events, rag). Pushes into `context_feed`.
- **Content** pulls context + archetype, dispatches a generator, critic-ranks, enqueues to `content_queue`.
- **Workers** drain queues and call `IGClient`.

State lives in `data/brain.db` (SQLite WAL). No external services.

## Code style

- Python 3.11+ (`from __future__ import annotations` everywhere).
- Type hints on everything public.
- `ruff` + default rules. `make lint` before commit.
- Imports grouped stdlib → third-party → `src.*`.
- No unnecessary comments. Docstrings only where the WHY isn't obvious.

## What's easy to contribute

- **New content formats** — add a generator under `src/content/generators/`, register in `pipeline._dispatch`, add a field to `FormatMix`. See `story_carousel.py` as a recent example.
- **New LLM providers** — `src/core/llm.py` is a small router; adding a provider is usually ~40 lines.
- **New brain modules** — drop under `src/brain/`, push to `context_feed`, wire a scheduled job in `orchestrator.py`.
- **New anti-detection measures** — `src/plugins/human_mimic.py`. All behaviour toggles live on `cfg.human_mimic`.

## What to be careful with

- **Licence gates** — adding a new dependency on a non-commercial model/component means adding a gate in `src/core/config.py` that blocks it under `commercial=true`. Search for `_lora_commercial_gate` for the pattern.
- **Breaking `niche.yaml` compat** — don't rename existing fields; deprecate + alias instead. Users' configs are the interface contract.
- **Defaults** — new features default OFF unless they're safe-by-default. Aggressive engagement, post-cooldown overrides, critic thresholds, etc. should be opt-in.

## PR checklist

- [ ] Tests pass: `make test`
- [ ] Lint clean: `make lint`
- [ ] If touching `niche.yaml` schema: `test_config.py` covers the new field
- [ ] If adding a licence-sensitive dep: gate is added in `config.py`
- [ ] CHANGELOG blurb in PR description (not a file — we use the merge log)
- [ ] No emoji or auto-generated comments unless the surrounding style already has them

## Security issues

Don't file public issues for security bugs — see [SECURITY.md](SECURITY.md).

## License

By contributing, you agree that your contributions are licensed under the repository's [MIT License](LICENSE).
