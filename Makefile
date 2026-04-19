# instagram-ai-agent — convenience targets.
# All commands assume you're inside a local clone and have `.venv` active
# (`source .venv/bin/activate` — or use `make install` on first run).

PY := .venv/bin/python
PIP := .venv/bin/pip
IG := .venv/bin/ig-agent

.DEFAULT_GOAL := help

help: ## show this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage: make \033[1m<target>\033[0m\n\nTargets:\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

install: ## first-run: venv + deps + playwright + default assets
	./install.sh
	@echo "\n\033[1mNext:\033[0m make setup   \033[2m(or: ig-agent setup)\033[0m"

setup: ## one-command onboarding — deps + niche + AI key (quick mode, 4 questions)
	$(IG) setup

setup-full: ## deep-config onboarding (~15 questions: voice / palette / formats / schedule / optional keys)
	$(IG) setup --full

init: ## legacy 40-question wizard — prefer `make setup` for most users
	$(IG) init

login: ## verify Instagram credentials + persist session
	$(IG) login

generate: ## one-shot: generate N posts (usage: make generate N=3)
	$(IG) generate --count $(or $(N),1)

review: ## walk pending-review items, approve / reject
	$(IG) review

run: ## start the full orchestrator (brain + generator + poster + engager)
	$(IG) run

status: ## agent pulse (heartbeat, backoff, queue, next posts)
	$(IG) status

pause: ## halt all IG writes + generation (brain keeps running)
	$(IG) pause

resume: ## clear pause state
	$(IG) resume

doctor: ## diagnostic self-check — run when something's off
	$(IG) doctor

dashboard: ## start the local read-only web dashboard on :8080
	$(IG) dashboard

test: ## run the full test suite
	$(PY) -m pytest -q

lint: ## ruff lint + check
	$(PY) -m ruff check src tests

update: ## pull latest + refresh deps
	git pull --ff-only
	$(PIP) install --quiet -e .

clean: ## remove venv + caches (keeps data/ and niche.yaml)
	rm -rf .venv __pycache__ .pytest_cache .ruff_cache *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

.PHONY: help install setup setup-full init login generate review run pause resume status dashboard doctor test lint update clean
