#!/usr/bin/env bash
# instagram-ai-agent one-shot installer.
#
# Usage (anywhere):
#   curl -fsSL https://raw.githubusercontent.com/alsk1992/instagram-ai-agent/main/install.sh | bash
#
# What it does:
#   1. Checks python 3.11+ and ffmpeg are on PATH
#   2. Clones the repo into ./instagram-ai-agent (or cd's in if already cloned)
#   3. Creates a .venv, pip installs the agent + playwright chromium + fonts
#   4. Drops you into the interactive setup wizard
#
# Idempotent — rerunning just refreshes deps.

set -euo pipefail

REPO_URL="https://github.com/alsk1992/instagram-ai-agent.git"
REPO_DIR="instagram-ai-agent"

# ─── Colours ────────────────────────────────────────────────
if [[ -t 1 ]]; then
  BOLD='\033[1m'; GREEN='\033[32m'; YELLOW='\033[33m'; RED='\033[31m'; DIM='\033[2m'; RESET='\033[0m'
else
  BOLD=''; GREEN=''; YELLOW=''; RED=''; DIM=''; RESET=''
fi

say()  { printf "${BOLD}==>${RESET} %s\n" "$*"; }
ok()   { printf "${GREEN}✓${RESET} %s\n" "$*"; }
warn() { printf "${YELLOW}⚠${RESET}  %s\n" "$*"; }
die()  { printf "${RED}✗${RESET}  %s\n" "$*" >&2; exit 1; }

# ─── Prerequisite checks ─────────────────────────────────────
say "Checking prerequisites"

PYTHON_BIN=""
for candidate in python3.12 python3.11 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then
    ver="$("$candidate" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "")"
    if [[ "$ver" == "3.11" || "$ver" == "3.12" ]]; then
      PYTHON_BIN="$candidate"
      break
    fi
  fi
done
if [[ -z "$PYTHON_BIN" ]]; then
  die "Python 3.11 or 3.12 required.
      Install:
        macOS:  brew install python@3.12
        Ubuntu: sudo apt install python3.12 python3.12-venv
        Arch:   sudo pacman -S python"
fi
ok "Python: $PYTHON_BIN ($(${PYTHON_BIN} --version 2>&1))"

if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
  die "ffmpeg + ffprobe required for reels.
      Install:
        macOS:  brew install ffmpeg
        Ubuntu: sudo apt install ffmpeg
        Arch:   sudo pacman -S ffmpeg"
fi
ok "ffmpeg: $(ffmpeg -version 2>&1 | head -1 | cut -d' ' -f1-3)"

if ! command -v git >/dev/null 2>&1; then
  die "git required. Install it first."
fi

# ─── Clone / cd ──────────────────────────────────────────────
if [[ -f "pyproject.toml" ]] && grep -q 'name = "ig-agent"' pyproject.toml 2>/dev/null; then
  say "Running from an existing ig-agent checkout — skipping clone"
  REPO_DIR="$(pwd)"
elif [[ -d "$REPO_DIR/.git" ]]; then
  say "Updating existing clone in $REPO_DIR"
  cd "$REPO_DIR"
  git pull --ff-only || warn "git pull failed — continuing with existing code"
else
  say "Cloning $REPO_URL → $REPO_DIR"
  git clone --depth 1 "$REPO_URL" "$REPO_DIR"
  cd "$REPO_DIR"
fi
ok "Working in: $(pwd)"

# ─── venv + install ──────────────────────────────────────────
say "Creating virtualenv in .venv"
if [[ ! -d ".venv" ]]; then
  "$PYTHON_BIN" -m venv .venv
  ok "Created .venv"
else
  ok "Reusing existing .venv"
fi

# shellcheck disable=SC1091
source .venv/bin/activate

say "Installing ig-agent + dependencies (this is the slow step)"
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -e .
ok "Package installed"

# ─── Playwright chromium ────────────────────────────────────
say "Installing Playwright chromium (first run only)"
python -m playwright install chromium >/dev/null 2>&1 || warn "Playwright chromium install had warnings"
python -m playwright install-deps chromium >/dev/null 2>&1 || true
ok "Playwright ready"

# ─── Default assets (fonts + sample meme templates) ─────────
say "Fetching commercial-safe default assets"
if [[ -f "scripts/gen_default_assets.py" ]]; then
  python scripts/gen_default_assets.py >/dev/null 2>&1 || warn "Default asset generation had warnings"
fi
mkdir -p data/fonts
if [[ ! -f "data/fonts/Archivo Black.ttf" ]]; then
  curl -fsSL -o "data/fonts/Archivo Black.ttf" \
    https://github.com/google/fonts/raw/main/ofl/archivoblack/ArchivoBlack-Regular.ttf \
    2>/dev/null || warn "Archivo Black font fetch failed (non-fatal)"
fi
if [[ ! -f "data/fonts/Inter.ttf" ]]; then
  curl -fsSL -o "data/fonts/Inter.ttf" \
    "https://github.com/google/fonts/raw/main/ofl/inter/Inter%5Bopsz%2Cwght%5D.ttf" \
    2>/dev/null || warn "Inter font fetch failed (non-fatal)"
fi
ok "Assets ready"

# ─── Next steps ─────────────────────────────────────────────
printf "\n${GREEN}${BOLD}✓ Install complete.${RESET}\n\n"
printf "${BOLD}Next steps${RESET} (activate the venv first if new shell):\n"
printf "  ${DIM}cd %s${RESET}\n" "$REPO_DIR"
printf "  ${DIM}source .venv/bin/activate${RESET}\n\n"
printf "  ${BOLD}1.${RESET} ig-agent init              ${DIM}# interactive wizard → niche.yaml + .env${RESET}\n"
printf "  ${BOLD}2.${RESET} ig-agent login             ${DIM}# verify IG credentials + persist session${RESET}\n"
printf "  ${BOLD}3.${RESET} ig-agent generate -n 3     ${DIM}# make 3 posts into the queue${RESET}\n"
printf "  ${BOLD}4.${RESET} ig-agent review            ${DIM}# approve / reject them${RESET}\n"
printf "  ${BOLD}5.${RESET} ig-agent run               ${DIM}# start the full orchestrator${RESET}\n\n"
printf "Need at least one free LLM key: ${BOLD}OPENROUTER_API_KEY${RESET} (https://openrouter.ai/keys)\n"
printf "Docs + niche.yaml examples: ${BOLD}https://github.com/alsk1992/instagram-ai-agent${RESET}\n"
