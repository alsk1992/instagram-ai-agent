#!/usr/bin/env bash
# One-shot dev bootstrap: venv + deps + playwright browsers + fonts + default meme bg.
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -d ".venv" ]; then
  python3.11 -m venv .venv 2>/dev/null || python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .

# Playwright browser + system deps
python -m playwright install chromium
python -m playwright install-deps chromium || true

# Default assets
python scripts/gen_default_assets.py

# Google Fonts (commercial-safe, SIL-OFL)
mkdir -p data/fonts
if [ ! -f "data/fonts/Archivo Black.ttf" ]; then
  curl -fsSL -o "data/fonts/Archivo Black.ttf" \
    https://github.com/google/fonts/raw/main/ofl/archivoblack/ArchivoBlack-Regular.ttf
fi
if [ ! -f "data/fonts/Inter.ttf" ]; then
  curl -fsSL -o "data/fonts/Inter.ttf" \
    "https://github.com/google/fonts/raw/main/ofl/inter/Inter%5Bopsz%2Cwght%5D.ttf"
fi

echo ""
echo "=== bootstrap complete ==="
echo "  1. ig-agent init          # interactive setup"
echo "  2. ig-agent login         # verify IG auth"
echo "  3. ig-agent generate --count 3   # make some content"
echo "  4. ig-agent review        # approve queue items"
echo "  5. ig-agent run           # start the full loop"
