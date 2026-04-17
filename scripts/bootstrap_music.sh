#!/usr/bin/env bash
# Seed data/music/ from public-domain sources.
#
# By default we do NOT clone the full SoundSafari/CC0-1.0-Music repo (it's
# large). Instead we add a README pointing the user at the two canonical
# CC0 sources. Pass --fetch to actually clone.
set -euo pipefail

cd "$(dirname "$0")/.."

MUSIC="data/music"
mkdir -p "$MUSIC/cache" "$MUSIC/fitness" "$MUSIC/lofi" "$MUSIC/ambient" "$MUSIC/upbeat" "$MUSIC/cinematic"

cat > "$MUSIC/README.md" <<'EOF'
# Music library

Drop commercial-safe audio files here (mp3/m4a/wav/ogg/flac/opus). They'll be
picked by the reel generator in preference to any external music API.

Organise by genre folder (`lofi/`, `upbeat/`, etc.) so the query matcher can
weight by niche keyword.

## Safe sources (CC0 / public domain, no attribution required)

- SoundSafari / CC0-1.0-Music  https://github.com/SoundSafari/CC0-1.0-Music
- FreePD                       https://freepd.com  (browser download)
- Free Music Archive CC0 list  https://freemusicarchive.org/genre/Cc0_1-0
- Pixabay Music                https://pixabay.com/music (commercial-OK, no attribution)

## Attribution-required (safe with credit — we don't auto-credit, so keep these out if you forget)

- Uppbeat                      https://uppbeat.io (free tier is NOT commercial for IG)
- YouTube Audio Library "Attribution" tracks

## Never drop in here

- MusicGen outputs (non-commercial weights)
- Stable Audio Open 1.0 outputs (non-commercial)
- Copyrighted tracks of any kind
EOF

if [[ "${1:-}" == "--fetch" ]]; then
  if [ ! -d "$MUSIC/_soundsafari" ]; then
    echo "Cloning SoundSafari/CC0-1.0-Music (large — first time only)..."
    git clone --depth 1 https://github.com/SoundSafari/CC0-1.0-Music "$MUSIC/_soundsafari"
  fi
  find "$MUSIC/_soundsafari" -type f \( -iname '*.mp3' -o -iname '*.wav' -o -iname '*.ogg' -o -iname '*.flac' \) -print0 \
    | while IFS= read -r -d '' file; do
        ln -sf "$(realpath "$file")" "$MUSIC/$(basename "$file")" 2>/dev/null || true
      done
  echo "Linked $(ls "$MUSIC" | grep -Ec '\.(mp3|wav|ogg|flac)$') tracks into data/music/"
else
  echo "Set up data/music/. Run with --fetch to clone the SoundSafari CC0 corpus."
fi
