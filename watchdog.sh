#!/usr/bin/env bash
# Watchdog — restart orchestrator if it crashes. Poll every 60s.

set -euo pipefail

cd "$(dirname "$0")"

PIDFILE="logs/orchestrator.pid"
LOG="logs/watchdog.log"
mkdir -p logs

while true; do
  if [ -f "$PIDFILE" ]; then
    PID=$(cat "$PIDFILE")
    if kill -0 "$PID" 2>/dev/null; then
      sleep 60
      continue
    fi
  fi
  echo "[$(date -u +%FT%TZ)] orchestrator not running — restarting" >> "$LOG"
  ./start.sh >> "$LOG" 2>&1
  sleep 60
done
