#!/usr/bin/env bash
# Start the ig-agent orchestrator as a single long-lived process.
#
# The orchestrator runs generator, brain (trend+competitor+watch), poster,
# engager, and health probes inside one asyncio loop via APScheduler.

set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo "No .venv found. Run scripts/bootstrap.sh first."
  exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

mkdir -p logs
LOG="logs/orchestrator.log"

echo "=== ig-agent starting ==="
nohup python -m instagram_ai_agent.orchestrator >> "$LOG" 2>&1 &
PID=$!
echo "$PID" > logs/orchestrator.pid
echo "PID: $PID"
echo "Log: tail -f $LOG"
echo "Stop: kill \$(cat logs/orchestrator.pid)"
