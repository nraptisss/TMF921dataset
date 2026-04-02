#!/usr/bin/env bash
set -euo pipefail

PID="$1"
LOG_PATH="$2"
STATUS_PATH="$3"
INTERVAL_SECONDS="${4:-20}"

while kill -0 "$PID" 2>/dev/null; do
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  stat="$(ps -p "$PID" -o stat=,etime=,pcpu=,pmem= || true)"
  last="$(tail -n 1 "$LOG_PATH" | tr -d '\r' || true)"
  echo "$ts | $stat | $last" > "$STATUS_PATH"
  sleep "$INTERVAL_SECONDS"
done

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) | process_exited" > "$STATUS_PATH"
