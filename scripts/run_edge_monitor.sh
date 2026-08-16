#!/usr/bin/env bash
# Runner for the edge health monitor. Resolves the repo root from its own
# location, activates the venv if present, runs the monitor, and (on a GO
# signal) fires an optional alert command. Measurement only — never trades.
#
# Env:
#   EDGE_ALERT_CMD   optional shell command run when the edge revives (exit 10),
#                    e.g. a curl to a webhook / telegram. Receives no args.
#   EDGE_MONITOR_ARGS  extra args passed to edge_monitor.py (e.g. "--slippage-bps 4")
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

python scripts/edge_monitor.py ${EDGE_MONITOR_ARGS:-} "$@"
rc=$?

if [ "$rc" -eq 10 ]; then
  echo "[edge-monitor] GO signal: edge revived — see data/edge_monitor_status.json"
  if [ -n "${EDGE_ALERT_CMD:-}" ]; then
    eval "$EDGE_ALERT_CMD" || echo "[edge-monitor] alert command failed" >&2
  fi
  exit 0
elif [ "$rc" -eq 0 ]; then
  exit 0
else
  echo "[edge-monitor] error (exit $rc)" >&2
  exit "$rc"
fi
