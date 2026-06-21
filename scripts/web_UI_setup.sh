#!/usr/bin/env bash
#
# web_UI_setup.sh — create the project virtualenv and install dependencies.
#
# Cross-platform: Linux, Windows + WSL (Ubuntu), macOS.
# (Native Windows PowerShell is intentionally out of scope.)
#
# The virtualenv is always created at <repo-root>/.venv because
# .vscode/launch.json hardcodes ${workspaceFolder}/.venv/bin/python.
#
# Usage:
#   ./scripts/web_UI_setup.sh            # create .venv and install the project (editable)
#   ./scripts/web_UI_setup.sh --run      # ...then start the web UI on :8000
#   ./scripts/web_UI_setup.sh --recreate # delete an existing .venv and rebuild from scratch
#   PYTHON=python3.12 ./scripts/web_UI_setup.sh   # force a specific interpreter
#
set -euo pipefail

# --- locate repo root (script lives in <repo>/scripts) ---------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
MIN_PY="3.11"

# --- args ------------------------------------------------------------------
RUN=0
RECREATE=0
for arg in "$@"; do
  case "$arg" in
    --run)      RUN=1 ;;
    --recreate) RECREATE=1 ;;
    -h|--help)
      grep -E '^#( |$)' "$0" | sed -E 's/^# ?//'
      exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

log()  { printf '\033[1;32m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!! \033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31mxx \033[0m %s\n' "$*" >&2; exit 1; }

# --- find a Python >= MIN_PY ----------------------------------------------
# Honors $PYTHON if set; otherwise probes common interpreter names.
py_ok() {
  "$1" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 11) else 1)' \
    >/dev/null 2>&1
}

PYTHON_BIN=""
if [[ -n "${PYTHON:-}" ]]; then
  command -v "$PYTHON" >/dev/null 2>&1 || die "PYTHON=$PYTHON not found on PATH"
  py_ok "$PYTHON" || die "PYTHON=$PYTHON is older than ${MIN_PY}"
  PYTHON_BIN="$PYTHON"
else
  for cand in python3.13 python3.12 python3.11 python3 python; do
    if command -v "$cand" >/dev/null 2>&1 && py_ok "$cand"; then
      PYTHON_BIN="$cand"; break
    fi
  done
fi
[[ -n "$PYTHON_BIN" ]] || die "No Python >= ${MIN_PY} found. Install one (pyenv / brew / apt) or set PYTHON=..."

PY_VER="$("$PYTHON_BIN" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])')"
log "Using interpreter: $(command -v "$PYTHON_BIN")  (Python ${PY_VER})"

# --- (re)create the venv ---------------------------------------------------
if [[ -d "$VENV_DIR" && "$RECREATE" -eq 1 ]]; then
  log "Removing existing .venv (--recreate)"
  rm -rf "$VENV_DIR"
fi

if [[ -d "$VENV_DIR" ]]; then
  log ".venv already exists — reusing it (pass --recreate to rebuild)"
else
  log "Creating virtualenv at ${VENV_DIR}"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

VENV_PY="${VENV_DIR}/bin/python"
[[ -x "$VENV_PY" ]] || die "venv python not found at ${VENV_PY} (venv creation failed?)"

# --- install ---------------------------------------------------------------
log "Upgrading pip"
"$VENV_PY" -m pip install --upgrade pip >/dev/null

log "Installing project (editable) + dependencies"
"$VENV_PY" -m pip install -e "$REPO_ROOT"

# --- verify ----------------------------------------------------------------
log "Verifying imports"
"$VENV_PY" -c 'import backtest, fastapi, uvicorn; print("  imports OK:", backtest.__file__)'

cat <<EOF

$(log "Done.")
  Interpreter : ${VENV_PY}
  Activate    : source .venv/bin/activate
  Run web UI  : python -m backtest web --port 8000   (http://localhost:8000)
  VSCode      : reload window, then F5 on the "Web UI" launch config.
EOF

# --- optional run ----------------------------------------------------------
if [[ "$RUN" -eq 1 ]]; then
  log "Starting web UI on http://localhost:8000  (Ctrl+C to stop)"
  cd "$REPO_ROOT"
  exec "$VENV_PY" -m backtest web --port 8000
fi
