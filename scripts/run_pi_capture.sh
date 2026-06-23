#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f "$ROOT_DIR/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$ROOT_DIR/.venv/bin/activate"
elif [[ -f "$ROOT_DIR/.venv/Scripts/activate" ]]; then
  # shellcheck source=/dev/null
  source "$ROOT_DIR/.venv/Scripts/activate"
else
  "$ROOT_DIR/build.sh"
  # shellcheck source=/dev/null
  source "$ROOT_DIR/.venv/bin/activate"
fi

oracle-report capture --output-dir "${1:-$ROOT_DIR/runs/pi-capture}"
