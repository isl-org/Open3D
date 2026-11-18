#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 -m venv "$ROOT/.venv"
"$ROOT/.venv/bin/python" -m pip install --upgrade pip wheel
"$ROOT/.venv/bin/python" -m pip install -r "$ROOT/requirements.txt"

printf '\nReady. Run:\n  source "%s/.venv/bin/activate"\n  python "%s/slam_demo.py"\n' "$ROOT" "$ROOT"
