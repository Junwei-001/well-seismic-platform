#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ ! -x .venv/bin/python ]]; then
  echo "尚未安装环境，请先运行 ./scripts/setup.sh" >&2
  exit 1
fi
if [[ ! -f frontend/dist/index.html ]]; then
  echo "前端尚未构建，请重新运行 ./scripts/setup.sh" >&2
  exit 1
fi

export WELL_SEISMIC_HOST="${WELL_SEISMIC_HOST:-127.0.0.1}"
export WELL_SEISMIC_PORT="${WELL_SEISMIC_PORT:-8000}"
export PYTHONPATH="$PROJECT_ROOT/src"
exec .venv/bin/python -m well_seismic.api

