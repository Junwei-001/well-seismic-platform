#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -d .git ]]; then
  git lfs install
  git lfs pull
else
  printf '当前为独立发布文件夹，使用随包模型权重。\n'
fi

python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements/development.txt

(
  cd frontend
  npm ci
  npm run build
)

.venv/bin/python tools/verify_release.py --runtime
printf '\n安装完成。运行 ./scripts/run.sh 启动平台。\n'
