#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash minimal_sgy/run.sh INPUT.sgy [OUTPUT_DIR] [extra arguments...]"
  exit 2
fi

input_path=$1
shift
output_dir=output/sgy_inference
if [[ $# -gt 0 && $1 != -* ]]; then
  output_dir=$1
  shift
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

python minimal_sgy/infer.py \
  --input "$input_path" \
  --output-dir "$output_dir" \
  "$@"
