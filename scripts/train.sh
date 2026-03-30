#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
DEFAULT_PYTHON=python3
if [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
  DEFAULT_PYTHON="$CONDA_PREFIX/bin/python"
fi
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON}"

TRAIN_ARGS=()
data_dir_set=false
output_dir_set=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir|--output-dir)
      if [[ $# -lt 2 ]]; then
        printf "Missing value for %s\n" "$1" >&2
        exit 1
      fi
      if [[ "$1" == "--data-dir" ]]; then
        data_dir_set=true
      else
        output_dir_set=true
      fi
      TRAIN_ARGS+=("$1" "$2")
      shift 2
      ;;
    *)
      TRAIN_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "$data_dir_set" == false ]]; then
  TRAIN_ARGS+=(--data-dir "$PROJECT_ROOT/data")
fi

if [[ "$output_dir_set" == false ]]; then
  TRAIN_ARGS+=(--output-dir "$PROJECT_ROOT/artifacts")
fi

exec "$PYTHON_BIN" "$PROJECT_ROOT/train.py" "${TRAIN_ARGS[@]}"
