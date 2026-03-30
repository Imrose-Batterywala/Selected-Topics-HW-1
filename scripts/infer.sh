#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
DEFAULT_PYTHON=python3
if [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
  DEFAULT_PYTHON="$CONDA_PREFIX/bin/python"
fi
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON}"

PREDICT_ARGS=()
checkpoint_set=false
test_dir_set=false
output_set=false
submission_path="$PROJECT_ROOT/submission.zip"
prediction_path="$PROJECT_ROOT/prediction.csv"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint|--test-dir|--output)
      if [[ $# -lt 2 ]]; then
        printf "Missing value for %s\n" "$1" >&2
        exit 1
      fi
      if [[ "$1" == "--checkpoint" ]]; then
        checkpoint_set=true
      elif [[ "$1" == "--test-dir" ]]; then
        test_dir_set=true
      else
        output_set=true
        prediction_path="$2"
      fi
      PREDICT_ARGS+=("$1" "$2")
      shift 2
      ;;
    --submission)
      if [[ $# -lt 2 ]]; then
        printf "Missing value for %s\n" "$1" >&2
        exit 1
      fi
      submission_path="$2"
      shift 2
      ;;
    *)
      PREDICT_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "$checkpoint_set" == false ]]; then
  PREDICT_ARGS+=(--checkpoint "$PROJECT_ROOT/artifacts/best_model.pt")
fi

if [[ "$test_dir_set" == false ]]; then
  PREDICT_ARGS+=(--test-dir "$PROJECT_ROOT/data/test")
fi

if [[ "$output_set" == false ]]; then
  PREDICT_ARGS+=(--output "$prediction_path")
fi

"$PYTHON_BIN" "$PROJECT_ROOT/predict.py" "${PREDICT_ARGS[@]}"

mkdir -p "$(dirname -- "$submission_path")"
rm -f "$submission_path"

"$PYTHON_BIN" - "$prediction_path" "$submission_path" <<'PY'
from pathlib import Path
import sys
import zipfile

prediction_path = Path(sys.argv[1]).resolve()
submission_path = Path(sys.argv[2]).resolve()

with zipfile.ZipFile(submission_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    archive.write(prediction_path, arcname=prediction_path.name)
PY

printf "Created %s\n" "$submission_path"
