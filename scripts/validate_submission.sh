#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/_3dgs_train_profiles.sh"

DATA_ROOTS=()
PRED_DIR=""
PYTHON_EXE=""

while (($# > 0)); do
  case "$1" in
    --data-root)
      shift
      [[ $# -gt 0 ]] || die "--data-root requires a value"
      append_csv_items "$1" DATA_ROOTS
      ;;
    --pred-dir)
      shift
      PRED_DIR="${1:-}"
      ;;
    --python)
      shift
      PYTHON_EXE="${1:-}"
      ;;
    -h|--help)
      cat <<'EOF'
Usage: validate_submission.sh --data-root phase1/public_set --pred-dir outputs/render [options]
  --python PATH
EOF
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
  shift
done

(( ${#DATA_ROOTS[@]} > 0 )) || die "At least one --data-root is required"
[[ -n "${PRED_DIR}" ]] || die "--pred-dir is required"

PYTHON_EXE="$(resolve_3dgs_python "${PYTHON_EXE}")"
export PYTHONPATH="${REPO_ROOT}/src"

"${PYTHON_EXE}" -m bts_baseline.validate_submission --data-root "${DATA_ROOTS[@]}" --pred-dir "${PRED_DIR}"
