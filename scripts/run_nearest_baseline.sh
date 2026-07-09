#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/_3dgs_train_profiles.sh"

DATA_ROOTS=()
OUT_DIR=""
ZIP_PATH=""
ORIENTATION_WEIGHT="0.5"
PYTHON_EXE=""

while (($# > 0)); do
  case "$1" in
    --data-root)
      shift
      [[ $# -gt 0 ]] || die "--data-root requires a value"
      append_csv_items "$1" DATA_ROOTS
      ;;
    --out-dir)
      shift
      OUT_DIR="${1:-}"
      ;;
    --zip|--zip-path)
      shift
      ZIP_PATH="${1:-}"
      ;;
    --orientation-weight)
      shift
      ORIENTATION_WEIGHT="${1:-}"
      ;;
    --python)
      shift
      PYTHON_EXE="${1:-}"
      ;;
    -h|--help)
      cat <<'EOF'
Usage: run_nearest_baseline.sh --data-root phase1/public_set --out-dir outputs/nearest [options]
  --zip PATH
  --orientation-weight FLOAT
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
[[ -n "${OUT_DIR}" ]] || die "--out-dir is required"

PYTHON_EXE="$(resolve_3dgs_python "${PYTHON_EXE}")"
export PYTHONPATH="${REPO_ROOT}/src"

args=( -m bts_baseline.nearest_pose_baseline --data-root )
args+=( "${DATA_ROOTS[@]}" )
args+=( --out-dir "${OUT_DIR}" --orientation-weight "${ORIENTATION_WEIGHT}" )
if [[ -n "${ZIP_PATH}" ]]; then
  args+=( --zip "${ZIP_PATH}" )
fi

"${PYTHON_EXE}" "${args[@]}"
