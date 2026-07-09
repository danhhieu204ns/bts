#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/_3dgs_train_profiles.sh"

DATA_ROOTS=()
MODEL_ROOT=""
OUT_DIR=""
SCENE=""
ITERATION="-1"
SH_DEGREE="3"
BACKGROUND="black"
ANTIALIASING=0
SCALING_MODIFIER="1.0"
PYTHON_EXE=""
ZIP_PATH=""

while (($# > 0)); do
  case "$1" in
    --data-root)
      shift
      [[ $# -gt 0 ]] || die "--data-root requires a value"
      append_csv_items "$1" DATA_ROOTS
      ;;
    --model-root)
      shift
      MODEL_ROOT="${1:-}"
      ;;
    --out-dir)
      shift
      OUT_DIR="${1:-}"
      ;;
    --scene)
      shift
      SCENE="${1:-}"
      ;;
    --iteration)
      shift
      ITERATION="${1:-}"
      ;;
    --sh-degree)
      shift
      SH_DEGREE="${1:-}"
      ;;
    --background)
      shift
      BACKGROUND="${1:-}"
      ;;
    --antialiasing)
      ANTIALIASING=1
      ;;
    --scaling-modifier)
      shift
      SCALING_MODIFIER="${1:-}"
      ;;
    --python)
      shift
      PYTHON_EXE="${1:-}"
      ;;
    --zip|--zip-path)
      shift
      ZIP_PATH="${1:-}"
      ;;
    -h|--help)
      cat <<'EOF'
Usage: render_3dgs_submission.sh --data-root phase1/public_set --model-root outputs/models --out-dir outputs/render [options]
  --scene NAME
  --iteration N
  --sh-degree N
  --background black|white
  --antialiasing
  --scaling-modifier FLOAT
  --zip PATH
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
[[ -n "${MODEL_ROOT}" ]] || die "--model-root is required"
[[ -n "${OUT_DIR}" ]] || die "--out-dir is required"
[[ "${BACKGROUND}" == "black" || "${BACKGROUND}" == "white" ]] || die "Invalid --background: ${BACKGROUND}"

PYTHON_EXE="$(resolve_3dgs_python "${PYTHON_EXE}")"
export PYTHONPATH="${REPO_ROOT}/src"

args=( -m bts_baseline.render_3dgs_submission --data-root )
args+=( "${DATA_ROOTS[@]}" )
args+=( --model-root "${MODEL_ROOT}" --out-dir "${OUT_DIR}" --iteration "${ITERATION}" --sh-degree "${SH_DEGREE}" --background "${BACKGROUND}" )
if [[ -n "${SCENE}" ]]; then
  args+=( --scene "${SCENE}" )
fi
if (( ANTIALIASING )); then
  args+=( --antialiasing )
fi
if [[ "${SCALING_MODIFIER}" != "1.0" ]]; then
  args+=( --scaling-modifier "${SCALING_MODIFIER}" )
fi
if [[ -n "${ZIP_PATH}" ]]; then
  args+=( --zip "${ZIP_PATH}" )
fi

"${PYTHON_EXE}" "${args[@]}"
