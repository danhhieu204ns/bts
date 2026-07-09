#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/_3dgs_train_profiles.sh"

DATA_ROOTS=()
OUT_DIR=""
MODEL_DIR="outputs/point_splat_models"
SCENE=""
BACKGROUND="nearest"
SPLAT_RADIUS="3"
ALPHA="0.85"
MAX_POINTS="0"
ZIP_PATH=""
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
    --model-dir)
      shift
      MODEL_DIR="${1:-}"
      ;;
    --scene)
      shift
      SCENE="${1:-}"
      ;;
    --background)
      shift
      BACKGROUND="${1:-}"
      ;;
    --splat-radius)
      shift
      SPLAT_RADIUS="${1:-}"
      ;;
    --alpha)
      shift
      ALPHA="${1:-}"
      ;;
    --max-points)
      shift
      MAX_POINTS="${1:-}"
      ;;
    --zip|--zip-path)
      shift
      ZIP_PATH="${1:-}"
      ;;
    --python)
      shift
      PYTHON_EXE="${1:-}"
      ;;
    -h|--help)
      cat <<'EOF'
Usage: run_point_splat_baseline.sh --data-root phase1/public_set --out-dir outputs/point_splat [options]
  --model-dir PATH
  --scene NAME
  --background nearest|solid
  --splat-radius N
  --alpha FLOAT
  --max-points N
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
[[ -n "${OUT_DIR}" ]] || die "--out-dir is required"
[[ "${BACKGROUND}" == "nearest" || "${BACKGROUND}" == "solid" ]] || die "Invalid --background: ${BACKGROUND}"

PYTHON_EXE="$(resolve_3dgs_python "${PYTHON_EXE}")"
export PYTHONPATH="${REPO_ROOT}/src"

args=( -m bts_baseline.point_splat_baseline --data-root )
args+=( "${DATA_ROOTS[@]}" )
args+=( --out-dir "${OUT_DIR}" --model-dir "${MODEL_DIR}" --background "${BACKGROUND}" --splat-radius "${SPLAT_RADIUS}" --alpha "${ALPHA}" --max-points "${MAX_POINTS}" )
if [[ -n "${SCENE}" ]]; then
  args+=( --scene "${SCENE}" )
fi
if [[ -n "${ZIP_PATH}" ]]; then
  args+=( --zip "${ZIP_PATH}" )
fi

"${PYTHON_EXE}" "${args[@]}"
