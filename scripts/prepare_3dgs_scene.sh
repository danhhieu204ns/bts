#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/_3dgs_train_profiles.sh"

DATA_ROOTS=()
OUT_ROOT="prepared/3dgs_data"
SCENE=""
COPY_MODE="hardlink"
CAMERA_MODE="pinhole"
PYTHON_EXE=""

while (($# > 0)); do
  case "$1" in
    --data-root)
      shift
      [[ $# -gt 0 ]] || die "--data-root requires a value"
      append_csv_items "$1" DATA_ROOTS
      ;;
    --out-root)
      shift
      OUT_ROOT="${1:-}"
      ;;
    --scene)
      shift
      SCENE="${1:-}"
      ;;
    --copy-mode)
      shift
      COPY_MODE="${1:-}"
      ;;
    --camera-mode)
      shift
      CAMERA_MODE="${1:-}"
      ;;
    --python)
      shift
      PYTHON_EXE="${1:-}"
      ;;
    -h|--help)
      cat <<'EOF'
Usage: prepare_3dgs_scene.sh --data-root phase1/public_set [options]
  --out-root PATH
  --scene NAME[,NAME...]
  --copy-mode hardlink|copy
  --camera-mode pinhole|copy
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
[[ "${COPY_MODE}" == "hardlink" || "${COPY_MODE}" == "copy" ]] || die "Invalid --copy-mode: ${COPY_MODE}"
[[ "${CAMERA_MODE}" == "pinhole" || "${CAMERA_MODE}" == "copy" ]] || die "Invalid --camera-mode: ${CAMERA_MODE}"

PYTHON_EXE="$(resolve_3dgs_python "${PYTHON_EXE}")"
export PYTHONPATH="${REPO_ROOT}/src"

args=( -m bts_baseline.prepare_3dgs_scene --data-root )
args+=( "${DATA_ROOTS[@]}" )
args+=( --out-root "${OUT_ROOT}" --copy-mode "${COPY_MODE}" --camera-mode "${CAMERA_MODE}" )
if [[ -n "${SCENE}" ]]; then
  args+=( --scene "${SCENE}" )
fi

"${PYTHON_EXE}" "${args[@]}"
