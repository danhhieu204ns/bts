#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/_3dgs_train_profiles.sh"

PREPARED_SCENE=""
MODEL_DIR=""
PRESET="l40s-quality"
ITERATIONS=0
RESOLUTION=0
OPTIMIZER_TYPE="profile"
ANTIALIASING_SPECIFIED=0
ANTIALIASING_VALUE=0
NO_ANTIALIASING=0
SAVE_ITERATIONS=()
TEST_ITERATIONS=()
CHECKPOINT_ITERATIONS=()
DENSIFY_UNTIL_ITER=0
DENSIFY_UNTIL_SPECIFIED=0
DENSIFY_GRAD_THRESHOLD="-1.0"
DENSIFICATION_INTERVAL=0
OPACITY_RESET_INTERVAL=0
START_CHECKPOINT=""
EXTRA_ARGS=()
PYTHON_EXE=""
CUDA_VISIBLE_DEVICES_VALUE=""
SKIP_TRAIN_EVAL=0
QUIET=0

while (($# > 0)); do
  case "$1" in
    --prepared-scene)
      shift
      PREPARED_SCENE="${1:-}"
      ;;
    --model-dir)
      shift
      MODEL_DIR="${1:-}"
      ;;
    --preset)
      shift
      PRESET="${1:-}"
      ;;
    --iterations)
      shift
      ITERATIONS="${1:-0}"
      ;;
    --resolution)
      shift
      RESOLUTION="${1:-0}"
      ;;
    --optimizer-type)
      shift
      OPTIMIZER_TYPE="${1:-}"
      ;;
    --antialiasing)
      ANTIALIASING_SPECIFIED=1
      ANTIALIASING_VALUE=1
      ;;
    --no-antialiasing)
      NO_ANTIALIASING=1
      ;;
    --save-iterations)
      shift
      [[ $# -gt 0 ]] || die "--save-iterations requires a value"
      SAVE_ITERATIONS=()
      append_csv_items "$1" SAVE_ITERATIONS
      ;;
    --test-iterations)
      shift
      [[ $# -gt 0 ]] || die "--test-iterations requires a value"
      TEST_ITERATIONS=()
      append_csv_items "$1" TEST_ITERATIONS
      ;;
    --checkpoint-iterations)
      shift
      [[ $# -gt 0 ]] || die "--checkpoint-iterations requires a value"
      CHECKPOINT_ITERATIONS=()
      append_csv_items "$1" CHECKPOINT_ITERATIONS
      ;;
    --densify-until-iter)
      shift
      DENSIFY_UNTIL_ITER="${1:-0}"
      DENSIFY_UNTIL_SPECIFIED=1
      ;;
    --densify-grad-threshold)
      shift
      DENSIFY_GRAD_THRESHOLD="${1:-}"
      ;;
    --densification-interval)
      shift
      DENSIFICATION_INTERVAL="${1:-0}"
      ;;
    --opacity-reset-interval)
      shift
      OPACITY_RESET_INTERVAL="${1:-0}"
      ;;
    --start-checkpoint)
      shift
      START_CHECKPOINT="${1:-}"
      ;;
    --extra-arg)
      shift
      [[ $# -gt 0 ]] || die "--extra-arg requires a value"
      EXTRA_ARGS+=("$1")
      ;;
    --python)
      shift
      PYTHON_EXE="${1:-}"
      ;;
    --cuda-visible-devices)
      shift
      CUDA_VISIBLE_DEVICES_VALUE="${1:-}"
      ;;
    --skip-train-eval)
      SKIP_TRAIN_EVAL=1
      ;;
    --quiet)
      QUIET=1
      ;;
    -h|--help)
      cat <<'EOF'
Usage: train_3dgs_scene.sh --prepared-scene prepared/3dgs_public/hcm0031 --model-dir outputs/model [options]
  --preset local-r2-7k|l40s-fast|l40s-quality|l40s-bts-quality|custom
  --iterations N
  --resolution N
  --optimizer-type profile|default|sparse_adam
  --antialiasing
  --no-antialiasing
  --save-iterations 7000,15000
  --test-iterations 7000,15000
  --checkpoint-iterations 15000,30000
  --densify-until-iter N
  --densify-grad-threshold FLOAT
  --densification-interval N
  --opacity-reset-interval N
  --start-checkpoint PATH
  --extra-arg VALUE
  --python PATH
  --cuda-visible-devices IDS
  --skip-train-eval
  --quiet
EOF
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
  shift
done

[[ -n "${PREPARED_SCENE}" ]] || die "--prepared-scene is required"
[[ -n "${MODEL_DIR}" ]] || die "--model-dir is required"

PYTHON_EXE="$(resolve_3dgs_python "${PYTHON_EXE}")"
TRAIN_PY="${REPO_ROOT}/external/gaussian-splatting/train.py"
[[ -f "${TRAIN_PY}" ]] || die "official 3DGS train.py not found: ${TRAIN_PY}"
export PYTHONPATH="${REPO_ROOT}/external/gaussian-splatting"

if [[ -n "${CUDA_VISIBLE_DEVICES_VALUE}" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}"
fi

resolve_3dgs_train_settings \
  "${PRESET}" \
  "${ITERATIONS}" \
  "${RESOLUTION}" \
  "${OPTIMIZER_TYPE}" \
  "${ANTIALIASING_SPECIFIED}" \
  "${ANTIALIASING_VALUE}" \
  "${NO_ANTIALIASING}" \
  SAVE_ITERATIONS \
  TEST_ITERATIONS \
  CHECKPOINT_ITERATIONS \
  "${DENSIFY_UNTIL_ITER}" \
  "${DENSIFY_GRAD_THRESHOLD}" \
  "${DENSIFICATION_INTERVAL}" \
  "${OPACITY_RESET_INTERVAL}" \
  EXTRA_ARGS

if [[ "${SETTINGS_OPTIMIZER_TYPE}" == "sparse_adam" ]] && ! test_3dgs_sparse_adam_available "${PYTHON_EXE}"; then
  warn "Sparse Adam is not available in diff_gaussian_rasterization. Falling back to optimizer=default."
  SETTINGS_OPTIMIZER_TYPE="default"
fi

if (( DENSIFY_UNTIL_SPECIFIED )); then
  SETTINGS_DENSIFY_UNTIL_ITER="${DENSIFY_UNTIL_ITER}"
fi

if (( SKIP_TRAIN_EVAL )); then
  SETTINGS_TEST_ITERATIONS=( $((SETTINGS_ITERATIONS + 1)) )
fi

echo "TRAIN scene preset=${SETTINGS_PRESET} iterations=${SETTINGS_ITERATIONS} resolution=${SETTINGS_RESOLUTION} optimizer=${SETTINGS_OPTIMIZER_TYPE} antialiasing=${SETTINGS_ANTIALIASING} densify_until=${SETTINGS_DENSIFY_UNTIL_ITER}"
if (( ${#SETTINGS_EXTRA_ARGS[@]} > 0 )); then
  echo "Extra train args: ${SETTINGS_EXTRA_ARGS[*]}"
fi

TRAIN_ARGS=()
add_3dgs_train_options TRAIN_ARGS "${START_CHECKPOINT}" "${QUIET}" -s "${PREPARED_SCENE}" -m "${MODEL_DIR}"

"${PYTHON_EXE}" "${TRAIN_PY}" "${TRAIN_ARGS[@]}"
