#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/_3dgs_train_profiles.sh"

PREPARED_ROOT=""
MODEL_ROOT=""
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
SCENES=()
SKIP_TRAIN_EVAL=0
FORCE=0

while (($# > 0)); do
  case "$1" in
    --prepared-root)
      shift
      PREPARED_ROOT="${1:-}"
      ;;
    --model-root)
      shift
      MODEL_ROOT="${1:-}"
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
    --scene)
      shift
      [[ $# -gt 0 ]] || die "--scene requires a value"
      append_csv_items "$1" SCENES
      ;;
    --skip-train-eval)
      SKIP_TRAIN_EVAL=1
      ;;
    --force)
      FORCE=1
      ;;
    -h|--help)
      cat <<'EOF'
Usage: train_3dgs_batch.sh --prepared-root prepared/3dgs_public --model-root outputs/models [options]
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
  --scene hcm0031,hcm0034
  --skip-train-eval
  --force
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

[[ -n "${PREPARED_ROOT}" ]] || die "--prepared-root is required"
[[ -n "${MODEL_ROOT}" ]] || die "--model-root is required"
[[ -d "${PREPARED_ROOT}" ]] || die "Prepared root not found: ${PREPARED_ROOT}"

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

echo "TRAIN batch preset=${SETTINGS_PRESET} iterations=${SETTINGS_ITERATIONS} resolution=${SETTINGS_RESOLUTION} optimizer=${SETTINGS_OPTIMIZER_TYPE} antialiasing=${SETTINGS_ANTIALIASING} densify_until=${SETTINGS_DENSIFY_UNTIL_ITER}"
if (( ${#SETTINGS_EXTRA_ARGS[@]} > 0 )); then
  echo "Extra train args: ${SETTINGS_EXTRA_ARGS[*]}"
fi

mapfile -t SCENE_DIRS < <(find "${PREPARED_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
(( ${#SCENE_DIRS[@]} > 0 )) || die "No prepared scenes found under ${PREPARED_ROOT}"

if (( ${#SCENES[@]} > 0 )); then
  FILTERED_SCENES=()
  declare -A WANTED_SCENES=()
  for scene in "${SCENES[@]}"; do
    WANTED_SCENES["${scene,,}"]=1
  done
  for scene in "${SCENE_DIRS[@]}"; do
    if [[ -n "${WANTED_SCENES[${scene,,}]:-}" ]]; then
      FILTERED_SCENES+=("${scene}")
    fi
  done
  SCENE_DIRS=("${FILTERED_SCENES[@]}")
fi

(( ${#SCENE_DIRS[@]} > 0 )) || die "No prepared scenes matched the requested --scene filter"

for scene in "${SCENE_DIRS[@]}"; do
  MODEL_DIR="${MODEL_ROOT}/${scene}"
  DONE_PATH="${MODEL_DIR}/point_cloud/iteration_${SETTINGS_ITERATIONS}/point_cloud.ply"
  LOG_DIR="${MODEL_DIR}/logs"
  LOG_NAME="train_${SETTINGS_ITERATIONS}_r${SETTINGS_RESOLUTION}_${SETTINGS_PRESET}.log"
  LOG_PATH="${LOG_DIR}/${LOG_NAME}"

  if (( ! FORCE )) && [[ -f "${DONE_PATH}" ]]; then
    echo "SKIP ${scene}: ${DONE_PATH} exists"
    continue
  fi

  mkdir -p "${LOG_DIR}"
  echo "TRAIN ${scene}: log=${LOG_PATH}"

  TRAIN_ARGS=()
  add_3dgs_train_options TRAIN_ARGS "${START_CHECKPOINT}" 1 -s "${PREPARED_ROOT}/${scene}" -m "${MODEL_DIR}"

  if ! invoke_3dgs_logged_process "${PYTHON_EXE}" "${LOG_PATH}" "${TRAIN_PY}" "${TRAIN_ARGS[@]}"; then
    tail -n 80 "${LOG_PATH}" || true
    die "3DGS train failed for ${scene}. Log: ${LOG_PATH}"
  fi

  if [[ ! -f "${DONE_PATH}" ]]; then
    tail -n 80 "${LOG_PATH}" || true
    die "3DGS train finished but output is missing for ${scene}: ${DONE_PATH}"
  fi

  echo "DONE ${scene}: ${DONE_PATH}"
done
