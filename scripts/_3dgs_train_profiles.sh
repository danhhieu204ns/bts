#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

die() {
  echo "Error: $*" >&2
  exit 1
}

warn() {
  echo "Warning: $*" >&2
}

resolve_3dgs_python() {
  local requested="${1:-}"
  local candidate

  if [[ -n "${requested}" ]]; then
    echo "${requested}"
    return 0
  fi
  if [[ -n "${PYTHON:-}" ]]; then
    echo "${PYTHON}"
    return 0
  fi

  for candidate in \
    "${REPO_ROOT}/.venv/bin/python" \
    "${REPO_ROOT}/.venv/bin/python3"
  do
    if [[ -x "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
  done

  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi
  if command -v python >/dev/null 2>&1; then
    command -v python
    return 0
  fi

  die "Could not find a Python interpreter. Pass --python or export PYTHON."
}

append_csv_items() {
  local input="${1:-}"
  local out_name="$2"
  local -n out_ref="${out_name}"
  local item

  IFS=',' read -r -a __items <<< "${input}"
  for item in "${__items[@]}"; do
    item="${item#"${item%%[![:space:]]*}"}"
    item="${item%"${item##*[![:space:]]}"}"
    if [[ -n "${item}" ]]; then
      out_ref+=("${item}")
    fi
  done
}

get_3dgs_train_profile() {
  local preset="$1"

  case "${preset}" in
    local-r2-7k)
      PROFILE_ITERATIONS=7000
      PROFILE_RESOLUTION=2
      PROFILE_OPTIMIZER_TYPE="default"
      PROFILE_ANTIALIASING=0
      PROFILE_SAVE_ITERATIONS=(7000)
      PROFILE_TEST_ITERATIONS=(7000)
      PROFILE_CHECKPOINT_ITERATIONS=()
      PROFILE_DENSIFY_UNTIL_ITER=0
      PROFILE_DENSIFY_GRAD_THRESHOLD=-1.0
      PROFILE_DENSIFICATION_INTERVAL=0
      PROFILE_OPACITY_RESET_INTERVAL=0
      PROFILE_EXTRA_ARGS=()
      ;;
    l40s-fast)
      PROFILE_ITERATIONS=15000
      PROFILE_RESOLUTION=1
      PROFILE_OPTIMIZER_TYPE="sparse_adam"
      PROFILE_ANTIALIASING=1
      PROFILE_SAVE_ITERATIONS=(7000 15000)
      PROFILE_TEST_ITERATIONS=(7000 15000)
      PROFILE_CHECKPOINT_ITERATIONS=(15000)
      PROFILE_DENSIFY_UNTIL_ITER=12000
      PROFILE_DENSIFY_GRAD_THRESHOLD=0.00025
      PROFILE_DENSIFICATION_INTERVAL=100
      PROFILE_OPACITY_RESET_INTERVAL=3000
      PROFILE_EXTRA_ARGS=()
      ;;
    l40s-quality)
      PROFILE_ITERATIONS=30000
      PROFILE_RESOLUTION=1
      PROFILE_OPTIMIZER_TYPE="sparse_adam"
      PROFILE_ANTIALIASING=1
      PROFILE_SAVE_ITERATIONS=(7000 15000 30000)
      PROFILE_TEST_ITERATIONS=(7000 15000 30000)
      PROFILE_CHECKPOINT_ITERATIONS=(15000 30000)
      PROFILE_DENSIFY_UNTIL_ITER=15000
      PROFILE_DENSIFY_GRAD_THRESHOLD=0.0002
      PROFILE_DENSIFICATION_INTERVAL=100
      PROFILE_OPACITY_RESET_INTERVAL=3000
      PROFILE_EXTRA_ARGS=()
      ;;
    l40s-bts-quality)
      PROFILE_ITERATIONS=30000
      PROFILE_RESOLUTION=1
      PROFILE_OPTIMIZER_TYPE="sparse_adam"
      PROFILE_ANTIALIASING=0
      PROFILE_SAVE_ITERATIONS=(7000 15000 20000 30000)
      PROFILE_TEST_ITERATIONS=(7000 15000 20000 30000)
      PROFILE_CHECKPOINT_ITERATIONS=(15000 30000)
      PROFILE_DENSIFY_UNTIL_ITER=20000
      PROFILE_DENSIFY_GRAD_THRESHOLD=0.00012
      PROFILE_DENSIFICATION_INTERVAL=75
      PROFILE_OPACITY_RESET_INTERVAL=3000
      PROFILE_EXTRA_ARGS=(--lambda_dssim 0.15)
      ;;
    custom)
      PROFILE_ITERATIONS=30000
      PROFILE_RESOLUTION=1
      PROFILE_OPTIMIZER_TYPE="default"
      PROFILE_ANTIALIASING=0
      PROFILE_SAVE_ITERATIONS=(30000)
      PROFILE_TEST_ITERATIONS=(30000)
      PROFILE_CHECKPOINT_ITERATIONS=()
      PROFILE_DENSIFY_UNTIL_ITER=0
      PROFILE_DENSIFY_GRAD_THRESHOLD=-1.0
      PROFILE_DENSIFICATION_INTERVAL=0
      PROFILE_OPACITY_RESET_INTERVAL=0
      PROFILE_EXTRA_ARGS=()
      ;;
    *)
      die "Unsupported preset: ${preset}"
      ;;
  esac
}

normalize_3dgs_iteration_list() {
  local values_name="$1"
  local defaults_name="$2"
  local final_iteration="$3"
  local include_final="$4"
  local out_name="$5"
  local -n values_ref="${values_name}"
  local -n defaults_ref="${defaults_name}"
  local -n out_ref="${out_name}"
  local items=()
  local value

  if (( ${#values_ref[@]} > 0 )); then
    items=("${values_ref[@]}")
  else
    items=("${defaults_ref[@]}")
  fi
  if (( include_final )) && (( final_iteration > 0 )); then
    items+=("${final_iteration}")
  fi

  mapfile -t out_ref < <(
    for value in "${items[@]}"; do
      [[ -n "${value}" ]] || continue
      if (( value > 0 )); then
        printf '%s\n' "${value}"
      fi
    done | sort -n -u
  )
}

resolve_3dgs_train_settings() {
  local preset="$1"
  local iterations="$2"
  local resolution="$3"
  local optimizer_type="$4"
  local antialiasing_specified="$5"
  local antialiasing_value="$6"
  local no_antialiasing="$7"
  local save_name="$8"
  local test_name="$9"
  local checkpoint_name="${10}"
  local densify_until_iter="${11}"
  local densify_grad_threshold="${12}"
  local densification_interval="${13}"
  local opacity_reset_interval="${14}"
  local extra_args_name="${15}"

  local -n save_ref="${save_name}"
  local -n test_ref="${test_name}"
  local -n checkpoint_ref="${checkpoint_name}"
  local -n extra_args_ref="${extra_args_name}"

  get_3dgs_train_profile "${preset}"

  SETTINGS_PRESET="${preset}"
  if (( iterations > 0 )); then
    SETTINGS_ITERATIONS="${iterations}"
  else
    SETTINGS_ITERATIONS="${PROFILE_ITERATIONS}"
  fi

  if (( resolution > 0 )); then
    SETTINGS_RESOLUTION="${resolution}"
  else
    SETTINGS_RESOLUTION="${PROFILE_RESOLUTION}"
  fi

  if [[ "${optimizer_type}" == "profile" ]]; then
    SETTINGS_OPTIMIZER_TYPE="${PROFILE_OPTIMIZER_TYPE}"
  else
    SETTINGS_OPTIMIZER_TYPE="${optimizer_type}"
  fi

  if (( no_antialiasing )); then
    SETTINGS_ANTIALIASING=0
  elif (( antialiasing_specified )); then
    SETTINGS_ANTIALIASING="${antialiasing_value}"
  else
    SETTINGS_ANTIALIASING="${PROFILE_ANTIALIASING}"
  fi

  if (( densify_until_iter > 0 )); then
    SETTINGS_DENSIFY_UNTIL_ITER="${densify_until_iter}"
  else
    SETTINGS_DENSIFY_UNTIL_ITER="${PROFILE_DENSIFY_UNTIL_ITER}"
  fi

  if [[ "${densify_grad_threshold}" != "-1.0" ]] && [[ "${densify_grad_threshold}" != "-1" ]]; then
    SETTINGS_DENSIFY_GRAD_THRESHOLD="${densify_grad_threshold}"
  else
    SETTINGS_DENSIFY_GRAD_THRESHOLD="${PROFILE_DENSIFY_GRAD_THRESHOLD}"
  fi

  if (( densification_interval > 0 )); then
    SETTINGS_DENSIFICATION_INTERVAL="${densification_interval}"
  else
    SETTINGS_DENSIFICATION_INTERVAL="${PROFILE_DENSIFICATION_INTERVAL}"
  fi

  if (( opacity_reset_interval > 0 )); then
    SETTINGS_OPACITY_RESET_INTERVAL="${opacity_reset_interval}"
  else
    SETTINGS_OPACITY_RESET_INTERVAL="${PROFILE_OPACITY_RESET_INTERVAL}"
  fi

  normalize_3dgs_iteration_list save_ref PROFILE_SAVE_ITERATIONS "${SETTINGS_ITERATIONS}" 1 SETTINGS_SAVE_ITERATIONS
  normalize_3dgs_iteration_list test_ref PROFILE_TEST_ITERATIONS "${SETTINGS_ITERATIONS}" 1 SETTINGS_TEST_ITERATIONS
  normalize_3dgs_iteration_list checkpoint_ref PROFILE_CHECKPOINT_ITERATIONS "${SETTINGS_ITERATIONS}" 0 SETTINGS_CHECKPOINT_ITERATIONS

  SETTINGS_EXTRA_ARGS=("${PROFILE_EXTRA_ARGS[@]}")
  if (( ${#extra_args_ref[@]} > 0 )); then
    SETTINGS_EXTRA_ARGS+=("${extra_args_ref[@]}")
  fi
}

test_3dgs_sparse_adam_available() {
  local python_exe="$1"
  "${python_exe}" - <<'PY' >/dev/null 2>&1
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except Exception:
    raise SystemExit(1)
raise SystemExit(0)
PY
}

add_3dgs_train_options() {
  local out_name="$1"
  local start_checkpoint="$2"
  local quiet="$3"
  shift 3
  local -n out_ref="${out_name}"
  local arg

  out_ref=("$@")
  out_ref+=(
    -r "${SETTINGS_RESOLUTION}"
    --iterations "${SETTINGS_ITERATIONS}"
    --optimizer_type "${SETTINGS_OPTIMIZER_TYPE}"
    --test_iterations
  )
  for arg in "${SETTINGS_TEST_ITERATIONS[@]}"; do
    out_ref+=("${arg}")
  done
  out_ref+=(--save_iterations)
  for arg in "${SETTINGS_SAVE_ITERATIONS[@]}"; do
    out_ref+=("${arg}")
  done
  out_ref+=(--disable_viewer)

  if (( SETTINGS_ANTIALIASING )); then
    out_ref+=(--antialiasing)
  fi
  if [[ "${SETTINGS_DENSIFY_UNTIL_ITER}" =~ ^-?[0-9]+$ ]] && (( SETTINGS_DENSIFY_UNTIL_ITER >= 0 )); then
    out_ref+=(--densify_until_iter "${SETTINGS_DENSIFY_UNTIL_ITER}")
  fi
  if awk "BEGIN { exit !(${SETTINGS_DENSIFY_GRAD_THRESHOLD} >= 0) }"; then
    out_ref+=(--densify_grad_threshold "${SETTINGS_DENSIFY_GRAD_THRESHOLD}")
  fi
  if [[ "${SETTINGS_DENSIFICATION_INTERVAL}" =~ ^[0-9]+$ ]] && (( SETTINGS_DENSIFICATION_INTERVAL > 0 )); then
    out_ref+=(--densification_interval "${SETTINGS_DENSIFICATION_INTERVAL}")
  fi
  if [[ "${SETTINGS_OPACITY_RESET_INTERVAL}" =~ ^[0-9]+$ ]] && (( SETTINGS_OPACITY_RESET_INTERVAL > 0 )); then
    out_ref+=(--opacity_reset_interval "${SETTINGS_OPACITY_RESET_INTERVAL}")
  fi
  if (( ${#SETTINGS_CHECKPOINT_ITERATIONS[@]} > 0 )); then
    out_ref+=(--checkpoint_iterations)
    for arg in "${SETTINGS_CHECKPOINT_ITERATIONS[@]}"; do
      out_ref+=("${arg}")
    done
  fi
  if [[ -n "${start_checkpoint}" ]]; then
    out_ref+=(--start_checkpoint "${start_checkpoint}")
  fi
  if (( quiet )); then
    out_ref+=(--quiet)
  fi
  if (( ${#SETTINGS_EXTRA_ARGS[@]} > 0 )); then
    out_ref+=("${SETTINGS_EXTRA_ARGS[@]}")
  fi
}

invoke_3dgs_logged_process() {
  local file_path="$1"
  local log_path="$2"
  shift 2

  mkdir -p "$(dirname "${log_path}")"
  "${file_path}" "$@" >"${log_path}" 2>&1
}
