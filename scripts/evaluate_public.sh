#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/_3dgs_train_profiles.sh"

DATA_ROOTS=()
PRED_DIR=""
LPIPS_MODE="auto"
LPIPS_NET="alex"
DEVICE="auto"
PSNR_MAX=(30.0 35.0 40.0)
OUTPUT_CSV=""
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
    --lpips)
      shift
      LPIPS_MODE="${1:-}"
      ;;
    --lpips-net)
      shift
      LPIPS_NET="${1:-}"
      ;;
    --device)
      shift
      DEVICE="${1:-}"
      ;;
    --psnr-max)
      shift
      [[ $# -gt 0 ]] || die "--psnr-max requires a value"
      PSNR_MAX=()
      append_csv_items "$1" PSNR_MAX
      ;;
    --output-csv)
      shift
      OUTPUT_CSV="${1:-}"
      ;;
    --python)
      shift
      PYTHON_EXE="${1:-}"
      ;;
    -h|--help)
      cat <<'EOF'
Usage: evaluate_public.sh --data-root phase1/public_set --pred-dir outputs/render [options]
  --lpips auto|on|off
  --lpips-net alex|vgg|squeeze
  --device auto|cuda|cpu
  --psnr-max 30,35,40
  --output-csv PATH
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

args=( -m bts_baseline.evaluate_public --data-root )
args+=( "${DATA_ROOTS[@]}" )
args+=( --pred-dir "${PRED_DIR}" --lpips "${LPIPS_MODE}" --lpips-net "${LPIPS_NET}" --device "${DEVICE}" --psnr-max )
args+=( "${PSNR_MAX[@]}" )
if [[ -n "${OUTPUT_CSV}" ]]; then
  args+=( --output-csv "${OUTPUT_CSV}" )
fi

"${PYTHON_EXE}" "${args[@]}"
