#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/_3dgs_train_profiles.sh"

PYTHON_EXE=""
TORCH_CUDA_ARCH_LIST_VALUE="${TORCH_CUDA_ARCH_LIST:-}"
CLEAN=0
FORCE_REINSTALL=0

while (($# > 0)); do
  case "$1" in
    --python)
      shift
      PYTHON_EXE="${1:-}"
      ;;
    --torch-cuda-arch-list)
      shift
      TORCH_CUDA_ARCH_LIST_VALUE="${1:-}"
      ;;
    --clean)
      CLEAN=1
      ;;
    --force-reinstall)
      FORCE_REINSTALL=1
      ;;
    -h|--help)
      cat <<'EOF'
Usage: build_3dgs_extensions.sh [options]
  --python PATH
  --torch-cuda-arch-list 8.9
  --clean
  --force-reinstall
EOF
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
  shift
done

PYTHON_EXE="$(resolve_3dgs_python "${PYTHON_EXE}")"
GS_ROOT="${REPO_ROOT}/external/gaussian-splatting"
[[ -d "${GS_ROOT}" ]] || die "official 3DGS repo not found: ${GS_ROOT}"

if [[ -n "${TORCH_CUDA_ARCH_LIST_VALUE}" ]]; then
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST_VALUE}"
fi

export PYTHONPATH="${REPO_ROOT}/src"

echo "Using Python: ${PYTHON_EXE}"
if [[ -n "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
  echo "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
fi

"${PYTHON_EXE}" - <<'PY'
import torch
print("torch", torch.__version__)
print("torch cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
PY

EXTENSIONS=(
  "submodules/simple-knn"
  "submodules/diff-gaussian-rasterization"
  "submodules/fused-ssim"
)

for extension in "${EXTENSIONS[@]}"; do
  EXT_PATH="${GS_ROOT}/${extension}"
  [[ -d "${EXT_PATH}" ]] || die "Extension path not found: ${EXT_PATH}"

  echo
  echo "Building ${extension}"

  if (( CLEAN )); then
    rm -rf "${EXT_PATH}/build"
    find "${EXT_PATH}" -maxdepth 1 -type d -name '*.egg-info' -exec rm -rf {} +
  fi

  PIP_ARGS=( -m pip install --no-build-isolation --no-cache-dir -v )
  if (( FORCE_REINSTALL )); then
    PIP_ARGS+=( --force-reinstall )
  fi
  PIP_ARGS+=( -e "${EXT_PATH}" )

  "${PYTHON_EXE}" "${PIP_ARGS[@]}"
done
