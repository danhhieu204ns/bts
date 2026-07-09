#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/_3dgs_train_profiles.sh"

DATA_ROOTS=("phase1/public_set" "phase1/private_set1")
OUTPUT="reports/data_report.md"
PYTHON_EXE=""

while (($# > 0)); do
  case "$1" in
    --data-root)
      shift
      [[ $# -gt 0 ]] || die "--data-root requires a value"
      DATA_ROOTS=()
      append_csv_items "$1" DATA_ROOTS
      ;;
    --output)
      shift
      OUTPUT="${1:-}"
      ;;
    --python)
      shift
      PYTHON_EXE="${1:-}"
      ;;
    -h|--help)
      cat <<'EOF'
Usage: analyze_data.sh [options]
  --data-root phase1/public_set,phase1/private_set1
  --output reports/data_report.md
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

PYTHON_EXE="$(resolve_3dgs_python "${PYTHON_EXE}")"
export PYTHONPATH="${REPO_ROOT}/src"

"${PYTHON_EXE}" -m bts_baseline.analyze_data --data-root "${DATA_ROOTS[@]}" --output "${OUTPUT}"
