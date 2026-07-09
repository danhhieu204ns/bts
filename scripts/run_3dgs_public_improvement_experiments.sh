#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/_3dgs_train_profiles.sh"

DATA_ROOT="phase1/public_set"
PREPARED_ROOT="prepared/3dgs_public"
BASE_PRED_DIR="outputs/3dgs_public_quality"
BASELINE_METRICS_CSV="reports/3dgs_public_quality_metrics.csv"
MODEL_ROOT="outputs/3dgs_models_public_bts_targeted"
EXPERIMENT_ROOT="outputs/3dgs_public_improvement"
TRAIN_PRESET="l40s-bts-quality"
LPIPS_MODE="on"
PYTHON_EXE=""
CUDA_VISIBLE_DEVICES_VALUE=""
FORCE=0
SKIP_TRAIN=0
SKIP_RENDER=0
SKIP_EVAL=0
SCENES=("HCM0204" "hcm0031" "hcm0034")
VARIANTS=(
  "aa_on:1.00"
  "aa_on:0.95"
  "aa_off:1.00"
  "aa_off:0.95"
)

usage() {
  cat <<'EOF'
Usage: run_3dgs_public_improvement_experiments.sh [options]
  --data-root PATH
  --prepared-root PATH
  --base-pred-dir PATH
  --baseline-metrics-csv PATH
  --model-root PATH
  --experiment-root PATH
  --train-preset local-r2-7k|l40s-fast|l40s-quality|l40s-bts-quality|custom
  --scene NAME1,NAME2
  --variant aa_on:1.00
  --lpips auto|on|off
  --python PATH
  --cuda-visible-devices IDS
  --skip-train
  --skip-render
  --skip-eval
  --force

Default variants:
  aa_on:1.00
  aa_on:0.95
  aa_off:1.00
  aa_off:0.95
EOF
}

while (($# > 0)); do
  case "$1" in
    --data-root)
      shift
      DATA_ROOT="${1:-}"
      ;;
    --prepared-root)
      shift
      PREPARED_ROOT="${1:-}"
      ;;
    --base-pred-dir)
      shift
      BASE_PRED_DIR="${1:-}"
      ;;
    --baseline-metrics-csv)
      shift
      BASELINE_METRICS_CSV="${1:-}"
      ;;
    --model-root)
      shift
      MODEL_ROOT="${1:-}"
      ;;
    --experiment-root)
      shift
      EXPERIMENT_ROOT="${1:-}"
      ;;
    --train-preset)
      shift
      TRAIN_PRESET="${1:-}"
      ;;
    --scene)
      shift
      [[ $# -gt 0 ]] || die "--scene requires a value"
      SCENES=()
      append_csv_items "$1" SCENES
      ;;
    --variant)
      shift
      [[ $# -gt 0 ]] || die "--variant requires a value"
      if [[ ${#VARIANTS[@]} -eq 4 && "${VARIANTS[0]}" == "aa_on:1.00" && "${VARIANTS[1]}" == "aa_on:0.95" ]]; then
        VARIANTS=()
      fi
      VARIANTS+=("$1")
      ;;
    --lpips)
      shift
      LPIPS_MODE="${1:-}"
      ;;
    --python)
      shift
      PYTHON_EXE="${1:-}"
      ;;
    --cuda-visible-devices)
      shift
      CUDA_VISIBLE_DEVICES_VALUE="${1:-}"
      ;;
    --skip-train)
      SKIP_TRAIN=1
      ;;
    --skip-render)
      SKIP_RENDER=1
      ;;
    --skip-eval)
      SKIP_EVAL=1
      ;;
    --force)
      FORCE=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
  shift
done

[[ -d "${DATA_ROOT}" ]] || die "Data root not found: ${DATA_ROOT}"
[[ -d "${PREPARED_ROOT}" ]] || die "Prepared root not found: ${PREPARED_ROOT}"
[[ -d "${BASE_PRED_DIR}" ]] || die "Base prediction dir not found: ${BASE_PRED_DIR}"
(( ${#SCENES[@]} > 0 )) || die "At least one scene is required"
(( ${#VARIANTS[@]} > 0 )) || die "At least one variant is required"
[[ "${LPIPS_MODE}" == "auto" || "${LPIPS_MODE}" == "on" || "${LPIPS_MODE}" == "off" ]] || die "Invalid --lpips: ${LPIPS_MODE}"

PYTHON_EXE="$(resolve_3dgs_python "${PYTHON_EXE}")"
if [[ -n "${CUDA_VISIBLE_DEVICES_VALUE}" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}"
fi

RENDER_ROOT="${EXPERIMENT_ROOT}/renders"
CANDIDATE_ROOT="${EXPERIMENT_ROOT}/candidates"
REPORT_ROOT="${EXPERIMENT_ROOT}/reports"
mkdir -p "${RENDER_ROOT}" "${CANDIDATE_ROOT}" "${REPORT_ROOT}"

join_csv() {
  local IFS=","
  echo "$*"
}

scale_tag() {
  local scale="$1"
  local tag="${scale/./}"
  printf '%s' "${tag}"
}

variant_name() {
  local aa_spec="$1"
  local scale="$2"
  local aa_tag="aa_off"
  if [[ "${aa_spec}" == "aa_on" ]]; then
    aa_tag="aa_on"
  fi
  printf 'bts_%s_s%s' "${aa_tag}" "$(scale_tag "${scale}")"
}

copy_base_prediction_tree() {
  local candidate_dir="$1"
  rm -rf "${candidate_dir}"
  mkdir -p "${candidate_dir}"
  cp -a "${BASE_PRED_DIR}/." "${candidate_dir}/"
}

overlay_scene_predictions() {
  local scene_render_dir="$1"
  local candidate_dir="$2"
  local scene_name
  for scene_name in "${SCENES[@]}"; do
    rm -rf "${candidate_dir}/${scene_name}"
    mkdir -p "${candidate_dir}"
    cp -a "${scene_render_dir}/${scene_name}" "${candidate_dir}/${scene_name}"
  done
}

evaluate_if_needed() {
  local pred_dir="$1"
  local metrics_csv="$2"
  if (( ! FORCE )) && [[ -f "${metrics_csv}" ]]; then
    echo "SKIP eval: ${metrics_csv} exists"
    return
  fi
  "${SCRIPT_DIR}/evaluate_public.sh" \
    --data-root "${DATA_ROOT}" \
    --pred-dir "${pred_dir}" \
    --lpips "${LPIPS_MODE}" \
    --output-csv "${metrics_csv}" \
    --python "${PYTHON_EXE}"
}

echo "Experiment root: ${EXPERIMENT_ROOT}"
echo "Scenes: $(join_csv "${SCENES[@]}")"
echo "Variants: $(join_csv "${VARIANTS[@]}")"

if (( ! SKIP_TRAIN )); then
  echo "== Train targeted scenes with preset ${TRAIN_PRESET} =="
  train_args=(
    --prepared-root "${PREPARED_ROOT}"
    --model-root "${MODEL_ROOT}"
    --preset "${TRAIN_PRESET}"
    --scene "$(join_csv "${SCENES[@]}")"
    --python "${PYTHON_EXE}"
  )
  if (( FORCE )); then
    train_args+=( --force )
  fi
  if [[ -n "${CUDA_VISIBLE_DEVICES_VALUE}" ]]; then
    train_args+=( --cuda-visible-devices "${CUDA_VISIBLE_DEVICES_VALUE}" )
  fi
  "${SCRIPT_DIR}/train_3dgs_batch.sh" "${train_args[@]}"
fi

if (( ! SKIP_EVAL )); then
  echo "== Ensure baseline metrics exist =="
  if [[ ! -f "${BASELINE_METRICS_CSV}" ]] || (( FORCE )); then
    evaluate_if_needed "${BASE_PRED_DIR}" "${BASELINE_METRICS_CSV}"
  else
    echo "SKIP baseline eval: ${BASELINE_METRICS_CSV} exists"
  fi
fi

for variant in "${VARIANTS[@]}"; do
  IFS=":" read -r aa_spec scale <<<"${variant}"
  [[ -n "${aa_spec:-}" && -n "${scale:-}" ]] || die "Invalid variant: ${variant}. Expected aa_on:1.00"
  [[ "${aa_spec}" == "aa_on" || "${aa_spec}" == "aa_off" ]] || die "Invalid AA spec in variant: ${variant}"

  variant_id="$(variant_name "${aa_spec}" "${scale}")"
  scene_render_dir="${RENDER_ROOT}/${variant_id}"
  candidate_dir="${CANDIDATE_ROOT}/${variant_id}"
  metrics_csv="${REPORT_ROOT}/${variant_id}_metrics.csv"

  if (( ! SKIP_RENDER )); then
    echo "== Render ${variant_id} =="
    rm -rf "${scene_render_dir}"
    mkdir -p "${scene_render_dir}"
    for scene_name in "${SCENES[@]}"; do
      render_args=(
        --data-root "${DATA_ROOT}"
        --model-root "${MODEL_ROOT}"
        --out-dir "${scene_render_dir}"
        --scene "${scene_name}"
        --scaling-modifier "${scale}"
        --python "${PYTHON_EXE}"
      )
      if [[ "${aa_spec}" == "aa_on" ]]; then
        render_args+=( --antialiasing )
      fi
      "${SCRIPT_DIR}/render_3dgs_submission.sh" "${render_args[@]}"
    done
  elif [[ ! -d "${scene_render_dir}" ]]; then
    die "Render dir not found for ${variant_id}: ${scene_render_dir}. Remove --skip-render or generate renders first."
  fi

  echo "== Build candidate ${variant_id} =="
  copy_base_prediction_tree "${candidate_dir}"
  overlay_scene_predictions "${scene_render_dir}" "${candidate_dir}"

  if (( ! SKIP_EVAL )); then
    echo "== Evaluate ${variant_id} =="
    evaluate_if_needed "${candidate_dir}" "${metrics_csv}"
  fi
done

if (( ! SKIP_EVAL )); then
  SUMMARY_CSV="${REPORT_ROOT}/summary.csv"
  SCENE_SUMMARY_CSV="${REPORT_ROOT}/scene_summary.csv"
  SUMMARY_MD="${REPORT_ROOT}/summary.md"

  "${PYTHON_EXE}" - <<'PY' "${BASELINE_METRICS_CSV}" "${REPORT_ROOT}" "${SUMMARY_CSV}" "${SCENE_SUMMARY_CSV}" "${SUMMARY_MD}" "$(join_csv "${SCENES[@]}")"
import csv
import math
import sys
from pathlib import Path

baseline_csv = Path(sys.argv[1])
report_root = Path(sys.argv[2])
summary_csv = Path(sys.argv[3])
scene_summary_csv = Path(sys.argv[4])
summary_md = Path(sys.argv[5])
target_scenes = [item for item in sys.argv[6].split(",") if item]


def load_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as fid:
        return list(csv.DictReader(fid))


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def aggregate(rows):
    by_scene = {}
    for row in rows:
      scene = row["scene"]
      by_scene.setdefault(scene, {"psnr": [], "ssim": [], "lpips": []})
      by_scene[scene]["psnr"].append(float(row["psnr"]))
      if row["ssim"]:
        by_scene[scene]["ssim"].append(float(row["ssim"]))
      if row["lpips"]:
        by_scene[scene]["lpips"].append(float(row["lpips"]))

    scene_summary = {}
    for scene, metrics in by_scene.items():
      psnr = mean(metrics["psnr"])
      ssim = mean(metrics["ssim"])
      lpips = mean(metrics["lpips"])
      score35 = float("nan")
      if not any(math.isnan(v) for v in (psnr, ssim, lpips)):
        score35 = 0.4 * (1.0 - lpips) + 0.3 * ssim + 0.3 * min(max(psnr / 35.0, 0.0), 1.0)
      scene_summary[scene] = {"psnr": psnr, "ssim": ssim, "lpips": lpips, "score35": score35}

    overall = {
      "psnr": mean([v["psnr"] for v in scene_summary.values()]),
      "ssim": mean([v["ssim"] for v in scene_summary.values()]),
      "lpips": mean([v["lpips"] for v in scene_summary.values()]),
      "score35": mean([v["score35"] for v in scene_summary.values()]),
    }
    target_overall = {
      "psnr": mean([scene_summary[s]["psnr"] for s in target_scenes if s in scene_summary]),
      "ssim": mean([scene_summary[s]["ssim"] for s in target_scenes if s in scene_summary]),
      "lpips": mean([scene_summary[s]["lpips"] for s in target_scenes if s in scene_summary]),
      "score35": mean([scene_summary[s]["score35"] for s in target_scenes if s in scene_summary]),
    }
    return scene_summary, overall, target_overall


baseline_rows = load_rows(baseline_csv)
baseline_scene, baseline_overall, baseline_target = aggregate(baseline_rows)
variant_paths = sorted(path for path in report_root.glob("bts_*_metrics.csv") if path.is_file())

summary_rows = []
scene_rows = []
for path in variant_paths:
    variant_rows = load_rows(path)
    variant_scene, variant_overall, variant_target = aggregate(variant_rows)
    variant_name = path.name.removesuffix("_metrics.csv")
    summary_rows.append(
        {
            "variant": variant_name,
            "overall_psnr": variant_overall["psnr"],
            "overall_ssim": variant_overall["ssim"],
            "overall_lpips": variant_overall["lpips"],
            "overall_score35": variant_overall["score35"],
            "delta_overall_psnr": variant_overall["psnr"] - baseline_overall["psnr"],
            "delta_overall_ssim": variant_overall["ssim"] - baseline_overall["ssim"],
            "delta_overall_lpips": variant_overall["lpips"] - baseline_overall["lpips"],
            "delta_overall_score35": variant_overall["score35"] - baseline_overall["score35"],
            "target_psnr": variant_target["psnr"],
            "target_ssim": variant_target["ssim"],
            "target_lpips": variant_target["lpips"],
            "target_score35": variant_target["score35"],
            "delta_target_psnr": variant_target["psnr"] - baseline_target["psnr"],
            "delta_target_ssim": variant_target["ssim"] - baseline_target["ssim"],
            "delta_target_lpips": variant_target["lpips"] - baseline_target["lpips"],
            "delta_target_score35": variant_target["score35"] - baseline_target["score35"],
        }
    )
    for scene in sorted(target_scenes):
        if scene not in variant_scene or scene not in baseline_scene:
            continue
        scene_rows.append(
            {
                "variant": variant_name,
                "scene": scene,
                "psnr": variant_scene[scene]["psnr"],
                "ssim": variant_scene[scene]["ssim"],
                "lpips": variant_scene[scene]["lpips"],
                "score35": variant_scene[scene]["score35"],
                "delta_psnr": variant_scene[scene]["psnr"] - baseline_scene[scene]["psnr"],
                "delta_ssim": variant_scene[scene]["ssim"] - baseline_scene[scene]["ssim"],
                "delta_lpips": variant_scene[scene]["lpips"] - baseline_scene[scene]["lpips"],
                "delta_score35": variant_scene[scene]["score35"] - baseline_scene[scene]["score35"],
            }
        )

summary_rows.sort(key=lambda item: item["delta_overall_score35"], reverse=True)

summary_csv.parent.mkdir(parents=True, exist_ok=True)
with summary_csv.open("w", encoding="utf-8", newline="") as fid:
    writer = csv.DictWriter(fid, fieldnames=list(summary_rows[0].keys()) if summary_rows else [])
    if summary_rows:
        writer.writeheader()
        writer.writerows(summary_rows)

with scene_summary_csv.open("w", encoding="utf-8", newline="") as fid:
    writer = csv.DictWriter(fid, fieldnames=list(scene_rows[0].keys()) if scene_rows else [])
    if scene_rows:
        writer.writeheader()
        writer.writerows(scene_rows)

lines = []
lines.append("# 3DGS Public Improvement Summary")
lines.append("")
lines.append(f"Baseline metrics: `{baseline_csv}`")
lines.append("")
if summary_rows:
    best = summary_rows[0]
    lines.append(
        "Best overall candidate: "
        f"`{best['variant']}` | delta Score@35={best['delta_overall_score35']:+.6f} | "
        f"delta PSNR={best['delta_overall_psnr']:+.4f}"
    )
    lines.append("")
    lines.append("| Variant | dScore@35 | dPSNR | dSSIM | dLPIPS | Target dScore@35 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for item in summary_rows:
        lines.append(
            f"| `{item['variant']}` | {item['delta_overall_score35']:+.6f} | "
            f"{item['delta_overall_psnr']:+.4f} | {item['delta_overall_ssim']:+.4f} | "
            f"{item['delta_overall_lpips']:+.4f} | {item['delta_target_score35']:+.6f} |"
        )
else:
    lines.append("No variant metrics found.")

summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Wrote {summary_csv}")
print(f"Wrote {scene_summary_csv}")
print(f"Wrote {summary_md}")
PY

  echo "== Summary files =="
  echo "${SUMMARY_CSV}"
  echo "${SCENE_SUMMARY_CSV}"
  echo "${SUMMARY_MD}"
fi
