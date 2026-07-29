#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SBVC_REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PUT_ROOT="${REPO_ROOT}/third_party/PUT"

MANIFEST="${REPO_ROOT}/exp_data/01_main_256_places2_rebinned36500_full/sample_manifest.csv"
OUT_DIR="${REPO_ROOT}/outputs/places2_fv_reproduction/rebinned36500_full/LatentCodes_SBVC_qkv_r64_last20"
EVAL_DIR="${REPO_ROOT}/outputs/places2_fv_reproduction/rebinned36500_full/LatentCodes_SBVC_qkv_r64_last20_eval"
REPORT_MD="${REPO_ROOT}/outputs/places2_fv_reproduction/rebinned36500_full/LATENT_CODES_SBVC_QKV_R64_LAST20_FULL36500_REPORT.md"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S %Z"
}

echo "[$(timestamp)] START Latent Codes-SBVC qkv r64 last20 full rebinned36500"
echo "MANIFEST=${MANIFEST}"
echo "OUT_DIR=${OUT_DIR}"
echo "EVAL_DIR=${EVAL_DIR}"

env PYTHONUNBUFFERED=1 PYTHONPATH="${REPO_ROOT}/tools" \
  LATENT_SBVC_ENABLE=1 \
  LATENT_SBVC_R=64 \
  LATENT_SBVC_MODE=qkv \
  LATENT_SBVC_LAYER_IDS=20-39 \
  LATENT_SBVC_ROUTE_CACHE=1 \
  LATENT_SBVC_COLLECT_STATS=1 \
  LATENT_SBVC_PROFILE=0 \
  "${PYTHON_BIN}" "${REPO_ROOT}/tools/baseline_adapters/run_latent_codes_places2_fv.py" \
    --manifest "${MANIFEST}" \
    --output-dir "${OUT_DIR}" \
    --seed 20260502 \
    --gpu 0

echo "[$(timestamp)] INFERENCE_DONE"

env PYTHONUNBUFFERED=1 PYTHONPATH="${REPO_ROOT}/tools:${PUT_ROOT}" \
  "${PYTHON_BIN}" "${REPO_ROOT}/tools/baseline_adapters/eval_places2_fv_outputs.py" \
    --manifest "${MANIFEST}" \
    --completed-dir "${OUT_DIR}/completed_single" \
    --output-dir "${EVAL_DIR}" \
    --gpu 0 \
    --fid-num-workers 8

echo "[$(timestamp)] EVAL_DONE"

"${PYTHON_BIN}" "${REPO_ROOT}/tools/summarize_latent_codes_sbvc_full_eval.py" \
  --manifest "${MANIFEST}" \
  --run-dir "${OUT_DIR}" \
  --eval-dir "${EVAL_DIR}" \
  --output-md "${REPORT_MD}" \
  --config-id "qkv_r64_last20_route_cache_on"

echo "[$(timestamp)] REPORT_DONE ${REPORT_MD}"
