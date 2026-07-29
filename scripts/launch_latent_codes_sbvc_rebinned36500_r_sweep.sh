#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SBVC_REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PUT_ROOT="${REPO_ROOT}/third_party/PUT"
MANIFEST="${REPO_ROOT}/exp_data/01_main_256_places2_rebinned36500_full/sample_manifest.csv"
OUT_ROOT="${REPO_ROOT}/outputs/places2_fv_reproduction/rebinned36500_full"
LOG_ROOT="${OUT_ROOT}/latent_codes_sbvc_r_sweep_logs"
WAIT_PID_FILE="${REPO_ROOT}/exp_data/02_r_sweep_rebinned36500_full/launcher.pid"
GPU="${1:-0}"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S %Z"
}

wait_for_pid_file() {
  if [[ ! -f "${WAIT_PID_FILE}" ]]; then
    echo "[$(timestamp)] WAIT_PID_FILE missing, continue immediately: ${WAIT_PID_FILE}"
    return 0
  fi
  local wait_pid
  wait_pid="$(cat "${WAIT_PID_FILE}")"
  if [[ -z "${wait_pid}" ]]; then
    echo "[$(timestamp)] WAIT_PID_FILE empty, continue immediately: ${WAIT_PID_FILE}"
    return 0
  fi
  echo "[$(timestamp)] Waiting for upstream PUT R-sweep PID ${wait_pid}"
  while kill -0 "${wait_pid}" >/dev/null 2>&1; do
    sleep 300
  done
  echo "[$(timestamp)] Upstream PUT R-sweep PID ${wait_pid} finished"
}

mkdir -p "${LOG_ROOT}"

echo "[$(timestamp)] START Latent Codes-SBVC full36500 R-sweep queue"
echo "MANIFEST=${MANIFEST}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "GPU=${GPU}"
echo "R_VALUES=32 48 80"
echo "NOTE=r64 already exists and is reused as anchor"

wait_for_pid_file

for R in 32 48 80; do
  RUN_DIR="${OUT_ROOT}/LatentCodes_SBVC_qkv_r${R}_last20"
  EVAL_DIR="${OUT_ROOT}/LatentCodes_SBVC_qkv_r${R}_last20_eval"
  REPORT_MD="${OUT_ROOT}/LATENT_CODES_SBVC_QKV_R${R}_LAST20_FULL36500_REPORT.md"
  LOG_PATH="${LOG_ROOT}/qkv_r${R}_last20.log"

  {
    echo "[$(timestamp)] START qkv_r${R}_last20"
    echo "RUN_DIR=${RUN_DIR}"
    echo "EVAL_DIR=${EVAL_DIR}"
    echo "REPORT_MD=${REPORT_MD}"
  } | tee -a "${LOG_PATH}"

  env PYTHONUNBUFFERED=1 PYTHONPATH="${REPO_ROOT}/tools" \
    LATENT_SBVC_ENABLE=1 \
    LATENT_SBVC_R="${R}" \
    LATENT_SBVC_MODE=qkv \
    LATENT_SBVC_LAYER_IDS=20-39 \
    LATENT_SBVC_ROUTE_CACHE=1 \
    LATENT_SBVC_COLLECT_STATS=1 \
    LATENT_SBVC_PROFILE=0 \
    "${PYTHON_BIN}" "${REPO_ROOT}/tools/baseline_adapters/run_latent_codes_places2_fv.py" \
      --manifest "${MANIFEST}" \
      --output-dir "${RUN_DIR}" \
      --seed 20260502 \
      --gpu "${GPU}" \
      2>&1 | tee -a "${LOG_PATH}"

  echo "[$(timestamp)] INFERENCE_DONE qkv_r${R}_last20" | tee -a "${LOG_PATH}"

  env PYTHONUNBUFFERED=1 PYTHONPATH="${REPO_ROOT}/tools:${PUT_ROOT}" \
    "${PYTHON_BIN}" "${REPO_ROOT}/tools/baseline_adapters/eval_places2_fv_outputs.py" \
      --manifest "${MANIFEST}" \
      --completed-dir "${RUN_DIR}/completed_single" \
      --output-dir "${EVAL_DIR}" \
      --gpu "${GPU}" \
      --fid-num-workers 8 \
      2>&1 | tee -a "${LOG_PATH}"

  echo "[$(timestamp)] EVAL_DONE qkv_r${R}_last20" | tee -a "${LOG_PATH}"

  "${PYTHON_BIN}" "${REPO_ROOT}/tools/summarize_latent_codes_sbvc_full_eval.py" \
    --manifest "${MANIFEST}" \
    --run-dir "${RUN_DIR}" \
    --eval-dir "${EVAL_DIR}" \
    --output-md "${REPORT_MD}" \
    --config-id "qkv_r${R}_last20_route_cache_on" \
    2>&1 | tee -a "${LOG_PATH}"

  echo "[$(timestamp)] REPORT_DONE ${REPORT_MD}" | tee -a "${LOG_PATH}"
done

echo "[$(timestamp)] DONE Latent Codes-SBVC full36500 R-sweep queue"
