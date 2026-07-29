#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SBVC_REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PUT_ROOT="${REPO_ROOT}/third_party/PUT"
CHECKPOINT="${PUT_CHECKPOINT:?Set PUT_CHECKPOINT to the official PUT transformer checkpoint}"
IMAGE_ROOT="${SBVC_IMAGE_ROOT:?Set SBVC_IMAGE_ROOT to the Places365 validation image root}"
MASK_ROOT="${SBVC_MASK_ROOT:?Set SBVC_MASK_ROOT to the evaluation mask root}"
SUBSET_DIR="${REPO_ROOT}/exp_data/01_main_256_places2_rebinned36500_full"
OUT_ROOT="${REPO_ROOT}/exp_data/02_r_sweep_rebinned36500_full"
GPU="${1:-0}"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S %Z"
}

mkdir -p "${OUT_ROOT}/logs"

echo "[$(timestamp)] START PUT-SBVC full36500 R-sweep"
echo "SUBSET_DIR=${SUBSET_DIR}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "GPU=${GPU}"

for R in 128 160 192 256; do
  OUT_DIR="${OUT_ROOT}/sbvc_r${R}"
  LOG_PATH="${OUT_ROOT}/logs/sbvc_r${R}.log"
  mkdir -p "${OUT_DIR}"
  {
    echo "[$(timestamp)] START sbvc_r${R}"
    echo "OUT_DIR=${OUT_DIR}"
  } | tee -a "${LOG_PATH}"

  env PYTHONUNBUFFERED=1 PYTHONPATH="${REPO_ROOT}/tools:${PUT_ROOT}" \
    "${PYTHON_BIN}" "${REPO_ROOT}/tools/phase7_main256_rebinned_protocol_eval.py" \
      --put-root "${PUT_ROOT}" \
      --checkpoint "${CHECKPOINT}" \
      --image-root "${IMAGE_ROOT}" \
      --mask-root "${MASK_ROOT}" \
      --output-dir "${OUT_DIR}" \
      --reuse-existing-subset-dir "${SUBSET_DIR}" \
      --dataset-name "Places2/NaturalScene rebinned36500_full PUT-SBVC r${R}" \
      --validation-source-label "rebinned36500_full manifest SHA 679e2d0b8759241e3e07ccb9a555606f77048fb294bbd34cae268fce57bd59b9" \
      --image-selection-mode recursive_image_root_scan \
      --mask-selection-mode deterministic_cycle \
      --count-01-10 6084 \
      --count-10-20 6083 \
      --count-20-30 6083 \
      --count-30-40 6083 \
      --count-40-50 6084 \
      --count-50-60 6083 \
      --methods sbvc_r224_zero_pad_fastpath \
      --safe-tome-r "${R}" \
      --global-tome-r 224 \
      --seed 20260502 \
      --gpu "${GPU}" \
      --fid-batch-size 64 \
      --fid-num-workers 4 \
      --resume-existing \
      2>&1 | tee -a "${LOG_PATH}"

  echo "[$(timestamp)] DONE sbvc_r${R}" | tee -a "${LOG_PATH}"
done

echo "[$(timestamp)] DONE PUT-SBVC full36500 R-sweep"
