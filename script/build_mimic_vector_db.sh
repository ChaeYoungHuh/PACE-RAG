#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

INPUT_CSV="${1:-${ROOT_DIR}/data/mimic_train.csv}"
OUTPUT_DIR="${2:-${ROOT_DIR}/vector_db/mimic_train_faiss_index}"

echo "========================================"
echo "Building MIMIC vector DB"
echo "Input : ${INPUT_CSV}"
echo "Output: ${OUTPUT_DIR}"
echo "Python: ${PYTHON_BIN}"
echo "========================================"

"${PYTHON_BIN}" "${ROOT_DIR}/script/build_mimic_vector_db.py" \
  --input_csv "${INPUT_CSV}" \
  --output_dir "${OUTPUT_DIR}"

echo "Done."
