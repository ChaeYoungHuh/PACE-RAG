#!/bin/bash
set -euo pipefail

# MIMIC Enhanced RAG runner (clean standalone copy)
# Usage:
#   bash script/run_mimic_enhanced_rag.sh [temperature]
#
# Optional environment variables:
#   LLM_MODEL=qwen3:8b
#   SEED=42
#   RETRIEVE_PATIENTS=7
#   THRESHOLD=0.9
#   MAX_SAMPLES=5
#   DATA_PATH=/path/to/mimic_test.csv
#   VECTOR_DB_PATH=/path/to/mimic_train_faiss_index
#   PYTHON_BIN=python

TEMP="${1:-0.0}"
ROOT_DIR="$(cd -- "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

LLM_MODEL="${LLM_MODEL:-qwen3:8b}"
SEED="${SEED:-42}"
RETRIEVE_PATIENTS="${RETRIEVE_PATIENTS:-7}"
THRESHOLD="${THRESHOLD:-0.9}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11436}"

DATA_PATH="${DATA_PATH:-${ROOT_DIR}/data/mimic_test.csv}"
VECTOR_DB_PATH="${VECTOR_DB_PATH:-${ROOT_DIR}/vector_db/mimic_train_faiss_index}"

OUT_DIR="${ROOT_DIR}/output/one_step_rag"
LOG_DIR="${ROOT_DIR}/output/logs"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

TS="$(date +%Y%m%d_%H%M%S)"
TEMP_TAG="${TEMP//./p}"
BASE_NAME="mimic_enhanced_rag_temp${TEMP_TAG}_thr${THRESHOLD}_rp${RETRIEVE_PATIENTS}_${TS}"
OUTPUT_FILE="${OUT_DIR}/${BASE_NAME}.json"
LOG_FILE="${LOG_DIR}/${BASE_NAME}.log"

echo "========================================"
echo "Run MIMIC Enhanced RAG"
echo "Model      : ${LLM_MODEL}"
echo "Temperature: ${TEMP}"
echo "Seed       : ${SEED}"
echo "Data       : ${DATA_PATH}"
echo "Vector DB  : ${VECTOR_DB_PATH}"
echo "Max samples: ${MAX_SAMPLES}"
echo "OLLAMA_HOST: ${OLLAMA_HOST}"
echo "Output     : ${OUTPUT_FILE}"
echo "Log        : ${LOG_FILE}"
echo "========================================"

cd "${ROOT_DIR}"

OLLAMA_HOST="${OLLAMA_HOST}" "${PYTHON_BIN}" ./main.py \
  --data_path "${DATA_PATH}" \
  --vector_db "${VECTOR_DB_PATH}" \
  --output_file "${OUTPUT_FILE}" \
  --llm_model "${LLM_MODEL}" \
  --temperature "${TEMP}" \
  --seed "${SEED}" \
  --retrieve_patients "${RETRIEVE_PATIENTS}" \
  --threshold "${THRESHOLD}" \
  --max_samples "${MAX_SAMPLES}" \
  > "${LOG_FILE}" 2>&1 &

echo "Started PID: $!"
echo "Tail logs: tail -f ${LOG_FILE}"
