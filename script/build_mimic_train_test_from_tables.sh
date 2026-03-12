#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

# Expected raw table files (place them under data/raw by default)
ADMISSIONS_CSV="${1:-${ROOT_DIR}/data/raw/admissions.csv}"
D_ICD_DIAGNOSES_CSV="${2:-${ROOT_DIR}/data/raw/d_icd_diagnoses.csv}"
# Accept both diagnosis_icd.csv and diagnoses_icd.csv naming variants.
if [ -n "${3:-}" ]; then
  DIAGNOSIS_ICD_CSV="$3"
else
  if [ -f "${ROOT_DIR}/data/raw/diagnosis_icd.csv" ]; then
    DIAGNOSIS_ICD_CSV="${ROOT_DIR}/data/raw/diagnosis_icd.csv"
  else
    DIAGNOSIS_ICD_CSV="${ROOT_DIR}/data/raw/diagnoses_icd.csv"
  fi
fi
DEMO_ATC_MAPPING_CSV="${4:-${ROOT_DIR}/data/raw/demo_atc_mapping.csv}"
PRESCRIPTIONS_CSV="${5:-${ROOT_DIR}/data/raw/prescriptions.csv}"

TARGET_TEST_VISITS="${TARGET_TEST_VISITS:-1000}"
SEED="${SEED:-42}"

# Debug option:
# MED_ROWS_LIMIT=20 bash script/build_mimic_train_test_from_tables.sh
MED_ROWS_LIMIT="${MED_ROWS_LIMIT:-0}"
ADMISSIONS_ROWS_LIMIT="${ADMISSIONS_ROWS_LIMIT:-0}"

echo "========================================"
echo "Build MIMIC train/test from raw tables"
echo "admissions       : ${ADMISSIONS_CSV}"
echo "d_icd_diagnoses  : ${D_ICD_DIAGNOSES_CSV}"
echo "diagnosis_icd    : ${DIAGNOSIS_ICD_CSV}"
echo "demo_atc_mapping : ${DEMO_ATC_MAPPING_CSV}"
echo "prescriptions    : ${PRESCRIPTIONS_CSV}"
echo "target test visits: ${TARGET_TEST_VISITS}"
echo "seed              : ${SEED}"
echo "med rows limit    : ${MED_ROWS_LIMIT}"
echo "admissions limit  : ${ADMISSIONS_ROWS_LIMIT}"
echo "========================================"

"${PYTHON_BIN}" "${ROOT_DIR}/script/build_mimic_train_test_from_tables.py" \
  --admissions_csv "${ADMISSIONS_CSV}" \
  --diagnosis_icd_csv "${DIAGNOSIS_ICD_CSV}" \
  --d_icd_diagnoses_csv "${D_ICD_DIAGNOSES_CSV}" \
  --prescriptions_csv "${PRESCRIPTIONS_CSV}" \
  --demo_atc_mapping_csv "${DEMO_ATC_MAPPING_CSV}" \
  --output_full_csv "${ROOT_DIR}/data/mimic_full_admissions.csv" \
  --output_train_csv "${ROOT_DIR}/data/mimic_train.csv" \
  --output_test_csv "${ROOT_DIR}/data/mimic_test.csv" \
  --target_test_visits "${TARGET_TEST_VISITS}" \
  --seed "${SEED}" \
  --med_rows_limit "${MED_ROWS_LIMIT}" \
  --admissions_rows_limit "${ADMISSIONS_ROWS_LIMIT}"

echo "Done."
