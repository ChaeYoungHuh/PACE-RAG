#!/usr/bin/env python3
"""
Build MIMIC train/test admission-level datasets from raw tables.

Required input tables:
- admissions
- diagnosis_icd (or diagnoses_icd)
- d_icd_diagnoses
- prescriptions
- demo_atc_mapping (NDC -> ATC name)

Outputs:
- data/mimic_train.csv          (retrieval pool)
- data/mimic_test.csv           (evaluation set)

Split policy:
- Patient-level split (subject_id based)
- Random shuffle with seed
- Add patients to test until total test visits >= target_test_visits
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _pick_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    if required:
        raise ValueError(f"Missing required column. candidates={candidates}, available={sorted(cols)}")
    return None


def _dedup_keep_order(values):
    seen = set()
    out = []
    for v in values:
        s = str(v).strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _build_diagnosis_table(diagnosis_icd: pd.DataFrame, d_icd_diagnoses: pd.DataFrame) -> pd.DataFrame:
    """Join diagnosis codes with long titles and aggregate per admission."""
    dx = _normalize_cols(diagnosis_icd)
    dx_ref = _normalize_cols(d_icd_diagnoses)

    sub_col = _pick_col(dx, ["subject_id"])
    hadm_col = _pick_col(dx, ["hadm_id"])
    seq_col = _pick_col(dx, ["seq_num", "seqno"], required=False)
    code_col = _pick_col(dx, ["icd_code", "icd9_code"])
    ver_col = _pick_col(dx, ["icd_version"], required=False)

    ref_code_col = _pick_col(dx_ref, ["icd_code", "icd9_code"])
    ref_ver_col = _pick_col(dx_ref, ["icd_version"], required=False)
    title_col = _pick_col(dx_ref, ["long_title", "icd_name", "diagnosis_name"])

    join_left = [code_col]
    join_right = [ref_code_col]
    if ver_col and ref_ver_col:
        join_left.append(ver_col)
        join_right.append(ref_ver_col)

    merged = dx.merge(
        dx_ref[[*join_right, title_col]].drop_duplicates(),
        left_on=join_left,
        right_on=join_right,
        how="left",
    )
    merged["diag_text"] = merged[title_col].fillna(merged[code_col]).astype(str)

    sort_cols = [sub_col, hadm_col]
    if seq_col:
        sort_cols.append(seq_col)
    merged = merged.sort_values(sort_cols)

    agg = (
        merged.groupby([sub_col, hadm_col], as_index=False)["diag_text"]
        .apply(lambda s: _dedup_keep_order(s.tolist()))
        .rename(columns={"diag_text": "diagnoses"})
    )
    agg = agg.rename(columns={sub_col: "subject_id", hadm_col: "hadm_id"})
    return agg


def _normalize_ndc(value) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _build_medication_table(
    prescriptions: pd.DataFrame,
    demo_atc_mapping: pd.DataFrame,
    med_rows_limit: int,
) -> pd.DataFrame:
    """
    Build admission-level medication labels using:
    prescriptions.ndc + demo_atc_mapping(ndc -> atc_name).
    """
    rx = _normalize_cols(prescriptions)
    mapping = _normalize_cols(demo_atc_mapping)

    if med_rows_limit and med_rows_limit > 0:
        rx = rx.head(med_rows_limit).copy()

    rx_sub_col = _pick_col(rx, ["subject_id"])
    rx_hadm_col = _pick_col(rx, ["hadm_id"])
    rx_ndc_col = _pick_col(rx, ["ndc"])

    map_ndc_col = _pick_col(mapping, ["ndc"])
    map_atc_name_col = _pick_col(
        mapping,
        ["atc_name", "atc_class_name", "atc_class", "atc_description", "atc_code"],
    )

    rx["ndc_norm"] = rx[rx_ndc_col].apply(_normalize_ndc)
    mapping["ndc_norm"] = mapping[map_ndc_col].apply(_normalize_ndc)
    mapping["atc_text"] = mapping[map_atc_name_col].astype(str).str.strip()
    mapping = mapping[mapping["atc_text"].ne("") & mapping["atc_text"].ne("nan")].copy()
    mapping = mapping[["ndc_norm", "atc_text"]].drop_duplicates()

    joined = rx.merge(mapping, on="ndc_norm", how="left")
    joined = joined[joined["atc_text"].notna()].copy()

    if joined.empty:
        print(
            "[WARN] No prescription rows were mapped to ATC names. "
            "Check whether NDC formats match between prescriptions and demo_atc_mapping."
        )
        return pd.DataFrame(columns=["subject_id", "hadm_id", "medications"])

    agg = (
        joined.groupby([rx_sub_col, rx_hadm_col], as_index=False)["atc_text"]
        .apply(lambda s: _dedup_keep_order(s.tolist()))
        .rename(columns={"atc_text": "medications"})
    )
    agg = agg.rename(columns={rx_sub_col: "subject_id", rx_hadm_col: "hadm_id"})
    return agg


def build_dataset(
    admissions_path: Path,
    diagnosis_icd_path: Path,
    d_icd_diagnoses_path: Path,
    prescriptions_path: Path,
    demo_atc_mapping_path: Path,
    med_rows_limit: int,
    admissions_rows_limit: int,
) -> pd.DataFrame:
    admissions = _normalize_cols(pd.read_csv(admissions_path))
    diagnosis_icd = _normalize_cols(pd.read_csv(diagnosis_icd_path))
    d_icd_diagnoses = pd.read_csv(d_icd_diagnoses_path)
    prescriptions = _normalize_cols(pd.read_csv(prescriptions_path))
    demo_atc_mapping = _normalize_cols(pd.read_csv(demo_atc_mapping_path))

    adm_sub = _pick_col(admissions, ["subject_id"])
    adm_hadm = _pick_col(admissions, ["hadm_id"])
    adm_time = _pick_col(admissions, ["admittime", "admit_time"])

    adm = admissions[[adm_sub, adm_hadm, adm_time]].drop_duplicates().copy()
    adm = adm.rename(columns={adm_sub: "subject_id", adm_hadm: "hadm_id", adm_time: "admittime"})
    if admissions_rows_limit and admissions_rows_limit > 0:
        adm = adm.head(admissions_rows_limit).copy()

    # For fast smoke tests, keep diagnosis/medication tables only for selected admissions.
    selected_hadm_ids = set(adm["hadm_id"].astype(str).tolist())
    dx_hadm_col = _pick_col(diagnosis_icd, ["hadm_id"])
    diagnosis_icd = diagnosis_icd[diagnosis_icd[dx_hadm_col].astype(str).isin(selected_hadm_ids)].copy()

    rx_hadm_col = _pick_col(prescriptions, ["hadm_id"])
    prescriptions = prescriptions[prescriptions[rx_hadm_col].astype(str).isin(selected_hadm_ids)].copy()

    dx_agg = _build_diagnosis_table(diagnosis_icd, d_icd_diagnoses)
    med_agg = _build_medication_table(
        prescriptions=prescriptions,
        demo_atc_mapping=demo_atc_mapping,
        med_rows_limit=med_rows_limit,
    )

    out = adm.merge(dx_agg, on=["subject_id", "hadm_id"], how="left")
    out = out.merge(med_agg, on=["subject_id", "hadm_id"], how="left")

    out["diagnoses"] = out["diagnoses"].apply(lambda x: x if isinstance(x, list) else [])
    out["medications"] = out["medications"].apply(lambda x: x if isinstance(x, list) else [])

    before_filter = len(out)
    out = out[(out["diagnoses"].map(len) > 0) & (out["medications"].map(len) > 0)].copy()
    dropped = before_filter - len(out)
    if dropped > 0:
        print(f"[INFO] Dropped {dropped:,} admissions with empty diagnoses or medications.")
    out = out.sort_values(["subject_id", "admittime", "hadm_id"])

    out["diagnoses"] = out["diagnoses"].apply(str)
    out["medications"] = out["medications"].apply(str)
    return out


def split_by_patient_until_visits(
    df: pd.DataFrame,
    target_test_visits: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Patient-level split: fill test with patients until visits >= target."""
    patient_counts = df.groupby("subject_id").size().to_dict()
    patient_ids = list(patient_counts.keys())
    random.Random(seed).shuffle(patient_ids)

    test_patients = []
    test_visits = 0
    for pid in patient_ids:
        if test_visits >= target_test_visits:
            break
        test_patients.append(pid)
        test_visits += int(patient_counts[pid])

    test_set = set(test_patients)
    test_df = df[df["subject_id"].isin(test_set)].copy()
    train_df = df[~df["subject_id"].isin(test_set)].copy()
    return train_df, test_df


def main():
    parser = argparse.ArgumentParser(description="Build mimic_train/test.csv from raw MIMIC tables.")
    parser.add_argument("--admissions_csv", required=True)
    parser.add_argument("--diagnosis_icd_csv", required=True)
    parser.add_argument("--d_icd_diagnoses_csv", required=True)
    parser.add_argument("--prescriptions_csv", required=True)
    parser.add_argument("--demo_atc_mapping_csv", required=True)
    parser.add_argument("--output_train_csv", default="data/mimic_train.csv")
    parser.add_argument("--output_test_csv", default="data/mimic_test.csv")
    parser.add_argument("--output_full_csv", default="data/mimic_full_admissions.csv")
    parser.add_argument("--target_test_visits", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--med_rows_limit",
        type=int,
        default=0,
        help="Debug mode: use only first N rows from demo_atc_mapping (0=all).",
    )
    parser.add_argument(
        "--admissions_rows_limit",
        type=int,
        default=0,
        help="Debug mode: use only first N admission rows before table joins (0=all).",
    )
    args = parser.parse_args()

    df = build_dataset(
        admissions_path=Path(args.admissions_csv),
        diagnosis_icd_path=Path(args.diagnosis_icd_csv),
        d_icd_diagnoses_path=Path(args.d_icd_diagnoses_csv),
        prescriptions_path=Path(args.prescriptions_csv),
        demo_atc_mapping_path=Path(args.demo_atc_mapping_csv),
        med_rows_limit=args.med_rows_limit,
        admissions_rows_limit=args.admissions_rows_limit,
    )

    train_df, test_df = split_by_patient_until_visits(
        df,
        target_test_visits=args.target_test_visits,
        seed=args.seed,
    )

    output_train = Path(args.output_train_csv)
    output_test = Path(args.output_test_csv)
    output_full = Path(args.output_full_csv)
    output_train.parent.mkdir(parents=True, exist_ok=True)
    output_test.parent.mkdir(parents=True, exist_ok=True)
    output_full.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_full, index=False)
    train_df.to_csv(output_train, index=False)
    test_df.to_csv(output_test, index=False)

    print(f"Full admissions: {len(df):,} -> {output_full}")
    print(f"Train (retrieval pool): {len(train_df):,} -> {output_train}")
    print(f"Test: {len(test_df):,} -> {output_test}")
    print(f"Unique patients (train/test): {train_df['subject_id'].nunique():,}/{test_df['subject_id'].nunique():,}")


if __name__ == "__main__":
    main()
