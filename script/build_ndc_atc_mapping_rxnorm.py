#!/usr/bin/env python3
"""
Build (or extend) an NDC -> ATC mapping table using RxNorm/RxClass APIs.

Output CSV columns:
- ndc
- atc_code
- atc_name
- status

This script is resumable:
- If output CSV exists, already mapped NDCs are skipped.
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm


RXNAV_BASE = "https://rxnav.nlm.nih.gov/REST"


def normalize_ndc(value) -> str:
    """Normalize NDC into a compact string."""
    if value is None:
        return ""
    ndc = str(value).strip()
    if ndc.endswith(".0"):
        ndc = ndc[:-2]
    return ndc


def fetch_atc_by_ndc(ndc: str, timeout_sec: float = 8.0):
    """
    NDC -> RxCUI -> ATC lookup.
    Returns (atc_code, atc_name, status).
    """
    if not ndc:
        return "", "", "empty_ndc"

    try:
        rxcui_resp = requests.get(
            f"{RXNAV_BASE}/ndcstatus.json",
            params={"ndc": ndc},
            timeout=timeout_sec,
        )
        rxcui_resp.raise_for_status()
        rxcui_data = rxcui_resp.json()

        rxcui = (
            rxcui_data.get("ndcStatus", {}) or {}
        ).get("rxcui", "")
        if not rxcui:
            return "", "", "no_rxcui"

        atc_resp = requests.get(
            f"{RXNAV_BASE}/rxclass/class/byRxcui.json",
            params={"rxcui": rxcui, "relaSource": "ATC"},
            timeout=timeout_sec,
        )
        atc_resp.raise_for_status()
        atc_data = atc_resp.json()

        infos = (((atc_data.get("rxclassDrugInfoList", {}) or {}).get("rxclassDrugInfo", [])) or [])
        if not infos:
            return "", "", "no_atc_link"

        concept = infos[0].get("rxclassMinConceptItem", {}) or {}
        return concept.get("classId", ""), concept.get("className", ""), "ok"
    except requests.RequestException as exc:
        return "", "", f"http_error:{exc.__class__.__name__}"
    except Exception as exc:  # keep robust for long running jobs
        return "", "", f"error:{exc.__class__.__name__}"


def load_unique_ndcs(input_csv: Path, ndc_column: str) -> list[str]:
    """Load unique NDC values from a CSV."""
    ndcs = set()
    for chunk in pd.read_csv(input_csv, usecols=[ndc_column], chunksize=100_000):
        values = chunk[ndc_column].dropna().tolist()
        for v in values:
            n = normalize_ndc(v)
            if n:
                ndcs.add(n)
    return sorted(ndcs)


def load_existing_mapping(output_csv: Path) -> dict:
    """Load existing mapping for resume mode."""
    if not output_csv.exists():
        return {}
    df = pd.read_csv(output_csv, dtype={"ndc": str})
    return {str(r["ndc"]): r for _, r in df.iterrows()}


def save_rows(output_csv: Path, rows: list[dict], append: bool) -> None:
    """Write batch rows to CSV."""
    if not rows:
        return
    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_csv, index=False, mode="a" if append else "w", header=not append)


def main():
    parser = argparse.ArgumentParser(description="Build NDC->ATC mapping via RxNorm API.")
    parser.add_argument("--input_csv", required=True, help="CSV that contains NDC column.")
    parser.add_argument("--ndc_column", default="ndc", help="NDC column name in input CSV.")
    parser.add_argument("--output_csv", required=True, help="Output mapping CSV path.")
    parser.add_argument("--sleep_sec", type=float, default=0.1, help="Delay between API calls.")
    parser.add_argument("--batch_size", type=int, default=100, help="Intermediate save interval.")
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    all_ndcs = load_unique_ndcs(input_csv, args.ndc_column)
    existing = load_existing_mapping(output_csv)
    to_process = [n for n in all_ndcs if n not in existing]

    print(f"Total unique NDCs: {len(all_ndcs):,}")
    print(f"Already mapped    : {len(existing):,}")
    print(f"To process        : {len(to_process):,}")

    append = output_csv.exists()
    buffer = []
    for i, ndc in enumerate(tqdm(to_process, desc="Mapping NDC -> ATC", unit="ndc"), 1):
        atc_code, atc_name, status = fetch_atc_by_ndc(ndc)
        buffer.append(
            {
                "ndc": ndc,
                "atc_code": atc_code,
                "atc_name": atc_name,
                "status": status,
            }
        )
        if i % args.batch_size == 0:
            save_rows(output_csv, buffer, append=append)
            append = True
            buffer = []
        time.sleep(args.sleep_sec)

    save_rows(output_csv, buffer, append=append)
    print(f"Saved mapping CSV: {output_csv}")


if __name__ == "__main__":
    main()
