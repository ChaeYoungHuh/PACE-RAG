#!/usr/bin/env python3
"""
Build a FAISS vector database for MIMIC Enhanced RAG.

Input CSV format (required columns):
- subject_id
- hadm_id
- diagnoses
- medications

This script stores:
- faiss_mimic_diagnoses_index
- faiss_mimic_medications_index

under the target vector DB directory.
"""

import argparse
import ast
from pathlib import Path

import pandas as pd
from langchain_community.vectorstores import FAISS

import sys


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


ROOT = _project_root()
sys.path.insert(0, str(ROOT))

from recommend_plan import embedding_model  # noqa: E402


def _safe_list(value):
    """Parse a list-like CSV cell robustly."""
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass
    # Fallback for comma-separated plain text
    return [p.strip() for p in text.split(",") if p.strip()]


def build_mimic_vector_db(input_csv: Path, output_dir: Path) -> None:
    """Build and save MIMIC FAISS indices."""
    df = pd.read_csv(input_csv)

    required = {"subject_id", "hadm_id", "diagnoses", "medications"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    diagnoses_texts = []
    medications_texts = []
    metadatas = []

    for row in df.to_dict("records"):
        diagnoses = _safe_list(row.get("diagnoses"))
        medications = _safe_list(row.get("medications"))

        diagnoses_texts.append(", ".join(diagnoses) if diagnoses else "None")
        medications_texts.append(", ".join(medications) if medications else "None")

        metadatas.append(
            {
                "subject_id": str(row.get("subject_id")),
                "hadm_id": str(row.get("hadm_id")),
                "diagnoses": diagnoses,
                "medications": medications,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    diagnoses_store = FAISS.from_texts(
        texts=diagnoses_texts,
        embedding=embedding_model,
        metadatas=metadatas,
    )
    medications_store = FAISS.from_texts(
        texts=medications_texts,
        embedding=embedding_model,
        metadatas=metadatas,
    )

    diagnoses_store.save_local(str(output_dir), "faiss_mimic_diagnoses_index")
    medications_store.save_local(str(output_dir), "faiss_mimic_medications_index")

    print(f"Built MIMIC vector DB at: {output_dir}")
    print("- faiss_mimic_diagnoses_index")
    print("- faiss_mimic_medications_index")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MIMIC FAISS vector DB.")
    parser.add_argument(
        "--input_csv",
        type=str,
        default=str(ROOT / "data" / "mimic_train.csv"),
        help="Path to MIMIC train CSV.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(ROOT / "vector_db" / "mimic_train_faiss_index"),
        help="Directory to store FAISS indices.",
    )
    args = parser.parse_args()

    build_mimic_vector_db(Path(args.input_csv), Path(args.output_dir))


if __name__ == "__main__":
    main()
