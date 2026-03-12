# PACE-RAG: Patient-Aware Contextual & Evidence-based Policy RAG for Clinical Drug Recommendation

PACE-RAG is a framework designed to predict and verify personalized medication plans for complex hospital inpatients using Retrieval-Augmented Generation. This repository provides the complete pipeline to process raw MIMIC-IV data, build a vector database of patient cases, and execute the PACE-RAG inference pipeline.

## Prerequisites

- Python 3.8+
- [Ollama](https://ollama.com/) (for local LLM inference, e.g., `qwen3:8b`) or OpenAI API access.
- Required Python packages: `pandas`, `tqdm`, `langchain`, `langchain-openai`, `langchain-ollama`, `langchain-community`, `faiss-cpu`, `sentence-transformers`.

## Directory Structure

Before running the pipeline, ensure your directory structure looks like this:

```text
.
├── data/
│   └── raw/                   <-- Place your MIMIC-IV CSV files here
├── output/
│   ├── logs/                  <-- Execution logs will be saved here
│   └── one_step_rag/          <-- Final JSON results will be saved here
├── script/                    <-- Execution scripts
├── vector_db/                 <-- Generated FAISS vector database will be saved here
├── main.py                    <-- Core PACE-RAG execution script
├── prompt.py                  <-- LLM Prompts and parsers
├── recommend_plan.py          <-- Embedding model configuration
└── README.md                  <-- This file
```

## Step-by-Step Usage Guide

### Step 1: Prepare Raw Data
Place the following raw MIMIC-IV tables into the `data/raw/` directory:
1. `admissions.csv`
2. `d_icd_diagnoses.csv`
3. `diagnoses_icd.csv` (or `diagnosis_icd.csv`)
4. `prescriptions.csv`
5. `demo_atc_mapping.csv` (NDC to ATC mapping dictionary)

### Step 2: Build Dataset
Convert the raw MIMIC-IV tables into admission-level datasets (`mimic_train.csv` and `mimic_test.csv`) with ATC-mapped medications. This script splits the data at the patient level to prevent data leakage.

```bash
# You can adjust TARGET_TEST_VISITS or use row limits for quick testing
TARGET_TEST_VISITS=1000 bash script/build_mimic_train_test_from_tables.sh
```
*Output: `data/mimic_train.csv` and `data/mimic_test.csv` will be generated.*

### Step 3: Build Vector Database
Create the FAISS vector database from the training set. This database is used by PACE-RAG to retrieve similar historical patient cases based on clinical focus keywords.

```bash
bash script/build_mimic_vector_db.sh
```
*Output: The FAISS index will be saved in the `vector_db/` directory.*

### Step 4: Run PACE-RAG
Execute the main PACE-RAG pipeline on the test dataset. This step extracts focus keywords, retrieves similar cases, generates an initial draft, and refines it using the delta verifier.

```bash
# Make sure your Ollama server is running, or configure OpenAI credentials.
# Default model is qwen3:8b. You can override it via environment variables.

LLM_MODEL=qwen3:8b OLLAMA_HOST=127.0.0.1:11436 bash script/run_mimic_enhanced_rag.sh 0.0
```

### Output Format
The final results are saved as a JSON file in `output/one_step_rag/`. The JSON contains:
- `summary`: Aggregate metrics (Macro F1, Precision, Recall).
- `results`: Patient-level predictions including:
  - `diagnoses` & `active_history`: Patient context.
  - `rag_logging`: Retrieved similar cases and extracted focus keywords.
  - `rag_tendency_by_focus`: Analyzed treatment patterns from similar cases.
  - `final_answer`: The verified medication plan (`final_prescription`) and `audit_log`.
  - `doctor_summary`: A human-readable clinical summary of the decision-making process.
