import json
import re
import statistics
import argparse
import pandas as pd
import ast
from typing import List, Dict
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS

# Import custom modules from the local bundle
from prompt import (
    LLM_extract_focus_keywords_MIMIC,
    call_LLM_simple_prescription_with_reason_prompt_MIMIC,
    call_LLM_rag_tendency_analyzer_MIMIC,
    call_LLM_delta_verifier_MIMIC,
    call_LLM_doctor_summary,
    parse_model_output
)
from recommend_plan import embedding_model

def safe_float(value, default=0.0):
    """Convert score-like values to float safely."""
    try:
        return float(value)
    except Exception:
        return float(default)

def parse_list_cell(value):
    """Parse CSV list-like cells safely."""
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
    return [p.strip() for p in text.split(",") if p.strip()]

def normalize_drug_name_mimic(name: str) -> str:
    """
    Normalize drug names for MIMIC (lowercase, alphanumeric only).
    Used for consistent F1 score calculation.
    """
    cleaned = re.sub(r"[^a-z0-9\s]", " ", (name or "").lower())
    return " ".join(cleaned.split())

def list_medicine_answer(answer, ground_truth_list):
    """
    Extract and normalize medication list from model output.
    Compares model output with ground truth to identify TP, FP, FN.
    """
    if isinstance(answer, dict):
        v = answer.get("final_prescription") or answer.get("final_prescription_list") or []
        model_drugs_list = [str(x).strip() for x in v] if isinstance(v, list) else []
    elif isinstance(answer, list):
        model_drugs_list = [str(x).strip() for x in answer]
    else:
        model_drugs_list = parse_model_output(answer)
    
    gt_norm = {normalize_drug_name_mimic(s) for s in ground_truth_list if normalize_drug_name_mimic(s)}
    pred_norm = {normalize_drug_name_mimic(s) for s in model_drugs_list if normalize_drug_name_mimic(s)}
    
    tp_norm = gt_norm & pred_norm
    fn_norm = gt_norm - pred_norm
    fp_norm = pred_norm - gt_norm
    
    return {
        "ground_truth_list": ground_truth_list, 
        "model_response_answer": model_drugs_list, 
        "TruePositive": [gt for gt in ground_truth_list if normalize_drug_name_mimic(gt) in tp_norm],
        "FalseNegative": [gt for gt in ground_truth_list if normalize_drug_name_mimic(gt) in fn_norm],
        "FalsePositive": [pred for pred in model_drugs_list if normalize_drug_name_mimic(pred) in fp_norm]
    }


def _build_doctor_summary_final_payload(final_answer_list, final_answer):
    """
    Match original doctor-summary payload behavior:
    if verifier returns structured final_prescription, prefer it.
    """
    if isinstance(final_answer, dict):
        fp = final_answer.get("final_prescription") or final_answer.get("final_prescription_list")
        if isinstance(fp, list):
            return fp
    if isinstance(final_answer_list, dict):
        mr = final_answer_list.get("model_response_answer")
        if isinstance(mr, list):
            return mr
    return final_answer_list

def compute_f1_score_mimic(metrics):
    """Compute F1, Precision, and Recall for MIMIC results."""
    tp = len(metrics.get("TruePositive", []))
    fp = len(metrics.get("FalsePositive", []))
    fn = len(metrics.get("FalseNegative", []))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1, precision, recall

def update_and_print_summary(results: List[Dict]) -> Dict:
    """
    Calculate and print aggregate metrics (Macro F1, etc.) for all processed samples.
    """
    f1s, precs, recs = [], [], []
    if1s, iprecs, irecs = [], [], []
    
    for res in results:
        m = res.get("metrics", {})
        f1s.append(m.get("f1", 0))
        precs.append(m.get("precision", 0))
        recs.append(m.get("recall", 0))
        if1s.append(m.get("initial_f1", 0))
        iprecs.append(m.get("initial_precision", 0))
        irecs.append(m.get("initial_recall", 0))
        
    summary = {
        "num_samples": len(results),
        "macro_f1": round(statistics.mean(f1s), 4) if f1s else 0,
        "macro_precision": round(statistics.mean(precs), 4) if precs else 0,
        "macro_recall": round(statistics.mean(recs), 4) if recs else 0,
        "initial_macro_f1": round(statistics.mean(if1s), 4) if if1s else 0,
    }
    print(f"\n>>> Current Summary (n={len(results)}): F1={summary['macro_f1']}, Init_F1={summary['initial_macro_f1']}")
    return summary

def main():
    parser = argparse.ArgumentParser(description="MIMIC-IV Enhanced RAG Prescription Prediction")
    parser.add_argument("--data_path", type=str, required=True, help="Path to mimic_test.csv")
    parser.add_argument("--vector_db", type=str, required=True, help="Path to MIMIC FAISS index")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSON path")
    parser.add_argument("--llm_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.9, help="RAG similarity threshold")
    parser.add_argument("--retrieve_patients", type=int, default=7, help="Top-K per focus keyword")
    parser.add_argument("--max_samples", type=int, default=0, help="Debug mode: process only first N rows (0=all).")
    args = parser.parse_args()

    # Initialize LLM (OpenAI or Ollama)
    if args.llm_model.startswith(("gpt-", "o1-", "o3-")):
        llm = ChatOpenAI(
            model=args.llm_model,
            temperature=args.temperature,
            model_kwargs={"seed": args.seed} if args.seed is not None else {}
        )
    else:
        llm = OllamaLLM(model=args.llm_model, temperature=args.temperature, seed=args.seed)

    # Load MIMIC data
    df = pd.read_csv(args.data_path)
    # Group by subject_id to handle historical visits
    subject_groups = {sid: group.sort_values("admittime").to_dict("records") 
                      for sid, group in df.groupby("subject_id")}
    
    # Load FAISS once (outside patient loop) for performance and reproducibility.
    vs = FAISS.load_local(
        args.vector_db,
        embedding_model,
        index_name="faiss_mimic_diagnoses_index",
        allow_dangerous_deserialization=True,
    )

    if args.max_samples and args.max_samples > 0:
        df = df.head(args.max_samples).copy()

    results = []
    summary = {
        "num_samples": 0,
        "macro_f1": 0,
        "macro_precision": 0,
        "macro_recall": 0,
        "initial_macro_f1": 0,
    }
    total_patients = len(df)

    for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=total_patients, desc="Processing MIMIC Enhanced RAG"), start=1):
        subject_id, hadm_id = row["subject_id"], row["hadm_id"]
        patient_key = f"{subject_id}_{hadm_id}"
        stage_logs = []
        def log_stage(stage_name: str, detail: str = ""):
            msg = f"[STAGE][{idx}/{total_patients}][{patient_key}] {stage_name}"
            if detail:
                msg += f" | {detail}"
            print(msg, flush=True)
            stage_logs.append(msg)

        log_stage("START", "patient processing started")
        diagnoses = parse_list_cell(row["diagnoses"])
        ground_truth_meds = parse_list_cell(row["medications"])
        diagnoses_text = ", ".join(diagnoses) if diagnoses else "None"
        
        # 1. History Processing (Active History & Recent Visits)
        history = subject_groups[subject_id]
        curr_idx = next(i for i, r in enumerate(history) if r["hadm_id"] == hadm_id)
        prev_visits = history[:curr_idx]
        
        # Active History: Medications from the immediate previous visit
        active_history_list = parse_list_cell(prev_visits[-1]["medications"]) if prev_visits else []
        
        # Recent Visit History: Up to 3 previous visits for context
        recent_visit_history = []
        for v in prev_visits[-3:]:
            recent_visit_history.append({
                "visit": f"Visit ({v['admittime']})",
                "symptoms": ", ".join(parse_list_cell(v["diagnoses"])),
                "prescription": parse_list_cell(v["medications"])
            })
        log_stage("HISTORY_READY", f"active_history={len(active_history_list)}, recent_visits={len(recent_visit_history)}")

        # 2. Focus keyword extraction
        focus_queries = LLM_extract_focus_keywords_MIMIC(diagnoses, llm)
        log_stage("FOCUS_EXTRACT_DONE", f"focus_keywords={len(focus_queries)}")

        # 3. RAG search (targeted retrieval)
        rag_patients = []
        similar_docs_map = {}
        query_level_hits = []
        # Always include full diagnosis string query.
        # If keyword extraction fails, this still performs retrieval deterministically.
        search_queries = [", ".join(diagnoses)] + [q for q in focus_queries if q]
        for q in search_queries:
            hits = vs.similarity_search_with_relevance_scores(q, k=args.retrieve_patients)
            query_hits = []
            for doc, score in hits:
                score_f = safe_float(score)
                passed_threshold = score_f >= args.threshold
                if passed_threshold:
                    query_hits.append({
                        "subject_id": doc.metadata.get("subject_id"),
                        "hadm_id": doc.metadata.get("hadm_id"),
                        "score": score_f,
                        "diagnoses": doc.metadata.get("diagnoses", []),
                        "medications": doc.metadata.get("medications", []),
                    })
                    key = (doc.metadata["subject_id"], doc.metadata["hadm_id"])
                    # Keep the best score if the same patient is retrieved via multiple keywords
                    if key not in similar_docs_map or score_f > similar_docs_map[key][1]:
                        similar_docs_map[key] = (doc, score_f)
            query_level_hits.append({
                "query": q,
                "topk": args.retrieve_patients,
                "threshold": args.threshold,
                "hits": query_hits,
            })
        log_stage("RAG_RETRIEVE_DONE", f"queries={len(search_queries)}, passed_hits={sum(len(qh.get('hits', [])) for qh in query_level_hits)}")
        
        # Sort by similarity score and limit to top-K
        for doc, score in sorted(similar_docs_map.values(), key=lambda x: x[1], reverse=True)[:args.retrieve_patients]:
                         rag_patients.append({
                             "content": doc.page_content,
                "medications": doc.metadata["medications"],
                "score": safe_float(score),
                "subject_id": doc.metadata.get("subject_id"),
                "hadm_id": doc.metadata.get("hadm_id"),
                "diagnoses": doc.metadata.get("diagnoses", []),
            })
        log_stage("RAG_SELECT_DONE", f"selected_similar_patients={len(rag_patients)}")

        # 4. RAG tendency by focus (only when RAG evidence exists)
        rag_tendency_by_focus = []

        # Global tendency over all retrieved similar patients
        if rag_patients:
            overall_tendency = call_LLM_rag_tendency_analyzer_MIMIC(
                rag_patients=rag_patients,
                diagnoses=diagnoses,
                llm=llm,
            )
            rag_tendency_by_focus.append({
                "focus": "diagnoses_all",
                "source": "full_diagnoses",
                "num_cases": len(rag_patients),
                "tendency": overall_tendency,
            })

        # Per-focus tendency only when that focus produced RAG hits
        query_to_hits = {item.get("query"): item.get("hits", []) for item in query_level_hits}
        for focus in focus_queries:
            focus_hits = query_to_hits.get(focus, [])
            if not focus_hits:
                # No RAG signal for this focus → do not add JSON entry
                continue
            focus_cases = []
            for h in focus_hits:
                focus_cases.append({
                    "content": ", ".join(h.get("diagnoses", [])) if isinstance(h.get("diagnoses"), list) else str(h.get("diagnoses", "")),
                    "medications": h.get("medications", []),
                    "score": h.get("score", 0.0),
                    "subject_id": h.get("subject_id"),
                    "hadm_id": h.get("hadm_id"),
                })
            if not focus_cases:
                continue
            focus_tendency = call_LLM_rag_tendency_analyzer_MIMIC(
                rag_patients=focus_cases,
                diagnoses=[focus],
                llm=llm,
            )
            rag_tendency_by_focus.append({
                "focus": focus,
                "source": "focus_keyword",
                "num_cases": len(focus_cases),
                "tendency": focus_tendency,
            })
        log_stage("RAG_TENDENCY_DONE", f"tendency_items={len(rag_tendency_by_focus)}")

        # 5. Initial Prescription (Drafting)
        # Generates a baseline prescription using current diagnoses and history
        initial_prescription_raw = call_LLM_simple_prescription_with_reason_prompt_MIMIC(
            diagnoses, active_history_list, llm, recent_visit_history=recent_visit_history
        )
        log_stage("INITIAL_DRAFT_DONE", "initial draft generated")
        
        # 6. Delta Verifier (Refinement via RAG)
        # Refines the draft by comparing it with similar historical cases (RAG context)
        final_result = call_LLM_delta_verifier_MIMIC(
            initial_prescription_raw, diagnoses, None, active_history_list, llm,
            rag_patients=rag_patients, rag_tendency_by_focus=rag_tendency_by_focus, recent_visit_history=recent_visit_history
        )
        log_stage("DELTA_VERIFIER_DONE", "final recommendation generated")
        patient_state = f"S: Patient admitted on {row.get('admittime', 'N/A')} | O: Diagnoses: {diagnoses} | A: Evaluation for {diagnoses}"
        doctor_summary = call_LLM_doctor_summary(
            patient_state=patient_state,
            initial_prescription=initial_prescription_raw,
            rag_tendency_by_focus=rag_tendency_by_focus,
            audit_log=final_result.get("audit_log", []) if isinstance(final_result, dict) else [],
            final_answer_list=_build_doctor_summary_final_payload(None, final_result),
                        llm=llm,
                    )
        log_stage("DOCTOR_SUMMARY_DONE", "doctor summary generated")

        # 7. Metrics & Evaluation
        init_metrics = list_medicine_answer(initial_prescription_raw, ground_truth_meds)
        final_metrics = list_medicine_answer(final_result, ground_truth_meds)

        if1, iprec, irec = compute_f1_score_mimic(init_metrics)
        f1, prec, rec = compute_f1_score_mimic(final_metrics)

        # Log results for this patient
        patient_result = {
            "patient_id": patient_key,
            "patient_input": f"Subjective : Patient admitted on {row.get('admittime', 'N/A')}\n\nObjective : Diagnoses: {diagnoses}\n\nAssessment : Evaluation for {diagnoses}\n\n",
            "subjective": f"Patient admitted on {row.get('admittime', 'N/A')}",
            "objective": f"Diagnoses: {diagnoses}",
            "assessment": f"Evaluation for {diagnoses}",
            "diagnoses": diagnoses,
            "ground_truth": ground_truth_meds,
            "ground_truth_list": ground_truth_meds,
            "summarized_history": "",
            "active_history_from_parser": active_history_list,
            "recent_visit_history": recent_visit_history,
            "rag_patients": rag_patients if rag_patients else None,
            "focus_areas": {"diagnoses": focus_queries} if focus_queries else {},
                "rag_tendency_by_focus": rag_tendency_by_focus,
            "active_history": active_history_list,
            "rag_logging": {
                "focus_keywords": focus_queries,
                "search_queries": search_queries,
                "topk": args.retrieve_patients,
                "threshold": args.threshold,
                "query_level_hits": query_level_hits,
                "selected_similar_patients": rag_patients,
            },
            "stage_logs": stage_logs,
            "draft_plan": initial_prescription_raw,
            "final_answer": final_result,
            "doctor_summary": doctor_summary,
            "initial_answer_list": init_metrics,
            "final_answer_list": final_metrics,
            "metrics": {
                "f1": f1, "precision": prec, "recall": rec, 
                "initial_f1": if1, "initial_precision": iprec, "initial_recall": irec
            }
        }
        results.append(patient_result)
        log_stage("METRICS_DONE", f"f1={f1:.4f}, initial_f1={if1:.4f}")
                
        # Periodic update and save
        summary = update_and_print_summary(results)
        with open(args.output_file, "w") as f:
            json.dump({"summary": summary, "results": results}, f, indent=2)
        log_stage("SAVE_DONE", f"saved_to={args.output_file}")

    # Ensure output exists even when there are zero rows.
    if not results:
        with open(args.output_file, "w") as f:
            json.dump({"summary": summary, "results": results}, f, indent=2)

    print(f"\nProcessing complete. Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
