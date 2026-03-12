from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json
import ast
import re
import os

# Qwen/Ollama reasoning suppression (if using reasoning models like Qwen-7B-Instruct)
QWEN_PREFIX = "/no_think\n"

def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from LLM output (for reasoning models)."""
    if not text or not isinstance(text, str):
        return text or ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

def apply_qwen_prefix(prompt: ChatPromptTemplate, llm) -> ChatPromptTemplate:
    """Apply Qwen-specific prefix to suppress internal reasoning output."""
    model_name = ""
    for attr in ("model", "model_name", "model_id"):
        model_name = getattr(llm, attr, model_name) or model_name
    env_model = os.environ.get("LLM_MODEL", "")
    target = f"{model_name} {env_model}".lower()
    
    # Only apply /no_think for Qwen models to ensure clean output format
    if "qwen" in target:
        return ChatPromptTemplate.from_messages([("system", QWEN_PREFIX)] + prompt.messages)
    return prompt

def _invoke_chain_with_full_input_log(prompt, llm, invoke_dict):
    """Invoke the LangChain LLM chain and return string output."""
    chain = prompt | llm | StrOutputParser()
    return chain.invoke(invoke_dict)

# --- MIMIC-IV Enhanced RAG Prompts ---

# 1. Focus Keyword Extractor: Extracts key clinical focus areas for targeted RAG retrieval.
LLM_focus_keyword_extractor_prompt_MIMIC = ChatPromptTemplate.from_messages([
    ("system", """You are a precise Medical Diagnosis Extractor for MIMIC-IV inpatients. Your ONLY job is to identify CURRENT ACTIVE diagnoses that are NEW, WORSENING, or UNRESOLVED and clearly matter for treatment.
You are NOT a doctor. Do NOT invent, infer, or hallucinate.

**STRICT EXTRACTION RULES:**
1. **THE 'EMPTY' RULE (CRITICAL):**
   - If the diagnoses list is empty, purely administrative/meta, or only screening/history/status phrases (e.g., "Inpatient", "Follow-up visit", "History of X")...
   - If nothing clearly reflects an acute or active condition needing management...
   -> YOU MUST OUTPUT AN EMPTY LIST: {{"keywords": []}}
2. **NO GENERIC LABELS OR FRAGMENTS:**
   - Do NOT output encounter/status/meta phrases alone (e.g., "Inpatient", "Hospitalization", "Follow-up visit").
   - Forbidden single bare words: "acute", "chronic", "failure", "infection" by themselves.
3. **KEEP FULL DIAGNOSIS PHRASES (LITERAL MATCH):**
   - Always keep full diagnosis phrases exactly as written (e.g., "Acute kidney failure", "Sepsis due to pneumonia").
   - Do NOT chop off qualifiers like "acute", "chronic", "unspecified".
   - Every keyword must be an exact substring from the diagnoses text (case-insensitive). No paraphrasing or rewording.
4. **LIMIT:** Max 5 phrases. Focus ONLY on acute/active issues that drive treatment decisions.

**OUTPUT FORMAT (JSON ONLY):**
{{"keywords": ["Diagnosis phrase 1", "Diagnosis phrase 2"]}}  OR  {{"keywords": []}}
No explanations, no markdown, no extra keys.
"""),
    ("user", """Diagnoses list:
{diagnoses}

Analyze the CURRENT text. If the patient is stable/improving/no acute symptoms, return {{"keywords": []}}. Otherwise, extract MAX 5 severe symptom phrases. Return JSON only.""")
])
# 2. Initial Prescription Generator: Drafts a baseline prescription based on diagnoses and history.
LLM_simple_prescription_with_reason_prompt_MIMIC = ChatPromptTemplate.from_messages([
    ("system", """You are a clinical medication specialist for complex hospital inpatients (MIMIC-IV dataset).

Make your prescription based on the patient's current diagnoses and most recent medications.
If there is a most recent medications list, start from that list.
Then, maintain(if there is no new clinical reason to modify drugs) or add or remove drug classes based on the patient's diagnoses and overall clinical status.
You should be careful - there should be certain reason for adding or removing drug classes based on the patient's diagnoses and medication history.

Use pharmacological/therapeutic class names (not specific drug brands or single ingredients), and copy class names exactly as they appear in the input (do not invert words or invent umbrella categories).

You should put one drug class name at every line!
FORMAT STRICT: Output ONLY the [START]...[END] block. No extra text.

STABILITY RULE (no most recent medications list):
- If the clinical picture is stable/improved/no major new problems, output ONLY a conservative minimal set of classes.
- Prefer continuing existing chronic medication classes when they are clearly indicated.
- Avoid starting broad new classes unless there is clear diagnostic support.
**OUTPUT:**
[START]
(Drug Class Name) | (short reason in 10 words or less)
(Drug Class Name) | (short reason in 10 words or less)
[END]
"""),
    ("user", """
**Patient Information:**
- Diagnoses: {diagnoses}
- Most Recent Medications: {medications}

Past Clinical Visits (up to 3 visits before current):
{recent_visit_history_text}

**Task:** Generate prescription. Should put at least one drug in output.
""")
])

# 3. RAG tendency analyzer + Delta verifier
LLM_rag_tendency_analyzer_MIMIC = ChatPromptTemplate.from_messages([
    ("system", """You are a highly focused Clinical Pattern Analyzer for complex hospital inpatients (MIMIC-IV).

**OBJECTIVE:** Analyze similar patient cases to identify EXACTLY which medication CLASSES physicians *NEWLY PRESCRIBED (ADDED)* to resolve the specific target diagnosis.

**STRICT ANALYSIS RULES:**
1. **FOCUS ONLY ON ADDITIONS:** Your ONLY job is to find drug classes that were *newly prescribed* (added) in the similar cases to treat the specific Target Clinical Focus.
2. **CAUSALITY CHECK (CRITICAL):** The drug class MUST have been added specifically for the Target Diagnosis. (Note: General inpatient care classes like 'Heparin group' or 'laxatives' that consistently accompany the diagnosis should also be extracted).
3. **IGNORE MAINTAINED DRUGS:** Do not extract drug classes that the patient was already taking.
4. **CAUTIOUS EMPTY DEFAULT:** If the cases do not explicitly show a new drug class being added to treat the target diagnosis, your `common_additions` MUST be empty `[]`.
5. **No Invention:** Rely ONLY on the provided text and copy exact pharmacological class names.

**OUTPUT FORMAT:** Output ONLY valid JSON.
{{"dominant_pattern": "ADD", "common_additions": ["Drug Class A"], "reasoning": "Brief 1-sentence reason based on cases."}}
"""),
    ("user", """
**Current Patient Diagnoses:** {diagnoses}
{keyword_focus_instruction}
**Similar Patient Cases:** {rag_patients}

Analyze explicitly what was ADDED. Output JSON only.""")
])

LLM_delta_verifier_prompt_MIMIC = ChatPromptTemplate.from_messages([
    ("system", """You are a strict Clinical Auditor checking a Draft Prescription against RAG Evidence for complex hospital inpatients (MIMIC-IV).
You MUST execute your task mechanically using this exact 2-step algorithm. Do not overthink.

=====================================================================
### DRUG CLASS FILTERING ALGORITHM

**STEP 1: PRESERVE ACTIVE HISTORY (MANDATORY)**
Look at the `Active History` list provided in the prompt.
- If `Active History` is empty or "None", you MUST NOT use the "KEPT" action. Skip to Step 2.
- Otherwise, you MUST put EVERY single drug class from `Active History` into your `final_prescription` array.
- You MUST create an {{"action": "KEPT", "drug": "..."}} entry in the `audit_log` for EACH of these classes.
- NEVER miss or drop a class from the Active History.

**STEP 2: EVALUATE DRAFT DRUG CLASSES**
Look at the `Initial Draft Prescription`. For each drug class that is NOT already in Active History:
- Does it conceptually match or exactly align with a specific pharmacological class in the RAG `common_additions` array? (Note: Draft may use broad terms like "Antidepressants", while RAG uses exact ATC classes like "Selective serotonin reuptake inhibitors").
  -> YES: ADD the exact RAG class name. (action: "ADDED")
  -> NO: REMOVE it. (action: "REMOVED")

=====================================================================
### CRITICAL RULES
- STRICT ADD GATE: You can ONLY add a new drug class if it is explicitly listed in `common_additions`. Being in `maintained_drug` does NOT justify adding a new class.
- SYNC RULE: If you log "REMOVED", it MUST NOT be in `final_prescription`.
- EMPTY FALLBACK: If `final_prescription` is completely empty, pick 1 or 2 drug classes from RAG's `common_additions` and ADD them.

=====================================================================
### REQUIRED JSON FORMAT (DO NOT ADD OTHER KEYS)
{{
  "final_prescription": ["Drug Class A", "Drug Class B"],
  "audit_log": [
    {{"action": "KEPT", "drug": "Drug Class A", "reason": "Patient is already taking it in active history."}},
    {{"action": "ADDED", "drug": "Drug Class B", "reason": "From RAG common_additions."}},
    {{"action": "REMOVED", "drug": "Drug Class C", "reason": "Not in RAG common_additions."}}
  ],
  "final_description": "Brief 1-sentence summary."
}}
"""),
    (
        "user",
        """**Patient Input (Diagnoses):**
{diagnoses}

**Active History (Currently taking. ALWAYS KEEP THESE):**
{active_history}

**Initial Draft Prescription (WARNING: This is just a guess. Default action is REMOVE unless proven by RAG):**
{initial_prescription}

**RAG Focus Tendencies (what the DB returned for those/similar symptoms; ordered by priority):**
{rag_focus_tendency}

**Past Clinical Visits:**
{recent_visit_history_text}

**FINAL EXECUTION TASK:**
Apply the 2-STEP DRUG CLASS FILTERING ALGORITHM.
1. Force-copy ALL Active History classes to `final_prescription` and log them as "KEPT" (unless history is empty).
2. Evaluate remaining Draft classes strictly against RAG `common_additions`, adopting the specific RAG terminology.

Output ONLY the strict JSON dictionary."""
    )
])

# --- Helper Functions ---

def parse_model_output(text: str) -> list:
    """Extract drug class names from the [START]...[END] block in model output."""
    text = strip_think_tags(text)
    match = re.search(r"\[START\](.*?)\[END\]", text, re.DOTALL)
    if not match:
        return []
    lines = match.group(1).strip().split("\n")
    results = []
    for line in lines:
        if "|" in line:
            results.append(line.split("|")[0].strip())
        else:
            results.append(line.strip())
    return [r for r in results if r]

def _format_recent_visit_history_text(recent_visit_history, max_visits=3):
    """Format recent visits: MIMIC = Symptoms | Prescription only."""
    if not recent_visit_history or not isinstance(recent_visit_history, list):
        return "No previous visit exists."
    visits = recent_visit_history[-max_visits:]
    lines = []
    for visit in visits:
        if not isinstance(visit, dict):
            lines.append(f"- Visit: {str(visit)}")
            continue
        visit_name = visit.get("visit", "Unknown visit")
        symptoms = visit.get("symptoms", "None")
        prescription = visit.get("prescription", [])
        if isinstance(prescription, list):
            prescription_txt = ", ".join([str(x) for x in prescription]) if prescription else "None"
        else:
            prescription_txt = str(prescription) if prescription else "None"
        lines.append(f"- {visit_name}: Symptoms={symptoms} | Prescription={prescription_txt}")
    return "\n".join(lines) if lines else "No previous visit exists."


def parse_json_garbage(text: str):
    """Best-effort JSON parse for noisy LLM outputs."""
    if not text:
        return None
    raw = text.strip()
    try:
        return json.loads(raw)
    except Exception:
        pass

    # Remove markdown fences if present
    if "```json" in raw:
        raw = raw.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in raw:
        raw = raw.split("```", 1)[1].split("```", 1)[0].strip()
    try:
        return json.loads(raw)
    except Exception:
        pass

    # Extract first JSON object region
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = raw[start:end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            # Try python-literal fallback
            try:
                return ast.literal_eval(candidate)
            except Exception:
                return None
    return None


def parse_prescription_to_list(text: str) -> list:
    """Parse [START]...[END] style output into class-name list."""
    parsed = parse_model_output(text)
    return [p.strip() for p in parsed if isinstance(p, str) and p.strip()]

def LLM_extract_focus_keywords_MIMIC(diagnoses, llm):
    """Extract focus keywords from diagnoses using LLM."""
    prompt = apply_qwen_prefix(LLM_focus_keyword_extractor_prompt_MIMIC, llm)
    res = _invoke_chain_with_full_input_log(prompt, llm, {"diagnoses": ", ".join(diagnoses)})
    res = strip_think_tags(res).strip()
    if res.lower() == "none" or not res:
        return []
    return [k.strip() for k in res.split(",") if k.strip()]

def call_LLM_simple_prescription_with_reason_prompt_MIMIC(diagnoses, medications, llm, recent_visit_history=None):
    """Generate the initial baseline prescription draft."""
    prompt = apply_qwen_prefix(LLM_simple_prescription_with_reason_prompt_MIMIC, llm)
    invoke_dict = {
        "diagnoses": ", ".join(diagnoses),
        "medications": medications,
        "recent_visit_history_text": _format_recent_visit_history_text(recent_visit_history)
    }
    return _invoke_chain_with_full_input_log(prompt, llm, invoke_dict)


def call_LLM_rag_tendency_analyzer_MIMIC(rag_patients, diagnoses, llm):
    if not rag_patients:
        return {
            "dominant_pattern": "MAINTAIN",
            "common_additions": [],
            "reasoning": "No relevant similar cases were retrieved for this focus.",
        }

    def format_case(case):
        if isinstance(case, dict):
            diagnoses_text = case.get("content", "")
            meds = case.get("medications", [])
            if isinstance(meds, str) and meds.startswith("["):
                try:
                    meds = ast.literal_eval(meds)
                except Exception:
                    pass
            meds_text = ", ".join(meds) if isinstance(meds, list) else str(meds)
            return f"Diagnoses: {diagnoses_text}\nMedications: {meds_text}"
        return str(case)

    formatted_cases = "\n\n".join([format_case(case) for case in rag_patients[:5]])
    focus_txt = ", ".join(str(d).strip() for d in diagnoses if str(d).strip()) if isinstance(diagnoses, (list, tuple)) else str(diagnoses or "").strip()
    keyword_focus_instruction = (
        "\n==================================================\n"
        f"TARGET CLINICAL DIAGNOSIS: >>> {focus_txt} <<<\n"
        "* IMPORTANT: Filter the 'Similar Patient Cases' below. Only pay attention to how physicians treated THIS EXACT TARGET.\n"
        "* Ignore medications given for other unrelated diagnoses in those cases.\n"
        "==================================================\n"
    )

    prompt = apply_qwen_prefix(LLM_rag_tendency_analyzer_MIMIC, llm)
    result = _invoke_chain_with_full_input_log(
        prompt,
        llm,
        {
            "rag_patients": formatted_cases,
            "diagnoses": focus_txt,
            "keyword_focus_instruction": keyword_focus_instruction,
        },
    ).strip()

    try:
        if "```json" in result:
            result = result.split("```json")[1].split("```")[0].strip()
        elif "```" in result:
            result = result.split("```")[1].split("```")[0].strip()
        parsed = parse_json_garbage(result)
        if isinstance(parsed, dict) and "dominant_pattern" in parsed:
            return parsed
        loaded = json.loads(result)
        if isinstance(loaded, dict) and "dominant_pattern" in loaded:
            return loaded
        return {
            "dominant_pattern": "MAINTAIN",
            "common_additions": [],
            "reasoning": "Unable to extract a stable pattern from retrieved cases.",
        }
    except Exception:
        return {
            "dominant_pattern": "MAINTAIN",
            "common_additions": [],
            "reasoning": "RAG tendency analysis failed; defaulting to maintenance pattern.",
        }

def call_LLM_delta_verifier_MIMIC(
    initial_prescription,
    diagnoses,
    _,
    active_history,
    llm,
    rag_patients=None,
    rag_tendency_by_focus=None,
    recent_visit_history=None,
):
    """Refine initial prescription with the same MIMIC delta-verifier prompt structure."""
    active_history_txt = active_history if active_history and len(active_history) > 0 else "None"

    def _format_focus_tendency(focus_items, max_items=6):
        if not focus_items or not isinstance(focus_items, list):
            return "None"
        lines = []
        for item in focus_items[:max_items]:
            if not isinstance(item, dict):
                continue
            focus = item.get("focus")
            tendency = item.get("tendency", {})
            if not focus:
                continue
            pattern = tendency.get("dominant_pattern", "N/A")
            additions = tendency.get("common_additions", [])
            reasoning = tendency.get("reasoning") or "N/A"
            if not isinstance(reasoning, str):
                reasoning = str(reasoning) if reasoning else "N/A"
            if not isinstance(additions, list):
                additions = [str(additions)] if additions else []
            add_txt = ", ".join([str(x).strip() for x in additions[:4] if str(x).strip()]) if additions else "None"
            lines.append(f"- Focus: {focus} | Pattern: {pattern} | Add: {add_txt} | Reasoning: {reasoning}")
        return "\n".join(lines) if lines else "None"

    prompt = apply_qwen_prefix(LLM_delta_verifier_prompt_MIMIC, llm)
    recent_visit_history_text = _format_recent_visit_history_text(recent_visit_history, max_visits=3)
    result = _invoke_chain_with_full_input_log(prompt, llm, {
        "diagnoses": ", ".join(diagnoses),
        "active_history": active_history_txt,
        "initial_prescription": initial_prescription,
        "rag_focus_tendency": _format_focus_tendency(rag_tendency_by_focus),
        "recent_visit_history_text": recent_visit_history_text,
    }).strip()

    try:
        if "```json" in result:
            result = result.split("```json")[1].split("```")[0].strip()
        elif "```" in result:
            result = result.split("```")[1].split("```")[0].strip()
        parsed = parse_json_garbage(result)

        if isinstance(parsed, dict):
            final_prescription = parsed.get("final_prescription", [])
            if isinstance(final_prescription, str):
                final_prescription = [s.strip() for s in final_prescription.split(",")]
            final_prescription = [str(s).split("|")[0].strip() for s in final_prescription if isinstance(s, str) and s.strip()]
            if not final_prescription:
                final_prescription = parse_prescription_to_list(initial_prescription)

            audit_log = parsed.get("audit_log", [])
            if isinstance(audit_log, list):
                for entry in audit_log:
                    if isinstance(entry, dict) and "drug" in entry and isinstance(entry["drug"], str):
                        entry["drug"] = entry["drug"].split("|")[0].strip()

            return {
                "final_prescription": final_prescription,
                "audit_log": audit_log if isinstance(audit_log, list) else [],
                "final_description": parsed.get("final_description", ""),
                "raw_output": result,
            }
        raise ValueError("Failed to parse JSON")
    except Exception as e:
        initial_drugs = parse_prescription_to_list(initial_prescription)
        return {
            "final_prescription": initial_drugs,
            "audit_log": [
                {
                    "action": "KEPT",
                    "drug": "initial",
                    "reason": f"Initial draft maintained due to verification error: {e}",
                }
            ],
            "final_description": "Initial recommendation was preserved because the verification step could not be reliably completed.",
            "raw_output": result,
        }


# --- Doctor Summary (for JSON output) ---
LLM_doctor_summary_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Senior Clinical Consultant summarizing cases for attending physicians.
Format your output EXACTLY as follows:

* Patient summary *
(Detailed description of patient's current symptoms, stability, and clinical trajectory in 2-3 sentences.)

* Key word *
(Comma-separated list of clinical focus keywords extracted from symptoms/diagnoses.)

* Clinical Evidence *
(Concise summary of treatment patterns found in similar historical cases. Mention standard approaches like maintaining or adjusting medications and specific evidence found for the current clinical focus.)

* Prescribe *
(Drug Name) : (Detailed clinical rationale. Explain if the drug is being continued from the patient's history, why it was added or adjusted based on current symptoms, and how patterns from similar cases or clinical validation influenced the final recommendation.)

Rules:
- Use ONLY provided inputs. No hallucinations.
- Ensure the * Prescribe * section covers EVERY drug in the final recommendation list.
- No internal monologue or <think> tags.
- Output plain text structure only.
"""),
    ("user", """
**Patient State:**
{patient_state}

**Initial Draft (Proposed Plan):**
{initial_prescription}

**Clinical Evidence from Similar Cases:**
{rag_tendency_by_focus}

**Clinical Validation Log:**
{audit_log}

**Final Recommended Medications:**
{final_answer_list}
""")
])


def call_LLM_doctor_summary(patient_state, initial_prescription, rag_tendency_by_focus, audit_log, final_answer_list, llm):
    try:
        prompt = apply_qwen_prefix(LLM_doctor_summary_prompt, llm)
        result = _invoke_chain_with_full_input_log(prompt, llm, {
            "patient_state": patient_state or "None",
            "initial_prescription": initial_prescription or "None",
            "rag_tendency_by_focus": rag_tendency_by_focus if rag_tendency_by_focus is not None else "None",
            "audit_log": audit_log if audit_log is not None else "None",
            "final_answer_list": final_answer_list if final_answer_list is not None else "None",
        }).strip()

        if "<think>" in result and "</think>" in result:
            result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()
        elif "<think>" in result:
            result = result.split("<think>")[-1].split("</think>")[-1].strip()

        if "```" in result:
            result = result.replace("```", "").strip()
        return result
    except Exception as e:
        print(f"DEBUG: Doctor Summary Error: {e}")
        return ""
