from __future__ import annotations

import json
import re
from typing import Iterable

STAGE1_KEYS_9: list[str] = [
    "DEMOGRAPHICS",
    "VITALS",
    "LABS",
    "PROBLEMS",
    "SYMPTOMS",
    "MEDICATIONS",
    "PROCEDURES",
    "UTILIZATION",
    "DISPOSITION",
]

STAGE1_KEYS_7: list[str] = [
    "DEMOGRAPHICS",
    "VITALS",
    "LABS",
    "PROBLEMS",
    "SYMPTOMS",
    "MEDICATIONS",
    "PROCEDURES",
]

STAGE1_ALLOWED_KEYS: set[str] = set(STAGE1_KEYS_9)


def resolve_stage1_keys(keys: Iterable[str]) -> list[str]:
    out: list[str] = []
    for k in keys:
        kk = str(k).strip()
        if not kk:
            continue
        if kk not in STAGE1_ALLOWED_KEYS:
            raise ValueError(f"Unknown Stage1 key: {kk}")
        out.append(kk)
    if not out:
        raise ValueError("Stage1 keys resolved to empty list.")
    # Ensure uniqueness while preserving order
    seen: set[str] = set()
    uniq: list[str] = []
    for k in out:
        if k in seen:
            continue
        seen.add(k)
        uniq.append(k)
    return uniq


def build_stage1_system_prompt(keys: Iterable[str] | None = None, *, crf_items: Iterable[str] | None = None) -> str:
    """
    Stage 1 (SGR-style) summary contract for CRF-filling:
    keep the same 9 readmission-oriented categories as a stable scaffold.

    IMPORTANT: This is an intermediate representation. The final submission is
    still the official 134-item CRF JSON (Stage 2).
    """
    resolved = resolve_stage1_keys(keys or STAGE1_KEYS_9)
    keys_str = ", ".join([f'"{k}"' for k in resolved])
    items_str = ""
    if crf_items is not None:
        items_list = [str(i).strip() for i in crf_items if str(i).strip()]
        if items_list:
            items_list_str = "\n".join([f"- {i}" for i in items_list])
            items_str = f"""

CRF ONTOLOGY (OPTIONAL REFERENCE):
- If you choose to name CRF items explicitly, use the EXACT item string (copy from this list).
- Do NOT try to output the official controlled vocabulary here; Stage 2 will map.

Allowed item strings:
{items_list_str}
"""
    return f"""You are an expert clinical summarization engine.

Task: Read ONE clinical note and output a CRF-focused structured summary as ONE VALID JSON object.

DETERMINISTIC APPENDIX (IMPORTANT):
- The input note may include a section titled "CRF Objective Appendix (deterministic)".
- If present, you MUST copy each appendix line verbatim into the appropriate cluster(s) using the same "Key: Value" text.
- The appendix is trusted evidence extracted from the note; do NOT paraphrase it.

SGR (STRICT, CASCADE):
You MUST follow this internal extraction order (do NOT output these steps, only the final JSON):
1) Objective first: extract VITALS + LABS numeric values (or "measured" if qualitative only).
2) Tests/procedures: imaging/monitoring and whether abnormality is present/absent (do not assume).
3) Problems/diagnoses: only explicitly stated.
4) Symptoms: present vs denied (explicit).
5) Medications/interventions: administered vs chronic, explicitly stated.
6) Utilization + disposition if stated.
7) Final audit: remove duplicates, remove anything not supported by the note.

ABG/LABS CYCLE (MANDATORY, INTERNAL):
- Do a focused pass over the note looking ONLY for ABG + key labs:
  ph, pa02, pac02, hc03-, lactates, hemoglobin, platelets, leukocytes,
  c-reactive protein, blood sodium, blood potassium, blood glucose,
  creatinine, transaminases, inr, troponin
- If any are present anywhere in the note, include them under LABS as lines:
  "<item>: <value>"
- CRITICAL: Do NOT output any of these items with "not stated". If a value is not present, OMIT it.

CRITICAL JSON RULES (STRICT):
1) Output MUST be a single valid JSON object.
2) The FIRST character of the output MUST be "{{".
3) Output JSON only (no markdown, no code fences, no preface).
4) Do NOT output analysis, chain-of-thought, or "thinking".
5) Use EXACTLY these keys (no extra keys, no missing keys):
   {keys_str}
6) TERMINATION: after the closing "}}", output a blank line and STOP. (Whitespace after JSON is allowed.)

VALUE FORMAT (STRICT):
- Each value MUST be a string.
- If a cluster has no supported facts in the note: use exactly "not stated" as the entire cluster value.
- If you include multiple lines, use "\\n" (two characters) inside the JSON string.
- Prefer short "Key: Value" lines inside each category string.
- NEVER output nested JSON objects or arrays as values (no dicts, no lists). If you need subfields, write multiple "Key: Value" lines in the string.

LINE FORMAT HARD RULE (CRITICAL FOR STAGE 2):
- In every cluster string, every non-empty line MUST contain ":" in the form "Key: Value".
- Do NOT output bare list items without ":" (no bullet lists, no plain item names).

SPARSE RULE (CRITICAL: AVOID TRUNCATION + FP):
- Inside clusters, include ONLY facts that are supported by the note.
- NEVER write per-item "not stated" lines (e.g., do NOT output "ph: not stated", "pneumonia: not stated", etc.).
- If an item is not stated, OMIT it. Missing items will be treated as "unknown" downstream.

CONTENT (STAGE-2 CONDITIONING):
- Stage 1 is an intermediate compressed representation; Stage 2 will produce the official 134-item CRF JSON.
- Prioritize compression: keep only CRF-relevant facts, numeric values, and explicit negations.
- Inside each category string, include only facts supported by the note.
- Use the note's wording and concrete values (numbers, units, imaging findings).
- Do NOT guess, hallucinate, or infer from absence.
- Do NOT force the official controlled vocabulary here; Stage 2 + deterministic mapping handle it.
- When possible, start each line with the EXACT CRF item name (verbatim), then a colon, then the raw value/evidence.
  - Example: "presence of dyspnea: present" or "spo2: 95% RA"
  - If you are not sure about the exact CRF item string, write a normal "Key: Value" line instead (do not invent item strings).

OBJECTIVE HARD RULES (VITALS/LABS):
- Include ALL objective values you can find in the note, even if normal.
- Output ONLY measurements that are present in the note (do not list missing measurements).
- If an objective measurement is mentioned without a number (e.g. "CRP negative", "troponin normal"):
  write the most literal token you can:
  - If the note uses a shorthand like "CRP neg" / "PCR neg": write "c-reactive protein: PCRneg"
  - If the note uses inequalities like "0.08 > 0.08": preserve them exactly (e.g. "troponin: 0.08 > 0.08")
  - Otherwise write "<item>: measured"
- For blood pressure: preserve raw "SBP/DBP" if available (e.g. "140/85 mmHg").

ABBREVIATION EXPANSION (DO THIS IN STAGE 1):
- If you see abbreviations, write the canonical CRF item key:
  WBC -> leukocytes
  Hb/Hgb -> hemoglobin
  PLT/PLTs -> platelets
  Na -> blood sodium
  K/K+ -> blood potassium
  Cr/Creat -> creatinine
  INR -> inr
  CRP/PCR -> c-reactive protein
  Tn/TnT/trop -> troponin
  pO2/PaO2 -> pa02
  pCO2/PaCO2 -> pac02
  HCO3/HCO3- -> hc03-
  Lac/Lactate -> lactates
  SpO2/SatO2 -> spo2
  HR -> heart rate
  RR -> respiratory rate
  Temp/T -> body temperature

KEYS (VITALS/LABS):
- Prefer canonical CRF keys when you know them (e.g., "hemoglobin", "platelets", "leukocytes", "creatinine", "troponin", "blood sodium", "blood potassium").
- If you are unsure, write a short key as in the note (Stage 2 will canonicalize).

VITALS (OUTPUT RAW; STAGE2 NORMALIZES):
- In VITALS, prefer raw measurements exactly as written:
  - spo2: keep percent value (e.g. "80% RA" or "80%")
  - blood pressure: keep SBP/DBP if present (e.g. "140/85 mmHg")
  - heart rate: keep bpm if present (e.g. "67 bpm")
  - respiratory rate: keep numeric if present (e.g. "18")
  - body temperature: keep numeric if present (e.g. "37.5") or "apyretic/afebrile"
  - level of consciousness: if GCS/AVPU/alertness is stated, write "level of consciousness: A/V/P"

TEST ABNORMALITY ITEMS (IMPORTANT):
- For items like "ecg, any abnormality", "chest rx, any abnormalities", "brain ct scan, any abnormality", etc:
  - If the note says abnormal findings: write "<item>: present"
  - If the note says normal/no abnormality: write "<item>: absent"
  - If only performed but no result: do NOT guess; omit (or write "not stated" by leaving it out of the cluster line list).

PROCEDURES KEY RULE (STRICT; FN REDUCTION):
- In "PROCEDURES", prefer using the EXACT CRF procedure/test item string as the KEY, e.g.:
  - ecg, any abnormality: present
  - chest rx, any abnormalities: present
  - cardiac ultrasound, any abnormality: present
  - brain ct scan, any abnormality: absent
- Do NOT use generic labels like "ECG:" or "Chest X-ray:" as keys in PROCEDURES.
- Do NOT invent new item names (e.g., do NOT write "echocardiogram, any abnormality").

PROBLEMS (RADICAL FN REDUCTION; CRF-ONLY):
- In "PROBLEMS", DO NOT list arbitrary comorbidities (e.g., hypertension/diabetes/CAD) unless they map to a CRF PROBLEMS item below.
- Instead, run this internal checklist over the note and include ONLY items you can support.
- Use EXACT CRF item strings as keys, and write a minimal value like "present"/"yes" (Stage 2 will normalize).
- NEVER output non-CRF items in PROBLEMS. If you cannot map a condition to a CRF item, omit it.
- CRITICAL: Do NOT fill the checklist with "absent"/"denied"/"no" by default.
  - If the note does not explicitly rule something out, omit it (Stage 2 will treat missing as "unknown").

CRF PROBLEMS OUTPUT (DO NOT ENUMERATE):
- Output ONLY problems that are explicitly stated in the note and match the official CRF problems/diagnoses.
- Do NOT output checklist items with "not stated"/"unknown".

PROBLEMS NORMALIZATION HINTS (USE ONLY WHEN EXPLICIT):
- COPD (chronic) -> chronic pulmonary disease: yes
- CKD / chronic kidney disease / ESRD -> chronic renal failure: yes
- Dialysis -> chronic dialysis: yes
- Cancer / neoplasm -> active neoplasia: yes
- CHF / decompensated HF -> heart failure: present (and/or chronic cardiac failure: yes if clearly chronic)
- AF / atrial fibrillation / arrhythmia -> arrhythmia: present
- PE -> pulmonary embolism: present
- ACS / MI -> acute coronary syndrome: present

PATIENT HISTORY / DEMOGRAPHICS CYCLE (MANDATORY, INTERNAL; FN-FIRST):
- Do a focused pass over the note to capture CRF "Patient History" items.
- Put these items under DEMOGRAPHICS as short "Key: Value" lines.
- Use the EXACT CRF item string as the Key whenever possible.
- Only include items that are explicitly stated or clearly supported by the note.
- For binary history items, use values like: "present" | "yes" (Stage 2 will normalize).
- For duration items, output ONLY: "short" | "long" (do not output exact minutes/seconds).
- CRITICAL: Do NOT output "absent"/"no"/"denied" unless the note explicitly states absence.

CRF HISTORY OUTPUT (DO NOT ENUMERATE):
- Output ONLY history/therapy items that are explicitly stated.
- High-priority history flags (often present): history of allergy, antihypertensive therapy, anticoagulants or antiplatelet drug therapy, poly-pharmacological therapy, cardiovascular diseases.

MEDICATIONS (CRF-ONLY; STOP REPETITION):
- In "MEDICATIONS", do NOT list brand names or long home-med lists.
- Only include these CRF medication/intervention concepts if explicitly stated:
  - administration of diuretics
  - administration of steroids
  - administration of bronchodilators
  - administration of oxygen/ventilation
  - blood transfusions
  - administration of fluids
- Use lines like "<item>: administered" ONLY when explicit. Otherwise omit (missing => unknown).

HOME MEDS RULE (STRICT; AVOID DRIFT):
- If the note lists chronic/home medications (e.g., ACE inhibitors, aspirin, beta blockers):
  - Do NOT list them under MEDICATIONS.
  - Instead, capture their CRF history flags under DEMOGRAPHICS when supported:
    - antihypertensive therapy
    - anticoagulants or antiplatelet drug therapy
    - poly-pharmacological therapy (use "present" if MANY chronic meds are listed)

PROCEDURES / UTILIZATION (CRF-ONLY):
- In "PROCEDURES", only include CRF procedure/test items (ECG/EEG/imaging/monitoring + the special tests).
- In "UTILIZATION", only include:
  - neurologist consultation
  - cardio-pulmonary resuscitation
- CRITICAL: Do NOT output "absent"/"no" utilization items unless explicitly stated; prefer omitting.

ANTI-DRIFT / STABILITY RULES (MUST FOLLOW):
- NEVER repeat the same line or item multiple times. Deduplicate aggressively.
- Keep each field compact, but do not drop objective facts:
  - VITALS: max 12 lines
  - LABS: max 18 lines
  - others: max 8 lines
  - Max ~700 characters per field.
- MEDICATIONS / PROCEDURES: list at most 10 unique entries total; if more, write "… (more)" once.
- If you start repeating, STOP and output the JSON object immediately.
- If you are unsure, prefer omitting a line over repeating or inventing.

ANTI-LEAKAGE RULE (CRITICAL):
- Do NOT copy example-looking strings from this instruction.
- Output ONLY facts supported by the clinical note. If a fact is not in the note, OMIT it (or write "not stated" at the cluster level).

FORMAT SKELETON (structure only; do not copy values):
{{
  "DEMOGRAPHICS": "not stated",
  "VITALS": "not stated",
  "LABS": "not stated",
  "PROBLEMS": "not stated",
  "SYMPTOMS": "not stated",
  "MEDICATIONS": "not stated",
  "PROCEDURES": "not stated",
  "UTILIZATION": "not stated",
  "DISPOSITION": "not stated"
}}

CLUSTER GUIDANCE (9 clusters):
- DEMOGRAPHICS: patient context, PMH, social/housing, baseline.
- VITALS: RR/Temp/HR/BP/SpO2 and related severity cues.
- LABS: blood gas / labs / biomarkers and key abnormal/normal statements.
- PROBLEMS: diagnoses and clinical problems (acute/chronic).
- SYMPTOMS: presenting symptoms and course (dyspnea, chest pain, etc.).
- MEDICATIONS: CRF interventions administered (NOT home med lists).
- PROCEDURES: imaging/tests/procedures performed.
- UTILIZATION: ED events, consults, monitoring, escalation.
- DISPOSITION: discharge outcome/follow-up if stated (else not stated).{items_str}
"""


def _strip_thinking_wrappers(text: str) -> str:
    # MedGemma / Gemma-family sometimes emits <unusedXX> wrappers for thoughts.
    s = text.strip()
    s = re.sub(r"<unused\\d+>thought.*?(?:<unused\\d+>|$)", "", s, flags=re.DOTALL | re.IGNORECASE)
    s = re.sub(r"<unused\\d+>", "", s, flags=re.IGNORECASE)
    return s.strip()


def has_thinking_leak(text: str) -> bool:
    """
    Best-effort detector for "thinking mode" artifacts leaking into the output.
    We treat these as undesirable for Stage 1 stability.
    """
    s = text.strip()
    brace = s.find("{")
    prefix = s if brace == -1 else s[:brace]
    prefix_lower = prefix.lower()
    return ("<unused" in prefix_lower) or ("thought" in prefix_lower) or ("thinking" in prefix_lower)


def extract_json_object(text: str) -> dict:
    s = _strip_thinking_wrappers(text)

    if "```json" in s:
        s = s.split("```json", 1)[1]
        s = s.split("```", 1)[0].strip()
    elif "```" in s:
        s = s.split("```", 1)[1]
        s = s.split("```", 1)[0].strip()

    if "{" not in s:
        raise ValueError("No JSON object start '{' found.")
    start = s.find("{")
    end = s.rfind("}")
    if end > start:
        return json.loads(s[start : end + 1])

    # Repair path: model sometimes truncates before printing the final "}" due to repetition.
    # We backtrack to the longest prefix that forms valid JSON (then coerce_stage1_summary fills missing keys).
    cand = s[start:]
    max_backtrack = max(0, len(cand) - 20000)  # cap work on extremely long responses
    for cut in range(len(cand), max_backtrack, -1):
        chunk = cand[:cut].strip()
        if not chunk:
            continue
        if not chunk.endswith("}"):
            chunk2 = chunk + "}"
        else:
            chunk2 = chunk
        try:
            obj = json.loads(chunk2)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    raise ValueError("No JSON object end '}' found.")


def coerce_stage1_summary(obj: dict, keys: Iterable[str] | None = None) -> dict[str, str]:
    """
    Make the Stage1 output safe and stable:
    - Fill missing keys with "not stated"
    - Drop extra keys
    - Coerce values to strings
    - Flatten nested dict/list values into "Key: Value" lines (string-only contract)
    """
    resolved = resolve_stage1_keys(keys or STAGE1_KEYS_9)

    def _flatten(v: object) -> str:
        if v is None:
            return "not stated"
        if isinstance(v, str):
            s = v.strip()
            return s if s else "not stated"
        if isinstance(v, dict):
            lines: list[str] = []
            for kk, vv in v.items():
                kks = str(kk).strip()
                if not kks:
                    continue
                if vv is None:
                    continue
                if isinstance(vv, dict):
                    inner_parts: list[str] = []
                    for k2, v2 in vv.items():
                        k2s = str(k2).strip()
                        v2s = str(v2).strip() if v2 is not None else ""
                        if not k2s or not v2s:
                            continue
                        inner_parts.append(f"{k2s} {v2s}")
                    inner = "; ".join(inner_parts).strip()
                    if inner:
                        lines.append(f"{kks}: {inner}")
                    else:
                        lines.append(f"{kks}: not stated")
                    continue
                if isinstance(vv, list):
                    inner = ", ".join([str(x).strip() for x in vv if str(x).strip()])
                    if inner:
                        lines.append(f"{kks}: {inner}")
                    continue
                vvs = str(vv).strip()
                if not vvs:
                    continue
                lines.append(f"{kks}: {vvs}")
            return "\n".join(lines).strip() if lines else "not stated"
        if isinstance(v, list):
            lines = [str(x).strip() for x in v if str(x).strip()]
            return "\n".join(lines).strip() if lines else "not stated"
        s = str(v).strip()
        return s if s else "not stated"

    out: dict[str, str] = {}
    for k in resolved:
        v = obj.get(k, "not stated")
        out[k] = _flatten(v)
    return out


def is_string_only_summary(obj: object, keys: Iterable[str] | None = None) -> bool:
    """
    Validate the Stage1 strict contract:
    - top-level object is a dict
    - contains all required keys
    - each required value is a *string* (no nested objects/arrays)
    """
    if not isinstance(obj, dict):
        return False
    resolved = resolve_stage1_keys(keys or STAGE1_KEYS_9)
    for k in resolved:
        if k not in obj:
            return False
        if not isinstance(obj.get(k), str):
            return False
    return True


def stage1_summary_to_text(summary: dict[str, str], keys: Iterable[str] | None = None) -> str:
    """
    Compact, stable text view for Stage2 conditioning.
    """
    parts: list[str] = []
    resolved = resolve_stage1_keys(keys or summary.keys() or STAGE1_KEYS_9)
    for k in resolved:
        parts.append(f"## {k}")
        parts.append(summary.get(k, "not stated"))
    return "\n".join(parts) + "\n"
