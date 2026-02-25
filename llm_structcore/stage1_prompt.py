from __future__ import annotations

"""
Stage 1 prompt for the CL4Health2026 CRF:filling dyspnea task.

Design goals:
- 100% JSON stability (single object, fixed keys)
- Compressed, cluster-scoped evidence for Stage 2 (CRF model + deterministic mapping)
- Avoid overloading Stage 1 with controlled-vocab constraints
"""

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

def detect_present_clusters(stage1_summary: dict[str, object]) -> list[str]:
    """
    Return Stage 1 clusters that are meaningfully present for a given document.

    Convention:
    - Absent clusters are either missing in the JSON or have the exact value "not stated".
    - Any other non-empty string is treated as present.
    """
    present: list[str] = []
    for k in STAGE1_KEYS_9:
        if k not in stage1_summary:
            continue
        v = stage1_summary.get(k)
        if v is None:
            continue
        s = str(v).strip()
        if not s:
            continue
        if s.lower() == "not stated":
            continue
        present.append(k)
    return present


def build_stage1_crf_system_prompt() -> str:
    keys_str = ", ".join([f'"{k}"' for k in STAGE1_KEYS_9])
    return f"""You are an expert clinical information compression engine.

Goal: Convert ONE clinical note into a compact, reliable intermediate representation for a downstream CRF-filling model.
This is Stage 1 only (NOT the final CRF output).

OUTPUT CONTRACT (STRICT):
1) Output MUST be a single valid JSON object.
2) Output JSON only: no markdown, no code fences, no extra text.
3) The FIRST character MUST be "{{".
4) Do NOT output analysis, chain-of-thought, or "thinking".
5) Use EXACTLY these keys (no extra keys, no missing keys):
   {keys_str}

VALUE FORMAT (STRICT):
- Each value MUST be a string.
- If not stated: use exactly "not stated".
- If multiple lines are needed: use "\\n" (two characters) inside the JSON string.
- Prefer short "Key: Value" lines inside each category string.

CONTENT (STAGE-2 CONDITIONING):
- Stage 1 is an intermediate compressed representation; Stage 2 will produce the official 134-item CRF JSON.
- Prioritize compression: keep only CRF-relevant facts, numeric values, and explicit negations.
- Use the note’s wording and concrete values (numbers, units, test names, findings).
- Do NOT guess or infer from absence. If uncertain, state uncertainty in plain text.
- When possible, start each line with the EXACT CRF item name (verbatim), then a colon, then the raw value/evidence.
  - Example: "presence of dyspnea: present" or "spo2: 95% RA"
  - If you are not sure about the exact CRF item string, write a normal "Key: Value" line instead (do not invent item strings).

ANTI-DRIFT / STABILITY RULES (MUST FOLLOW):
- NEVER repeat the same line multiple times. Deduplicate aggressively.
- Keep each field SHORT:
  - Max 8 lines per field.
  - Max ~400 characters per field.
- MEDICATIONS / PROCEDURES: list at most 10 unique entries total; if more, add "… (more)" once.
- If you start repeating yourself, STOP and output the JSON object immediately.

STYLE EXAMPLE (illustrative placeholders, not from the note):
{{
  "DEMOGRAPHICS": "Age: 77\\nSocial: lives with family",
  "VITALS": "SpO2: 95% RA\\nBP: 160/90",
  "LABS": "WBC: 24530\\nCRP: 0.69",
  "PROBLEMS": "Pneumonia\\nCOPD exacerbation",
  "SYMPTOMS": "Dyspnea: present\\nChest pain: denied",
  "MEDICATIONS": "Oxygen: administered\\nSteroids: given",
  "PROCEDURES": "Chest X-ray: performed, abnormal",
  "UTILIZATION": "ED: recurrent events\\nConsult: neurology",
  "DISPOSITION": "Disposition: not stated"
}}

CLUSTER GUIDANCE (9 clusters):
- DEMOGRAPHICS: patient context, PMH, social/housing, baseline.
- VITALS: RR/Temp/HR/BP/SpO2 and severity cues.
- LABS: blood gas / labs / biomarkers and key abnormal/normal statements.
- PROBLEMS: diagnoses and clinical problems (acute/chronic).
- SYMPTOMS: presenting symptoms and course.
- MEDICATIONS: chronic meds and treatments administered.
- PROCEDURES: imaging/tests/procedures performed.
- UTILIZATION: ED events, consults, monitoring, escalation.
- DISPOSITION: discharge outcome/follow-up if stated (else not stated).
"""


def build_stage1_crf_user_prompt(note_text: str) -> str:
    return f"Clinical Note:\\n{note_text}\\n\\nReturn the Stage 1 JSON now:"
