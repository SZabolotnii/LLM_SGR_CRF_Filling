"""
Shared prompt builder for the CL4Health 2026 CRF Filling task.

Design:
- Train / run the model with the EXACT official controlled vocabulary
  so model output == submission format.
- Keep post-processing deterministic and minimal (edge-case cleanup only).
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────
# Item → category mappings (derived from NLP-FBK/dyspnea-valid-options)
# ──────────────────────────────────────────────────────────────────

CHRONIC_ITEMS: list[str] = [
    "chronic pulmonary disease",
    "chronic respiratory failure",
    "chronic cardiac failure",
    "chronic renal failure",
    "chronic metabolic failure",
    "chronic rheumatologic disease",
    "chronic dialysis",
]

NEOPLASIA_ITEMS: list[str] = ["active neoplasia"]

LAB_ITEMS: list[str] = [
    "spo2",
    "ph",
    "pa02",
    "pac02",
    "hc03-",
    "lactates",
    "hemoglobin",
    "platelets",
    "leukocytes",
    "c-reactive protein",
    "blood sodium",
    "blood potassium",
    "blood glucose",
    "creatinine",
    "transaminases",
    "inr",
    "troponin",
    "bnp or nt-pro-bnp",
    "d-dimer",
    "blood calcium",
    "serum creatinine kinase",
    "blood alcohol",
    "blood drug dosage",
    "urine drug test",
]

POSNEG_ITEMS: list[str] = [
    "carotid sinus massage",
    "supine-to-standing systolic blood pressure test",
    "blood in the stool",
    "sars-cov-2 swab test",
]

DURATION_ITEMS: list[str] = [
    "duration of the patient's consciousness recovery",
    "duration of the patient's unconsciousness",
]

# Single-item categories
CONSCIOUSNESS_ITEM = "level of consciousness"
AUTONOMY_ITEM = "level of autonomy (mobility)"
RESP_RATE_ITEM = "respiratory rate"
BODY_TEMP_ITEM = "body temperature"
HEART_RATE_ITEM = "heart rate"
BLOOD_PRESSURE_ITEM = "blood pressure"
RESP_DISTRESS_ITEM = "presence of respiratory distress"


def build_system_instruction(crf_items: list[str], *, sparse_output: bool = True) -> str:
    """
    Build the system instruction for the CRF extraction task.
    Uses the official controlled vocabulary.
    """
    items_list_str = "\n".join([f"- {i}" for i in crf_items])

    chronic_str = ", ".join([f'"{i}"' for i in crf_items if i in CHRONIC_ITEMS])
    neoplasia_str = ", ".join([f'"{i}"' for i in crf_items if i in NEOPLASIA_ITEMS])
    lab_str = ", ".join([f'"{i}"' for i in crf_items if i in LAB_ITEMS])
    posneg_str = ", ".join([f'"{i}"' for i in crf_items if i in POSNEG_ITEMS])
    duration_str = ", ".join([f'"{i}"' for i in crf_items if i in DURATION_ITEMS])

    sparse_rule = ""
    if sparse_output:
        sparse_rule = (
            "\n\nSPARSE OUTPUT (RECOMMENDED):\n"
            "- Only include items when you can output a value different from \"unknown\".\n"
            "- Missing items will be treated as \"unknown\" by the downstream normalizer.\n"
        )

    return f"""You are an expert clinical NLP extraction engine for the CL4Health 2026 shared task on CRF filling.
Your task is to review ONE clinical discharge note and extract the values for each Case Report Form (CRF) item listed below.

## Required CRF Items:
{items_list_str}

## Answer Format Rules (STRICT — follow exactly):

### 1) Binary items (presence / absence / yes-no questions)
For ALL items NOT listed in the special categories below, use:
- "y" if present/confirmed
- "n" if explicitly absent
- "unknown" if not mentioned / not inferable

### 2) Chronic disease items ({chronic_str})
Use: "certainly chronic" | "possibly chronic" | "certainly not chronic" | "unknown"

### 3) Active neoplasia ({neoplasia_str})
Use: "certainly active" | "possibly active" | "certainly not active" | "unknown"

### 4) Lab and vital sign measurements ({lab_str})
Extract the numeric value exactly as it appears (e.g. "7.6", "126", "96%").
If the item is measured but no value is given, use "measured".
If undocumented, use "unknown".

### 5) Positive/negative tests ({posneg_str})
Use: "pos" | "neg" | "unknown"

### 6) Duration items ({duration_str})
Use: "short" | "long" | "unknown"

### 7) Level of consciousness ("{CONSCIOUSNESS_ITEM}")
Use the AVPU scale: "A" (alert) | "V" (verbal) | "P" (pain) | "unknown"

### 8) Level of autonomy ("{AUTONOMY_ITEM}")
Use: "walking independently" | "walking with auxiliary aids" | "walking with physical assistance" | "bedridden" | "unknown"

### 9) Respiratory rate ("{RESP_RATE_ITEM}")
Use: "bradypneic" | "eupneic" | "tachypneic" | "unknown"

### 10) Body temperature ("{BODY_TEMP_ITEM}")
Use: "hypothermic" | "normothermic" | "hyperthermic" | "unknown"

### 11) Heart rate ("{HEART_RATE_ITEM}")
Use: "bradycardic" | "normocardic" | "tachycardic" | "unknown"

### 12) Blood pressure ("{BLOOD_PRESSURE_ITEM}")
Use: "hypotensive" | "normotensive" | "hypertensive" | "unknown"

### 13) Presence of respiratory distress ("{RESP_DISTRESS_ITEM}")
Use: "current" | "past" | "unknown"

### GENERAL RULES
- If the item is NOT explicitly stated or cannot be inferred from the note → use "unknown".
- NEVER guess, hallucinate, or infer from absence.

### Output format
Return a single valid JSON object ONLY in this EXACT shape:

{{
  "predictions": [
    {{"item": "<exact CRF item string>", "prediction": "<value>"}}
  ]
}}

Rules:
- "item" MUST be one of the CRF item strings listed above (exact match).
- "prediction" MUST follow the allowed vocab for that item.
- If you output multiple predictions, each CRF item MUST appear at most once.
- No markdown fences, no explanation, no extra text.{sparse_rule}"""


def build_user_prompt(note_text: str) -> str:
    return f"## Clinical Note:\n{note_text}\n\n## Extract CRF Items to JSON:\n"
