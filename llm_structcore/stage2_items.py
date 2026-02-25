from __future__ import annotations

"""
Stage 2 prompt for CL4Health2026 CRF:filling.

This variant keeps the item list grouped into sections aligned to the Stage 1 clusters:
DEMOGRAPHICS, VITALS, LABS, PROBLEMS, SYMPTOMS, MEDICATIONS, PROCEDURES, UTILIZATION, DISPOSITION.

Important: Stage 2 should consume the Stage 1 structured summary (not the raw note).
"""

from collections.abc import Iterable

# Official CRF items grouped into clusters aligned with Stage 1.
# Total items: 134 (union of all cluster lists below).
CRF_ITEMS_BY_CLUSTER: dict[str, list[str]] = {
    "DEMOGRAPHICS": [
        "duration of the patient's consciousness recovery",
        "duration of the patient's unconsciousness",
        "first episod of epilepsy",
        "known history of epilepsy",
        "history of allergy",
        "history of recent trauma",
        "pregnancy",
        "history of drug abuse",
        "history of alcohol abuse",
        "anticoagulants or antiplatelet drug therapy",
        "presence of prodromal symptoms",
        "compliance with antiepileptic therapy",
        "antiepileptic therapy already in place",
        "poly-pharmacological therapy",
        "diffuse vascular disease",
        "neuropsychiatric disorders",
        "presence of pacemaker",
        "presence of defibrillator",
        "antihypertensive therapy",
        "cardiovascular diseases",
        "neurodegenerative diseases",
        "peripheral neuropathy",
        "immunosuppression",
        "palliative care",
        "problematic family context",
        "need but absence of a caregiver",
        "homelessness",
        "living alone",
    ],
    "VITALS": [
        "level of autonomy (mobility)",
        "level of consciousness",
        "respiratory rate",
        "body temperature",
        "heart rate",
        "blood pressure",
        "presence of respiratory distress",
        "spo2",
    ],
    "LABS": [
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
        "sars-cov-2 swab test",
    ],
    "PROBLEMS": [
        "chronic pulmonary disease",
        "chronic respiratory failure",
        "chronic cardiac failure",
        "chronic renal failure",
        "chronic metabolic failure",
        "chronic rheumatologic disease",
        "active neoplasia",
        "chronic dialysis",
        "heart failure",
        "pneumonia",
        "copd exacerbation",
        "acute pulmonary edema",
        "asthma exacerbation",
        "respiratory failure",
        "intoxication",
        "covid 19",
        "influenza and various infections",
        "pneumothorax",
        "situational syncope",
        "epilepsy / epileptic seizure",
        "pulmonary embolism",
        "arrhythmia",
        "cardiac tamponade",
        "aortic dissection",
        "acute coronary syndrome",
        "hemorrhage",
        "severe anemia",
        "concussive head trauma",
    ],
    "SYMPTOMS": [
        "tloc during effort",
        "tloc while supine",
        "drowsiness, confusion, disorientation as postcritical state",
        "stiffness during the episode",
        "drooling during the episode",
        "tonic-clonic seizures",
        "pale skin during the episode",
        "eye deviation during the episode",
        "situation description, like coughing, prolonged periods of straining, sudden abdominal pain, phlebotomy",
        "chest pain",
        "head or other districts trauma",
        "tongue bite",
        "agitation",
        "foreign body in the airways",
        "improvement of dyspnea",
        "presence of dyspnea",
        "dementia",
        "general condition deterioration",
        "ab ingestis pneumonia",
        "further seizures in the ed",
        "improvement of patient’s conditions",
    ],
    "MEDICATIONS": [
        "administration of diuretics",
        "administration of steroids",
        "administration of bronchodilators",
        "administration of oxygen/ventilation",
        "blood transfusions",
        "administration of fluids",
    ],
    "PROCEDURES": [
        "ecg, any abnormality",
        "ecg monitoring, any abnormality",
        "eeg, any abnormality",
        "thoracic ultrasound, any abnormalities",
        "chest rx, any abnormalities",
        "gastroscopy , any abnormalities",
        "brain ct scan, any abnormality",
        "brain mri, any abnormality",
        "cardiac ultrasound, any abnormality",
        "chest ct scan, any abnormality",
        "pulmonary scintigraphy, any abnormality",
        "abdomen ct scan, any abnormality",
        "compression ultrasound (cus), any abnormality",
        "carotid sinus massage",
        "supine-to-standing systolic blood pressure test",
        "blood in the stool",
        "performance of thoracentesis",
    ],
    "UTILIZATION": [
        "neurologist consultation",
        "cardio-pulmonary resuscitation",
    ],
    "DISPOSITION": [],
}

STAGE2_CLUSTER_ORDER: list[str] = [
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


def crf_items_for_clusters(clusters: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for c in clusters:
        for item in CRF_ITEMS_BY_CLUSTER.get(c, []):
            if item in seen:
                continue
            seen.add(item)
            out.append(item)
    return out


def _clusters_block(allowed_clusters: Iterable[str]) -> str:
    cl = [c for c in STAGE2_CLUSTER_ORDER if c in set(allowed_clusters)]
    if not cl:
        cl = STAGE2_CLUSTER_ORDER[:]
    return ", ".join(cl)


def _items_block(allowed_clusters: Iterable[str] | None) -> str:
    allowed_set = set(allowed_clusters) if allowed_clusters is not None else set(STAGE2_CLUSTER_ORDER)
    parts: list[str] = []
    for c in STAGE2_CLUSTER_ORDER:
        if c not in allowed_set:
            continue
        items = CRF_ITEMS_BY_CLUSTER.get(c) or []
        if not items:
            continue
        items_list = "\n".join([f"- {i}" for i in items])
        parts.append(f"## {c}\n{items_list}\n")
    return "\n".join(parts).strip()


def build_stage2_system_prompt_clustered(*, allowed_clusters: Iterable[str] | None = None) -> str:
    """
    Build Stage 2 system prompt.

    Strategy: include the full ontology ONLY for clusters that are present in Stage 1.
    If `allowed_clusters` is None, include all clusters.
    """
    clusters_str = _clusters_block(allowed_clusters or STAGE2_CLUSTER_ORDER)
    items_block = _items_block(allowed_clusters) or "(no items provided)"
    prompt = """You are an expert clinical NLP expert. Your task is to review the provided Stage 1 structured summary (derived from a clinical discharge note) and extract specific CRF item values into the official CRF JSON format.

The Stage 1 input is organized into these clusters:
DEMOGRAPHICS, VITALS, LABS, PROBLEMS, SYMPTOMS, MEDICATIONS, PROCEDURES, UTILIZATION, DISPOSITION.

This document’s Stage 1 summary contains evidence for these clusters:
__STAGE2_ALLOWED_CLUSTERS__

OUTPUT FORMAT (STRICT):
1) Output MUST be a single valid JSON object.
2) Output JSON only: no markdown, no code fences, no extra text.
3) The FIRST character MUST be "{".
4) The JSON object MUST have EXACTLY ONE top-level key: "predictions".
5) "predictions" MUST be a JSON array of objects, each with EXACTLY two keys: "item" and "prediction".
6) Do NOT output any other keys (no item->value dict, no extra metadata).
7) HARD PREFIX LOCK: if you output anything, the first characters MUST be exactly: {"predictions":[

Schema:
{
  "predictions": [
    {"item": "<exact CRF item name>", "prediction": "<value>"}
  ]
}

ANTI-DRIFT / STABILITY RULES (MUST FOLLOW):
- NEVER repeat the same item multiple times.
- NEVER invent new item names. "item" must EXACTLY match one of the CRF item names listed below.
- NEVER use medication names, lab names, or free text as JSON keys. Only use the schema keys: predictions/item/prediction.
- Keep output SPARSE: omit items that would be "unknown".
- If you start repeating, STOP and output the JSON object immediately.
- If you cannot extract any non-unknown items, output exactly: {"predictions":[]}

Compatibility fallback (only if you absolutely cannot follow the schema above):
- You may output a single JSON object mapping { "<exact CRF item name>": "<value>", ... }.
- Do NOT include any keys other than CRF item names in this fallback object.

CRITICAL GATING RULE:
- Extract ONLY information explicitly stated in the provided Stage 1 summary.
- Do NOT use outside medical knowledge.
- Do NOT infer from absence.

Stage 1 synonym/abbreviation mapping (use when clearly supported by the Stage 1 text):
- SpO2 / O2 sat -> spo2
- BP -> blood pressure
- HR -> heart rate
- RR -> respiratory rate
- Temp / T -> body temperature
- pH -> ph
- pO2 -> pa02
- pCO2 -> pac02
- HCO3- / HCO3 -> hc03-
- Lactate / Lac -> lactates
- WBC -> leukocytes
- CRP -> c-reactive protein
- Na+ -> blood sodium
- K+ -> blood potassium
- Glucose -> blood glucose

If Stage 1 provides numeric vitals, map to categorical vocab using standard thresholds:
- heart rate: <60 bradycardic, 60–100 normocardic, >100 tachycardic
- respiratory rate: <12 bradypneic, 12–20 eupneic, >20 tachypneic
- body temperature (°C): <36 hypothermic, 36–37.9 normothermic, >=38 hyperthermic
- blood pressure (mmHg): SBP<90 or DBP<60 hypotensive; SBP>=140 or DBP>=90 hypertensive; otherwise normotensive
- level of consciousness: if only GCS is provided, approximate AVPU as: 15->A, 9–14->V, <=8->P

FEW-SHOT EXAMPLES (FORMAT ONLY; values are generic placeholders):

Example 1 (Sparse output; binary + pos/neg + measurement):
Input (Stage 1 summary excerpt):
## VITALS
spo2: stated
## SYMPTOMS
presence of dyspnea: present
## PROCEDURES
sars-cov-2 swab test: negative

Correct output:
{
  "predictions": [
    {"item": "presence of dyspnea", "prediction": "y"},
    {"item": "spo2", "prediction": "measured"},
    {"item": "sars-cov-2 swab test", "prediction": "neg"}
  ]
}

Example 2 (Chronic + respiratory distress time):
Input (Stage 1 summary excerpt):
## PROBLEMS
COPD: chronic
## VITALS
presence of respiratory distress: current

Correct output:
{
  "predictions": [
    {"item": "chronic pulmonary disease", "prediction": "possibly chronic"},
    {"item": "presence of respiratory distress", "prediction": "current"}
  ]
}

Example 3 (Duration + consciousness):
Input (Stage 1 summary excerpt):
## SYMPTOMS
duration of the patient's unconsciousness: prolonged
## VITALS
level of consciousness: alert

Correct output:
{
  "predictions": [
    {"item": "duration of the patient's unconsciousness", "prediction": "long"},
    {"item": "level of consciousness", "prediction": "A"}
  ]
}

You must extract values for the following CRF items (EXACT item names; do not paraphrase).
Only these items are allowed in your output:

__STAGE2_ITEMS_BLOCK__

## ANSWER FORMAT RULES

Follow these rules EXACTLY for each item type:

1) Binary items (default for most items):
For all items NOT listed in categories 2-13 below, use:
- \"y\" = present, confirmed, or yes
- \"n\" = explicitly stated as absent or no
- \"unknown\" = not mentioned or cannot be inferred

2) Chronic disease items:
For: \"chronic pulmonary disease\", \"chronic respiratory failure\", \"chronic cardiac failure\", \"chronic renal failure\", \"chronic metabolic failure\", \"chronic rheumatologic disease\", \"chronic dialysis\"
Use ONLY: \"certainly chronic\" | \"possibly chronic\" | \"certainly not chronic\" | \"unknown\"

3) Active neoplasia:
For: \"active neoplasia\"
Use ONLY: \"certainly active\" | \"possibly active\" | \"certainly not active\" | \"unknown\"

4) Lab values and vital sign measurements:
For: \"spo2\", \"ph\", \"pa02\", \"pac02\", \"hc03-\", \"lactates\", \"hemoglobin\", \"platelets\", \"leukocytes\", \"c-reactive protein\", \"blood sodium\", \"blood potassium\", \"blood glucose\", \"creatinine\", \"transaminases\", \"inr\", \"troponin\", \"bnp or nt-pro-bnp\", \"d-dimer\", \"blood calcium\", \"serum creatinine kinase\", \"blood alcohol\", \"blood drug dosage\", \"urine drug test\"
- Extract the exact numeric value as written (percent values like \"95%\" are allowed if written that way)
- If measured but no value given, use \"measured\"
- If not documented, use \"unknown\"
- Do NOT add units unless they are part of the value as written

5) Positive/negative tests:
For: \"carotid sinus massage\", \"supine-to-standing systolic blood pressure test\", \"blood in the stool\", \"sars-cov-2 swab test\"
Use ONLY: \"pos\" | \"neg\" | \"unknown\"

6) Duration items:
For: \"duration of the patient's consciousness recovery\", \"duration of the patient's unconsciousness\"
Use ONLY: \"short\" | \"long\" | \"unknown\"

7) Level of consciousness:
For: \"level of consciousness\"
Use ONLY: \"A\" | \"V\" | \"P\" | \"unknown\"

8) Level of autonomy:
For: \"level of autonomy (mobility)\"
Use ONLY: \"walking independently\" | \"walking with auxiliary aids\" | \"walking with physical assistance\" | \"bedridden\" | \"unknown\"

9) Respiratory rate:
For: \"respiratory rate\"
Use ONLY: \"bradypneic\" | \"eupneic\" | \"tachypneic\" | \"unknown\"

10) Body temperature:
For: \"body temperature\"
Use ONLY: \"hypothermic\" | \"normothermic\" | \"hyperthermic\" | \"unknown\"

11) Heart rate:
For: \"heart rate\"
Use ONLY: \"bradycardic\" | \"normocardic\" | \"tachycardic\" | \"unknown\"

12) Blood pressure:
For: \"blood pressure\"
Use ONLY: \"hypotensive\" | \"normotensive\" | \"hypertensive\" | \"unknown\"

13) Presence of respiratory distress:
For: \"presence of respiratory distress\"
Use ONLY: \"current\" | \"past\" | \"unknown\"

## GENERAL EXTRACTION PRINCIPLES
- Extract ONLY information explicitly stated in the provided Stage 1 summary.
- NEVER guess, hallucinate, or assume information not present.
- Be precise with terminology: item names must EXACTLY match the CRF item names listed above.

## OUTPUT FORMAT (STRICT)
You must output a single valid JSON object with NO markdown, NO tags, NO extra text:

{
  \"predictions\": [
    {\"item\": \"<exact CRF item name>\", \"prediction\": \"<value>\"}
  ]
}

Rules:
- Each \"item\" must EXACTLY match one of the CRF item names listed above.
- Each \"prediction\" must use ONLY the allowed vocabulary for that item type.
- Each CRF item may appear AT MOST ONCE in the predictions array.

## SPARSE OUTPUT (STRONGLY RECOMMENDED)
- ONLY include items where you can provide a value OTHER than \"unknown\".
- OMIT items that would have \"unknown\".
- The downstream system will treat any missing item as \"unknown\".
"""

    return prompt.replace("__STAGE2_ALLOWED_CLUSTERS__", clusters_str).replace("__STAGE2_ITEMS_BLOCK__", items_block)
