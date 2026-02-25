from __future__ import annotations

import re
from typing import Any

from llm_structcore.stage2_items import CRF_ITEMS_BY_CLUSTER

_KEY_CANON_MAP_LOWER: dict[str, str] = {
    "first episode of epilepsy": "first episod of epilepsy",
    "neurodegenerative disease": "neurodegenerative diseases",
    "presence of chest pain": "chest pain",
}

_ABBREV_KEY_MAP_LOWER: dict[str, str] = {
    "spo2": "spo2",
    "sp02": "spo2",
    "sat02": "spo2",
    "sato2": "spo2",
    "o2 sat": "spo2",
    "o2 saturation": "spo2",
    "bp": "blood pressure",
    "blood pressure (bp)": "blood pressure",
    "heart rate": "heart rate",
    "hr": "heart rate",
    "resp rate": "respiratory rate",
    "rr": "respiratory rate",
    "respirations": "respiratory rate",
    "temp": "body temperature",
    "t": "body temperature",
    "temperature": "body temperature",
    "wbc": "leukocytes",
    "white blood cells": "leukocytes",
    "hgb": "hemoglobin",
    "hb": "hemoglobin",
    "plt": "platelets",
    "crp": "c-reactive protein",
    "c reactive protein": "c-reactive protein",
    "na": "blood sodium",
    "sodium": "blood sodium",
    "k": "blood potassium",
    "potassium": "blood potassium",
    "trop": "troponin",
    "bnp": "bnp or nt-pro-bnp",
    "nt-pro-bnp": "bnp or nt-pro-bnp",
    "pro-bnp": "bnp or nt-pro-bnp",
    "ecg": "ecg, any abnormality",
    "ekg": "ecg, any abnormality",
    "chest x-ray": "chest rx, any abnormalities",
    "chest xray": "chest rx, any abnormalities",
    "cxr": "chest rx, any abnormalities",
    "echo": "cardiac ultrasound, any abnormality",
    "echocardiogram": "cardiac ultrasound, any abnormality",
    "eeg": "eeg, any abnormality",
    "brain ct": "brain ct scan, any abnormality",
    "ct brain": "brain ct scan, any abnormality",
    "ct head": "brain ct scan, any abnormality",
    "head ct": "brain ct scan, any abnormality",
    "chest ct": "chest ct scan, any abnormality",
    "ct chest": "chest ct scan, any abnormality",
    "pci": "percutaneous coronary intervention",
    "cabg": "coronary artery bypass grafting",
    "copd": "chronic pulmonary disease",
    "asthma": "chronic pulmonary disease",
    "chf": "chronic cardiac failure",
    "heart failure": "chronic cardiac failure",
    "ckd": "chronic renal failure",
    "pe": "pulmonary embolism",
    "acs": "acute coronary syndrome",
    "tloc": "tloc during effort",  # Approximation
    "pacemaker": "presence of pacemaker",
    "defibrillator": "presence of defibrillator",
    "homeless": "homelessness",
    "lives alone": "living alone",
    "living alone": "living alone",
    "palliative": "palliative care",
}

_LINE_KV_RE = re.compile(r"^\s*([^:]{1,120})\s*:\s*(.*?)\s*$")

_SPECULATION_CUES_RE = re.compile(
    r"(?i)(?:\b(?:possible|possibly|probable|likely|unlikely|suspect(?:ed)?|"
    r"consider|consistent with|suggest(?:s|ed)?|vs\.?|rule\s*out|r/o|"
    r"differential|cannot exclude|may represent|concerning for)\b|\?)"
)

_ACS_SUPPORT_RE = re.compile(
    r"(?i)\b("
    r"stemi|nstemi|myocardial\s+infarction|acute\s+coronary\s+syndrome|acs|"
    r"heart\s+attack|active\s+ischemia|ischemic\s+changes|coronary\s+occlusion"
    r")\b"
)

_DYSPNEA_SUPPORT_RE = re.compile(
    r"(?i)\b("
    r"dyspn(?:ea|oea)|shortness\s+of\s+breath|sob|breathless(?:ness)?|"
    r"respiratory\s+distress|work\s+of\s+breathing|tachypnea|"
    r"dispnea|mancanza\s+di\s+respiro|fame\s+d'aria|affanno|respiro\s+corto"
    r")\b"
)

_DYSPNEA_NEGATION_RE = re.compile(
    r"(?i)\b("
    r"no\s+(?:dyspn(?:ea|oea)|shortness\s+of\s+breath|sob|dispnea)|"
    r"without\s+(?:dyspn(?:ea|oea)|shortness\s+of\s+breath|sob|dispnea)|"
    r"nega\s+(?:dispnea|mancanza\s+di\s+respiro)|"
    r"senza\s+(?:dispnea|mancanza\s+di\s+respiro)"
    r")\b"
)

_CANCER_EVIDENCE_RE = re.compile(
    r"(?i)\b("
    r"cancer|carcinoma|sarcoma|melanoma|lymphoma|leukemia|malignan(?:cy|t)|"
    r"tumou?r|metastat(?:ic|es)|met(?:s)?\b"
    r")\b"
)

_CANCER_ACTIVE_CUES_RE = re.compile(
    r"(?i)\b("
    r"active|progressi(?:ve|ng)|chemo(?:therapy)?|radiotherapy|radiation|"
    r"palliative|hospice|untreated|current(?:ly)\s+treated"
    r")\b"
)

_CANCER_INACTIVE_CUES_RE = re.compile(
    r"(?i)\b("
    r"excision|excised|resected|resection|removed|mastectomy|prostatectomy|"
    r"tumou?r\s+resection|cancer\s+resection|in\s+remission|remission|"
    r"no\s+evidence\s+of\s+disease|ned"
    r")\b"
)

_ARRHYTHMIA_SUPPORT_RE = re.compile(
    r"(?i)\b("
    r"atrial\s+fibrillation|a\.?\s*fib|afib|atrial\s+flutter|"
    r"svt|supraventricular\s+tachycardia|ventricular\s+tachycardia|\bvt\b|"
    r"ventricular\s+fibrillation|\bvf\b|arrhythmia|dysrhythmia|heart\s+block|"
    r"bradycardia|tachycardia|irregular(?:ly\s+irregular)?\s+rhythm"
    r")\b"
)

_ARRHYTHMIA_NEGATION_RE = re.compile(
    r"(?i)\b("
    r"normal\s+sinus\s+rhythm|nsr|regular\s+rate\s+and\s+rhythm|rrr|"
    r"no\s+arrhythmia|sinus\s+rhythm"
    r")\b"
)

_MEDS_ANTIPLATELET_ANTICOAG = {
    "aspirin", "asa", "clopidogrel", "plavix", "ticagrelor", "brilinta",
    "prasugrel", "effient", "warfarin", "coumadin", "apixaban", "eliquis",
    "rivaroxaban", "xarelto", "dabigatran", "pradaxa", "edoxaban", "savaysa",
    "enoxaparin", "lovenox", "heparin", "fondaparinux", "arixtra",
}

_MEDS_ANTIHYPERTENSIVES = {
    "enalapril", "lisinopril", "ramipril", "perindopril", "captopril",
    "losartan", "valsartan", "irbesartan", "candesartan", "olmesartan", "telmisartan",
    "amlodipine", "nifedipine", "diltiazem", "verapamil", "felodipine",
    "metoprolol", "atenolol", "bisoprolol", "carvedilol", "nebivolol", "propranolol",
    "hydrochlorothiazide", "hctz", "chlorthalidone", "indapamide",
    "spironolactone", "eplerenone", "furosemide", "torsemide", "bumetanide",
    "clonidine", "methyldopa", "doxazosin", "terazosin",
}

_CARDIO_DX_KEYWORDS = {
    "hypertension", "htn", "atrial fibrillation", "af", "heart failure",
    "chf", "coronary", "cad", "ischemic", "myocardial", "mi", "angina",
    "cardiomyopathy", "valvular",
}


def _build_allowed_key_canon_map(allowed_items: set[str]) -> dict[str, str]:
    m: dict[str, str] = {}
    for item in allowed_items:
        it = str(item).strip()
        if not it:
            continue
        m[it] = it
        m[it.lower()] = it
        m[it.replace("’", "'")] = it
        m[it.replace("’", "'").lower()] = it
        m[it.replace("'", "’")] = it
        m[it.replace("'", "’").lower()] = it
    return m


def _canon_item_key(
    k: str,
    *,
    allowed_canon_map: dict[str, str],
    umls_alias_map: dict[str, str] | None = None,
) -> str:
    s = (k or "").strip()
    if not s:
        return s
    s = re.sub(r"^[\\-\\*•]\s*", "", s).strip()
    low = s.lower().strip()
    if low.startswith("p "):
        s = s[2:].strip()
        low = s.lower().strip()
    if low in _KEY_CANON_MAP_LOWER:
        s = _KEY_CANON_MAP_LOWER[low]
        low = s.lower().strip()
    if low in _ABBREV_KEY_MAP_LOWER:
        s = _ABBREV_KEY_MAP_LOWER[low]
        low = s.lower().strip()
    for pref in ("current ", "past "):
        if low.startswith(pref):
            s = s[len(pref) :].strip()
            low = s.lower().strip()
            break
    if s in allowed_canon_map:
        return allowed_canon_map[s]
    if low in allowed_canon_map:
        return allowed_canon_map[low]
    if umls_alias_map:
        if s in umls_alias_map:
            return umls_alias_map[s]
        if low in umls_alias_map:
            return umls_alias_map[low]
    return s


def _extract_kv_from_stage1_text(
    stage1_text: str,
    *,
    allowed_items: set[str],
    allowed_canon_map: dict[str, str],
    umls_alias_map: dict[str, str] | None = None,
) -> dict[str, str]:
    kv: dict[str, str] = {}
    for raw_line in (stage1_text or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("## "):
            continue
        line2 = re.sub(r"^[-*•]\s*", "", line).strip()
        if not line2:
            continue
        m = _LINE_KV_RE.match(line2)
        if not m:
            continue
        k_raw, v_raw = m.group(1).strip(), m.group(2).strip()
        if not k_raw or not v_raw:
            continue
        if v_raw.lower() in ("not stated", "unknown"):
            continue
        k = _canon_item_key(k_raw, allowed_canon_map=allowed_canon_map, umls_alias_map=umls_alias_map)
        if k not in allowed_items:
            continue
        if k in kv:
            continue
        kv[k] = v_raw
    return kv


def _token_set_from_stage1_medications(stage1_summary: dict[str, object]) -> set[str]:
    meds_raw = str(stage1_summary.get("MEDICATIONS") or "")
    toks: set[str] = set()
    for raw_line in meds_raw.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^[-*•]\s*", "", line).strip()
        if not line:
            continue
        m = _LINE_KV_RE.match(line)
        if m:
            key = re.sub(r"\s+", " ", m.group(1).strip().lower())
            val = re.sub(r"\s+", " ", m.group(2).strip().lower())
            if key:
                toks.add(key)
            if val:
                toks.add(val)
            for t in re.findall(r"[a-z0-9]+", f"{key} {val}".strip()):
                if len(t) >= 3:
                    toks.add(t)
            continue
        low = re.sub(r"\s+", " ", line.lower()).strip()
        if low:
            toks.add(low)
        for t in re.findall(r"[a-z0-9]+", low):
            if len(t) >= 3:
                toks.add(t)
    return toks


def _derive_from_stage1_text(
    *,
    stage1_summary: dict[str, object],
    stage1_text: str,
    kv: dict[str, str],
    allowed_items: set[str],
) -> dict[str, str]:
    out = dict(kv)
    t = (stage1_text or "").lower()
    meds = _token_set_from_stage1_medications(stage1_summary)
    meds_join = " ".join(sorted(meds))

    def has_kw(kw: str) -> bool:
        k = kw.lower().strip()
        if not k:
            return False
        if len(k) <= 3:
            return bool(re.search(rf"(?<![a-z0-9]){re.escape(k)}(?![a-z0-9])", t))
        return k in t

    if "poly-pharmacological therapy" in allowed_items and "poly-pharmacological therapy" not in out:
        if any(has_kw(k) for k in ("poly-pharmacological therapy", "polypharmacy", "polytherapy")):
            out["poly-pharmacological therapy"] = "present"
        else:
            meds_lines = [ln for ln in str(stage1_summary.get("MEDICATIONS") or "").splitlines() if ln.strip()]
            per_med = [ln for ln in meds_lines if re.match(r"(?i)^medication\s*:", ln.strip())]
            if len({ln.strip().lower() for ln in per_med}) >= 8:
                out["poly-pharmacological therapy"] = "present"

    if "antihypertensive therapy" in allowed_items and "antihypertensive therapy" not in out:
        if any(m in meds for m in _MEDS_ANTIHYPERTENSIVES) or any(m in meds_join for m in _MEDS_ANTIHYPERTENSIVES):
            out["antihypertensive therapy"] = "present"

    if "anticoagulants or antiplatelet drug therapy" in allowed_items and "anticoagulants or antiplatelet drug therapy" not in out:
        if any(m in meds for m in _MEDS_ANTIPLATELET_ANTICOAG) or any(m in meds_join for m in _MEDS_ANTIPLATELET_ANTICOAG):
            out["anticoagulants or antiplatelet drug therapy"] = "present"

    if "cardiovascular diseases" in allowed_items and "cardiovascular diseases" not in out:
        if any(has_kw(k) for k in _CARDIO_DX_KEYWORDS) or ("antihypertensive therapy" in out) or ("anticoagulants or antiplatelet drug therapy" in out):
            out["cardiovascular diseases"] = "present"

    if "chronic metabolic failure" in allowed_items and "chronic metabolic failure" not in out:
        if any(has_kw(k) for k in ("diabetes", "dm", "metformin")):
            out["chronic metabolic failure"] = "chronic"

    if "diffuse vascular disease" in allowed_items and "diffuse vascular disease" not in out:
        if any(has_kw(k) for k in ("cad", "coronary", "triple vessel", "multivessel", "three-vessel", "3-vessel")):
            out["diffuse vascular disease"] = "present"

    if "history of allergy" in allowed_items and "history of allergy" not in out:
        if "allergy" in t or "allergies" in t or "nkda" in t:
            if any(neg in t for neg in ("no known allergies", "no allergies", "nkda")):
                out["history of allergy"] = "absent"
            else:
                out["history of allergy"] = "present"

    if "active neoplasia" in allowed_items and "active neoplasia" not in out:
        if _CANCER_EVIDENCE_RE.search(stage1_text or ""):
            out["active neoplasia"] = "possibly active"

    if "transaminases" in allowed_items and "transaminases" not in out:
        m_ast = re.search(r"(?<![a-z0-9])ast(?![a-z0-9])[^0-9]{0,12}([-+]?\\d+(?:[\.,]\d+)?)", t)
        if m_ast:
            out["transaminases"] = m_ast.group(1).replace(",", ".")

    return out


def _strip_kv_lines(stage1_text: str, keys_lower: set[str]) -> str:
    if not stage1_text:
        return ""
    out_lines: list[str] = []
    for raw_line in stage1_text.splitlines():
        line = raw_line.strip()
        if not line:
            out_lines.append(raw_line)
            continue
        m = _LINE_KV_RE.match(re.sub(r"^[-*•]\s*", "", line).strip())
        if not m:
            out_lines.append(raw_line)
            continue
        k = m.group(1).strip().lower()
        if k in keys_lower:
            continue
        out_lines.append(raw_line)
    return "\n".join(out_lines)


def _is_positive_binary(v: str) -> bool:
    return (v or "").strip().lower() in {"y", "yes", "true", "present", "positive", "pos", "performed", "done", "given", "administered"}

def _is_negative_binary(v: str) -> bool:
    return (v or "").strip().lower() in {"n", "no", "false", "absent", "negative", "neg", "denied"}


def _apply_fp_gates(
    *,
    stage1_summary: dict[str, object],
    stage1_text: str,
    kv: dict[str, str],
) -> tuple[dict[str, str], dict[str, str]]:
    out = dict(kv)
    notes: dict[str, str] = {}
    evidence_text = _strip_kv_lines(
        stage1_text,
        keys_lower={"active neoplasia", "arrhythmia", "acute coronary syndrome"},
    )

    if "active neoplasia" in out:
        low_v = str(out["active neoplasia"]).strip().lower()
        if not _CANCER_EVIDENCE_RE.search(evidence_text or ""):
            out.pop("active neoplasia", None)
            notes["active neoplasia"] = "dropped: no cancer evidence"
        else:
            if _CANCER_ACTIVE_CUES_RE.search(evidence_text or ""):
                out["active neoplasia"] = "certainly active"
            elif _CANCER_INACTIVE_CUES_RE.search(evidence_text or ""):
                out["active neoplasia"] = "certainly not active"
            elif low_v in {"yes", "y", "present", "active", "certainly active"}:
                out["active neoplasia"] = "possibly active"
            elif low_v in {"no", "n", "absent", "inactive", "certainly not active"}:
                out["active neoplasia"] = "certainly not active"
            else:
                out["active neoplasia"] = "possibly active"

    if "arrhythmia" in out:
        v = str(out.get("arrhythmia") or "")
        if _is_positive_binary(v):
            if _ARRHYTHMIA_NEGATION_RE.search(evidence_text or ""):
                out.pop("arrhythmia", None)
            elif not _ARRHYTHMIA_SUPPORT_RE.search(evidence_text or ""):
                out.pop("arrhythmia", None)
        elif _is_negative_binary(v) and not _ARRHYTHMIA_NEGATION_RE.search(evidence_text or ""):
            out.pop("arrhythmia", None)

    if "acute coronary syndrome" in out:
        v = str(out.get("acute coronary syndrome") or "")
        if _is_positive_binary(v):
            if _SPECULATION_CUES_RE.search(v):
                out.pop("acute coronary syndrome", None)
            else:
                has_acs = bool(_ACS_SUPPORT_RE.search(evidence_text or ""))
                supportive = _is_positive_binary(out.get("chest pain", "")) or str(out.get("troponin", "")).strip() or _is_positive_binary(out.get("ecg, any abnormality", ""))
                if not has_acs or not supportive:
                    out.pop("acute coronary syndrome", None)
        elif _is_negative_binary(v) and not re.search(r"(?i)\bno\s+(?:acs|acute\s+coronary\s+syndrome|stemi|nstemi|mi)\b", evidence_text or ""):
            out.pop("acute coronary syndrome", None)

    if "presence of dyspnea" in out:
        v = str(out.get("presence of dyspnea") or "")
        if _is_positive_binary(v):
            if _DYSPNEA_NEGATION_RE.search(evidence_text or ""):
                out.pop("presence of dyspnea", None)
            elif not _DYSPNEA_SUPPORT_RE.search(evidence_text or ""):
                out.pop("presence of dyspnea", None)
        elif _is_negative_binary(v) and not _DYSPNEA_NEGATION_RE.search(evidence_text or ""):
            out.pop("presence of dyspnea", None)

    return out, notes

def compile_stage1_to_stage2(
    summary: dict[str, object],
    stage1_text: str,
    allowed_items_global: set[str],
    fp_gates: bool = True,
    umls_alias_map: dict[str, str] | None = None
) -> dict[str, str]:
    from llm_structcore.stage1_prompt import detect_present_clusters
    present_clusters = detect_present_clusters(summary)
    
    allowed_items_doc: list[str] = []
    for c in present_clusters:
        allowed_items_doc.extend([i for i in (CRF_ITEMS_BY_CLUSTER.get(c) or []) if i in allowed_items_global])
    allowed_items_doc.extend([i for i in (CRF_ITEMS_BY_CLUSTER.get("DEMOGRAPHICS") or []) if i in allowed_items_global])
    allowed_items_doc_set = set(allowed_items_doc) or allowed_items_global

    allowed_canon_map = _build_allowed_key_canon_map(allowed_items_doc_set)

    kv = _extract_kv_from_stage1_text(
        stage1_text, allowed_items=allowed_items_doc_set, allowed_canon_map=allowed_canon_map, umls_alias_map=umls_alias_map
    )
    kv = _derive_from_stage1_text(
        stage1_summary=summary, stage1_text=stage1_text, kv=kv, allowed_items=allowed_items_doc_set
    )
    if fp_gates:
        kv, _ = _apply_fp_gates(stage1_summary=summary, stage1_text=stage1_text, kv=kv)
        
    from llm_structcore.crf_prompt import CHRONIC_ITEMS, NEOPLASIA_ITEMS, DURATION_ITEMS
    chronic_set = set(CHRONIC_ITEMS)
    neoplasia_set = set(NEOPLASIA_ITEMS)
    duration_set = set(DURATION_ITEMS)

    out_adjusted = {}
    for item, value in kv.items():
        v = str(value).strip()
        low = v.lower()
        if item in chronic_set:
            if low in ("present", "yes", "y", "chronic"): out_adjusted[item] = "yes"; continue
            if low in ("absent", "no", "n", "not chronic"): out_adjusted[item] = "no"; continue
        if item in neoplasia_set:
            if low in ("present", "yes", "y", "active"): out_adjusted[item] = "yes"; continue
            if low in ("absent", "no", "n", "not active", "inactive"): out_adjusted[item] = "no"; continue
        if item in duration_set:
            if low in ("short", "long"): out_adjusted[item] = low; continue
            if any(u in low for u in ("sec", "min")): out_adjusted[item] = "short"; continue
            if any(u in low for u in ("hour", "day")): out_adjusted[item] = "long"; continue
        out_adjusted[item] = v

    return out_adjusted
