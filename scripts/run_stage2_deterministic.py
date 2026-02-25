#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from llm_structcore.normalize_predictions import coerce_predictions_mapping, predictions_to_submission  # noqa: E402
from llm_structcore.scoring import score_records  # noqa: E402
from llm_structcore.stage1_sgr import stage1_summary_to_text  # noqa: E402
from llm_structcore.submission import validate_submission_structure, write_jsonl, zip_codabench_jsonl  # noqa: E402
from llm_structcore.crf_prompt import CHRONIC_ITEMS, NEOPLASIA_ITEMS, DURATION_ITEMS  # noqa: E402

from llm_structcore.stage1_prompt import STAGE1_KEYS_9, detect_present_clusters  # noqa: E402
from llm_structcore.stage2_items import CRF_ITEMS_BY_CLUSTER  # noqa: E402
from llm_structcore.umls_mapping_utils import build_umls_alias_map  # noqa: E402

_KEY_CANON_MAP_LOWER: dict[str, str] = {
    # Known organizer typo: episod (official) vs episode (model)
    "first episode of epilepsy": "first episod of epilepsy",
    # Pluralization drift
    "neurodegenerative disease": "neurodegenerative diseases",
    # Common drift
    "presence of chest pain": "chest pain",
}

_ABBREV_KEY_MAP_LOWER: dict[str, str] = {
    # Vitals / objective abbreviations
    "spo2": "spo2",
    "sp02": "spo2",
    "sat02": "spo2",
    "sato2": "spo2",
    "o2 sat": "spo2",
    "o2 saturation": "spo2",
    "bp": "blood pressure",
    "hr": "heart rate",
    "hart rate": "heart rate",
    "rr": "respiratory rate",
    "temp": "body temperature",
    "temperature": "body temperature",
    "t": "body temperature",
    "gcs": "level of consciousness",
    "avpu": "level of consciousness",
    # Labs / blood gas
    "ph": "ph",
    "pco2": "pac02",
    "paco2": "pac02",
    "po2": "pa02",
    "pao2": "pa02",
    "hco3-": "hc03-",
    "hco3": "hc03-",
    "lac": "lactates",
    "lactate": "lactates",
    "hb": "hemoglobin",
    "hgb": "hemoglobin",
    "plt": "platelets",
    "plts": "platelets",
    "wbc": "leukocytes",
    "crp": "c-reactive protein",
    "pcr": "c-reactive protein",
    "tnt": "troponin",
    "trop": "troponin",
    "na+": "blood sodium",
    "na": "blood sodium",
    "k+": "blood potassium",
    "k": "blood potassium",
    # Common short-forms for symptoms/diagnoses
    "dyspnea": "presence of dyspnea",
    # Procedures / imaging shorthand (Stage1 drift recovery)
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
    "head ct": "brain ct scan, any abnormality",
    "brain mri": "brain mri, any abnormality",
    "chest ct": "chest ct scan, any abnormality",
    "abdomen ct": "abdomen ct scan, any abnormality",
    "abdominal ct": "abdomen ct scan, any abnormality",
    # Patient history shorthand
    "allergies": "history of allergy",
    "allergy": "history of allergy",
    "drug abuse": "history of drug abuse",
    "alcohol abuse": "history of alcohol abuse",
    "anticoagulants": "anticoagulants or antiplatelet drug therapy",
    "antiplatelet": "anticoagulants or antiplatelet drug therapy",
    "polypharmacy": "poly-pharmacological therapy",
    "polytherapy": "poly-pharmacological therapy",
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
    r"cannot\s*exclude|may\s*represent)\b|\?)"
)

_CANCER_EVIDENCE_RE = re.compile(
    r"(?i)\b("
    r"cancer|carcinoma|adenocarcinoma|neoplasm|tumou?r|malignan(?:t|cy)|"
    r"metasta(?:sis|tic)|lymphoma|leukem(?:ia|ic)|myeloma|sarcoma|"
    r"neoplasia|tumore|metastasi"
    r")\b"
)

_CANCER_ACTIVE_CUES_RE = re.compile(
    r"(?i)\b("
    r"metasta(?:sis|tic)|active\s+(?:cancer|malignan(?:cy|t)|neoplasm|disease)|"
    r"ongoing\s+(?:chemo|chemotherapy|radiotherapy|radiation|immunotherapy)|"
    r"on\s+(?:chemo|chemotherapy|radiotherapy|radiation|immunotherapy)|"
    r"currently\s+(?:on|receiving)\s+(?:chemo|chemotherapy|radiotherapy|radiation|immunotherapy)"
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
    r"irregular\s+rhythm|irregularly\s+irregular|palpitat(?:ion|ions)|"
    r"ectopic|pvc|premature\s+ventricular|"
    r"aritmi[ae]|fibrillazione\s+atriale|flutter\s+atriale"
    r")\b"
)

_ARRHYTHMIA_NEGATION_RE = re.compile(
    r"(?i)\b("
    r"no\s+(?:arrhythmia|af|afib|atrial\s+fibrillation)|sinus\s+rhythm|"
    r"regular\s+rhythm"
    r")\b"
)

_ACS_SUPPORT_RE = re.compile(
    r"(?i)\b("
    r"acute\s+coronary\s+syndrome|\bacs\b|\bnstemi\b|\bstemi\b|"
    r"myocardial\s+infarction|\bami\b|unstable\s+angina|"
    r"sindrome\s+coronarica\s+acuta|infarto\s+miocardico|angina\s+instabile"
    r")\b"
)

_DYSPNEA_SUPPORT_RE = re.compile(
    r"(?i)\b("
    r"dyspn(?:ea|oea)|shortness\s+of\s+breath|\bsob\b|"
    r"dispnea|mancanza\s+di\s+respiro"
    r")\b"
)
_DYSPNEA_NEGATION_RE = re.compile(
    r"(?i)\b("
    r"no\s+(?:dyspn(?:ea|oea)|shortness\s+of\s+breath|sob|dispnea|mancanza\s+di\s+respiro)|"
    r"denies\s+(?:dyspn(?:ea|oea)|shortness\s+of\s+breath|sob|dispnea)|"
    r"without\s+(?:dyspn(?:ea|oea)|shortness\s+of\s+breath|sob|dispnea)|"
    r"nega\s+(?:dispnea|mancanza\s+di\s+respiro)|"
    r"senza\s+(?:dispnea|mancanza\s+di\s+respiro)"
    r")\b"
)

_MEDS_ANTIHYPERTENSIVES = {
    # ACEi / ARB
    "enalapril",
    "lisinopril",
    "ramipril",
    "perindopril",
    "captopril",
    "losartan",
    "valsartan",
    "irbesartan",
    "candesartan",
    "olmesartan",
    "telmisartan",
    # CCB
    "amlodipine",
    "nifedipine",
    "diltiazem",
    "verapamil",
    # Beta blockers (brand + generic)
    "metoprolol",
    "seloken",
    "bisoprolol",
    "carvedilol",
    "atenolol",
    "nebivolol",
    # Diuretics often used for HTN/HF (keep conservative)
    "hydrochlorothiazide",
    "hctz",
}

_MEDS_ANTIPLATELET_ANTICOAG = {
    # Antiplatelets
    "aspirin",
    "asa",
    "cardioasa",
    "clopidogrel",
    "ticagrelor",
    "prasugrel",
    "dipyridamole",
    # DOACs / VKAs
    "warfarin",
    "coumadin",
    "apixaban",
    "eliquis",
    "rivaroxaban",
    "xarelto",
    "dabigatran",
    "pradaxa",
    "edoxaban",
    "lixiana",
    # Heparins
    "heparin",
    "enoxaparin",
    "clexane",
    "dalteparin",
    "fondaparinux",
    "arixtra",
}

_CARDIO_DX_KEYWORDS = {
    "hypertension",
    "htn",
    "atrial fibrillation",
    "af",
    "heart failure",
    "chf",
    "coronary",
    "cad",
    "ischemic",
    "myocardial",
    "mi",
    "angina",
    "cardiomyopathy",
    "valvular",
}


def _hf_rows(dataset: str, *, split: str, offset: int, length: int) -> dict[str, Any]:
    """
    Fetch rows from HF datasets-server.

    Note: /rows endpoint has a max `length` (commonly ~100). For larger `length`, we page.
    """
    length_i = int(length)
    offset_i = int(offset)
    if length_i <= 0:
        return {"rows": []}

    page_size = 100
    out_rows: list[dict[str, Any]] = []
    remaining = length_i
    while remaining > 0:
        take = min(page_size, remaining)
        qs = urllib.parse.urlencode(
            {
                "dataset": dataset,
                "config": "default",
                "split": split,
                "offset": str(offset_i),
                "length": str(take),
            }
        )
        url = f"https://datasets-server.huggingface.co/rows?{qs}"
        req = urllib.request.Request(url, headers={"User-Agent": "medgemma-crf-stage2-det/1.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read().decode("utf-8"))
        batch = data.get("rows") or []
        out_rows.extend(batch)
        if len(batch) < take:
            break
        offset_i += take
        remaining -= take

    return {"rows": out_rows}


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
    # Strip bullets commonly emitted by LLMs.
    s = re.sub(r"^[\\-\\*•]\\s*", "", s).strip()

    low = s.lower().strip()
    # Common drift: leading single-letter prefixes ("p platelets", "pO2" already handled via abbrev map).
    if low.startswith("p "):
        s = s[2:].strip()
        low = s.lower().strip()
    if low in _KEY_CANON_MAP_LOWER:
        s = _KEY_CANON_MAP_LOWER[low]
        low = s.lower().strip()
    if low in _ABBREV_KEY_MAP_LOWER:
        s = _ABBREV_KEY_MAP_LOWER[low]
        low = s.lower().strip()

    # Remove common accidental prefixes that are not part of the item string.
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
    """
    Deterministic extraction from Stage1 "Key: Value" lines.
    Conservative by design: emits only facts already present in Stage1.
    """
    kv: dict[str, str] = {}
    for raw_line in (stage1_text or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("## "):
            continue
        # Support bullet-ish lines.
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


    # Note: we intentionally avoid mapping bare (no ':') lines into CRF items here.
    # It tends to increase FP due to label noise and ambiguity (performed vs abnormal, home meds vs administered).


def _token_set_from_stage1_medications(stage1_summary: dict[str, object]) -> set[str]:
    """
    Extract a conservative set of medication name tokens from the Stage1 MEDICATIONS cluster.
    """
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
        # Non-KV medication line: keep conservative tokens from the whole line.
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
    """
    Derive a small set of high-FN CRF flags from Stage1 evidence text.
    This is still "evidence-only" because it uses ONLY Stage1 content.
    """
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

    # Poly-pharmacological therapy:
    # Avoid over-triggering on rich medication lists (high FP). Prefer explicit mention in Stage1 text.
    if "poly-pharmacological therapy" in allowed_items and "poly-pharmacological therapy" not in out:
        if any(has_kw(k) for k in ("poly-pharmacological therapy", "polypharmacy", "polytherapy")):
            out["poly-pharmacological therapy"] = "present"
        else:
            meds_lines = [ln for ln in str(stage1_summary.get("MEDICATIONS") or "").splitlines() if ln.strip()]
            per_med = [ln for ln in meds_lines if re.match(r"(?i)^medication\\s*:", ln.strip())]
            # Keep a high threshold for implicit trigger to protect precision.
            if len({ln.strip().lower() for ln in per_med}) >= 8:
                out["poly-pharmacological therapy"] = "present"

    # Antihypertensive therapy: detect common antihypertensive meds in Stage1 MEDICATIONS.
    if "antihypertensive therapy" in allowed_items and "antihypertensive therapy" not in out:
        if any(m in meds for m in _MEDS_ANTIHYPERTENSIVES) or any(m in meds_join for m in _MEDS_ANTIHYPERTENSIVES):
            out["antihypertensive therapy"] = "present"

    # Anticoagulants or antiplatelet drug therapy: detect common agents in Stage1 MEDICATIONS.
    if (
        "anticoagulants or antiplatelet drug therapy" in allowed_items
        and "anticoagulants or antiplatelet drug therapy" not in out
    ):
        if any(m in meds for m in _MEDS_ANTIPLATELET_ANTICOAG) or any(m in meds_join for m in _MEDS_ANTIPLATELET_ANTICOAG):
            out["anticoagulants or antiplatelet drug therapy"] = "present"

    # Cardiovascular diseases: detect direct PMH keywords, or infer from therapy flags.
    if "cardiovascular diseases" in allowed_items and "cardiovascular diseases" not in out:
        cardio_kw_hit = any(has_kw(k) for k in _CARDIO_DX_KEYWORDS)
        therapy_hit = ("antihypertensive therapy" in out) or ("anticoagulants or antiplatelet drug therapy" in out)
        if cardio_kw_hit or therapy_hit:
            out["cardiovascular diseases"] = "present"

    # Chronic metabolic failure (chronic disease): proxy by explicit diabetes/metformin mention.
    if "chronic metabolic failure" in allowed_items and "chronic metabolic failure" not in out:
        if any(has_kw(k) for k in ("diabetes", "dm", "metformin")):
            out["chronic metabolic failure"] = "chronic"

    # Diffuse vascular disease: proxy by explicit CAD/multivessel/coronary disease mention.
    if "diffuse vascular disease" in allowed_items and "diffuse vascular disease" not in out:
        if any(has_kw(k) for k in ("cad", "coronary", "triple vessel", "multivessel", "three-vessel", "3-vessel")):
            out["diffuse vascular disease"] = "present"

    # History of allergy: detect allergy mention; treat explicit negation as "absent".
    if "history of allergy" in allowed_items and "history of allergy" not in out:
        if "allergy" in t or "allergies" in t or "nkda" in t:
            if any(neg in t for neg in ("no known allergies", "no allergies", "nkda")):
                out["history of allergy"] = "absent"
            else:
                out["history of allergy"] = "present"

    # Active neoplasia: keep conservative. We only create a weak signal when cancer is explicitly mentioned in Stage1.
    # More precise gating happens later in _apply_fp_gates().
    if "active neoplasia" in allowed_items and "active neoplasia" not in out:
        if _CANCER_EVIDENCE_RE.search(stage1_text or ""):
            out["active neoplasia"] = "possibly active"

    # Transaminases: if Stage1 mentions AST/ALT but did not output "transaminases".
    if "transaminases" in allowed_items and "transaminases" not in out:
        m_ast = re.search(r"(?<![a-z0-9])ast(?![a-z0-9])[^0-9]{0,12}([-+]?\\d+(?:[\\.,]\\d+)?)", t)
        if m_ast:
            out["transaminases"] = m_ast.group(1).replace(",", ".")

    return out


def _strip_kv_lines(stage1_text: str, keys_lower: set[str]) -> str:
    """
    Remove "Key: Value" lines where Key matches keys_lower (case-insensitive).
    This avoids treating a self-reported Stage1 label line as its own evidence.
    """
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
    low = (v or "").strip().lower()
    return low in {"y", "yes", "true", "present", "positive", "pos", "performed", "done", "given", "administered"}


def _is_negative_binary(v: str) -> bool:
    low = (v or "").strip().lower()
    return low in {"n", "no", "false", "absent", "negative", "neg", "denied"}


def _apply_fp_gates(
    *,
    stage1_summary: dict[str, object],
    stage1_text: str,
    kv: dict[str, str],
) -> tuple[dict[str, str], dict[str, str]]:
    """
    Evidence-gated post-processing for a small set of high-FP items.
    Goal: reduce false positives from "teacher-like" Stage1 summaries without killing recall.
    Returns (new_kv, gate_notes_by_item).
    """
    _ = stage1_summary  # reserved for future cluster-aware gates
    out = dict(kv)
    notes: dict[str, str] = {}

    # Evidence text excludes the model's own KV label lines for these items.
    evidence_text = _strip_kv_lines(
        stage1_text,
        # NOTE: keep dyspnea self-label lines as evidence to avoid recall collapse.
        keys_lower={"active neoplasia", "arrhythmia", "acute coronary syndrome"},
    )

    # ── active neoplasia (neoplasia vocab)
    if "active neoplasia" in out:
        raw_v = str(out.get("active neoplasia") or "").strip()
        low_v = raw_v.lower()
        has_cancer_evidence = bool(_CANCER_EVIDENCE_RE.search(evidence_text or ""))
        if not has_cancer_evidence:
            out.pop("active neoplasia", None)
            notes["active neoplasia"] = "dropped: no cancer evidence outside self-label line"
        else:
            if _CANCER_ACTIVE_CUES_RE.search(evidence_text or ""):
                out["active neoplasia"] = "certainly active"
                notes["active neoplasia"] = "set: certainly active (active cues)"
            elif _CANCER_INACTIVE_CUES_RE.search(evidence_text or ""):
                out["active neoplasia"] = "certainly not active"
                notes["active neoplasia"] = "set: certainly not active (inactive cues)"
            else:
                # If Stage1 tries to force a yes/no, prefer "possibly active" absent explicit cues.
                if low_v in {"yes", "y", "present", "active", "certainly active"}:
                    out["active neoplasia"] = "possibly active"
                    notes["active neoplasia"] = "downgraded: certainly active -> possibly active (no activity cues)"
                elif low_v in {"no", "n", "absent", "inactive", "certainly not active"}:
                    out["active neoplasia"] = "certainly not active"
                    notes["active neoplasia"] = "kept: certainly not active"
                else:
                    out["active neoplasia"] = "possibly active"
                    notes["active neoplasia"] = "set: possibly active (cancer mentioned, no activity cues)"

    # ── arrhythmia (binary)
    if "arrhythmia" in out:
        v = str(out.get("arrhythmia") or "")
        if _is_positive_binary(v):
            if _ARRHYTHMIA_NEGATION_RE.search(evidence_text or ""):
                out.pop("arrhythmia", None)
                notes["arrhythmia"] = "dropped: negation cue present"
            elif not _ARRHYTHMIA_SUPPORT_RE.search(evidence_text or ""):
                out.pop("arrhythmia", None)
                notes["arrhythmia"] = "dropped: no supporting rhythm evidence outside self-label line"
        elif _is_negative_binary(v):
            # Avoid hallucinated negatives: keep only if explicit negation exists in evidence.
            if not _ARRHYTHMIA_NEGATION_RE.search(evidence_text or ""):
                out.pop("arrhythmia", None)
                notes["arrhythmia"] = "dropped: negative without explicit evidence"

    # ── acute coronary syndrome (binary)
    if "acute coronary syndrome" in out:
        v = str(out.get("acute coronary syndrome") or "")
        if _is_positive_binary(v):
            if _SPECULATION_CUES_RE.search(v):
                out.pop("acute coronary syndrome", None)
                notes["acute coronary syndrome"] = "dropped: speculative value"
            else:
                has_acs_evidence = bool(_ACS_SUPPORT_RE.search(evidence_text or ""))
                # Require at least one supportive acute marker already present in Stage1 (still Stage1-only).
                chest_pain = str(out.get("chest pain") or "")
                troponin = str(out.get("troponin") or "")
                ecg_abn = str(out.get("ecg, any abnormality") or "")
                supportive_marker = _is_positive_binary(chest_pain) or (troponin.strip().lower() not in {"", "unknown"}) or _is_positive_binary(ecg_abn)
                if not has_acs_evidence or not supportive_marker:
                    out.pop("acute coronary syndrome", None)
                    notes["acute coronary syndrome"] = "dropped: missing ACS evidence and/or supportive markers"
        elif _is_negative_binary(v):
            # Avoid hallucinated negatives: keep only if explicit evidence exists.
            if not re.search(r"(?i)\bno\s+(?:acs|acute\s+coronary\s+syndrome|stemi|nstemi|mi)\b", evidence_text or ""):
                out.pop("acute coronary syndrome", None)
                notes["acute coronary syndrome"] = "dropped: negative without explicit evidence"

    # ── presence of dyspnea (binary)
    if "presence of dyspnea" in out:
        v = str(out.get("presence of dyspnea") or "")
        if _is_positive_binary(v):
            if _DYSPNEA_NEGATION_RE.search(evidence_text or ""):
                out.pop("presence of dyspnea", None)
                notes["presence of dyspnea"] = "dropped: negation cue present"
            elif not _DYSPNEA_SUPPORT_RE.search(evidence_text or ""):
                out.pop("presence of dyspnea", None)
                notes["presence of dyspnea"] = "dropped: no dyspnea evidence outside self-label line"
        elif _is_negative_binary(v):
            # Avoid hallucinated negatives: require explicit negation phrase.
            if not _DYSPNEA_NEGATION_RE.search(evidence_text or ""):
                out.pop("presence of dyspnea", None)
                notes["presence of dyspnea"] = "dropped: negative without explicit evidence"

    return out, notes


def _adjust_values_for_official_vocab(kv: dict[str, str]) -> dict[str, str]:
    """
    Fix a few common value forms so the official normalizer can map them.
    """
    out: dict[str, str] = {}
    chronic_set = set(CHRONIC_ITEMS)
    neoplasia_set = set(NEOPLASIA_ITEMS)
    duration_set = set(DURATION_ITEMS)

    for item, value in kv.items():
        v = str(value).strip()
        low = v.lower()

        if item in chronic_set:
            if low in ("present", "yes", "y", "chronic"):
                out[item] = "yes"
                continue
            if low in ("absent", "no", "n", "not chronic"):
                out[item] = "no"
                continue

        if item in neoplasia_set:
            if low in ("present", "yes", "y", "active"):
                out[item] = "yes"
                continue
            if low in ("absent", "no", "n", "not active", "inactive"):
                out[item] = "no"
                continue

        if item in duration_set:
            if low in ("short", "long"):
                out[item] = low
                continue
            # Numeric durations (best-effort): treat seconds/minutes as "short"; hours/days as "long".
            if any(u in low for u in ("sec", "secs", "second", "seconds", "min", "mins", "minute", "minutes")):
                out[item] = "short"
                continue
            if any(u in low for u in ("hour", "hours", "day", "days")):
                out[item] = "long"
                continue

        out[item] = v

    return out


def _write_item_diagnostics(
    *,
    run_dir: Path,
    reference: list[dict[str, Any]],
    submission: list[dict[str, Any]],
    stage1_text_by_doc_plain: dict[str, str],
    not_available: str = "unknown",
    max_examples_per_item: int = 3,
) -> None:
    per_item = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "examples": []})

    for ref, sub in zip(reference, submission):
        doc_plain = str(ref["document_id"])
        sub_doc_id = str(sub["document_id"])
        sub_plain, _lang = sub_doc_id.split("_", 1)
        if sub_plain != doc_plain:
            continue

        ref_anns = ref.get("annotations") or []
        sub_preds = sub.get("predictions") or []
        if len(ref_anns) != len(sub_preds):
            continue

        stage1_text = stage1_text_by_doc_plain.get(doc_plain, "")
        stage1_lower = stage1_text.lower()

        for a, p in zip(ref_anns, sub_preds):
            item = str(a.get("item") or p.get("item") or "")
            if not item:
                continue
            t = str(a.get("ground_truth") or not_available)
            y = str(p.get("prediction") or not_available)
            if t != not_available and y != not_available and y == t:
                per_item[item]["tp"] += 1
            elif t == not_available and y != not_available:
                per_item[item]["fp"] += 1
                if len(per_item[item]["examples"]) < max_examples_per_item:
                    per_item[item]["examples"].append(
                        {
                            "document_id": doc_plain,
                            "type": "fp",
                            "gt": t,
                            "pred": y,
                            "stage1_support_hint": (item.lower() in stage1_lower),
                        }
                    )
            elif t != not_available and y == not_available:
                per_item[item]["fn"] += 1
                if len(per_item[item]["examples"]) < max_examples_per_item:
                    per_item[item]["examples"].append(
                        {
                            "document_id": doc_plain,
                            "type": "fn",
                            "gt": t,
                            "pred": y,
                            "stage1_support_hint": (item.lower() in stage1_lower),
                        }
                    )
            elif t != not_available and y != not_available and y != t:
                per_item[item]["fp"] += 1
                if len(per_item[item]["examples"]) < max_examples_per_item:
                    per_item[item]["examples"].append(
                        {
                            "document_id": doc_plain,
                            "type": "fp_mismatch",
                            "gt": t,
                            "pred": y,
                            "stage1_support_hint": (item.lower() in stage1_lower),
                        }
                    )

    items_sorted_fp = sorted(per_item.items(), key=lambda kv: (kv[1]["fp"], kv[1]["fn"], kv[1]["tp"]), reverse=True)
    items_sorted_fn = sorted(per_item.items(), key=lambda kv: (kv[1]["fn"], kv[1]["fp"], kv[1]["tp"]), reverse=True)
    all_items = [{"item": k, **v} for k, v in sorted(per_item.items(), key=lambda kv: kv[0])]
    out = {
        "items": all_items,
        "top_fp": [{"item": k, **v} for k, v in items_sorted_fp[:25]],
        "top_fn": [{"item": k, **v} for k, v in items_sorted_fn[:25]],
    }
    (run_dir / "item_diagnostics.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Human-friendly table for quick scanning.
    def _md_table(rows: list[dict[str, Any]], title: str) -> str:
        lines = [f"## {title}", "", "| item | tp | fp | fn | stage1_support_hint_any |", "|---|---:|---:|---:|:---:|"]
        for r in rows:
            examples = r.get("examples") or []
            hint_any = any(bool(ex.get("stage1_support_hint")) for ex in examples)
            lines.append(
                f"| {r.get('item','')} | {int(r.get('tp',0))} | {int(r.get('fp',0))} | {int(r.get('fn',0))} | {'yes' if hint_any else 'no'} |"
            )
        lines.append("")
        return "\n".join(lines)

    top_fn_rows = [{"item": k, **v} for k, v in items_sorted_fn[:25]]
    top_fp_rows = [{"item": k, **v} for k, v in items_sorted_fp[:25]]
    md = "\n".join(
        [
            "# CRF Stage2(det) diagnostics",
            "",
            _md_table(top_fn_rows, "Top FN (by count)"),
            _md_table(top_fp_rows, "Top FP (by count)"),
        ]
    )
    (run_dir / "item_diagnostics.md").write_text(md + "\n", encoding="utf-8")


def _write_doc_diff_md(
    *,
    doc_dir: Path,
    doc_plain: str,
    stage1_text: str,
    gt_annotations: list[dict[str, Any]],
    submission_record: dict[str, Any],
    not_available: str = "unknown",
) -> None:
    gt_by_item: dict[str, str] = {str(a.get("item") or ""): str(a.get("ground_truth") or not_available) for a in gt_annotations}
    pred_by_item: dict[str, str] = {
        str(p.get("item") or ""): str(p.get("prediction") or not_available) for p in (submission_record.get("predictions") or [])
    }
    stage1_lower = (stage1_text or "").lower()

    rows: list[tuple[str, str, str, str, str]] = []
    for item, gt in gt_by_item.items():
        if not item:
            continue
        pred = pred_by_item.get(item, not_available)
        if gt == not_available and pred == not_available:
            continue
        if gt != not_available and pred != not_available and pred == gt:
            status = "TP"
        elif gt == not_available and pred != not_available:
            status = "FP"
        elif gt != not_available and pred == not_available:
            status = "FN"
        elif gt != not_available and pred != not_available and pred != gt:
            status = "FP_mismatch"
        else:
            status = "?"
        hint = "yes" if (item.lower() in stage1_lower) else "no"
        rows.append((item, gt, pred, status, hint))

    # Sort: FN/FP first, then by item.
    order = {"FN": 0, "FP": 1, "FP_mismatch": 2, "TP": 3, "?": 4}
    rows.sort(key=lambda r: (order.get(r[3], 9), r[0]))

    lines = [
        f"# Doc diff: {doc_plain}",
        "",
        "| item | gt | pred | status | stage1_support_hint |",
        "|---|---|---|---|:---:|",
    ]
    for item, gt, pred, status, hint in rows:
        lines.append(f"| {item} | {gt} | {pred} | {status} | {hint} |")
    lines.append("")
    (doc_dir / "diff.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage1-run-dir", required=True)
    p.add_argument("--language", default="en", choices=["en", "it"])
    p.add_argument("--dataset-id", default="NLP-FBK/dyspnea-crf-train")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--use-umls-mapping", action="store_true")
    p.add_argument(
        "--umls-mapping-path",
        default=str(PROJECT_ROOT / "data" / "umls_crf_mapping.json"),
        help="Path to umls_crf_mapping.json",
    )
    p.add_argument("--out-dir", default="results/crf_stage2_det_from_stage1")
    grp = p.add_mutually_exclusive_group()
    grp.add_argument(
        "--fp-gates",
        dest="fp_gates",
        action="store_true",
        default=True,
        help="Apply evidence-gated filters for known high-FP items (default: enabled).",
    )
    grp.add_argument(
        "--no-fp-gates",
        dest="fp_gates",
        action="store_false",
        help="Disable evidence-gated FP filters (for ablations).",
    )
    p.add_argument("--zip", action="store_true", default=True, help="Write Codabench zip (default: enabled).")
    p.add_argument("--no-zip", dest="zip", action="store_false")
    p.add_argument("--write-docs", action="store_true", default=True, help="Write per-doc artifacts (default: enabled).")
    p.add_argument("--no-docs", dest="write_docs", action="store_false")
    args = p.parse_args()

    stage1_dir = Path(args.stage1_run_dir)
    if not stage1_dir.exists():
        raise SystemExit(f"Stage1 run dir not found: {stage1_dir}")

    hf = _hf_rows(args.dataset_id, split=args.language, offset=0, length=int(args.limit))
    rows = hf.get("rows") or []
    if not rows:
        raise SystemExit("No rows returned from HF datasets-server.")

    first_anns = rows[0]["row"].get("annotations") or []
    if not first_anns:
        raise SystemExit("HF record has no annotations; cannot infer CRF item list.")
    items = [a["item"] for a in first_anns]
    allowed_items_global = set(items)

    # Map Stage1 summaries by document_id (plain id).
    stage1_by_doc_plain: dict[str, dict[str, Any]] = {}
    for stage1_path in stage1_dir.glob("*.stage1_summary.json"):
        try:
            obj = json.loads(stage1_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        doc_id = str(obj.get("document_id") or stage1_path.name.split(".", 1)[0])
        doc_plain = doc_id.split("_", 1)[0]
        if doc_plain:
            stage1_by_doc_plain[doc_plain] = obj

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = out_root / f"run_{run_id}_{args.language}"
    run_dir.mkdir(parents=True, exist_ok=True)
    docs_dir: Path | None = None
    if bool(args.write_docs):
        docs_dir = run_dir / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)

    reference: list[dict[str, Any]] = []
    submission: list[dict[str, Any]] = []
    stage1_text_by_doc_plain: dict[str, str] = {}

    for r in rows:
        rec = r["row"]
        doc_id = str(rec.get("document_id") or "")
        doc_plain = doc_id.split("_", 1)[0]
        if not doc_plain:
            continue

        gt_doc = rec.get("annotations") or []
        reference.append({"document_id": doc_plain, "annotations": gt_doc})

        stage1_obj = stage1_by_doc_plain.get(doc_plain)
        if not stage1_obj:
            # Still emit an all-unknown record to keep the submission length stable.
            # This is important for Codabench uploads where missing docs can invalidate the submission.
            empty_preds: dict[str, Any] = {}
            sub_rec = predictions_to_submission(doc_plain, empty_preds, items, language=args.language)
            submission.append(sub_rec)
            stage1_text_by_doc_plain[doc_plain] = ""
            continue
        summary = stage1_obj.get("summary") or {}

        # Determine doc-scoped ontology: union of item lists for clusters present in Stage1.
        present_clusters = detect_present_clusters(summary)
        allowed_items_doc: list[str] = []
        for c in present_clusters:
            allowed_items_doc.extend([i for i in (CRF_ITEMS_BY_CLUSTER.get(c) or []) if i in allowed_items_global])
        # Always include DEMOGRAPHICS items (high-FN history flags can be supported by meds/PMH evidence in other clusters).
        allowed_items_doc.extend([i for i in (CRF_ITEMS_BY_CLUSTER.get("DEMOGRAPHICS") or []) if i in allowed_items_global])
        allowed_items_doc_set = set(allowed_items_doc) or allowed_items_global

        allowed_canon_map = _build_allowed_key_canon_map(allowed_items_doc_set)
        umls_alias_map = (
            build_umls_alias_map(mapping_path=args.umls_mapping_path, allowed_items=allowed_items_doc_set)
            if args.use_umls_mapping
            else {}
        )

        # Convert Stage1 summary JSON -> deterministic Stage1 Markdown view.
        stage1_text = stage1_summary_to_text(
            {k: str(summary.get(k, "not stated")) for k in STAGE1_KEYS_9},
            STAGE1_KEYS_9,
        )
        stage1_text_by_doc_plain[doc_plain] = stage1_text

        kv = _extract_kv_from_stage1_text(
            stage1_text,
            allowed_items=allowed_items_doc_set,
            allowed_canon_map=allowed_canon_map,
            umls_alias_map=umls_alias_map,
        )
        kv = _derive_from_stage1_text(
            stage1_summary=summary,
            stage1_text=stage1_text,
            kv=kv,
            allowed_items=allowed_items_doc_set,
        )
        gate_notes: dict[str, str] = {}
        if bool(args.fp_gates):
            kv, gate_notes = _apply_fp_gates(stage1_summary=summary, stage1_text=stage1_text, kv=kv)
        kv = _adjust_values_for_official_vocab(kv)

        # Convert to the canonical predictions mapping accepted by the normalizer/submission builder.
        preds_obj: dict[str, Any] = {"predictions": [{"item": k, "prediction": v} for k, v in kv.items()]}
        preds = coerce_predictions_mapping(preds_obj)
        sub_rec = predictions_to_submission(doc_plain, preds, items, language=args.language)
        submission.append(sub_rec)

        # ── Artifacts for manual inspection
        if docs_dir is not None:
            ddir = docs_dir / doc_plain
            ddir.mkdir(parents=True, exist_ok=True)
            (ddir / "stage1_summary.json").write_text(
                json.dumps(stage1_obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            (ddir / "stage1.md").write_text(stage1_text, encoding="utf-8")
            (ddir / "stage2_det_kv.json").write_text(
                json.dumps(
                    {"document_id": doc_id, "present_clusters": present_clusters, "kv": kv, "fp_gates": gate_notes},
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            (ddir / "stage2_submission_record.json").write_text(
                json.dumps(sub_rec, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            (ddir / "ground_truth.json").write_text(
                json.dumps({"document_id": doc_plain, "annotations": gt_doc}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            _write_doc_diff_md(
                doc_dir=ddir,
                doc_plain=doc_plain,
                stage1_text=stage1_text,
                gt_annotations=gt_doc,
                submission_record=sub_rec,
                not_available="unknown",
            )

    validate_submission_structure(submission, n_items=len(items))
    write_jsonl(run_dir / "submission.jsonl", submission)
    if bool(args.zip):
        zip_codabench_jsonl(run_dir / "submission.jsonl", run_dir / "submission_codabench.zip")

    score = score_records(reference, submission[: len(reference)], not_available="unknown")
    _write_item_diagnostics(
        run_dir=run_dir,
        reference=reference,
        submission=submission[: len(reference)],
        stage1_text_by_doc_plain=stage1_text_by_doc_plain,
        not_available="unknown",
    )
    (run_dir / "score.json").write_text(
        json.dumps({"macro_f1": score.macro_f1, "tp": score.tp, "fp": score.fp, "fn": score.fn}, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "stage1_run_dir": str(stage1_dir),
                "dataset_id": str(args.dataset_id),
                "limit": int(args.limit),
                "format": "deterministic_from_stage1",
                "use_umls_mapping": bool(args.use_umls_mapping),
                "umls_mapping_path": str(args.umls_mapping_path),
                "fp_gates": bool(args.fp_gates),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Wrote: {run_dir}")
    print(json.dumps({"macro_f1": score.macro_f1, "tp": score.tp, "fp": score.fp, "fn": score.fn}, indent=2))


if __name__ == "__main__":
    main()
