"""
normalize_predictions.py

Deterministic post-processing:
- Map common model mistakes into the official controlled vocabulary
- Ensure every record contains ALL 134 items
- Ensure final output matches the official expected structure
"""

from __future__ import annotations

import re
from typing import Iterable

from .crf_prompt import (
    CHRONIC_ITEMS,
    NEOPLASIA_ITEMS,
    LAB_ITEMS,
    POSNEG_ITEMS,
    DURATION_ITEMS,
    CONSCIOUSNESS_ITEM,
    AUTONOMY_ITEM,
    RESP_RATE_ITEM,
    BODY_TEMP_ITEM,
    HEART_RATE_ITEM,
    BLOOD_PRESSURE_ITEM,
    RESP_DISTRESS_ITEM,
)

# ──────────────────────────────────────────────────────────────────
# Fallback maps for common model mistakes
# ──────────────────────────────────────────────────────────────────

BINARY_FALLBACK = {
    "yes": "y",
    "no": "n",
    "true": "y",
    "false": "n",
    "present": "y",
    "absent": "n",
    "positive": "y",
    "negative": "n",
    "denied": "n",
    "not indicated": "n",
    "not performed": "n",
    "performed": "y",
    "done": "y",
    "given": "y",
    "administered": "y",
    "provided": "y",
    "y": "y",
    "n": "n",
}

CONSCIOUSNESS_FALLBACK = {
    "alert": "A",
    "a": "A",
    "awake": "A",
    "oriented": "A",
    "alert and oriented": "A",
    "verbal": "V",
    "v": "V",
    "responds to voice": "V",
    "pain": "P",
    "p": "P",
    "responds to pain": "P",
    "unresponsive": "unknown",
    "u": "unknown",
}

AUTONOMY_FALLBACK = {
    "independent": "walking independently",
    "ambulatory": "walking independently",
    "fully ambulatory": "walking independently",
    "with assistance": "walking with auxiliary aids",
    "assisted": "walking with auxiliary aids",
    "walker": "walking with auxiliary aids",
    "cane": "walking with auxiliary aids",
    "wheelchair": "bedridden",
    "non-ambulatory": "bedridden",
    "bed-bound": "bedridden",
    "immobile": "bedridden",
}

CHRONIC_FALLBACK = {
    "yes": "certainly chronic",
    "y": "certainly chronic",
    "no": "certainly not chronic",
    "n": "certainly not chronic",
    "chronic": "certainly chronic",
    "not chronic": "certainly not chronic",
}

NEOPLASIA_FALLBACK = {
    "yes": "certainly active",
    "y": "certainly active",
    "no": "certainly not active",
    "n": "certainly not active",
    "active": "certainly active",
    "not active": "certainly not active",
}

POSNEG_FALLBACK = {
    "positive": "pos",
    "yes": "pos",
    "y": "pos",
    "negative": "neg",
    "no": "neg",
    "n": "neg",
}

RESP_RATE_FALLBACK = {"normal": "eupneic", "tachypnea": "tachypneic", "bradypnea": "bradypneic"}
BODY_TEMP_FALLBACK = {
    "normal": "normothermic",
    "fever": "hyperthermic",
    "febrile": "hyperthermic",
    "afebrile": "normothermic",
    # Common shorthand in ED notes: "T nega/negative" ~= afebrile
    "negative": "normothermic",
    "nega": "normothermic",
}
HEART_RATE_FALLBACK = {"normal": "normocardic", "tachycardia": "tachycardic", "bradycardia": "bradycardic"}
BP_FALLBACK = {"normal": "normotensive", "high": "hypertensive", "low": "hypotensive"}
RESP_DISTRESS_FALLBACK = {"yes": "current", "y": "current", "present": "current", "no": "unknown", "n": "unknown", "absent": "unknown"}
DURATION_FALLBACK = {"brief": "short", "prolonged": "long"}

VALID_OPTIONS = {
    "binary": {"y", "n", "unknown"},
    "chronic": {"certainly chronic", "possibly chronic", "certainly not chronic", "unknown"},
    "neoplasia": {"certainly active", "possibly active", "certainly not active", "unknown"},
    "posneg": {"pos", "neg", "unknown"},
    "duration": {"short", "long", "unknown"},
    "consciousness": {"A", "V", "P", "unknown"},
    "autonomy": {"walking independently", "walking with auxiliary aids", "walking with physical assistance", "bedridden", "unknown"},
    "resp_rate": {"bradypneic", "eupneic", "tachypneic", "unknown"},
    "body_temp": {"hypothermic", "normothermic", "hyperthermic", "unknown"},
    "heart_rate": {"bradycardic", "normocardic", "tachycardic", "unknown"},
    "bp": {"hypotensive", "normotensive", "hypertensive", "unknown"},
    "resp_distress": {"current", "past", "unknown"},
}

UNKNOWN_VARIANTS = {
    "",
    "unknown",
    "not stated",
    "n/a",
    "na",
    "none",
    "null",
    "not specified",
    "not mentioned",
}

_NUM_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)?")


def _extract_numeric_token(value: str) -> str | None:
    """
    Best-effort numeric extractor:
    - "97% RA" -> "97%"
    - "37.5°C" -> "37.5"
    - "120/80 mmHg" -> "120/80"
    """
    s = (value or "").strip()
    if not s:
        return None

    m_bp = re.match(r"^\s*(\d+)\s*/\s*(\d+)\s*(?:mmhg)?\s*$", s, flags=re.IGNORECASE)
    if m_bp:
        return f"{m_bp.group(1)}/{m_bp.group(2)}"

    m = _NUM_RE.search(s)
    if not m:
        return None

    num = m.group(0).replace(",", ".")
    end = m.end()
    if end < len(s) and s[end] == "%":
        return f"{num}%"
    return num


def _is_numeric(value: str) -> bool:
    token = _extract_numeric_token(value)
    if token is None:
        return False
    return bool(re.match(r"^\s*[-+]?\d+(?:\.\d+)?%?\s*$", token)) or bool(re.match(r"^\s*\d+\s*/\s*\d+\s*$", token))


def _to_float(value: str) -> float | None:
    token = _extract_numeric_token(value)
    if not token:
        return None
    if "/" in token:
        return None
    try:
        return float(token.replace("%", ""))
    except Exception:
        return None


def _map_hr_category(value: str) -> str | None:
    x = _to_float(value)
    if x is None:
        return None
    if x < 60:
        return "bradycardic"
    if x <= 100:
        return "normocardic"
    return "tachycardic"


def _map_rr_category(value: str) -> str | None:
    x = _to_float(value)
    if x is None:
        return None
    if x < 12:
        return "bradypneic"
    if x <= 20:
        return "eupneic"
    return "tachypneic"


def _map_temp_category(value: str) -> str | None:
    x = _to_float(value)
    if x is None:
        return None
    if x < 36:
        return "hypothermic"
    if x < 38:
        return "normothermic"
    return "hyperthermic"


def _map_bp_category(value: str) -> str | None:
    token = _extract_numeric_token(value)
    if not token:
        return None
    if "/" in token:
        try:
            sbp_s, dbp_s = token.split("/", 1)
            sbp = float(sbp_s)
            dbp = float(dbp_s)
        except Exception:
            return None
        if sbp < 90 or dbp < 60:
            return "hypotensive"
        # Conservative hypertensive threshold to match organizer GT behavior observed on train10:
        # e.g., SBP=140/DBP=85 is labeled normotensive in GT.
        if sbp >= 180 or dbp >= 110:
            return "hypertensive"
        return "normotensive"
    x = _to_float(token)
    if x is None:
        return None
    if x < 90:
        return "hypotensive"
    if x >= 180:
        return "hypertensive"
    return "normotensive"


def _get_item_category(item_name: str) -> str:
    if item_name in CHRONIC_ITEMS:
        return "chronic"
    if item_name in NEOPLASIA_ITEMS:
        return "neoplasia"
    if item_name in LAB_ITEMS:
        return "lab"
    if item_name in POSNEG_ITEMS:
        return "posneg"
    if item_name in DURATION_ITEMS:
        return "duration"
    if item_name == CONSCIOUSNESS_ITEM:
        return "consciousness"
    if item_name == AUTONOMY_ITEM:
        return "autonomy"
    if item_name == RESP_RATE_ITEM:
        return "resp_rate"
    if item_name == BODY_TEMP_ITEM:
        return "body_temp"
    if item_name == HEART_RATE_ITEM:
        return "heart_rate"
    if item_name == BLOOD_PRESSURE_ITEM:
        return "bp"
    if item_name == RESP_DISTRESS_ITEM:
        return "resp_distress"
    return "binary"


def normalize_crf_value(value: object, item_name: str) -> str:
    v = str(value).strip()
    v_lower = v.lower()

    if v_lower in UNKNOWN_VARIANTS:
        return "unknown"

    category = _get_item_category(item_name)

    if category == "binary":
        if v_lower in BINARY_FALLBACK:
            return BINARY_FALLBACK[v_lower]
        if v in VALID_OPTIONS["binary"]:
            return v
        return "unknown"

    if category == "chronic":
        if v in VALID_OPTIONS["chronic"]:
            return v
        if v_lower in CHRONIC_FALLBACK:
            return CHRONIC_FALLBACK[v_lower]
        return "unknown"

    if category == "neoplasia":
        if v in VALID_OPTIONS["neoplasia"]:
            return v
        if v_lower in NEOPLASIA_FALLBACK:
            return NEOPLASIA_FALLBACK[v_lower]
        return "unknown"

    if category == "lab":
        # Preserve inequality-style troponin expressions used in the provided GT.
        if item_name == "troponin":
            if any(sym in v for sym in (">", "<", "=")):
                vv = v.strip()
                return vv[:32] if len(vv) > 32 else vv
        if v_lower == "measured":
            return "measured"
        token = _extract_numeric_token(v)
        if token is not None and _is_numeric(token):
            return token
        if v_lower in ("yes", "y", "present", "positive"):
            return "measured"
        # Keep a few common qualitative / shorthand lab values used in the provided GT.
        if item_name == "c-reactive protein":
            if v_lower in ("pcrneg", "pcr neg", "crp neg", "crp negative", "negative", "neg"):
                return "PCRneg"
            if "pcr" in v_lower and "neg" in v_lower:
                return "PCRneg"
        return "unknown"

    if category == "posneg":
        if v in VALID_OPTIONS["posneg"]:
            return v
        if v_lower in POSNEG_FALLBACK:
            return POSNEG_FALLBACK[v_lower]
        return "unknown"

    if category == "duration":
        if v in VALID_OPTIONS["duration"]:
            return v
        if v_lower in DURATION_FALLBACK:
            return DURATION_FALLBACK[v_lower]
        return "unknown"

    if category == "consciousness":
        if v in VALID_OPTIONS["consciousness"]:
            return v
        if v_lower in CONSCIOUSNESS_FALLBACK:
            return CONSCIOUSNESS_FALLBACK[v_lower]
        # Numeric GCS -> AVPU approximation (common in ED notes)
        gcs = _to_float(v)
        if gcs is not None:
            if gcs >= 15:
                return "A"
            if gcs >= 9:
                return "V"
            return "P"
        return "unknown"

    if category == "autonomy":
        if v in VALID_OPTIONS["autonomy"]:
            return v
        if v_lower in AUTONOMY_FALLBACK:
            return AUTONOMY_FALLBACK[v_lower]
        return "unknown"

    if category == "resp_rate":
        if v in VALID_OPTIONS["resp_rate"]:
            return v
        mapped = _map_rr_category(v)
        if mapped is not None:
            return mapped
        if v_lower in RESP_RATE_FALLBACK:
            return RESP_RATE_FALLBACK[v_lower]
        return "unknown"

    if category == "body_temp":
        if v in VALID_OPTIONS["body_temp"]:
            return v
        if "negative" in v_lower or "nega" in v_lower:
            return "normothermic"
        mapped = _map_temp_category(v)
        if mapped is not None:
            return mapped
        if v_lower in BODY_TEMP_FALLBACK:
            return BODY_TEMP_FALLBACK[v_lower]
        return "unknown"

    if category == "heart_rate":
        if v in VALID_OPTIONS["heart_rate"]:
            return v
        mapped = _map_hr_category(v)
        if mapped is not None:
            return mapped
        if v_lower in HEART_RATE_FALLBACK:
            return HEART_RATE_FALLBACK[v_lower]
        return "unknown"

    if category == "bp":
        if v in VALID_OPTIONS["bp"]:
            return v
        mapped = _map_bp_category(v)
        if mapped is not None:
            return mapped
        if v_lower in BP_FALLBACK:
            return BP_FALLBACK[v_lower]
        return "unknown"

    if category == "resp_distress":
        if v in VALID_OPTIONS["resp_distress"]:
            return v
        if v_lower in RESP_DISTRESS_FALLBACK:
            return RESP_DISTRESS_FALLBACK[v_lower]
        return "unknown"

    return "unknown"


def normalize_record(predictions: dict[str, object], items: Iterable[str]) -> dict[str, str]:
    return {item: normalize_crf_value(predictions.get(item, "unknown"), item_name=item) for item in items}


def predictions_to_submission(
    document_id: str,
    predictions: dict[str, object],
    items: list[str],
    *,
    language: str = "en",
) -> dict:
    doc_id_str = str(document_id)
    if not doc_id_str.endswith(f"_{language}"):
        doc_id_str = f"{doc_id_str}_{language}"

    normalized = normalize_record(predictions, items)
    return {
        "document_id": doc_id_str,
        "predictions": [{"item": item, "prediction": normalized[item]} for item in items],
    }


def coerce_predictions_mapping(model_obj: object) -> dict[str, object]:
    """
    Accept either:
    - dict[item -> value]
    - {"predictions": [{"item": ..., "prediction": ...}, ...]}
    and return a plain dict[item -> value].
    """
    if isinstance(model_obj, dict):
        preds = model_obj.get("predictions")
        if isinstance(preds, list):
            out: dict[str, object] = {}
            for p in preds:
                if not isinstance(p, dict):
                    continue
                item = p.get("item")
                if not item:
                    continue
                out[str(item)] = p.get("prediction", "unknown")
            return out
        return model_obj
    return {}
