from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path


def _variants(text: str) -> set[str]:
    s = " ".join(str(text).strip().split())
    if not s:
        return set()
    out = {
        s,
        s.lower(),
        s.replace("’", "'"),
        s.replace("’", "'").lower(),
        s.replace("'", "’"),
        s.replace("'", "’").lower(),
    }
    return {x for x in out if x}


def build_item_canon_map(items: Iterable[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        canonical = " ".join(str(item).strip().split())
        if not canonical:
            continue
        for v in _variants(canonical):
            out[v] = canonical
    return out


def load_umls_mapping_records(mapping_path: str | Path) -> list[dict[str, object]]:
    p = Path(mapping_path)
    if not p.exists():
        return []
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(raw, list):
        return []
    records: list[dict[str, object]] = []
    for rec in raw:
        if isinstance(rec, dict):
            records.append(rec)
    return records


def build_umls_alias_map(
    *,
    mapping_path: str | Path,
    allowed_items: Iterable[str],
    include_preferred_name: bool = True,
    include_alternatives: bool = True,
    include_query_used: bool = True,
) -> dict[str, str]:
    records = load_umls_mapping_records(mapping_path)
    if not records:
        return {}

    canon_map = build_item_canon_map(allowed_items)
    alias_to_item: dict[str, str] = {}

    for rec in records:
        item_raw = " ".join(str(rec.get("item", "")).strip().split())
        if not item_raw:
            continue
        canonical_item = canon_map.get(item_raw) or canon_map.get(item_raw.lower())
        if not canonical_item:
            continue

        alias_candidates: list[str] = [item_raw]
        if include_preferred_name:
            alias_candidates.append(str(rec.get("preferred_name", "") or ""))
        if include_query_used:
            alias_candidates.append(str(rec.get("query_used", "") or ""))
        if include_alternatives:
            alternatives = rec.get("alternatives", [])
            if isinstance(alternatives, list):
                for a in alternatives:
                    alias_candidates.append(str(a or ""))

        for alias in alias_candidates:
            for key in _variants(alias):
                if key not in alias_to_item:
                    alias_to_item[key] = canonical_item
    return alias_to_item


def canonicalize_item_name(
    item: str,
    *,
    item_canon_map: Mapping[str, str],
    umls_alias_map: Mapping[str, str] | None = None,
) -> str:
    s = " ".join(str(item).strip().split())
    if not s:
        return s
    for key in (s, s.lower(), s.replace("’", "'"), s.replace("’", "'").lower(), s.replace("'", "’"), s.replace("'", "’").lower()):
        if key in item_canon_map:
            return item_canon_map[key]
    if umls_alias_map:
        for key in (s, s.lower(), s.replace("’", "'"), s.replace("’", "'").lower(), s.replace("'", "’"), s.replace("'", "’").lower()):
            if key in umls_alias_map:
                return umls_alias_map[key]
    return s


def canonicalize_prediction_mapping(
    preds: Mapping[str, object] | None,
    *,
    allowed_items: Iterable[str],
    umls_alias_map: Mapping[str, str] | None = None,
) -> dict[str, str]:
    if not preds:
        return {}
    allowed_set = {str(x) for x in allowed_items}
    canon_map = build_item_canon_map(allowed_set)
    out: dict[str, str] = {}

    for key, value in preds.items():
        canonical = canonicalize_item_name(str(key), item_canon_map=canon_map, umls_alias_map=umls_alias_map)
        if canonical not in allowed_set:
            continue
        val = str(value).strip()
        if not val:
            continue
        if canonical not in out:
            out[canonical] = val
            continue
        if out[canonical].lower() == "unknown" and val.lower() != "unknown":
            out[canonical] = val
    return out
