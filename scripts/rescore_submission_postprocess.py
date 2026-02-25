#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from llm_structcore.crf_prompt import (
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
from llm_structcore.scoring import score_records
from llm_structcore.submission import validate_submission_structure


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def _dataset_id(split: str) -> str:
    if split == "test":
        return "NLP-FBK/dyspnea-crf-test"
    return "NLP-FBK/dyspnea-crf-development"


def _is_binary_item(item: str) -> bool:
    s = str(item)
    if s in CHRONIC_ITEMS:
        return False
    if s in NEOPLASIA_ITEMS:
        return False
    if s in LAB_ITEMS:
        return False
    if s in POSNEG_ITEMS:
        return False
    if s in DURATION_ITEMS:
        return False
    if s in {
        CONSCIOUSNESS_ITEM,
        AUTONOMY_ITEM,
        RESP_RATE_ITEM,
        BODY_TEMP_ITEM,
        HEART_RATE_ITEM,
        BLOOD_PRESSURE_ITEM,
        RESP_DISTRESS_ITEM,
    }:
        return False
    return True


def _apply_postprocess(
    submission: list[dict[str, Any]],
    *,
    drop_binary_negatives: bool,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rec in submission:
        preds = []
        for ann in (rec.get("predictions") or []):
            item = str(ann.get("item") or "")
            pred = str(ann.get("prediction") or "unknown")
            if drop_binary_negatives and _is_binary_item(item) and pred == "n":
                pred = "unknown"
            preds.append({"item": item, "prediction": pred})
        out.append({"document_id": rec.get("document_id"), "predictions": preds})
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Rescore an existing submission.jsonl with deterministic postprocessing.")
    p.add_argument("--run-dir", required=True, help="Directory containing submission.jsonl")
    p.add_argument("--split", default="dev", choices=["dev", "test"])
    p.add_argument("--language", default="en", choices=["en", "it"])
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--drop-binary-negatives", action="store_true", help="Map binary 'n' -> unknown (only for binary items).")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    sub_path = run_dir / "submission.jsonl"
    if not sub_path.exists():
        raise SystemExit(f"Missing: {sub_path}")

    submission = _load_jsonl(sub_path)
    submission_pp = _apply_postprocess(submission, drop_binary_negatives=bool(args.drop_binary_negatives))
    validate_submission_structure(submission_pp, n_items=len(submission_pp[0]["predictions"]))

    if args.split != "dev":
        print("Postprocessed submission written (no scoring on test split).")
        out_path = run_dir / "submission.postprocessed.jsonl"
        out_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in submission_pp) + "\n", encoding="utf-8")
        print(f"Wrote: {out_path}")
        return

    load_dotenv()
    hf_token = os.environ.get("HF_TOKEN") or None
    from datasets import load_dataset

    dsid = _dataset_id("dev")
    ds = load_dataset(dsid, split=args.language, token=hf_token)
    n_total = len(ds) if int(args.limit) <= 0 else min(len(ds), int(args.limit))

    ref = []
    for j in range(n_total):
        r = ds[j]
        ref.append({"document_id": str(r.get("document_id") or "").split("_", 1)[0], "annotations": r.get("annotations") or []})

    score = score_records(ref[: len(submission_pp)], submission_pp)
    print(json.dumps({"macro_f1": score.macro_f1, "tp": score.tp, "fp": score.fp, "fn": score.fn}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
