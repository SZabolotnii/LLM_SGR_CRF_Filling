#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from llm_structcore.scoring import score_records


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--submission-jsonl", required=True)
    p.add_argument("--ref-jsonl", default=None, help="Path to official dev_gt.jsonl (preferred).")
    p.add_argument("--language", default="en", choices=["en", "it"], help="Used only when --ref-jsonl is not provided.")
    args = p.parse_args()

    submission = _load_jsonl(Path(args.submission_jsonl))

    if args.ref_jsonl:
        reference = _load_jsonl(Path(args.ref_jsonl))
    else:
        from datasets import load_dataset

        ds = load_dataset("NLP-FBK/dyspnea-crf-development", split=args.language)
        reference = []
        for rec in ds:
            # Normalize to official dev_gt.jsonl schema:
            doc_id = str(rec.get("document_id") or rec.get("id") or "")
            plain_id = doc_id.split("_", 1)[0]
            anns = rec.get("annotations") or []
            reference.append(
                {
                    "document_id": plain_id,
                    "annotations": [{"item": a["item"], "ground_truth": a["ground_truth"]} for a in anns],
                }
            )

    score = score_records(reference, submission, not_available="unknown")
    print(f"Macro-F1: {score.macro_f1:.6f}")
    print(f"TP/FP/FN: {score.tp}/{score.fp}/{score.fn}")


if __name__ == "__main__":
    main()
