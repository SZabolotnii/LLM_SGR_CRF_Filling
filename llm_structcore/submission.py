from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def zip_codabench_jsonl(jsonl_path: Path, zip_path: Path) -> None:
    """
    Create a Codabench-style zip with a single JSONL file at archive root named:
    `mock_data_dev_codabench.jsonl`.
    """
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    data = jsonl_path.read_bytes()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("mock_data_dev_codabench.jsonl", data)


def validate_submission_structure(records: list[dict], *, n_items: int = 134) -> None:
    if not records:
        raise ValueError("Submission is empty.")

    for i, rec in enumerate(records):
        if "document_id" not in rec:
            raise ValueError(f"Record[{i}] missing 'document_id'.")
        if "predictions" not in rec:
            raise ValueError(f"Record[{i}] missing 'predictions'.")
        preds = rec["predictions"]
        if not isinstance(preds, list):
            raise ValueError(f"Record[{i}] 'predictions' must be a list.")
        if len(preds) != n_items:
            raise ValueError(f"Record[{i}] expected {n_items} predictions, got {len(preds)}.")
        for j, p in enumerate(preds):
            if not isinstance(p, dict):
                raise ValueError(f"Record[{i}].predictions[{j}] must be an object.")
            if "item" not in p or "prediction" not in p:
                raise ValueError(f"Record[{i}].predictions[{j}] missing 'item' or 'prediction'.")

