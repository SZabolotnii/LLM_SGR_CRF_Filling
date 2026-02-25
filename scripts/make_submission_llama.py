#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from llm_structcore.crf_prompt import build_system_instruction, build_user_prompt  # noqa: E402
from llm_structcore.llama_client import LlamaChatConfig, chat_completions  # noqa: E402
from llm_structcore.normalize_predictions import coerce_predictions_mapping, predictions_to_submission  # noqa: E402
from llm_structcore.stage1_sgr import extract_json_object  # noqa: E402
from llm_structcore.submission import validate_submission_structure, write_jsonl, zip_codabench_jsonl  # noqa: E402


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _append_jsonl(path: Path, rec: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _trim_text(text: str, *, max_chars: int, strategy: str) -> str:
    if max_chars <= 0:
        return text
    s = text or ""
    if len(s) <= max_chars:
        return s
    if strategy == "head":
        return s[:max_chars]
    if strategy == "tail":
        return s[-max_chars:]
    if strategy == "head_tail":
        half = max_chars // 2
        return s[:half] + "\n...\n" + s[-(max_chars - half) :]
    # default: middle
    start = max(0, (len(s) - max_chars) // 2)
    return s[start : start + max_chars]


def _dataset_id(split: str) -> str:
    if split == "test":
        return "NLP-FBK/dyspnea-crf-test"
    return "NLP-FBK/dyspnea-crf-development"


def _get_items_from_record(rec: dict[str, Any]) -> list[str]:
    anns = rec.get("annotations") or []
    if not anns:
        raise ValueError("Record has no annotations; cannot infer CRF items list.")
    return [a["item"] for a in anns]


def main() -> None:
    p = argparse.ArgumentParser(description="Create a Codabench-ready submission via llama-server (OpenAI-compatible).")
    p.add_argument("--split", default="test", choices=["test", "dev"])
    p.add_argument("--language", default="en", choices=["en", "it"])
    p.add_argument("--limit", type=int, default=0, help="Limit number of docs (0 = all).")
    p.add_argument("--stage2-url", default="http://127.0.0.1:1247")
    p.add_argument("--stage2-model", default="medgemma-ft-dyspnea-crf-q5_k_m")
    p.add_argument("--max-new-tokens", type=int, default=1200)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--timeout-s", type=int, default=600)
    p.add_argument("--json-mode", action="store_true", default=True, help="Send response_format=json_object (best-effort).")
    p.add_argument("--no-json-mode", dest="json_mode", action="store_false")
    p.add_argument("--max-note-chars", type=int, default=0, help="Trim note text before inference (0 disables).")
    p.add_argument("--trim-strategy", default="middle", choices=["middle", "head", "tail", "head_tail"])
    p.add_argument("--out-dir", default="submissions/llama_ft")
    p.add_argument("--run-dir", default="", help="Optional explicit run directory (overrides timestamp-based run_{ts}_{lang}).")
    p.add_argument("--resume", action="store_true", help="Resume from existing submission.jsonl in --run-dir.")
    p.add_argument("--zip", action="store_true", default=True, help="Write Codabench zip (default: enabled).")
    p.add_argument("--no-zip", dest="zip", action="store_false")
    args = p.parse_args()

    load_dotenv()
    hf_token = os.environ.get("HF_TOKEN") or None

    from datasets import load_dataset

    dsid = _dataset_id(args.split)
    ds = load_dataset(dsid, split=args.language, token=hf_token)
    n_total = len(ds) if int(args.limit) <= 0 else min(len(ds), int(args.limit))
    if n_total <= 0:
        raise SystemExit("No docs to process.")

    items = _get_items_from_record(ds[0])
    stage2_sys = build_system_instruction(items, sparse_output=True)

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    if str(args.run_dir).strip():
        run_dir = Path(str(args.run_dir))
    else:
        run_id = time.strftime("%Y%m%d_%H%M%S")
        run_dir = out_root / f"run_{run_id}_{args.language}"
    run_dir.mkdir(parents=True, exist_ok=True)

    submission_path = run_dir / "submission.jsonl"
    done: set[str] = set()
    submission: list[dict[str, Any]] = []
    if bool(args.resume):
        for rec in _load_jsonl(submission_path):
            doc_id = str(rec.get("document_id") or "")
            if doc_id:
                done.add(doc_id.split("_", 1)[0])
                submission.append(rec)
    else:
        if submission_path.exists():
            submission_path.unlink()

    cfg = LlamaChatConfig(
        base_url=str(args.stage2_url),
        model=str(args.stage2_model),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_tokens=int(args.max_new_tokens),
        timeout_s=int(args.timeout_s),
        json_mode=bool(args.json_mode),
    )

    for i in tqdm(range(n_total), desc=f"{dsid}:{args.language}"):
        rec = ds[i]
        doc_id = str(rec.get("document_id") or "")
        if not doc_id:
            continue
        doc_plain = doc_id.split("_", 1)[0]
        if doc_plain in done:
            continue

        note = str(rec.get("clinical_note") or "")
        note = _trim_text(note, max_chars=int(args.max_note_chars), strategy=str(args.trim_strategy))

        user_prompt = build_user_prompt(note)
        raw = chat_completions(cfg, [{"role": "system", "content": stage2_sys}, {"role": "user", "content": user_prompt}])
        (run_dir / f"{doc_id}.raw.txt").write_text(raw or "", encoding="utf-8")

        parse_err: str | None = None
        preds_map: dict[str, object] = {}
        try:
            obj = extract_json_object(raw)
            preds_map = coerce_predictions_mapping(obj)
        except Exception as e:
            parse_err = str(e)
            preds_map = {}

        (run_dir / f"{doc_id}.parse.json").write_text(
            json.dumps({"document_id": doc_id, "parse_error": parse_err, "preds": preds_map}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        sub_rec = predictions_to_submission(doc_plain, preds_map, items, language=args.language)
        submission.append(sub_rec)
        done.add(doc_plain)
        _append_jsonl(submission_path, sub_rec)

    validate_submission_structure(submission, n_items=len(items))
    write_jsonl(submission_path, submission)
    if bool(args.zip):
        zip_codabench_jsonl(submission_path, run_dir / "submission_codabench.zip")

    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "dataset_id": dsid,
                "language": args.language,
                "limit": int(args.limit),
                "stage2_url": cfg.base_url,
                "stage2_model": cfg.model,
                "max_note_chars": int(args.max_note_chars),
                "trim_strategy": str(args.trim_strategy),
                "json_mode": bool(args.json_mode),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Wrote: {run_dir}")


if __name__ == "__main__":
    main()

