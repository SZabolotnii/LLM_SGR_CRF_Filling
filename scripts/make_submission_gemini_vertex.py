#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
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


def _load_vertex_sa_credentials() -> tuple[object, str]:
    """
    Returns (google.auth.credentials.Credentials, project_id).
    """
    service_account_b64 = os.environ.get("GCP_SERVICE_ACCOUNT_B64")
    if not service_account_b64:
        raise ValueError("GCP_SERVICE_ACCOUNT_B64 is not set.")

    sa_json = base64.b64decode(service_account_b64).decode("utf-8")
    sa_info = json.loads(sa_json)
    project_id = str(sa_info.get("project_id") or "").strip()
    if not project_id:
        raise ValueError("Service account JSON has no project_id.")

    from google.oauth2 import service_account

    creds = service_account.Credentials.from_service_account_info(
        sa_info,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    return creds, project_id


def _vertex_generate_json(
    *,
    model_name: str,
    prompt: str,
    credentials: object,
    project_id: str,
    location: str,
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    timeout_s: int,
    attempts: int,
) -> str:
    import vertexai
    from vertexai.generative_models import GenerativeModel

    vertexai.init(project=project_id, location=location, credentials=credentials)
    model = GenerativeModel(model_name)

    last_err: Exception | None = None
    for attempt in range(1, int(attempts) + 1):
        try:
            # Vertex SDK doesn't provide a strict JSON response_format knob like OpenAI JSON mode.
            # We rely on prompt constraints + JSON extraction + retry.
            #
            # Note: the stable `vertexai.generative_models.GenerativeModel.generate_content()` API does not
            # expose a per-request timeout parameter. We keep `timeout_s` in the CLI for future-proofing
            # and log metadata, but do not enforce it here.
            resp = model.generate_content(
                prompt,
                generation_config={
                    "temperature": float(temperature),
                    "top_p": float(top_p),
                    "max_output_tokens": int(max_output_tokens),
                },
            )
            return str(getattr(resp, "text", "") or "")
        except Exception as e:
            last_err = e
            # Small backoff to avoid instant retries on quota errors.
            time.sleep(min(2.0 * attempt, 8.0))
            continue

    raise RuntimeError(f"Vertex generation failed after {attempts} attempts: {last_err}")


def main() -> None:
    p = argparse.ArgumentParser(description="Create a Codabench-ready submission via Gemini on Vertex AI.")
    p.add_argument("--split", default="test", choices=["test", "dev"])
    p.add_argument("--language", default="en", choices=["en", "it"])
    p.add_argument("--limit", type=int, default=0, help="Limit number of docs (0 = all).")
    p.add_argument("--model", default="gemini-2.0-flash")
    p.add_argument("--location", default="us-central1")
    p.add_argument("--max-output-tokens", type=int, default=1200)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--timeout-s", type=int, default=90)
    p.add_argument("--attempts", type=int, default=3)
    p.add_argument("--max-note-chars", type=int, default=0, help="Trim note text before inference (0 disables).")
    p.add_argument("--trim-strategy", default="middle", choices=["middle", "head", "tail", "head_tail"])
    p.add_argument("--out-dir", default="submissions/gemini_vertex")
    p.add_argument("--run-dir", default="", help="Optional explicit run directory (overrides timestamp-based run_{ts}_{lang}).")
    p.add_argument("--resume", action="store_true", help="Resume from existing submission.jsonl in --run-dir.")
    p.add_argument("--zip", action="store_true", default=True, help="Write Codabench zip (default: enabled).")
    p.add_argument("--no-zip", dest="zip", action="store_false")
    args = p.parse_args()

    load_dotenv()
    hf_token = os.environ.get("HF_TOKEN") or None

    creds, project_id = _load_vertex_sa_credentials()

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

    for i in tqdm(range(n_total), desc=f"{dsid}:{args.language}:{args.model}"):
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
        full_prompt = stage2_sys.strip() + "\n\n" + user_prompt.strip()

        raw = _vertex_generate_json(
            model_name=str(args.model),
            prompt=full_prompt,
            credentials=creds,
            project_id=project_id,
            location=str(args.location),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            max_output_tokens=int(args.max_output_tokens),
            timeout_s=int(args.timeout_s),
            attempts=int(args.attempts),
        )
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
                "model": str(args.model),
                "location": str(args.location),
                "project_id": project_id,
                "max_note_chars": int(args.max_note_chars),
                "trim_strategy": str(args.trim_strategy),
                "max_output_tokens": int(args.max_output_tokens),
                "temperature": float(args.temperature),
                "top_p": float(args.top_p),
                "timeout_s": int(args.timeout_s),
                "attempts": int(args.attempts),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Wrote: {run_dir}")


if __name__ == "__main__":
    main()
