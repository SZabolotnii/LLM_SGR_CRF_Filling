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

try:
    from openai import OpenAI  # type: ignore

    _OPENAI_AVAILABLE = True
except Exception:
    OpenAI = None  # type: ignore
    _OPENAI_AVAILABLE = False

from llm_structcore.crf_prompt import build_system_instruction, build_user_prompt  # noqa: E402
from llm_structcore.normalize_predictions import coerce_predictions_mapping, predictions_to_submission  # noqa: E402
from llm_structcore.scoring import score_records  # noqa: E402
from llm_structcore.stage1_sgr import extract_json_object  # noqa: E402
from llm_structcore.submission import validate_submission_structure, write_jsonl, zip_codabench_jsonl  # noqa: E402

import re


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
    r"dispnea|mancanza\s+di\s+respiro|affanno|fiato\s+corto"
    r")\b"
)
_DYSPNEA_NEGATION_RE = re.compile(
    r"(?i)\b("
    r"no\s+(?:dyspn(?:ea|oea)|shortness\s+of\s+breath|sob|dispnea|mancanza\s+di\s+respiro|affanno|fiato\s+corto)|"
    r"denies\s+(?:dyspn(?:ea|oea)|shortness\s+of\s+breath|sob|dispnea)|"
    r"without\s+(?:dyspn(?:ea|oea)|shortness\s+of\s+breath|sob|dispnea)|"
    r"nega\s+(?:dispnea|mancanza\s+di\s+respiro|affanno)|"
    r"senza\s+(?:dispnea|mancanza\s+di\s+respiro|affanno)"
    r")\b"
)


def _is_positive_binary(v: str) -> bool:
    low = (v or "").strip().lower()
    return low in {"y", "yes", "true", "present", "positive", "pos", "performed", "done", "given", "administered"}


def _is_negative_binary(v: str) -> bool:
    low = (v or "").strip().lower()
    return low in {"n", "no", "false", "absent", "negative", "neg", "denied"}


def _apply_fp_gates_note_only(preds: dict[str, object], note: str) -> dict[str, object]:
    """
    Evidence-gated post-processing for a small set of high-FP items.
    Uses ONLY the raw note as evidence (no Stage1).
    """
    t = str(note or "")
    out = dict(preds)

    # active neoplasia
    if "active neoplasia" in out:
        if not _CANCER_EVIDENCE_RE.search(t):
            out.pop("active neoplasia", None)
        else:
            if _CANCER_ACTIVE_CUES_RE.search(t):
                out["active neoplasia"] = "certainly active"
            elif _CANCER_INACTIVE_CUES_RE.search(t):
                out["active neoplasia"] = "certainly not active"
            else:
                out["active neoplasia"] = "possibly active"

    # arrhythmia
    if "arrhythmia" in out:
        v = str(out.get("arrhythmia") or "")
        if _is_positive_binary(v):
            if _ARRHYTHMIA_NEGATION_RE.search(t) or not _ARRHYTHMIA_SUPPORT_RE.search(t):
                out.pop("arrhythmia", None)
        elif _is_negative_binary(v):
            if not _ARRHYTHMIA_NEGATION_RE.search(t):
                out.pop("arrhythmia", None)

    # acute coronary syndrome
    if "acute coronary syndrome" in out:
        v = str(out.get("acute coronary syndrome") or "")
        if _is_positive_binary(v):
            if _SPECULATION_CUES_RE.search(v) or not _ACS_SUPPORT_RE.search(t):
                out.pop("acute coronary syndrome", None)
        elif _is_negative_binary(v):
            if not re.search(r"(?i)\bno\s+(?:acs|acute\s+coronary\s+syndrome|stemi|nstemi|mi)\b", t):
                out.pop("acute coronary syndrome", None)

    # presence of dyspnea
    if "presence of dyspnea" in out:
        v = str(out.get("presence of dyspnea") or "")
        if _is_positive_binary(v):
            if _DYSPNEA_NEGATION_RE.search(t) or not _DYSPNEA_SUPPORT_RE.search(t):
                out.pop("presence of dyspnea", None)
        elif _is_negative_binary(v):
            if not _DYSPNEA_NEGATION_RE.search(t):
                out.pop("presence of dyspnea", None)

    return out


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


def _get_openai_api_key() -> str | None:
    key = os.environ.get("GPT_API_KEY_OPEN_AI") or os.environ.get("OPENAI_API_KEY")
    if key:
        return str(key).strip()
    return None


def _openai_chat_json(
    *,
    client: Any,
    model: str,
    system: str,
    user: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout_s: float,
    verbosity: str,
    reasoning_effort: str,
) -> str:
    kwargs: dict[str, Any] = {
        "model": str(model),
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "response_format": {"type": "json_object"},
        "verbosity": str(verbosity),
        "store": False,
        "temperature": float(temperature),
        "top_p": float(top_p),
        # gpt-5.x uses max_completion_tokens (max_tokens is rejected).
        "max_completion_tokens": int(max_tokens),
    }
    eff = str(reasoning_effort or "").strip().lower()
    if eff and eff not in {"none", "off", "false", "0"}:
        kwargs["reasoning_effort"] = str(reasoning_effort)
    resp = client.chat.completions.create(timeout=float(timeout_s), **kwargs)
    return str(resp.choices[0].message.content or "")


def main() -> None:
    p = argparse.ArgumentParser(description="Create a Codabench-ready submission via OpenAI API.")
    p.add_argument("--split", default="test", choices=["test", "dev"])
    p.add_argument("--language", default="en", choices=["en", "it"])
    p.add_argument("--limit", type=int, default=0, help="Limit number of docs (0 = all).")
    p.add_argument("--model", default="gpt-5.2")
    p.add_argument("--verbosity", default="low", choices=["low", "medium", "high"])
    p.add_argument("--reasoning-effort", default="none", choices=["none", "low", "medium", "high"])
    p.add_argument("--max-new-tokens", type=int, default=1800)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--timeout-s", type=int, default=600)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--max-note-chars", type=int, default=0, help="Trim note text before inference (0 disables).")
    p.add_argument("--trim-strategy", default="middle", choices=["middle", "head", "tail", "head_tail"])
    p.add_argument("--fp-gates", action="store_true", default=True, help="Enable evidence-gated post-filters (default: enabled).")
    p.add_argument("--no-fp-gates", dest="fp_gates", action="store_false")
    p.add_argument(
        "--strict-no-inference",
        action="store_true",
        default=False,
        help="Add a stronger instruction: only extract explicitly stated facts; do not infer from context.",
    )
    p.add_argument("--out-dir", default="submissions/openai")
    p.add_argument("--run-dir", default="", help="Optional explicit run directory (overrides timestamp-based run_{ts}_{lang}).")
    p.add_argument("--resume", action="store_true", help="Resume from existing submission.jsonl in --run-dir.")
    p.add_argument("--zip", action="store_true", default=True, help="Write Codabench zip (default: enabled).")
    p.add_argument("--no-zip", dest="zip", action="store_false")
    args = p.parse_args()

    if not _OPENAI_AVAILABLE:
        raise SystemExit("openai package not available in venv; install openai>=2.0.0")

    load_dotenv()
    api_key = _get_openai_api_key()
    if not api_key:
        raise SystemExit("Missing GPT_API_KEY_OPEN_AI (or OPENAI_API_KEY).")

    hf_token = os.environ.get("HF_TOKEN") or None
    from datasets import load_dataset

    dsid = _dataset_id(args.split)
    ds = load_dataset(dsid, split=args.language, token=hf_token)
    n_total = len(ds) if int(args.limit) <= 0 else min(len(ds), int(args.limit))
    if n_total <= 0:
        raise SystemExit("No docs to process.")

    items = _get_items_from_record(ds[0])
    base_sys = build_system_instruction(items, sparse_output=True)
    if str(args.language) == "it":
        base_sys = (
            base_sys
            + "\n\nNOTE: The clinical note may be in Italian. Extract facts from Italian text, but keep CRF item names exactly as listed."
        )
    if bool(getattr(args, "strict_no_inference", False)):
        base_sys = (
            base_sys
            + "\n\nSTRICTNESS OVERRIDE:\n"
            + "- Extract ONLY facts that are explicitly stated in the note (or a direct numeric measurement).\n"
            + "- Do NOT infer diagnoses/conditions from treatments, test orders, past history templates, or clinician suspicion.\n"
            + "- If the note uses speculative language (possible/probable/likely/consider/rule out), output unknown/omit.\n"
            + "- Do NOT output negative answers unless the note explicitly negates the item."
        )

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

    client = OpenAI(api_key=api_key)  # type: ignore

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

        raw = ""
        parse_err: str | None = None
        preds_map: dict[str, object] = {}

        last_e: Exception | None = None
        for attempt in range(max(1, int(args.retries))):
            try:
                raw = _openai_chat_json(
                    client=client,
                    model=str(args.model),
                    system=base_sys,
                    user=user_prompt,
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    max_tokens=int(args.max_new_tokens),
                    timeout_s=float(args.timeout_s),
                    verbosity=str(args.verbosity),
                    reasoning_effort=str(args.reasoning_effort),
                )
                break
            except Exception as e:
                last_e = e
                raw = ""
                time.sleep(min(10.0, (2**attempt) * 1.5))
                continue

        (run_dir / f"{doc_id}.raw.txt").write_text(raw or (str(last_e) if last_e else ""), encoding="utf-8")

        try:
            obj = extract_json_object(raw)
            preds_map = coerce_predictions_mapping(obj)
            if bool(getattr(args, "fp_gates", True)):
                preds_map = _apply_fp_gates_note_only(preds_map, note)
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

    meta = {
        "dataset_id": dsid,
        "language": args.language,
        "limit": int(args.limit),
        "model": str(args.model),
        "max_note_chars": int(args.max_note_chars),
        "trim_strategy": str(args.trim_strategy),
        "max_new_tokens": int(args.max_new_tokens),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "verbosity": str(args.verbosity),
        "reasoning_effort": str(args.reasoning_effort),
    }
    if args.split == "dev":
        # Reference for scoring must be aligned to submission.jsonl order.
        # We score on the first N docs with GT (development split has GT).
        ref = []
        for j in range(n_total):
            r = ds[j]
            ref.append({"document_id": str(r.get("document_id") or "").split("_", 1)[0], "annotations": r.get("annotations") or []})
        score = score_records(ref, submission)
        meta["score"] = {"macro_f1": score.macro_f1, "tp": score.tp, "fp": score.fp, "fn": score.fn}

    (run_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote: {run_dir}")
    if args.split == "dev" and "score" in meta:
        print(json.dumps(meta["score"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
