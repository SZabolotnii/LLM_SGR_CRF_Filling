#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from llm_structcore.crf_prompt import build_system_instruction, build_user_prompt
from llm_structcore.normalize_predictions import coerce_predictions_mapping, predictions_to_submission
from llm_structcore.stage1_sgr import (
    build_stage1_system_prompt,
    coerce_stage1_summary,
    extract_json_object,
    has_thinking_leak,
    resolve_stage1_keys,
    STAGE1_KEYS_7,
    STAGE1_KEYS_9,
    stage1_summary_to_text,
)
from llm_structcore.submission import validate_submission_structure, write_jsonl, zip_codabench_jsonl


def _get_items_from_record(rec: dict) -> list[str]:
    if "annotations" in rec and isinstance(rec["annotations"], list) and rec["annotations"]:
        ann0 = rec["annotations"][0]
        if isinstance(ann0, dict) and "item" in ann0:
            return [a["item"] for a in rec["annotations"]]
    if "expected_crf_items" in rec and isinstance(rec["expected_crf_items"], list):
        return list(rec["expected_crf_items"])
    raise ValueError("Cannot infer CRF item list from dataset record.")


def _chat_prompt(tokenizer, messages: list[dict]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    # Fallback for non-chat tokenizers
    return "\n\n".join([m["content"] for m in messages]) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--language", default="en", choices=["en", "it"])
    p.add_argument("--split", default="test", choices=["test", "dev"])

    p.add_argument("--stage1-model-id", default="google/medgemma-1.5-4b-it")
    p.add_argument("--stage2-model-id", required=True, help="Fine-tuned CRF model (merged) OR base model if using adapters.")
    p.add_argument("--stage2-base-model-id", default=None, help="Stage2 base model id (use with --stage2-adapter-id).")
    p.add_argument("--stage2-adapter-id", default=None, help="Stage2 PEFT adapter id (use with --stage2-base-model-id).")

    p.add_argument("--out-jsonl", required=True)
    p.add_argument("--out-zip", default=None)
    p.add_argument("--max-new-tokens-s1", type=int, default=768)
    p.add_argument("--max-new-tokens-s2", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--limit", type=int, default=0, help="Limit number of docs (0 = all).")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--stage1-retries", type=int, default=2)
    p.add_argument("--stage1-profile", default="9", choices=["9", "7"], help="Stage1 key set size.")
    p.add_argument(
        "--stage1-keys",
        default="",
        help="Optional override: comma-separated Stage1 keys (must be subset of the 9 allowed keys).",
    )
    args = p.parse_args()

    load_dotenv()
    hf_token = os.environ.get("HF_TOKEN") or None

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    dataset_id = "NLP-FBK/dyspnea-crf-test" if args.split == "test" else "NLP-FBK/dyspnea-crf-development"
    ds = load_dataset(dataset_id, split=args.language)

    first = ds[0]
    items = _get_items_from_record(first)

    if args.stage1_keys.strip():
        stage1_keys = resolve_stage1_keys([k.strip() for k in args.stage1_keys.split(",")])
    else:
        stage1_keys = STAGE1_KEYS_9 if args.stage1_profile == "9" else STAGE1_KEYS_7

    stage1_sys = build_stage1_system_prompt(stage1_keys)
    stage2_sys = build_system_instruction(items, sparse_output=True)

    def load_model(model_id: str):
        try:
            tok = AutoTokenizer.from_pretrained(
                model_id, token=hf_token, trust_remote_code=args.trust_remote_code, use_fast=True
            )
        except Exception:
            tok = AutoTokenizer.from_pretrained(
                model_id, token=hf_token, trust_remote_code=args.trust_remote_code, use_fast=False
            )
        kwargs = {
            "token": hf_token,
            "trust_remote_code": args.trust_remote_code,
            "device_map": "auto",
            "low_cpu_mem_usage": True,
        }
        if torch.cuda.is_available():
            kwargs["torch_dtype"] = torch.bfloat16
        m = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        m.eval()
        return m, tok

    stage1_model, stage1_tok = load_model(args.stage1_model_id)

    if args.stage2_adapter_id:
        if not args.stage2_base_model_id:
            raise SystemExit("Provide --stage2-base-model-id together with --stage2-adapter-id.")
        stage2_model, stage2_tok = load_model(args.stage2_base_model_id)
        from peft import PeftModel

        stage2_model = PeftModel.from_pretrained(stage2_model, args.stage2_adapter_id, token=hf_token)
        stage2_model.eval()
    else:
        stage2_model, stage2_tok = load_model(args.stage2_model_id)

    def gen_kwargs(max_new_tokens: int) -> dict:
        kw = {"max_new_tokens": max_new_tokens}
        if args.temperature > 0:
            kw.update({"do_sample": True, "temperature": args.temperature, "top_p": args.top_p})
        else:
            kw["do_sample"] = False
        return kw

    out_records: list[dict] = []
    n_total = len(ds) if args.limit <= 0 else min(len(ds), args.limit)

    for i in tqdm(range(n_total), desc=f"2stage:{args.split}:{args.language}"):
        rec = ds[i]
        doc_id = rec.get("document_id") or rec.get("id")
        note = rec.get("clinical_note") or rec.get("note") or ""
        if not doc_id:
            raise ValueError(f"Missing document id for index {i}")

        # ── Stage 1: stable JSON summary over 9 categories
        stage1_summary: dict[str, str] | None = None
        for attempt in range(max(1, args.stage1_retries)):
            s1_messages = [
                {"role": "system", "content": stage1_sys},
                {"role": "user", "content": f"Clinical Note:\n{note}\n\nJSON Summary:"},
            ]
            s1_prompt = _chat_prompt(stage1_tok, s1_messages)
            s1_inp = stage1_tok(s1_prompt, return_tensors="pt")
            s1_inp = {k: v.to(stage1_model.device) for k, v in s1_inp.items()}

            s1_out = stage1_model.generate(
                **s1_inp,
                pad_token_id=stage1_tok.eos_token_id,
                **gen_kwargs(args.max_new_tokens_s1),
            )
            s1_text = stage1_tok.decode(s1_out[0][s1_inp["input_ids"].shape[-1] :], skip_special_tokens=True)
            try:
                if has_thinking_leak(s1_text):
                    raise ValueError("Thinking leak detected in Stage1 output prefix.")
                raw = extract_json_object(s1_text)
                stage1_summary = coerce_stage1_summary(raw, stage1_keys)
                break
            except Exception:
                stage1_summary = None
                continue

        if stage1_summary is None:
            stage1_summary = coerce_stage1_summary({}, stage1_keys)

        stage1_text = stage1_summary_to_text(stage1_summary, stage1_keys)

        # ── Stage 2: CRF extraction conditioned on Stage1 summary text (not the full note)
        s2_user = (
            "## Stage 1 Structured Summary (trusted intermediate)\n"
            f"{stage1_text}\n"
            "## Extract CRF Items to JSON:\n"
        )
        s2_messages = [{"role": "system", "content": stage2_sys}, {"role": "user", "content": s2_user}]
        s2_prompt = _chat_prompt(stage2_tok, s2_messages)
        s2_inp = stage2_tok(s2_prompt, return_tensors="pt")
        s2_inp = {k: v.to(stage2_model.device) for k, v in s2_inp.items()}

        s2_out = stage2_model.generate(
            **s2_inp,
            pad_token_id=stage2_tok.eos_token_id,
            **gen_kwargs(args.max_new_tokens_s2),
        )
        s2_text = stage2_tok.decode(s2_out[0][s2_inp["input_ids"].shape[-1] :], skip_special_tokens=True)

        try:
            preds_obj = extract_json_object(s2_text)
            preds = coerce_predictions_mapping(preds_obj)
        except Exception:
            preds = {}

        out_records.append(
            {
                **predictions_to_submission(str(doc_id), preds, items, language=args.language),
                "_debug_stage1_summary": stage1_summary,
            }
        )

    # Strip debug blocks before writing submission.
    submission_records = [{k: v for k, v in r.items() if not k.startswith("_debug_")} for r in out_records]
    validate_submission_structure(submission_records, n_items=len(items))

    out_jsonl = Path(args.out_jsonl)
    write_jsonl(out_jsonl, submission_records)
    print(f"Wrote JSONL: {out_jsonl} ({len(submission_records)} records)")

    if args.out_zip:
        out_zip = Path(args.out_zip)
        zip_codabench_jsonl(out_jsonl, out_zip)
        print(f"Wrote ZIP:   {out_zip}")


if __name__ == "__main__":
    main()
