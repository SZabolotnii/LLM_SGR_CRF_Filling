#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from llm_structcore.crf_prompt import build_system_instruction, build_user_prompt
from llm_structcore.normalize_predictions import coerce_predictions_mapping, predictions_to_submission
from llm_structcore.submission import validate_submission_structure, write_jsonl, zip_codabench_jsonl


def _extract_json_object(text: str) -> dict:
    s = text.strip()
    if "```json" in s:
        s = s.split("```json", 1)[1]
        s = s.split("```", 1)[0].strip()
    elif "```" in s:
        s = s.split("```", 1)[1]
        s = s.split("```", 1)[0].strip()

    # Strip MedGemma "thinking" wrappers if any.
    s = re.sub(r"<unused\\d+>thought.*?(?:<unused\\d+>|$)", "", s, flags=re.DOTALL)
    s = re.sub(r"<unused\\d+>", "", s)

    if "{" not in s:
        raise ValueError("No JSON object start '{' found in model output.")
    start = s.find("{")
    end = s.rfind("}")
    if end <= start:
        raise ValueError("No JSON object end '}' found in model output.")
    return json.loads(s[start : end + 1])


def _get_items_from_record(rec: dict) -> list[str]:
    if "annotations" in rec and isinstance(rec["annotations"], list) and rec["annotations"]:
        ann0 = rec["annotations"][0]
        if isinstance(ann0, dict) and "item" in ann0:
            return [a["item"] for a in rec["annotations"]]
    if "expected_crf_items" in rec and isinstance(rec["expected_crf_items"], list):
        return list(rec["expected_crf_items"])
    raise ValueError("Cannot infer CRF item list from dataset record.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default="en", choices=["en", "it"])
    parser.add_argument("--split", default="test", choices=["test", "dev"])
    parser.add_argument("--model-id", default=None, help="Hugging Face model id (fine-tuned or base).")
    parser.add_argument("--base-model-id", default=None, help="Base model id (use with --adapter-id).")
    parser.add_argument("--adapter-id", default=None, help="PEFT adapter id (use with --base-model-id).")
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--out-zip", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=0, help="Limit number of docs (0 = all).")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    hf_token = os.environ.get("HF_TOKEN") or None

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    if not args.model_id and not (args.base_model_id and args.adapter_id):
        raise SystemExit("Provide either --model-id OR (--base-model-id and --adapter-id).")

    dataset_id = "NLP-FBK/dyspnea-crf-test" if args.split == "test" else "NLP-FBK/dyspnea-crf-development"
    ds = load_dataset(dataset_id, split=args.language)

    first = ds[0]
    items = _get_items_from_record(first)
    sys_prompt = build_system_instruction(items, sparse_output=True)

    model_id = args.model_id or args.base_model_id
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, token=hf_token, trust_remote_code=args.trust_remote_code, use_fast=True
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, token=hf_token, trust_remote_code=args.trust_remote_code, use_fast=False
        )

    model_kwargs = {
        "token": hf_token,
        "trust_remote_code": args.trust_remote_code,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }
    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

    if args.adapter_id:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter_id, token=hf_token)
        model.eval()

    out_records: list[dict] = []
    n_total = len(ds) if args.limit <= 0 else min(len(ds), args.limit)
    for i in tqdm(range(n_total), desc=f"infer:{args.split}:{args.language}"):
        rec = ds[i]
        doc_id = rec.get("document_id") or rec.get("id")
        note = rec.get("clinical_note") or rec.get("note") or ""
        if not doc_id:
            raise ValueError(f"Missing document id for index {i}")

        user_prompt = build_user_prompt(note)
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]

        if hasattr(tokenizer, "apply_chat_template"):
            prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        else:
            prompt_text = sys_prompt + "\n\n" + user_prompt

        inputs = tokenizer(prompt_text, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        gen_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if args.temperature > 0:
            gen_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                }
            )
        else:
            gen_kwargs["do_sample"] = False

        gen = model.generate(
            **inputs,
            **gen_kwargs,
        )
        out_text = tokenizer.decode(gen[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)

        try:
            preds_obj = _extract_json_object(out_text)
            preds = coerce_predictions_mapping(preds_obj)
        except Exception:
            preds = {}

        out_records.append(
            predictions_to_submission(
                str(doc_id),
                preds,
                items,
                language=args.language,
            )
        )

    validate_submission_structure(out_records, n_items=len(items))

    out_jsonl = Path(args.out_jsonl)
    write_jsonl(out_jsonl, out_records)
    print(f"Wrote JSONL: {out_jsonl} ({len(out_records)} records)")

    if args.out_zip:
        out_zip = Path(args.out_zip)
        zip_codabench_jsonl(out_jsonl, out_zip)
        print(f"Wrote ZIP:   {out_zip}")


if __name__ == "__main__":
    main()
