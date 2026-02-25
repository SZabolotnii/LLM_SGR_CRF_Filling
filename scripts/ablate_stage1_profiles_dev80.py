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

from llm_structcore.crf_prompt import build_system_instruction
from llm_structcore.normalize_predictions import predictions_to_submission
from llm_structcore.scoring import score_records
from llm_structcore.stage1_sgr import (
    STAGE1_KEYS_7,
    STAGE1_KEYS_9,
    build_stage1_system_prompt,
    coerce_stage1_summary,
    extract_json_object,
    has_thinking_leak,
    stage1_summary_to_text,
)
from llm_structcore.submission import validate_submission_structure, write_jsonl, zip_codabench_jsonl


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _chat_prompt(tokenizer, messages: list[dict]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return "\n\n".join([m["content"] for m in messages]) + "\n"


def _get_doc_id_plain(doc_id: str) -> str:
    return str(doc_id).split("_", 1)[0]


def _load_processed_doc_ids_from_submission(path: Path) -> set[str]:
    if not path.exists():
        return set()
    processed: set[str] = set()
    for rec in _load_jsonl(path):
        doc_id = rec.get("document_id")
        if not doc_id:
            continue
        processed.add(_get_doc_id_plain(str(doc_id)))
    return processed


def _append_jsonl(path: Path, rec: dict) -> None:
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


def _get_items_from_dev_record(rec: dict) -> list[str]:
    anns = rec.get("annotations") or []
    if not anns:
        raise ValueError("Dev record has no annotations; cannot infer items.")
    return [a["item"] for a in anns]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--language", default="en", choices=["en", "it"])
    p.add_argument("--split", default="dev", choices=["dev", "train"], help="HF dataset split to evaluate.")
    p.add_argument("--backend", default="llama", choices=["llama", "transformers"])

    # llama-server backend (OpenAI-compatible)
    p.add_argument("--stage1-url", default="http://127.0.0.1:1245")
    p.add_argument("--stage1-model", default="medgemma-base")
    p.add_argument("--stage2-url", default="")
    p.add_argument("--stage2-model", default="")
    grp = p.add_mutually_exclusive_group()
    grp.add_argument(
        "--json-mode",
        dest="stage1_json_mode",
        action="store_true",
        default=True,
        help="Stage1: best-effort JSON-only mode via response_format (default: enabled).",
    )
    grp.add_argument(
        "--no-json-mode",
        dest="stage1_json_mode",
        action="store_false",
        help="Stage1: disable response_format JSON-only mode.",
    )
    p.add_argument(
        "--stage2-json-mode",
        action="store_true",
        default=False,
        help="Stage2: best-effort JSON-only mode via response_format (default: disabled).",
    )

    # transformers backend (optional)
    p.add_argument("--stage1-model-id", default="google/medgemma-1.5-4b-it")
    p.add_argument("--stage2-model-id", default="")
    p.add_argument("--stage2-base-model-id", default=None)
    p.add_argument("--stage2-adapter-id", default=None)
    p.add_argument("--max-new-tokens-s1", type=int, default=768)
    p.add_argument("--max-new-tokens-s2", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--limit", type=int, default=0, help="Limit docs (0 = all dev=80).")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--stage1-retries", type=int, default=2)
    p.add_argument("--max-note-chars", type=int, default=4000, help="Trim the clinical note for Stage1 speed/stability. 0 disables.")
    p.add_argument("--trim-strategy", default="middle", choices=["middle", "head", "tail", "head_tail"])
    p.add_argument("--resume", action="store_true", help="Resume from existing outputs in --out-dir.")
    p.add_argument("--out-dir", default="submissions/ablation_dev")
    p.add_argument("--zip", action="store_true", help="Also write Codabench ZIPs for each profile.")
    p.add_argument("--use-official-ref-jsonl", default="", help="Optional path to official dev_gt.jsonl (for exact parity).")
    args = p.parse_args()

    load_dotenv()
    hf_token = os.environ.get("HF_TOKEN") or None

    from datasets import load_dataset

    dataset_id = "NLP-FBK/dyspnea-crf-development" if args.split == "dev" else "NLP-FBK/dyspnea-crf-train"
    ds = load_dataset(dataset_id, split=args.language)
    n_total = len(ds) if args.limit <= 0 else min(len(ds), args.limit)

    items = _get_items_from_dev_record(ds[0])
    stage2_sys = build_system_instruction(items, sparse_output=True)

    if args.backend == "transformers":
        if not args.stage2_model_id and not (args.stage2_base_model_id and args.stage2_adapter_id):
            raise SystemExit("Transformers backend requires --stage2-model-id or (--stage2-base-model-id + --stage2-adapter-id).")

        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

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

    else:
        from llm_structcore.llama_client import LlamaChatConfig, chat_completions

        if not args.stage2_url:
            args.stage2_url = args.stage1_url
        if not args.stage2_model:
            args.stage2_model = args.stage1_model

        stage1_cfg = LlamaChatConfig(
            base_url=args.stage1_url,
            model=args.stage1_model,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens_s1,
            json_mode=bool(args.stage1_json_mode),
        )
        stage2_cfg = LlamaChatConfig(
            base_url=args.stage2_url,
            model=args.stage2_model,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens_s2,
            json_mode=bool(args.stage2_json_mode),
        )

    # Reference list (official schema): doc_id without suffix + ground_truth annotations.
    if args.use_official_ref_jsonl.strip():
        reference = _load_jsonl(Path(args.use_official_ref_jsonl))
        # Filter / align to dataset order and language if needed:
        ref_map = {str(r["document_id"]): r for r in reference}
        reference = []
        for i in range(n_total):
            plain = _get_doc_id_plain(ds[i]["document_id"])
            reference.append(ref_map[plain])
    else:
        reference = []
        for i in range(n_total):
            rec = ds[i]
            plain = _get_doc_id_plain(rec["document_id"])
            reference.append({"document_id": plain, "annotations": rec["annotations"]})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_summary: list[dict] = []

    for profile_name, stage1_keys in [("p9", STAGE1_KEYS_9), ("p7", STAGE1_KEYS_7)]:
        stage1_sys = build_stage1_system_prompt(stage1_keys)
        out_jsonl = out_dir / f"submission_{args.split}_{args.language}_{profile_name}.jsonl"

        processed = _load_processed_doc_ids_from_submission(out_jsonl) if args.resume else set()
        if processed:
            print(f"[{profile_name}] resume: {len(processed)} docs already in {out_jsonl}")

        for i in tqdm(range(n_total), desc=f"profile:{profile_name}"):
            rec = ds[i]
            doc_id = rec["document_id"]
            plain_doc_id = _get_doc_id_plain(doc_id)
            if plain_doc_id in processed:
                continue
            note = _trim_text(
                rec.get("clinical_note") or "",
                max_chars=args.max_note_chars,
                strategy=args.trim_strategy,
            )

            # Stage1
            stage1_summary: dict[str, str] | None = None
            for _attempt in range(max(1, args.stage1_retries)):
                s1_messages = [
                    {"role": "system", "content": stage1_sys},
                    {"role": "user", "content": f"Clinical Note:\n{note}\n\nJSON Summary:"},
                ]

                if args.backend == "transformers":
                    s1_prompt = _chat_prompt(stage1_tok, s1_messages)
                    s1_inp = stage1_tok(s1_prompt, return_tensors="pt")
                    s1_inp = {k: v.to(stage1_model.device) for k, v in s1_inp.items()}
                    s1_out = stage1_model.generate(
                        **s1_inp,
                        pad_token_id=stage1_tok.eos_token_id,
                        **gen_kwargs(args.max_new_tokens_s1),
                    )
                    s1_text = stage1_tok.decode(s1_out[0][s1_inp["input_ids"].shape[-1] :], skip_special_tokens=True)
                else:
                    s1_text = chat_completions(stage1_cfg, s1_messages)

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

            # Stage2
            s2_user = (
                "## Stage 1 Structured Summary (trusted intermediate)\n"
                f"{stage1_text}\n"
                "## Extract CRF Items to JSON:\n"
            )
            s2_messages = [{"role": "system", "content": stage2_sys}, {"role": "user", "content": s2_user}]

            if args.backend == "transformers":
                s2_prompt = _chat_prompt(stage2_tok, s2_messages)
                s2_inp = stage2_tok(s2_prompt, return_tensors="pt")
                s2_inp = {k: v.to(stage2_model.device) for k, v in s2_inp.items()}
                s2_out = stage2_model.generate(
                    **s2_inp,
                    pad_token_id=stage2_tok.eos_token_id,
                    **gen_kwargs(args.max_new_tokens_s2),
                )
                s2_text = stage2_tok.decode(s2_out[0][s2_inp["input_ids"].shape[-1] :], skip_special_tokens=True)
            else:
                s2_text = chat_completions(stage2_cfg, s2_messages)

            try:
                from llm_structcore.normalize_predictions import coerce_predictions_mapping

                preds_obj = extract_json_object(s2_text)
                preds = coerce_predictions_mapping(preds_obj)
            except Exception:
                preds = {}

            submission_rec = predictions_to_submission(str(doc_id), preds, items, language=args.language)
            _append_jsonl(out_jsonl, submission_rec)
            processed.add(plain_doc_id)

        submission_records = _load_jsonl(out_jsonl)
        validate_submission_structure(submission_records, n_items=len(items))
        score = score_records(reference, submission_records, not_available="unknown")

        out_zip = out_dir / f"submission_{args.split}_{args.language}_{profile_name}.zip"
        if args.zip:
            zip_codabench_jsonl(out_jsonl, out_zip)

        results_summary.append(
            {
                "profile": profile_name,
                "stage1_keys": stage1_keys,
                "macro_f1": score.macro_f1,
                "tp": score.tp,
                "fp": score.fp,
                "fn": score.fn,
                "jsonl": str(out_jsonl),
                "zip": str(out_zip) if args.zip else "",
            }
        )

        print(f"[{profile_name}] macro_f1={score.macro_f1:.6f} tp/fp/fn={score.tp}/{score.fp}/{score.fn}")

    # Print delta
    by_prof = {r["profile"]: r for r in results_summary}
    if "p9" in by_prof and "p7" in by_prof:
        delta = float(by_prof["p7"]["macro_f1"]) - float(by_prof["p9"]["macro_f1"])
        print(f"Δ macro_f1 (p7 - p9): {delta:+.6f}")

    summary_path = out_dir / f"ablation_summary_{args.split}_{args.language}.json"
    summary_path.write_text(json.dumps(results_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()
