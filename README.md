# MedGemma StructCore: Schema-Guided Condensation and Deterministic Compilation for CRF Filling

This repository contains the official codebase for the MedGemma StructCore submission to the **CRF-filling Shared Task at CL4Health (LREC-COLING 2026)**.

## Architecture

The system uses a **two-stage, contract-driven pipeline** to solve the extreme sparsity and false-positive sensitivity of clinical CRF filling:

1. **Stage 1 (LLM-based Condensation):** An LLM reads the clinical note and produces a stable, 9-category JSON summary (Schema-Guided Reasoning).
2. **Stage 2 (0-LLM Deterministic Compiler):** A deterministic Python compiler parses the summary, canonicalizes terms (optionally using a UMLS alias map), derives high-FN items, applies evidence-gated FP filters, and normalizes output to the 134-item controlled vocabulary.

## Repository Structure

- `llm_structcore/` - The core Python package with the Stage 1 prompts and Stage 2 compiler.
- `scripts/` - CLI entrypoints for running end-to-end inference across various backends (llama.cpp, vLLM/OpenAI-compat, Transformers, NVIDIA API, Gemini Vertex).
- `data/` - Static assets including the UMLS CRF alias mapping (`134/134` coverage) and the task ontology.
- `external/` - Organizer-provided evaluation scripts (`scoring.py`, `check_submission_format.py`).
- `paper/` - The LREC 2026 system description paper (LaTeX and Ukrainian source).

## Quickstart

Clone the repository and install dependencies:
```bash
git clone https://github.com/SZabolotnii/LLM_SGR_CRF_Filling.git
cd LLM_SGR_CRF_Filling
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 1) Run an End-to-End Submission
Requires a downloaded/fine-tuned model (e.g., `DocUA/CRF_Filling_CL4Health_LREC_2026`).

**Via llama.cpp (Local Edge AI):**
```bash
python3 scripts/make_submission_llama.py \
  --language en \
  --split test \
  --model-url http://127.0.0.1:8080 \
  --out-jsonl submissions/submission_test_en.jsonl \
  --out-zip submissions/submission_test_en.zip
```

**Via HF Transformers:**
```bash
python3 scripts/make_submission_two_stage_transformers.py \
  --language en \
  --split test \
  --stage1-model-id google/medgemma-1.5-4b-it \
  --stage2-model-id DocUA/CRF_Filling_CL4Health_LREC_2026 \
  --out-zip submissions/submission_test_en.zip
```

### 2) Local Scoring (Dev Set)
```bash
python3 scripts/score_submission.py \
  --ref-jsonl external/CRF-filling-CL4Health2026/development_data/dev_gt.jsonl \
  --submission-jsonl submissions/submission_dev_en.jsonl
```

## Reproducibility
All deterministic logic is contained in `llm_structcore/stage2_compiler.py`.
The Stage 1 SGR schema guarantees `json_parse_ok=100%` with MedGemma-1.5-4B without requiring JSON-mode sampling. 

## License
MIT License.
