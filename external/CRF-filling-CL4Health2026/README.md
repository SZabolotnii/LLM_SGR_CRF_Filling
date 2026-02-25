# CRF:Filling Shared Task @ CL4Health (LREC 2026)


# Codabench Submission Helper

This workspace includes a tiny utility to validate a result-only submission JSONL and create a clean ZIP ready to upload to Codabench.

1. Calculate the F1 for your submission locally
2. Create the ZIP file for automatic scoring on the [Codabench leaderboard](https://www.codabench.org/competitions/11984/#/results-tab) 

## Index
- [Expected JSONL Structure](#expected-jsonl-structure)
- [Local Scoring](#local-scoring)
  - [What it expects](#what-it-expects)
  - [Run](#run)
  - [Output](#output)
  - [Tips](#tips)
- [Validator Script](#validator-script)
  - [Usage](#usage)
  - [Output](#output-1)
- [Upload](#upload)


## Expected JSONL Structure
An example of the submission file participants must create is presented at `your_submission_data/example_submission_italian.jsonl`.

The submission file must have the following structure:
- One JSON object per line (JSONL)
- Each record has:
  - `id`: non-empty string (e.g., patient ID)
  - `predictions`: non-empty list of objects, each with:
    - `item`: non-empty string
    - `prediction`: string

Example (single line, truncated):
```json
{"id":"1234","predictions":[{"item":"Exam: haemoglobin ","prediction":" 8g/dl. "}, ... ]}
```

## Local Scoring
Use `scoring.py` to evaluate your submission locally.

### What it expects
- Input predictions: `your_submission_data/example_submission_italian.jsonl`
- Language: `en` or `it`

### Run
```bash
python3 scoring.py \
  --submission_path your_submission_data/example_submission_italian.jsonl \
  --language it
```

### Output
- Writes a JSON file with the metrics to `your_sumbmission_scores/scores.json` (created automatically).
- Prints a short summary to the terminal.

### Tips
- Make sure your `example_submission_italian.jsonl` follows the expected submission schema for this task.
- The reference file for development runs is fixed to `development_data/dev_gt.jsonl` inside the script.
- Use `--language it` for Italian submissions; `en` for English.



## Validator Script
`check_submission_format.py` validates the structure and writes a ZIP with the file at the archive root as `mock_data_dev_codabench.jsonl`.

### Usage
```bash
python3 check_submission_format.py your_submission_data/dev_submission_all_correct.jsonl --out your_submission_data/submission_validated.zip
```

- If validation passes, it prints the output path and exits with code 0.
- On failure, it prints a clear error and exits non‑zero.
- If your input filename differs, the script warns and still writes the ZIP with the required name.

### Output
The produced ZIP contains exactly one file at the root:
```
submission_validated.zip
└── mock_data_dev_codabench.jsonl
```

### Final Upload
- Upload the resulting ZIP to [Codabench](https://www.codabench.org/competitions/11984/#/participate-tab)
