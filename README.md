# Multimodal Fuzzy Agent — Phase 2 (Physics Text + Image)

This package continues the Physics QA project from Phase 1. The current purpose is to **establish a reproducible direct MLLM/VLM baseline (B0)** before adding fuzzy interpretations, multi-agent verification, and context-aware selection. Scope is **Text + Image only**; no audio/video. ManyModalQA is intentionally not used.

## Phase 1 decision

- The cleaned Phase 1 file has 1,731 samples with all required fields and locally readable referenced images.
- A manual visual spot-check found four image groups that appear inconsistent with their question/solution. These 23 affected records remain in `physics_dataset_clean_all.json` for provenance but are quarantined from the provisional candidate.
- The runnable Phase 2 candidate has **1,708 samples** and is stored in `data/processed/physics_dataset_candidate.json` and `.jsonl`.
- The original 1,731-record clean dataset is preserved in `data/processed/physics_dataset_clean_all.json`; the original workbook is preserved in `data/source/`.
- See `docs/phase1_readiness_report.md` for evidence, caveats, and the benchmark-readiness verdict. `data/audit/human_review_blind_sample_100.csv` and `data/audit/human_review_gold_key_100.csv` form a stratified two-stage audit queue; the blind file excludes reference answers and `cot` until independent solving is complete. See `docs/human_review_instructions.md`.

## Structure

```text
.
├── configs/baseline.yaml
├── data/
│   ├── source/Physics_dataset_with_image.xlsx
│   ├── raw/physics_dataset_unfiltered.json
│   ├── processed/physics_dataset_clean_all.json
│   ├── processed/physics_dataset_candidate.json
│   ├── processed/physics_dataset_candidate.jsonl
│   ├── images/                  # image files referenced by the dataset
│   ├── splits/split_assignments.json
│   ├── audit/                   # cleaning stats, duplicate groups, quarantine queue
│   ├── predictions/             # model outputs (generated locally)
│   └── results/                 # evaluation summaries (generated locally)
├── docs/phase1_readiness_report.md
├── docs/phase2_protocol.md
├── src/
│   ├── create_candidate_dataset.py
│   ├── audit_dataset.py
│   ├── create_splits.py
│   ├── baseline_openai.py
│   └── evaluate_predictions.py
└── tests/
```

## Install on Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env` and set `OPENAI_API_KEY` and `OPENAI_MODEL` to an available **vision-capable** model. `OPENAI_BASE_URL` is optional for an OpenAI-compatible endpoint. Do not commit the `.env` file or share API keys.

## Rebuild candidate and splits after a data decision

If the quarantine list or dataset is changed, regenerate the Phase 2 candidate and split rather than editing split assignments manually:

```powershell
python src/create_candidate_dataset.py
python src/create_splits.py --seed 42
python src/create_review_sample.py --sample-size 100 --seed 20261015
python src/audit_dataset.py --dataset data/processed/physics_dataset_candidate.json --image-dir data/images --split-file data/splits/split_assignments.json --out-dir data/audit/candidate_audit
```

Re-read the new audit summary before using a test split, then regenerate the manifest with `python src/create_manifest.py`. Changing membership after test inference means that the test protocol should be treated as changed.

## Run B0 on a small validation pilot

```powershell
python src/baseline_openai.py --split validation --limit 20 --output data/predictions/b0_validation_pilot.jsonl
python src/evaluate_predictions.py --predictions data/predictions/b0_validation_pilot.jsonl --split validation
```

Use validation to debug output parsing and freeze the prompt. After configuration is frozen, run the test set once:

```powershell
python src/baseline_openai.py --split test --output data/predictions/b0_test_frozen.jsonl
python src/evaluate_predictions.py --predictions data/predictions/b0_test_frozen.jsonl --split test
```

To resume an interrupted run, use the same output file with `--resume`. If replacing a previous run, pass `--overwrite`. Avoid mixing models or prompt versions in a single predictions file.

## Data integrity manifest

`data/audit/benchmark_manifest.json` records candidate/split/config SHA-256 checksums, split counts and the referenced image file hashes. `data/audit/image_manifest.csv` maps each image reference to its SHA-256, dimensions and the number of samples using it. Regenerate them with `python src/create_manifest.py` after changing the candidate or split.

## No-key smoke test

```powershell
python src/baseline_openai.py --split validation --limit 3 --smoke-test-from-gold --output data/predictions/smoke_test_from_gold.jsonl
python -m unittest discover -s tests -v
```

This mode copies the reference answer/unit to verify dataflow only. Output is explicitly marked `is_mock=true`; the evaluator excludes such rows by default. It is **not a baseline experiment** and its scores must not be reported.

## Important evaluation contract

- The model sees only `input.question` and the linked image. It must never see `reference.solution_cot`, `reference.answer`, or `reference.unit` at inference time.
- The evaluator uses `answer` as primary gold, `unit` for unit correctness, and does not demand that generated reasoning match `cot` word-for-word.
- Metrics include tolerant answer accuracy, exact answer accuracy, unit accuracy, joint answer+unit accuracy, parse rate, numeric errors, and Wilson confidence intervals.
- This is a **pilot benchmark candidate**, not a fully validated public benchmark. More human review is required before making strong scientific claims.
