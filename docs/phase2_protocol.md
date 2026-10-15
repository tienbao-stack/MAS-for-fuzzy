# Phase 2 Protocol — B0 Direct Multimodal Baseline

## Objective

Create a reproducible first experimental reference point for the Text + Image Physics QA task. No audio or video is in scope. ManyModalQA is not part of this project or its baseline comparison.

## Experimental setup

- **Dataset:** `data/processed/physics_dataset_candidate.json` (1,708 samples).
- **Split:** `data/splits/split_assignments.json`, seed 42.
- **Inputs to model:** `input.question` and `input.image.image_ref` resolved under `data/images/` only.
- **Forbidden inference fields:** `reference.solution_cot`, `reference.answer`, `reference.unit`. These are used only after inference.
- **B0:** one vision-capable MLLM/VLM call per sample; zero-shot direct solve; no fuzzy interpretations, no agents, no evidence-ranking module.
- **Reproducibility:** save model name, raw output, parsed answer/unit, status, latency, token usage where returned, prompt version, and all evaluation tolerances.

## Split policy

Connected components are formed when records share either (a) the SHA-256 hash of the exact image bytes, or (b) the same normalized question after whitespace normalization and case-folding. Each component is assigned to one split. This prevents exact image-pixel and exact-question overlap across splits. Near-duplicate templates still need future audit.

| Split | Samples |
|---|---:|
| Train | 1,195 |
| Validation | 257 |
| Test | 256 |

B0 does not train on the train split; it is included for future methods that may require development or training data. Use validation for prompt selection and debugging. Freeze prompt and configuration before one final test run.

## Metrics

1. **Answer accuracy with numeric tolerance:** for numeric answers, `abs(pred-gold) <= max(0.001, 0.01 * abs(gold))`; non-numeric answers use normalized exact text comparison.
2. **Exact answer accuracy:** numeric values match to `1e-9`, or normalized text is identical.
3. **Unit accuracy:** normalized predicted unit equals normalized gold unit.
4. **Joint answer+unit accuracy:** answer and unit both correct.
5. **Parse rate:** predicted answer could be parsed.
6. **Numeric MAE / relative error:** calculated only on scalar pairs that can be parsed numerically.
7. **95% Wilson interval:** reported for answer accuracy and joint accuracy.

Do not string-match generated explanation to `cot`; equivalent valid reasoning need not have identical wording. Reasoning quality requires a separately defined human rubric or a calibrated judge with human validation.

## Commands

Install (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set a vision-capable model and credentials in `.env`, then start with a small validation pilot:

```powershell
python src/baseline_openai.py --split validation --limit 20 --output data/predictions/b0_validation_pilot.jsonl
python src/evaluate_predictions.py --predictions data/predictions/b0_validation_pilot.jsonl --split validation
```

Once the model/prompt are stable, run the full validation split, review failures and freeze the protocol. Only then run the test set:

```powershell
python src/baseline_openai.py --split test --output data/predictions/b0_test_frozen.jsonl
python src/evaluate_predictions.py --predictions data/predictions/b0_test_frozen.jsonl --split test
```

If a run is interrupted, use the same output path and `--resume`. If a file exists and you intend to replace it, pass `--overwrite`; never combine different models/prompts in one file.

## Smoke-test command (not research results)

```powershell
python src/baseline_openai.py --split validation --limit 3 --smoke-test-from-gold --output data/predictions/smoke_test_from_gold.jsonl
python -m unittest discover -s tests -v
```

The smoke test copies reference answers only to verify file and evaluator plumbing. It is tagged `is_mock=true` and excluded from metrics unless the unsafe `--include-mock` option is deliberately passed. Do not report its score as model performance.

## Limits before a paper-quality benchmark report

The benchmark candidate is still not fully adjudicated. The current Phase 2 run can establish that the experimental pipeline and metric code work and can generate a baseline result, but the result should be labeled preliminary until a stratified human review verifies image-question-answer alignment and answer/unit correctness. No fuzzy-interpretation, agent-verification, or context-selection gold annotations are present yet.
