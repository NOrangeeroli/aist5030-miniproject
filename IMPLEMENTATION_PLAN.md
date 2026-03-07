# OFT Mini-Project Implementation Plan

## Task
- Finetune a pretrained LLM with OFT for schema-grounded Text-to-SQL generation.
- Input: natural language question + `CREATE TABLE ...` schema context.
- Output: SQL query.
- Primary dataset: `b-mc2/sql-create-context`.

## 1. Project Setup
- Create scripts:
  - `src/data_sql.py` for loading, splitting, subsampling, and prompt formatting.
  - `src/train_oft_sql.py` for OFT finetuning.
  - `src/eval_sql.py` for exact match, SQL parse success, and qualitative outputs.
  - `src/plot_metrics.py` for training/evaluation curves.
- Add config presets:
  - `configs/sql_4h.yaml` for the full run.
  - `configs/sql_smoke.yaml` for fast smoke testing.

## 2. Model and OFT Design
- Base model: `Qwen2.5-0.5B-Instruct` (fallback `Qwen-1.3B` if GPU permits).
- Use Hugging Face PEFT `OFTConfig` targeting linear layers in attention/MLP blocks.
- Freeze base model weights and train OFT adapters only.

## 3. Data Pipeline
- Use instruction-style prompts with strict output format: SQL-only response.
- Start with max sequence length around 384 (increase to 512 only if memory allows).
- Enforce a bounded training subset for runtime control:
  - Train: 15k to 25k examples.
  - Validation: 2k examples.
  - Test: 2k examples.
- Save split indices for reproducibility.

## 4. Training Strategy (Single GPU, ~4 Hours)
- Enable mixed precision (`bf16` if supported, else `fp16`).
- Use gradient accumulation with a small per-device batch size.
- Use warmup + cosine or linear learning-rate scheduler.
- Add early-stop guard on validation stagnation.
- Set a hard `max_steps` limit to keep runtime within the budget.

## 5. Evaluation Protocol
- Baseline: evaluate base model on the fixed test subset before finetuning.
- Post-finetuning: evaluate with the same test subset and decoding settings.
- Metrics:
  - Exact Match (EM) on normalized SQL strings.
  - SQL parse success rate.
  - Optional execution match (if feasible in available time).
- Export 8 to 12 qualitative before/after examples.

## 6. Report Artifacts (3 Pages)
- Training and validation loss curves.
- Baseline vs OFT metric table.
- Qualitative examples: question, schema, prediction before/after, gold SQL.
- Brief error analysis by failure type.

## 7. Test Checklist
- `[CPU]` Dataset load + prompt-formatting smoke test.
- `[CPU]` Evaluation script unit test on tiny handcrafted examples.
- `[CPU]` Plotting script test using dummy logs.
- `[GPU]` 20-step smoke training run using `configs/sql_smoke.yaml`.
- `[GPU]` Full bounded training run using `configs/sql_4h.yaml`.
- `[GPU]` Final evaluation + adapter reload consistency check.

## 8. Deliverables
- Reproducible project code + README run commands.
- Saved OFT adapter checkpoints.
- Metrics JSON/CSV and generated plots for the final report.
