# OFT Text-to-SQL Mini-Project

This repository implements an end-to-end pipeline to finetune a pretrained LLM with **OFT (Orthogonal Finetuning)** for schema-grounded Text-to-SQL generation on `b-mc2/sql-create-context`.

## Project layout

- `src/data_sql.py`: dataset loading, reproducible splitting, prompt formatting, SQL normalization.
- `src/train_oft_sql.py`: OFT adapter training with Hugging Face `transformers` + `peft`.
- `src/merge_oft_adapter.py`: merge OFT adapter back into full model weights for deployment/inference.
- `src/eval_sql.py`: baseline and post-finetuning evaluation (merges OFT first, then compares base vs merged model).
- `src/plot_metrics.py`: plotting training/evaluation artifacts.
- `configs/sql_smoke.yaml`: 20-step smoke configuration.
- `configs/sql_4h.yaml`: bounded full-run configuration for ~4h single GPU budget.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch transformers datasets peft accelerate sentencepiece sqlparse matplotlib pandas pyyaml
```

## 1) Data smoke test (CPU)

```bash
python src/data_sql.py --train-size 500 --val-size 100 --test-size 100 --split-index-path outputs/splits/split_indices.json
```

## 2) Smoke training (GPU)

```bash
python src/train_oft_sql.py --config configs/sql_smoke.yaml
```

## 3) Full bounded training run (GPU)

```bash
python src/train_oft_sql.py --config configs/sql_4h.yaml
```

## 4) Merge OFT adapter into full model weights

```bash
python src/merge_oft_adapter.py \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --adapter-path outputs/train_4h/best_adapter \
  --output-dir outputs/train_4h/merged_model
```

## 5) Baseline + merged-OFT evaluation (vLLM)

```bash
python src/eval_sql.py \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --adapter-path outputs/train_4h/best_adapter \
  --split-index-path outputs/splits/full_split_indices.json \
  --sample-count 2000 \
  --output-dir outputs/eval
```

`eval_sql.py` first merges the OFT adapter into full model weights (`outputs/eval/merged_oft_model` by default), then runs inference on:
- the base model
- the merged finetuned model (without runtime adapters)

Outputs:
- `outputs/eval/metrics.json`
- `outputs/eval/baseline_predictions.csv`
- `outputs/eval/oft_predictions.csv`
- `outputs/eval/qualitative_examples.json`

## 6) Plot report figures (CPU)

```bash
python src/plot_metrics.py \
  --trainer-log outputs/train_4h/trainer_state.json \
  --eval-metrics outputs/eval/metrics.json \
  --output-dir outputs/plots
```

Outputs:
- `outputs/plots/loss_curves.png`
- `outputs/plots/eval_metrics.png`

## Notes

- Exact match metric is computed on normalized SQL (lowercased, compacted whitespace, trailing semicolon removed).
- SQL parse success uses `sqlparse` parseability.
- Prompt style enforces SQL-only responses.
