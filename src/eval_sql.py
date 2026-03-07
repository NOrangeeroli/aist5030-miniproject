"""Evaluate baseline vs OFT-adapted model for text-to-SQL."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import sqlparse
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_sql import DATASET_NAME, PromptConfig, SplitConfig, build_or_load_splits, format_for_training, load_sql_dataset, normalize_sql


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", type=str, required=True)
    p.add_argument("--adapter-path", type=str, default=None, help="Optional trained adapter directory")
    p.add_argument("--split-index-path", type=str, default="outputs/splits/split_indices.json")
    p.add_argument("--test-size", type=int, default=2000)
    p.add_argument("--val-size", type=int, default=2000)
    p.add_argument("--train-size", type=int, default=20000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--sample-count", type=int, default=200)
    p.add_argument("--output-dir", type=str, default="outputs/eval")
    return p.parse_args()


def sql_parses(query: str) -> bool:
    try:
        parsed = sqlparse.parse(query)
        return len(parsed) > 0 and any(tok.value.strip() for tok in parsed[0].tokens)
    except Exception:
        return False


def generate_sql(model, tokenizer, prompt: str, max_new_tokens: int = 128) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return generated.strip()


def compute_metrics(rows: List[Dict[str, str]]) -> Dict[str, float]:
    n = len(rows)
    if n == 0:
        return {"exact_match": 0.0, "parse_success": 0.0}

    em = sum(1 for r in rows if normalize_sql(r["prediction"]) == normalize_sql(r["gold"])) / n
    parse_ok = sum(1 for r in rows if sql_parses(r["prediction"])) / n
    return {"exact_match": em, "parse_success": parse_ok}


def run_eval(model, tokenizer, test_ds, max_new_tokens: int, sample_count: int) -> List[Dict[str, str]]:
    rows = []
    upto = min(sample_count, len(test_ds))
    for i in range(upto):
        ex = test_ds[i]
        pred = generate_sql(model, tokenizer, ex["prompt"], max_new_tokens=max_new_tokens)
        rows.append(
            {
                "question": ex["question"],
                "schema": ex["context"],
                "gold": ex["answer"],
                "prediction": pred,
            }
        )
    return rows


def save_rows(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "schema", "gold", "prediction"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_sql_dataset(DATASET_NAME)
    splits = build_or_load_splits(
        ds,
        SplitConfig(train_size=args.train_size, val_size=args.val_size, test_size=args.test_size, seed=args.seed),
        split_index_path=args.split_index_path,
    )
    formatted = format_for_training(splits, PromptConfig())
    test_ds = formatted["test"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype="auto", device_map="auto")
    base_rows = run_eval(base_model, tokenizer, test_ds, args.max_new_tokens, args.sample_count)
    base_metrics = compute_metrics(base_rows)

    results: Dict[str, Any] = {"baseline": base_metrics}
    save_rows(out_dir / "baseline_predictions.csv", base_rows)

    if args.adapter_path:
        adapted_base = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype="auto", device_map="auto")
        oft_model = PeftModel.from_pretrained(adapted_base, args.adapter_path)
        oft_rows = run_eval(oft_model, tokenizer, test_ds, args.max_new_tokens, args.sample_count)
        oft_metrics = compute_metrics(oft_rows)
        results["oft"] = oft_metrics
        save_rows(out_dir / "oft_predictions.csv", oft_rows)

        qual = []
        for b, o in zip(base_rows[:12], oft_rows[:12]):
            qual.append(
                {
                    "question": b["question"],
                    "schema": b["schema"],
                    "gold": b["gold"],
                    "baseline_prediction": b["prediction"],
                    "oft_prediction": o["prediction"],
                }
            )
        with open(out_dir / "qualitative_examples.json", "w", encoding="utf-8") as f:
            json.dump(qual, f, indent=2)

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
