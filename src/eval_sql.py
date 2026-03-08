"""Evaluate baseline vs merged-OFT model for text-to-SQL generation."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import shutil
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    p.add_argument(
        "--merged-model-dir",
        type=str,
        default=None,
        help="Directory to store merged full-weights checkpoint (defaults to <output-dir>/merged_oft_model).",
    )
    p.add_argument(
        "--force-remerge",
        action="store_true",
        help="Delete existing merged checkpoint before recreating it.",
    )
    return p.parse_args()


def sql_parses(query: str) -> bool:
    try:
        parsed = sqlparse.parse(query)
        return len(parsed) > 0 and any(tok.value.strip() for tok in parsed[0].tokens)
    except Exception:
        return False


def compute_metrics(rows: List[Dict[str, str]]) -> Dict[str, float]:
    n = len(rows)
    if n == 0:
        return {"exact_match": 0.0, "parse_success": 0.0}

    em = sum(1 for r in rows if normalize_sql(r["prediction"]) == normalize_sql(r["gold"])) / n
    parse_ok = sum(1 for r in rows if sql_parses(r["prediction"])) / n
    return {"exact_match": em, "parse_success": parse_ok}


def run_eval_vllm(llm, test_ds, max_new_tokens: int, sample_count: int) -> List[Dict[str, str]]:
    upto = min(sample_count, len(test_ds))
    subset = [test_ds[i] for i in range(upto)]
    prompts = [row["prompt"] for row in subset]

    from vllm import SamplingParams

    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    outputs = llm.generate(prompts, sampling_params)

    rows = []
    for ex, out in zip(subset, outputs):
        pred = out.outputs[0].text.strip() if out.outputs else ""
        rows.append(
            {
                "question": ex["question"],
                "schema": ex["context"],
                "gold": ex["answer"],
                "prediction": pred,
            }
        )
    return rows


def run_eval_transformers(
    model,
    tokenizer,
    test_ds,
    max_new_tokens: int,
    sample_count: int,
) -> List[Dict[str, str]]:
    upto = min(sample_count, len(test_ds))
    subset = [test_ds[i] for i in range(upto)]
    rows = []
    for ex in subset:
        encoded = tokenizer(ex["prompt"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        pred = tokenizer.decode(out[0][encoded["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        rows.append(
            {
                "question": ex["question"],
                "schema": ex["context"],
                "gold": ex["answer"],
                "prediction": pred,
            }
        )
    return rows


def load_adapter_peft_type(adapter_path: str) -> Optional[str]:
    config_path = Path(adapter_path) / "adapter_config.json"
    if not config_path.exists():
        return None
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f).get("peft_type")


def try_build_vllm(model_name: str):
    try:
        from vllm import LLM

        return LLM(model=model_name, dtype="auto", trust_remote_code=True, enable_lora=False)
    except Exception as exc:
        warnings.warn(f"vLLM is unavailable in this environment ({exc}); falling back to Transformers generation.")
        return None


def merge_oft_adapter_into_base(model_name: str, adapter_path: str, merged_model_dir: Path, force_remerge: bool) -> Path:
    if merged_model_dir.exists() and force_remerge:
        shutil.rmtree(merged_model_dir)

    if merged_model_dir.exists() and (merged_model_dir / "config.json").exists():
        return merged_model_dir

    merged_model_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=(
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else (torch.float16 if torch.cuda.is_available() else torch.float32)
        ),
        device_map="auto",
        trust_remote_code=True,
    )
    adapted_model = PeftModel.from_pretrained(base_model, adapter_path)
    merged_model = adapted_model.merge_and_unload()

    merged_model.save_pretrained(merged_model_dir)
    tokenizer.save_pretrained(merged_model_dir)
    return merged_model_dir


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


    llm = try_build_vllm(args.model_name)

    if llm is not None:
        base_rows = run_eval_vllm(llm, test_ds, args.max_new_tokens, args.sample_count)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            dtype=(
                torch.bfloat16
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                else (torch.float16 if torch.cuda.is_available() else torch.float32)
            ),
            device_map="auto",
            trust_remote_code=True,
        )
        base_rows = run_eval_transformers(base_model, tokenizer, test_ds, args.max_new_tokens, args.sample_count)
    base_metrics = compute_metrics(base_rows)

    results: Dict[str, Any] = {"baseline": base_metrics}
    save_rows(out_dir / "baseline_predictions.csv", base_rows)
    del llm
    
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    
    merged_model_path: Optional[Path] = None
    if args.adapter_path:
        adapter_peft_type = load_adapter_peft_type(args.adapter_path)
        if adapter_peft_type != "OFT":
            warnings.warn(f"Expected OFT adapter but found {adapter_peft_type}; attempting PEFT merge anyway.")

        merged_dir = Path(args.merged_model_dir) if args.merged_model_dir else out_dir / "merged_oft_model"
        merged_model_path = merge_oft_adapter_into_base(
            model_name=args.model_name,
            adapter_path=args.adapter_path,
            merged_model_dir=merged_dir,
            force_remerge=args.force_remerge,
        )

    if merged_model_path is not None:
        merged_llm = try_build_vllm(str(merged_model_path))
        if merged_llm is not None:
            oft_rows = run_eval_vllm(merged_llm, test_ds, args.max_new_tokens, args.sample_count)
        else:
            tokenizer = AutoTokenizer.from_pretrained(str(merged_model_path), use_fast=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            merged_model = AutoModelForCausalLM.from_pretrained(
                str(merged_model_path),
                dtype=(
                    torch.bfloat16
                    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                    else (torch.float16 if torch.cuda.is_available() else torch.float32)
                ),
                device_map="auto",
                trust_remote_code=True,
            )
            oft_rows = run_eval_transformers(merged_model, tokenizer, test_ds, args.max_new_tokens, args.sample_count)

        oft_metrics = compute_metrics(oft_rows)
        results["oft"] = oft_metrics
        results["merged_model_path"] = str(merged_model_path)
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
