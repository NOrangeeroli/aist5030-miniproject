"""Train OFT adapters for schema-grounded text-to-SQL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from peft import OFTConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from data_sql import DATASET_NAME, PromptConfig, SplitConfig, build_or_load_splits, format_for_training, load_sql_dataset


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def supports_bf16() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def tokenize_function(examples, tokenizer, max_length: int):
    return tokenizer(examples["text"], truncation=True, max_length=max_length)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_sql_dataset(cfg.get("dataset_name", DATASET_NAME))
    split_cfg = SplitConfig(
        train_size=cfg["data"]["train_size"],
        val_size=cfg["data"]["val_size"],
        test_size=cfg["data"]["test_size"],
        seed=cfg["data"].get("seed", 42),
    )
    split_path = cfg["data"].get("split_index_path")
    ds = build_or_load_splits(dataset, split_cfg=split_cfg, split_index_path=split_path)
    ds = format_for_training(ds, PromptConfig(add_system=True, eos_token=cfg.get("eos_token", "")))

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized = ds.map(
        lambda x: tokenize_function(x, tokenizer=tokenizer, max_length=cfg["max_seq_len"]),
        batched=True,
        remove_columns=ds["train"].column_names,
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        torch_dtype=torch.bfloat16 if supports_bf16() else torch.float16,
        device_map="auto",
    )

    oft_cfg = OFTConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg["oft"]["r"],
        oft_block_size=cfg["oft"]["block_size"],
        target_modules=cfg["oft"]["target_modules"],
        module_dropout=cfg["oft"].get("module_dropout", 0.0),
    )
    model = get_peft_model(model, oft_cfg)
    model.print_trainable_parameters()

    use_bf16 = supports_bf16()
    train_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=cfg["train"]["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["train"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["train"]["gradient_accumulation_steps"],
        learning_rate=cfg["train"]["learning_rate"],
        num_train_epochs=cfg["train"].get("num_train_epochs", 1),
        max_steps=cfg["train"]["max_steps"],
        warmup_ratio=cfg["train"].get("warmup_ratio", 0.03),
        lr_scheduler_type=cfg["train"].get("lr_scheduler_type", "cosine"),
        logging_steps=cfg["train"].get("logging_steps", 20),
        eval_strategy=cfg["train"].get("eval_strategy", "steps"),
        eval_steps=cfg["train"].get("eval_steps", 100),
        save_steps=cfg["train"].get("save_steps", 100),
        save_total_limit=cfg["train"].get("save_total_limit", 2),
        bf16=use_bf16,
        fp16=not use_bf16,
        report_to=cfg["train"].get("report_to", []),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg["train"].get("early_stopping_patience", 3))],
    )

    train_result = trainer.train()
    trainer.save_model(str(output_dir / "best_adapter"))
    tokenizer.save_pretrained(str(output_dir / "best_adapter"))

    metrics = train_result.metrics
    metrics["train_samples"] = len(tokenized["train"])
    metrics["eval_samples"] = len(tokenized["validation"])

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    with open(output_dir / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
