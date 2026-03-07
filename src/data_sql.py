"""Utilities for schema-grounded Text-to-SQL data loading and prompt formatting."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from datasets import Dataset, DatasetDict, load_dataset

DATASET_NAME = "b-mc2/sql-create-context"

SYSTEM_PROMPT = (
    "You are a Text-to-SQL assistant. Given a question and SQL table schema, "
    "generate only the final SQL query. Do not include explanation."
)


@dataclass
class SplitConfig:
    train_size: int = 20000
    val_size: int = 2000
    test_size: int = 2000
    seed: int = 42


@dataclass
class PromptConfig:
    add_system: bool = True
    eos_token: str = ""


def _format_prompt(question: str, schema: str, answer_sql: Optional[str], cfg: PromptConfig) -> str:
    sections: List[str] = []
    if cfg.add_system:
        sections.append(f"### System\n{SYSTEM_PROMPT}")
    sections.append(f"### Question\n{question.strip()}")
    sections.append(f"### Schema\n{schema.strip()}")
    sections.append("### Response\n")

    if answer_sql is not None:
        return "\n\n".join(sections) + answer_sql.strip() + cfg.eos_token
    return "\n\n".join(sections)


def normalize_sql(sql: str) -> str:
    """Simple canonicalization for exact-match style comparison."""
    return " ".join(sql.strip().lower().rstrip(";").split())


def load_sql_dataset(dataset_name: str = DATASET_NAME) -> Dataset:
    ds = load_dataset(dataset_name, split="train")
    required_columns = {"question", "context", "answer"}
    missing = required_columns - set(ds.column_names)
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")
    return ds


def build_or_load_splits(
    dataset: Dataset,
    split_cfg: SplitConfig,
    split_index_path: Optional[str] = None,
) -> DatasetDict:
    n = len(dataset)
    needed = split_cfg.train_size + split_cfg.val_size + split_cfg.test_size
    if needed > n:
        raise ValueError(
            f"Requested split sizes ({needed}) exceed dataset size ({n}). "
            "Lower train/val/test sizes."
        )

    if split_index_path and Path(split_index_path).exists():
        with open(split_index_path, "r", encoding="utf-8") as f:
            idx_data = json.load(f)
    else:
        indices = list(range(n))
        rng = random.Random(split_cfg.seed)
        rng.shuffle(indices)
        idx_data = {
            "train": indices[: split_cfg.train_size],
            "validation": indices[
                split_cfg.train_size : split_cfg.train_size + split_cfg.val_size
            ],
            "test": indices[
                split_cfg.train_size + split_cfg.val_size : needed
            ],
            "meta": {
                "seed": split_cfg.seed,
                "train_size": split_cfg.train_size,
                "val_size": split_cfg.val_size,
                "test_size": split_cfg.test_size,
                "dataset_size": n,
            },
        }
        if split_index_path:
            Path(split_index_path).parent.mkdir(parents=True, exist_ok=True)
            with open(split_index_path, "w", encoding="utf-8") as f:
                json.dump(idx_data, f, indent=2)

    return DatasetDict(
        {
            "train": dataset.select(idx_data["train"]),
            "validation": dataset.select(idx_data["validation"]),
            "test": dataset.select(idx_data["test"]),
        }
    )


def with_formatted_text(dataset: Dataset, prompt_cfg: PromptConfig) -> Dataset:
    def _map(example: Dict[str, str]) -> Dict[str, str]:
        question = example["question"]
        schema = example["context"]
        answer = example.get("answer")
        return {
            "prompt": _format_prompt(question, schema, None, prompt_cfg),
            "text": _format_prompt(question, schema, answer, prompt_cfg),
            "target_sql": answer,
        }

    return dataset.map(_map)


def format_for_training(ds_dict: DatasetDict, prompt_cfg: PromptConfig) -> DatasetDict:
    return DatasetDict({k: with_formatted_text(v, prompt_cfg) for k, v in ds_dict.items()})


def preview_examples(ds: Dataset, n: int = 2) -> List[Dict[str, str]]:
    out = []
    for i in range(min(n, len(ds))):
        row = ds[i]
        out.append(
            {
                "question": row.get("question", ""),
                "context": row.get("context", "")[:300],
                "answer": row.get("answer", ""),
            }
        )
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test for SQL data pipeline")
    parser.add_argument("--train-size", type=int, default=500)
    parser.add_argument("--val-size", type=int, default=100)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-index-path", type=str, default="outputs/splits/split_indices.json")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dataset = load_sql_dataset()
    splits = build_or_load_splits(
        dataset,
        SplitConfig(
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            seed=args.seed,
        ),
        split_index_path=args.split_index_path,
    )
    formatted = format_for_training(splits, PromptConfig())
    print({k: len(v) for k, v in formatted.items()})
    for ex in preview_examples(formatted["train"], n=2):
        print(json.dumps(ex, indent=2)[:600])


if __name__ == "__main__":
    main()
