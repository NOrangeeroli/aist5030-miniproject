"""Merge a trained OFT adapter into base model weights and save a full checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", type=str, required=True, help="Base model path or HF model id")
    p.add_argument("--adapter-path", type=str, required=True, help="Path to trained OFT adapter")
    p.add_argument("--output-dir", type=str, required=True, help="Directory to save merged full model checkpoint")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path, use_fast=True)
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
    adapted_model = PeftModel.from_pretrained(base_model, args.adapter_path)
    merged_model = adapted_model.merge_and_unload()

    merged_model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Merged checkpoint saved to: {out_dir}")


if __name__ == "__main__":
    main()
