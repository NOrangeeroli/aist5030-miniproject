"""Plot training logs and baseline vs OFT metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--trainer-log", type=str, default="outputs/train/trainer_state.json")
    p.add_argument("--eval-metrics", type=str, default="outputs/eval/metrics.json")
    p.add_argument("--output-dir", type=str, default="outputs/plots")
    return p.parse_args()


def plot_training_curves(trainer_log_path: Path, out_dir: Path) -> None:
    if not trainer_log_path.exists():
        print(f"Skipping training curves: {trainer_log_path} not found")
        return

    with open(trainer_log_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    hist = state.get("log_history", [])
    if not hist:
        print("No log history found")
        return

    df = pd.DataFrame(hist)
    if "step" not in df.columns:
        return

    plt.figure(figsize=(8, 5))
    if "loss" in df.columns:
        sub = df[["step", "loss"]].dropna()
        plt.plot(sub["step"], sub["loss"], label="train loss")
    if "eval_loss" in df.columns:
        sub = df[["step", "eval_loss"]].dropna()
        plt.plot(sub["step"], sub["eval_loss"], label="val loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training/Validation Loss")
    plt.legend()
    plt.tight_layout()
    out = out_dir / "loss_curves.png"
    plt.savefig(out, dpi=150)
    plt.close()


def plot_eval_bars(metrics_path: Path, out_dir: Path) -> None:
    if not metrics_path.exists():
        print(f"Skipping eval bars: {metrics_path} not found")
        return

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    rows = []
    for model_name, vals in metrics.items():
        for metric_name, value in vals.items():
            rows.append({"model": model_name, "metric": metric_name, "value": value})
    if not rows:
        return

    df = pd.DataFrame(rows)
    pivot = df.pivot(index="metric", columns="model", values="value")
    ax = pivot.plot(kind="bar", figsize=(8, 5), rot=0)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Baseline vs OFT")
    plt.tight_layout()
    out = out_dir / "eval_metrics.png"
    plt.savefig(out, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_training_curves(Path(args.trainer_log), out_dir)
    plot_eval_bars(Path(args.eval_metrics), out_dir)


if __name__ == "__main__":
    main()
