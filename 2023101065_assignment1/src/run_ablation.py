"""
Runs all five configurations (C1-C5) back to back with identical
hyperparameters (except the one changed component per Table 1) and produces
a comparison table + bar plots in outputs/.

Usage:
    python -m src.run_ablation --data_path /path/to/dataset.csv --epochs 10
"""
import argparse
import json
import os
import subprocess
import sys

from .utils import plot_config_comparison

CONFIGS = ["C1", "C2", "C3", "C4", "C5"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--extra_args", default="", help="extra args forwarded to train.py, e.g. '--batch_size 16'")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for cfg in CONFIGS:
        cmd = [
            sys.executable, "-m", "src.train",
            "--config", cfg,
            "--data_path", args.data_path,
            "--epochs", str(args.epochs),
            "--output_dir", args.output_dir,
        ]
        if args.use_wandb:
            cmd.append("--use_wandb")
        if args.extra_args:
            cmd.extend(args.extra_args.split())
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    # aggregate
    all_metrics = {}
    for cfg in CONFIGS:
        path = os.path.join(args.output_dir, f"{cfg}_metrics.json")
        if os.path.exists(path):
            with open(path) as f:
                all_metrics[cfg] = json.load(f)

    with open(os.path.join(args.output_dir, "ablation_summary.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    for metric_name in ["bit_accuracy", "sequence_accuracy", "levenshtein", "train_time_sec", "peak_mem_mb"]:
        names = [c for c in CONFIGS if c in all_metrics]
        values = [all_metrics[c].get(metric_name, 0.0) for c in names]
        if values:
            plot_config_comparison(names, values, metric_name, os.path.join(args.output_dir, f"compare_{metric_name}.png"))

    print(json.dumps(all_metrics, indent=2))


if __name__ == "__main__":
    main()
