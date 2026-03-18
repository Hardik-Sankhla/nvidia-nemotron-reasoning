"""Unified YAML-driven CLI for train/eval/full-run modes."""

import argparse

from scripts.train_lora import run as train_run
from scripts.evaluate import run as eval_run
from scripts.full_run import run as full_run
from src.utils.config import load_yaml


def main():
    parser = argparse.ArgumentParser(description="Unified pipeline runner")
    parser.add_argument("--mode", choices=["train", "eval", "full"], required=True)
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    if args.mode == "train":
        train_run(cfg)
    elif args.mode == "eval":
        eval_run(cfg)
    else:
        full_run(cfg)


if __name__ == "__main__":
    main()
