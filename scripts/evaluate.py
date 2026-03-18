"""Evaluate pipeline driven by YAML config."""

import argparse

from src.models.nemotron_wrapper import NemotronWrapper
from src.reasoning.self_consistency import self_consistency
from src.evaluation.metrics import compute_accuracy
from src.utils.config import load_yaml


def run(config):
    model_cfg = config.get("model", {})
    eval_cfg = config.get("eval", {})

    model = NemotronWrapper(model_type=model_cfg.get("type", "local"), model_name=model_cfg.get("name"))

    dataset = eval_cfg.get("dataset", [])
    n = int(eval_cfg.get("self_consistency_n", 5))
    temperature = float(eval_cfg.get("temperature", 0.7))

    predictions = []
    truths = []
    for item in dataset:
        prompt = item["prompt"] + "\nLet's think step by step."
        pred = self_consistency(model, prompt, n=n, temperature=temperature)
        predictions.append(pred)
        truths.append(item["answer"])

    accuracy = compute_accuracy(predictions, truths)
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Evaluate with YAML config")
    parser.add_argument("--config", default="configs/eval.yaml", help="Path to YAML config")
    args = parser.parse_args()

    config = load_yaml(args.config)
    run(config)


if __name__ == '__main__':
    main()
