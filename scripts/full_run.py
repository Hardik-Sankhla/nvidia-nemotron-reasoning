"""End-to-end automation: run, evaluate, and update leaderboard."""

import argparse

from src.models.nemotron_wrapper import NemotronWrapper
from src.reasoning.self_consistency import self_consistency
from src.evaluation.metrics import compute_accuracy
from src.evaluation.leaderboard import update_leaderboard
from src.utils.config import load_yaml


def run(config):
    model_cfg = config.get("model", {})
    model = NemotronWrapper(model_type=model_cfg.get("type", "local"), model_name=model_cfg.get("name"))

    eval_cfg = config.get("eval", {})
    dataset = eval_cfg.get(
        "dataset",
        [{"prompt": "2+2", "answer": "4"}, {"prompt": "5+3", "answer": "8"}],
    )
    n = int(eval_cfg.get("self_consistency_n", 5))
    temperature = float(eval_cfg.get("temperature", 0.7))

    predictions = []
    ground_truths = []

    for item in dataset:
        prompt = item["prompt"] + "\nLet's think step by step."
        pred = self_consistency(model, prompt, n=n, temperature=temperature)
        predictions.append(pred)
        ground_truths.append(item["answer"])

    acc = compute_accuracy(predictions, ground_truths)
    print("Accuracy:", acc)

    exp_cfg = config.get("experiment", {})
    update_leaderboard(
        exp_id=exp_cfg.get("id", "exp_full_pipeline"),
        accuracy=acc,
        notes=exp_cfg.get("notes", "Full system run"),
        technique=exp_cfg.get("technique", "cot+self_consistency"),
        model=model_cfg.get("name", ""),
        lora_rank=config.get("lora", {}).get("rank", ""),
    )

    return acc


def main():
    parser = argparse.ArgumentParser(description="Run full pipeline with YAML config")
    parser.add_argument("--config", default="configs/full_run.yaml", help="Path to YAML config")
    args = parser.parse_args()

    config = load_yaml(args.config)
    run(config)


if __name__ == '__main__':
    main()
