"""End-to-end automation: run, evaluate, and update leaderboard."""

from src.models.nemotron_wrapper import NemotronWrapper
from src.reasoning.self_consistency import self_consistency
from src.evaluation.metrics import compute_accuracy
from src.evaluation.leaderboard import update_leaderboard


def main():
    model = NemotronWrapper()

    dataset = [
        {"prompt": "2+2", "answer": "4"},
        {"prompt": "5+3", "answer": "8"},
    ]

    predictions = []
    ground_truths = []

    for item in dataset:
        prompt = item["prompt"] + "\nLet's think step by step."

        pred = self_consistency(model, prompt, n=5)

        predictions.append(pred)
        ground_truths.append(item["answer"])

    acc = compute_accuracy(predictions, ground_truths)

    print("Accuracy:", acc)

    update_leaderboard("exp_full_pipeline", acc, "Full system run")


if __name__ == '__main__':
    main()
