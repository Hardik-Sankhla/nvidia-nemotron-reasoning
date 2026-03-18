"""Elite inference pipeline: CoT + self-consistency + optional synthetic hooks."""

from src.models.nemotron_wrapper import NemotronWrapper
from src.reasoning.self_consistency import self_consistency


def main():
    model = NemotronWrapper()

    prompt = "Solve: 15 + 27"

    enhanced_prompt = prompt + "\nLet's think step by step."

    final_answer = self_consistency(model, enhanced_prompt, n=5, temperature=0.7)

    print("Final Answer:", final_answer)


if __name__ == '__main__':
    main()
