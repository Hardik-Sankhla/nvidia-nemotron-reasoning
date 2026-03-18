"""Run the full multi-step pipeline: CoT -> model -> refine -> evaluate."""
from src.models.nemotron_wrapper import NemotronWrapper
from src.reasoning.cot import chain_of_thought
from src.evaluation.metrics import extract_answer


def main():
    model = NemotronWrapper()

    prompt = "Solve: 12 * 8"

    cot_prompt = chain_of_thought(prompt)

    response = model.generate(cot_prompt)

    answer = extract_answer(response)

    print("Final Answer:", answer)


if __name__ == '__main__':
    main()
