"""Prompt optimization loop utilities."""

from src.evaluation.metrics import extract_answer


def optimize_prompt(base_prompt):
    return [
        base_prompt + "\nLet's think step by step.",
        base_prompt + "\nBreak it into smaller steps.",
        base_prompt + "\nSolve carefully and verify the answer.",
    ]


def find_best_prompt(model, prompt, ground_truth):
    candidates = optimize_prompt(prompt)

    best_prompt = None
    best_score = -1

    for p in candidates:
        response = model.generate(p)
        pred = extract_answer(response)

        score = int(str(pred).strip() == str(ground_truth).strip())

        if score > best_score:
            best_score = score
            best_prompt = p

    return best_prompt
