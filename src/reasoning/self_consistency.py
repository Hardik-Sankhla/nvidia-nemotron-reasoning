"""Self-consistency evaluation helpers."""

from collections import Counter
from src.evaluation.metrics import extract_answer


def self_consistency(model, prompt, n=5, temperature=0.7):
    """Sample multiple responses and return the most common extracted answer."""
    answers = []

    for _ in range(max(1, n)):
        response = model.generate(prompt, temperature=temperature)
        answer = extract_answer(response)
        if answer is not None:
            answers.append(answer)

    if not answers:
        return None

    return Counter(answers).most_common(1)[0][0]


def aggregate_responses(responses):
    """Backwards-compatible helper used by older experiments."""
    return responses[0] if responses else None
