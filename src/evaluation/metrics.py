"""Metric wrappers for the NVIDIA Nemotron benchmark."""

import re


def extract_boxed_answer(text):
    """Extract answer in LaTeX \boxed{...} format when present."""
    if not text:
        return None
    matches = re.findall(r"\\boxed\{([^}]*)\}", text)
    if not matches:
        return None
    return matches[-1].strip()

def score_prediction(prediction, reference):
    # placeholder exact-match or numeric tolerance
    return prediction == reference


def extract_answer(text):
    """Extract the model's final answer using boxed extraction or fallback.

    This is a lightweight helper used by the pipeline script.
    """
    boxed = extract_boxed_answer(text)
    if boxed:
        return boxed
    # Fallback: last line or last token
    if not text:
        return None
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    return lines[-1] if lines else None


def compute_accuracy(predictions, ground_truths):
    """Compute exact-match accuracy."""
    if not ground_truths:
        return 0.0
    correct = 0
    for pred, truth in zip(predictions, ground_truths):
        if str(pred).strip() == str(truth).strip():
            correct += 1
    return correct / len(ground_truths)
