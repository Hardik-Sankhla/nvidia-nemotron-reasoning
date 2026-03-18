"""Metric wrappers for the NVIDIA Nemotron benchmark."""
def extract_boxed_answer(text):
    # Placeholder: extract content inside \boxed{} or fallback heuristics
    return None

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
