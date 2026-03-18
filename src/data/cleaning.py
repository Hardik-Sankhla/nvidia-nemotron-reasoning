"""Dataset cleaning utilities for prompt/answer data."""

def clean_dataset(dataset):
    """Clean raw records and return a list of normalized rows.

    Expected row keys: prompt, answer (answer can be empty for synthetic rows).
    """
    cleaned = []

    for item in dataset:
        prompt = item.get("prompt", "")
        answer = item.get("answer", "")

        if not prompt:
            continue

        if len(prompt) > 2000:
            continue

        cleaned.append({
            "prompt": prompt.strip(),
            "answer": str(answer).strip(),
            "synthetic_reasoning": item.get("synthetic_reasoning", ""),
        })

    return cleaned
