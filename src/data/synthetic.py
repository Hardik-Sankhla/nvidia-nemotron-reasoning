"""Synthetic data generation helpers for reasoning supervision."""

def generate_synthetic_sample(model, prompt):
    """Generate a synthetic reasoning trace for a single prompt.

    If `model` is None, a fallback template is used to keep pipeline scripts
    runnable for dry runs.
    """
    synthetic_prompt = (
        "Solve the problem step by step and explain reasoning:\n\n"
        f"{prompt}"
    )

    if model is None:
        response = "Step 1: Analyze prompt.\nStep 2: Compute result.\nFinal answer: [placeholder]"
    else:
        response = model.generate(synthetic_prompt)

    return {
        "prompt": prompt,
        "answer": "",
        "synthetic_reasoning": response,
    }


def generate_dataset(model, prompts, variations=1):
    """Batch synthetic generator with optional multiple variations per prompt."""
    data = []
    for p in prompts:
        for _ in range(max(1, variations)):
            data.append(generate_synthetic_sample(model, p))
    return data
