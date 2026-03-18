"""Higher-level reasoning pipeline helpers."""

from .reflection import reflection


def advanced_reasoning(model, prompt):
    """Run CoT and then refine output with reflection."""
    cot_prompt = prompt + "\nLet's think step by step."
    response = model.generate(cot_prompt)
    refined = reflection(model, response)
    return refined
