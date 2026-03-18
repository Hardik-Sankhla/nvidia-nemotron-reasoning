"""Chain-of-thought prompting utilities."""

def make_cot_prompt(example):
    return """Step-by-step reasoning:\n1. ...\n2. ...\n"""


def chain_of_thought(prompt):
    """Simple helper that appends a CoT instruction to a prompt."""
    return prompt + "\nLet's think step by step."
