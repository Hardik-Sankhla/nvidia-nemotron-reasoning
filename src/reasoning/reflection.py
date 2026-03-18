"""Reflection and critique-based improvement helpers."""
def reflection(model, prompt):
    initial = model.generate(prompt)

    critique_prompt = f"""
Question: {prompt}
Answer: {initial}

Critique the answer and improve it.
"""

    improved = model.generate(critique_prompt)

    return improved
