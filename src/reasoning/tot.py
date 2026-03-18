"""Tree-of-thought scaffolding (simple implementation)."""
def tree_of_thought(model, prompt, branches=3):
    """Generate multiple 'thought' branches and return the longest one (naive).

    Replace selection heuristic with scoring or critic in later improvements.
    """
    thoughts = []

    for _ in range(branches):
        thoughts.append(model.generate(prompt + "\nThink of another approach."))

    # naive selection for now: longest response
    return max(thoughts, key=len) if thoughts else None
