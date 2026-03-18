"""Evaluation harness (skeleton) for Nemotron LoRA adapters.

This script demonstrates how to load a LoRA adapter and run evaluation.
"""

def evaluate(adapter_path, cases):
    """Placeholder evaluation function.

    adapter_path: path to LoRA adapter
    cases: list of (prompt, expected)
    """
    # TODO: integrate vLLM inference and the NVIDIA Nemotron metric
    results = []
    for prompt, expected in cases:
        # run model, extract boxed answer
        results.append((prompt, None, False))
    return results

if __name__ == '__main__':
    print('Run evaluation with evaluate(adapter_path, cases)')
