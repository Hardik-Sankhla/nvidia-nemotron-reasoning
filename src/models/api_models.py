"""Phase 1: API-based / lightweight model wrappers for fast experimentation.

These helpers are intended for prompt engineering and fast iteration using
hosted APIs or small local models. They are NOT suitable for producing final
LoRA adapters required by the competition, but they let you iterate quickly.
"""

def generate_with_api(prompt, api_client, max_tokens=512, temperature=0.0):
    """Run a prompt through a hosted API client (HF, OpenAI, etc.).

    api_client should implement a simple `generate(prompt, **kwargs)` method.
    """
    return api_client.generate(prompt, max_tokens=max_tokens, temperature=temperature)
