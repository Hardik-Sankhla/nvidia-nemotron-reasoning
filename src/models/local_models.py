"""Phase 2: local model utilities for training and evaluation.

This module contains helpers and thin wrappers around Hugging Face / vLLM
loading and lightweight experiments when local or cloud GPUs are available.
Use these for LoRA training orchestration and local evaluation before
producing the final adapter.
"""

def load_small_model(model_name, device='cpu'):
    """Load a small model for rapid iteration (not the Nemotron submission model).

    Returns a model handle compatible with the rest of the repo's inference API.
    """
    # placeholder: user will wire HF/vLLM load here
    return None

def prepare_training(model, tokenizer, config):
    """Prepare model and optimizer for LoRA / PEFT training.

    This is a convenience hook to keep training code consistent across experiments.
    """
    return {
        'model': model,
        'tokenizer': tokenizer,
        'config': config,
    }
