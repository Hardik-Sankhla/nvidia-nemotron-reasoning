"""Phase 2: local model utilities for training and evaluation.

This module provides a lightweight Hugging Face wrapper for rapid iteration
and local inference. It is intended for Phase 1/2 experimentation (fast
inference and preparing for LoRA training). Replace the default model
name with any HF-compatible causal LM you have access to.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class LocalHFModel:
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.1"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )

    def generate(self, prompt, max_new_tokens=256, temperature=0.0):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


def load_small_model(model_name="mistralai/Mistral-7B-Instruct-v0.1"):
    return LocalHFModel(model_name=model_name)


def prepare_training(model, tokenizer, config):
    """Return a minimal training bundle.

    This hook is intentionally small; training orchestration should live in
    `scripts/train_lora.py` which uses PEFT/Trainer.
    """
    return {"model": model, "tokenizer": tokenizer, "config": config}
