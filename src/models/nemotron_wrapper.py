"""High-level wrapper that selects a model backend for inference.

This wrapper currently supports a `local` backend that uses
`src.models.local_models.LocalHFModel` for real HF inference.
It is a thin compatibility layer so the rest of the repo can call
`.generate(prompt)` uniformly.
"""
from .local_models import LocalHFModel


class NemotronWrapper:
    def __init__(self, model_type="local", model_name=None):
        if model_type == "local":
            self.model = LocalHFModel(model_name=model_name) if model_name else LocalHFModel()
        else:
            raise NotImplementedError(f"model_type {model_type} not implemented")

    def generate(self, prompt, **kwargs):
        return self.model.generate(prompt, **kwargs)


def load_base_model(*args, **kwargs):
    return NemotronWrapper(*args, **kwargs)
