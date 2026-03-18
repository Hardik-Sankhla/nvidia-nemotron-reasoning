NVIDIA Nemotron Reasoning — Experiments

This repository collects experiments, notebooks, and reproducible recipes for the "NVIDIA Nemotron Model Reasoning Challenge" (Kaggle). It contains pipelines, prompt recipes, and lightweight fine-tuning examples (LoRA) to evaluate reasoning performance on the provided benchmark.

Structure
- `docs/` — high-level docs and guides
- `notebooks/` — research notebooks and demos
- `experiments/` — experiment configs, logs, and adapters
- `src/` — evaluation and helper scripts
- `scripts/` — utility scripts for data prep and evaluation

See `docs/overview.md` for next steps.

---
Author: Hardik Sankhla

## Strategy & Phases

This repository follows a hybrid experimentation → training → optimization workflow recommended for the NVIDIA Nemotron Reasoning Challenge.

Phase 1 — Fast experimentation
- Use hosted APIs or small local models to iterate on prompts and reasoning strategies (CoT, ToT, self-consistency).
- Tools: API clients, small HF models, `src/models/api_models.py`.

Phase 2 — Local LoRA training
- Move to local/cloud GPUs to fine-tune LoRA adapters (rank ≤ 32) on curated/synthetic data.
- Tools: HF Transformers, PEFT (LoRA), TRL/Unsloth for faster experiments, `src/models/local_models.py`.

Phase 3 — Final submission
- Produce a LoRA adapter compatible with Nemotron-3-Nano and vLLM. Ensure the adapter includes `adapter_config.json` and exported weights.
- Tools: `src/models/nemotron_lora.py` contains helpers to export and validate adapters for vLLM.

Important: API-only workflows are useful for Phase 1 but cannot produce the final LoRA adapter required for competition submission.

## Repo layout update
The `src/models/` package now contains three phase-specific modules:
- `api_models.py` — quick prompt & API experimentation
- `local_models.py` — small-model training utilities and experiment orchestration
- `nemotron_lora.py` — production LoRA export and vLLM compatibility helpers (final submission)

