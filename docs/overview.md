# Overview

This repo supports experiments for the NVIDIA Nemotron Model Reasoning Challenge. Use `notebooks/` for interactive exploration and `experiments/` for reproducible runs. The `src/` folder contains evaluation harnesses and metric wrappers.

Workflow
- Prepare data in `data/` or mount Kaggle dataset locally.
- Build prompt templates in `notebooks/`.
- Train or produce LoRA adapters in `experiments/`.
- Evaluate with `src/eval.py` using vLLM and the provided metric.

Phased approach (recommended)

- Phase 1 — Fast experimentation: prototype prompts, CoT/ToT strategies, and small-model evaluation. Use `src/models/api_models.py` for API-driven experiments and `src/models/local_models.py` for small local models.
- Phase 2 — LoRA fine-tuning: run LoRA training (rank ≤ 32) on Nemotron or compatible checkpoints. Place experiments under `experiments/exp_00X` and record configs/results.
- Phase 3 — Export & validate: use `src/models/nemotron_lora.py` to export adapter artifacts and verify vLLM compatibility before zipping submission.

Notes
- The final submission MUST be a LoRA adapter compatible with Nemotron-3-Nano and vLLM. API-only methods cannot produce a valid submission.

