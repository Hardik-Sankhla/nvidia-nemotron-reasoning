# Overview

This repo supports experiments for the NVIDIA Nemotron Model Reasoning Challenge. Use `notebooks/` for interactive exploration and `experiments/` for reproducible runs. The `src/` folder contains evaluation harnesses and metric wrappers.

Workflow
- Prepare data in `data/` or mount Kaggle dataset locally.
- Build prompt templates in `notebooks/`.
- Train or produce LoRA adapters in `experiments/`.
- Evaluate with `src/eval.py` using vLLM and the provided metric.
