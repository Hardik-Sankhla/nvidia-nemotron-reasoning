# NVIDIA Nemotron 3 Reasoning Challenge — Novel Benchmark

## Overview

The NVIDIA Nemotron 3 Reasoning Challenge (hosted on Kaggle) is designed to advance high-quality, efficient AI reasoning by challenging participants to improve the accuracy of the NVIDIA Nemotron-3-Nano model on a novel logical reasoning benchmark developed by NVIDIA Research.

## What is the novel benchmark?

The novel benchmark is a specialized dataset of logical reasoning puzzles created to test a model's ability to identify and apply hidden transformation rules. Unlike standard benchmarks, this dataset emphasizes structured, hidden-rule discovery across multiple domains, including:

- Secretly altered rules: scenarios where constants, unit conversions, or encryption rules have been modified.
- Numerical systems: conversions between numbers expressed in different, unknown numeral systems.
- Logical puzzles: transformations applied to algebraic equations or bit-manipulation rules (for example, affecting 8-bit binary numbers).

The benchmark stresses the model's capacity to discover rules, generalize across instances, and produce reliably structured answers.

## Key components of the challenge

- **Base model:** NVIDIA-Nemotron-3-Nano-4B-BF16 (a compact model tuned for agentic tasks with a large context window).  
- **Goal:** Improve reasoning performance on the benchmark using techniques such as prompt engineering, data curation/filtering, synthetic data generation, reinforcement learning, or lightweight fine-tuning (e.g., LoRA).  
- **Evaluation:** Submissions are evaluated on a hidden test set of several hundred problems. Participants must submit a LoRA adapter compatible with Nemotron-3-Nano; the evaluation loads the base model with the submitted adapter and runs the official metric to determine accuracy.

## Submission requirements (summary)

- Submit a LoRA adapter of rank at most 32 packaged in `submission.zip` (include `adapter_config.json`).  
- Provide public reproducible documentation: a Kaggle notebook and write-up describing dataset preparation, prompting strategies, tuning/LoRA recipe, and evaluation steps (required for prize eligibility).

## Purpose and community goals

This challenge aims to create an open, reproducible environment for reasoning research where methods, datasets, and evaluation are shared so the community can compare, reproduce, and iterate on approaches. Clear documentation (notebooks + write-ups) is encouraged and required for awards/prize eligibility.

---
_Document created for the `nvidia-nemotron-reasoning` repo._
