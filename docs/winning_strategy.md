# Top 1% Winning Strategy

## Weekly execution plan

Week 1
- Establish baseline runs and leaderboard logging.
- Add CoT and self-consistency on a fixed eval slice.
- Freeze evaluation protocol to prevent metric drift.

Week 2
- Run prompt optimization sweeps.
- Introduce data cleaning and synthetic data generation.
- Compare synthetic variants by quality filters before scaling.

Week 3
- Start LoRA runs on clean + synthetic blends.
- Track rank, target modules, and training steps in leaderboard notes.
- Keep at least one daily reproducible run.

Week 4+
- Tighten selection loop: choose top prompt + top data mix + top LoRA config.
- Package and validate adapters for vLLM repeatedly.
- Prepare final notebook write-up with clear ablations and visuals.

## High-impact ablations

- CoT vs no-CoT
- Self-consistency n=1/3/5/10
- Synthetic ratio (0%, 20%, 40%, 60%)
- LoRA rank 8/16/32
- Prompt templates by domain

## Operational discipline

- Never merge untracked experiments into final claims.
- Log every run into `reports/leaderboard.csv`.
- Keep one stable config as anchor while exploring one variable at a time.
