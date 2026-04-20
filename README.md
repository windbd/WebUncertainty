# WebUncertainly
WebUncertainty is a modular Python framework implementing a dual-level uncertainty-driven approach for web task automation.

## Core Ideas

- **Task Uncertainty-Driven Adaptive Planning** (π_ana): Analyze planning uncertainty across three dimensions (environmental unfamiliarity, execution complexity, progress indicators). Switch between explicit forward decomposition (u_plan ≤ δ=0.4) and reactive single-step planning (u_plan > δ).

- **Action Uncertainty-Driven MCTS Reasoning** (π_reason): Use Confidence-induced Action Uncertainty (ConActU) to guide candidate generation (K=10), uncertainty-based pruning (EU > 0.65), and PUCT tree search (w_puct=5.0).

- **State Evaluation with Reward Modulation** (π_eval): Evaluate states using base feasibility score and estimated EU/AU. Apply 4×4 reward modulation matrix with 5 distinct rules for MCTS value estimation.
