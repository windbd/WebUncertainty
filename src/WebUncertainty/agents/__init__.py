"""The four LLM-calling roles from the WebUncertainty paper.

Each agent wraps one prompt template (:mod:`WebUncertainty.prompts`) plus
response validation around :func:`WebUncertainty.model.ask_llm`. Agents may
call the model; only :mod:`WebUncertainty.planning` and the browser-facing
``mcts.sampler`` ever mutate a real page.

- ``analysis_agent`` / ``planning_agent`` — Task Uncertainty-Driven Adaptive
  Planning (paper Section 3.1).
- ``reasoning_agent`` / ``evaluation_agent`` — Action Uncertainty-Driven MCTS
  Reasoning (paper Section 3.2).
"""
