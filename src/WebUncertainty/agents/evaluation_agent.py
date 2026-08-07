"""Evaluation Agent: scores one candidate action without executing it.

Wraps :mod:`WebUncertainty.prompts.evaluation`. Produces the base
feasibility score ``S_base`` the paper's Simulation phase either accepts
outright (``S_base >= tau``) or hands to
:func:`WebUncertainty.mcts.uncertainty.modulate_reward`, plus a predicted
description of the resulting page that deeper (hypothetical) search nodes
condition on. Never touches a real browser — see the ``mcts`` package
docstring for why that is the design, not an oversight.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from WebUncertainty import model as model_module
from WebUncertainty.mcts.tree import ActionCandidate
from WebUncertainty.prompts import evaluation

AskLLM = Callable[..., Any]

_SCORE_MIN, _SCORE_MAX = 0.0, 10.0


class EvaluationAgentError(RuntimeError):
    """Raised when the model response is missing a usable score/prediction."""


@dataclass(frozen=True)
class EvaluationResult:
    """``score`` is S_base on the paper's 0-10 scale."""

    score: float
    predicted_observation: str
    rationale: str = ""


class EvaluationAgent:
    """Calls the model to score one candidate action against a state."""

    def __init__(
        self,
        *,
        ask_llm: AskLLM = model_module.ask_llm,
        model: str | None = None,
        temperature: float = 0.3,
        max_retries: int = 3,
        timeout: float | None = None,
    ) -> None:
        self._ask_llm = ask_llm
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout

    def evaluate(
        self,
        *,
        subgoal: str,
        state_text: str,
        candidate: ActionCandidate,
    ) -> EvaluationResult:
        user_prompt = evaluation.build_user_prompt(
            subgoal=subgoal,
            state_text=state_text,
            action_description=candidate.describe(),
        )
        response = self._ask_llm(
            user_prompt,
            system=evaluation.SYSTEM_PROMPT,
            is_json=True,
            model=self.model,
            temperature=self.temperature,
            max_retries=self.max_retries,
            timeout=self.timeout,
        )
        if not isinstance(response, dict):
            raise EvaluationAgentError("model response was not a JSON object")

        try:
            score = float(response.get("score"))
        except (TypeError, ValueError) as exc:
            raise EvaluationAgentError("model response has a non-numeric 'score'") from exc
        score = max(_SCORE_MIN, min(_SCORE_MAX, score))

        predicted_observation = str(response.get("predicted_observation") or "").strip()
        if not predicted_observation:
            raise EvaluationAgentError(
                "model response is missing a non-empty 'predicted_observation'"
            )
        rationale = str(response.get("rationale") or "").strip()
        return EvaluationResult(
            score=score,
            predicted_observation=predicted_observation,
            rationale=rationale,
        )
