"""Planning Agent: explicit (one-shot decomposition) and implicit (reactive) modes.

Wraps :mod:`WebUncertainty.prompts.planning`. This agent only knows how to
run each mode — the decision of *which* mode to use for a given step (the
``task_uncertainty`` vs. ``delta`` comparison, paper Section 3.1) belongs to
the caller, :mod:`WebUncertainty.planning.adaptive_planner`, so that switch
threshold has exactly one place it can live.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from WebUncertainty import model as model_module
from WebUncertainty.prompts import planning
from WebUncertainty.react_agent.browser import PageObservation

AskLLM = Callable[..., Any]


class PlanningAgentError(RuntimeError):
    """Raised when the model response is missing a usable subgoal."""


@dataclass(frozen=True)
class PlanningResult:
    """``subgoal`` is what the reasoning stage resolves this step.

    ``remaining_subgoals`` is only populated in explicit mode — the rest of
    the one-shot decomposition, kept for logging/inspection; the adaptive
    planner re-runs analysis every step rather than blindly consuming this
    queue, so a later step can still swing back to implicit mode.
    """

    subgoal: str
    mode: Literal["explicit", "implicit"]
    remaining_subgoals: list[str] = field(default_factory=list)
    rationale: str = ""


class PlanningAgent:
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

    def plan_explicit(
        self,
        *,
        remaining_objectives: str,
        observation: PageObservation,
    ) -> PlanningResult:
        """One-shot decomposition; commits to the first subgoal (Section 3.1)."""
        response = self._call(
            planning.EXPLICIT_SYSTEM_PROMPT,
            remaining_objectives=remaining_objectives,
            observation=observation,
        )
        raw_subgoals = response.get("subgoals")
        if not isinstance(raw_subgoals, list):
            raise PlanningAgentError("model response is missing a 'subgoals' list")
        subgoals = [str(item).strip() for item in raw_subgoals if str(item).strip()]
        if not subgoals:
            raise PlanningAgentError("model response has no usable subgoal")
        return PlanningResult(
            subgoal=subgoals[0],
            mode="explicit",
            remaining_subgoals=subgoals[1:],
            rationale=str(response.get("rationale") or "").strip(),
        )

    def plan_implicit(
        self,
        *,
        remaining_objectives: str,
        observation: PageObservation,
    ) -> PlanningResult:
        """Proposes only the immediate next subgoal (Section 3.1)."""
        response = self._call(
            planning.IMPLICIT_SYSTEM_PROMPT,
            remaining_objectives=remaining_objectives,
            observation=observation,
        )
        subgoal = str(response.get("subgoal") or "").strip()
        if not subgoal:
            raise PlanningAgentError("model response is missing a non-empty 'subgoal'")
        return PlanningResult(
            subgoal=subgoal,
            mode="implicit",
            rationale=str(response.get("rationale") or "").strip(),
        )

    def _call(
        self,
        system_prompt: str,
        *,
        remaining_objectives: str,
        observation: PageObservation,
    ) -> dict:
        user_prompt = planning.build_user_prompt(
            remaining_objectives=remaining_objectives,
            observation_text=observation.render(),
        )
        response = self._ask_llm(
            user_prompt,
            system=system_prompt,
            is_json=True,
            model=self.model,
            temperature=self.temperature,
            max_retries=self.max_retries,
            timeout=self.timeout,
        )
        if not isinstance(response, dict):
            raise PlanningAgentError("model response was not a JSON object")
        return response
