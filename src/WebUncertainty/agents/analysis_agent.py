"""Analysis Agent: estimates task uncertainty before each planning step.

Wraps :mod:`WebUncertainty.prompts.analysis`. The resulting
``task_uncertainty`` (``u_plan``) is what
:mod:`WebUncertainty.planning.adaptive_planner` compares against ``delta``
to choose the explicit or implicit planner for this step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from WebUncertainty import model as model_module
from WebUncertainty.prompts import analysis
from WebUncertainty.react_agent.browser import PageObservation

AskLLM = Callable[..., Any]


class AnalysisAgentError(RuntimeError):
    """Raised when the model response is missing objectives/uncertainty."""


@dataclass(frozen=True)
class AnalysisResult:
    """``task_uncertainty`` is ``u_plan`` in [0, 1] (paper Eq. in Section 3.1)."""

    remaining_objectives: str
    task_uncertainty: float
    rationale: str = ""


class AnalysisAgent:
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

    def analyze(
        self,
        *,
        task: str,
        observation: PageObservation,
        history: list[dict],
    ) -> AnalysisResult:
        user_prompt = analysis.build_user_prompt(
            task=task,
            observation_text=observation.render(),
            history=history,
        )
        response = self._ask_llm(
            user_prompt,
            system=analysis.SYSTEM_PROMPT,
            is_json=True,
            model=self.model,
            temperature=self.temperature,
            max_retries=self.max_retries,
            timeout=self.timeout,
        )
        if not isinstance(response, dict):
            raise AnalysisAgentError("model response was not a JSON object")

        remaining_objectives = str(response.get("remaining_objectives") or "").strip()
        if not remaining_objectives:
            raise AnalysisAgentError(
                "model response is missing a non-empty 'remaining_objectives'"
            )
        try:
            task_uncertainty = float(response.get("task_uncertainty"))
        except (TypeError, ValueError) as exc:
            raise AnalysisAgentError(
                "model response has a non-numeric 'task_uncertainty'"
            ) from exc
        task_uncertainty = max(0.0, min(1.0, task_uncertainty))

        return AnalysisResult(
            remaining_objectives=remaining_objectives,
            task_uncertainty=task_uncertainty,
            rationale=str(response.get("rationale") or "").strip(),
        )
