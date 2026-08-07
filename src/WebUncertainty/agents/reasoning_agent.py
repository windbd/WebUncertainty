"""Reasoning Agent: proposes candidate actions with confidence scores.

Wraps :mod:`WebUncertainty.prompts.reasoning` around
:func:`WebUncertainty.model.ask_llm` and turns the response into
:class:`WebUncertainty.mcts.tree.ActionCandidate` objects — grounded (and
safety-checked) via
:meth:`WebUncertainty.react_agent.agent.AgentAction.from_payload` for the
real root page, or free-text for a hypothetical deeper node.

A malformed individual candidate (bad element id, missing confidence, ...)
is dropped rather than failing the whole expansion — the Reasoning Agent
asked for up to ``k_candidates`` alternatives precisely so search still has
something to work with if one of them doesn't pan out. Only an *empty*
result after dropping is treated as an error.
"""

from __future__ import annotations

from typing import Any, Callable

from WebUncertainty import model as model_module
from WebUncertainty.mcts.tree import ActionCandidate
from WebUncertainty.mcts.uncertainty import normalize_confidences
from WebUncertainty.prompts import reasoning
from WebUncertainty.react_agent.agent import ActionError, AgentAction
from WebUncertainty.react_agent.browser import PageObservation

AskLLM = Callable[..., Any]


class ReasoningAgentError(RuntimeError):
    """Raised when the Reasoning Agent returns no usable candidate at all."""


class ReasoningAgent:
    """Calls the model for candidate actions and validates the response."""

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

    def propose_grounded(
        self,
        *,
        subgoal: str,
        observation: PageObservation,
        k_candidates: int,
    ) -> list[ActionCandidate]:
        """Propose candidates grounded against a real page observation."""
        user_prompt = reasoning.build_grounded_user_prompt(
            subgoal=subgoal,
            observation_text=observation.render(),
            k_candidates=k_candidates,
        )
        payload = self._call(reasoning.SYSTEM_PROMPT, user_prompt)
        raw_candidates = _extract_candidate_list(payload)[:k_candidates]

        confidences: list[float] = []
        actions: list[AgentAction] = []
        for item in raw_candidates:
            confidence = _extract_confidence(item)
            if confidence is None:
                continue
            try:
                action = AgentAction.from_payload(item, observation)
            except ActionError:
                continue
            confidences.append(confidence)
            actions.append(action)

        if not actions:
            raise ReasoningAgentError(
                "no candidate in the model response grounded to a valid action"
            )
        priors = normalize_confidences(confidences)
        return [
            ActionCandidate(
                confidence=confidence,
                prior=prior,
                reason=action.reason,
                grounded_action=action,
            )
            for confidence, prior, action in zip(confidences, priors, actions)
        ]

    def propose_hypothetical(
        self,
        *,
        subgoal: str,
        predicted_observation: str,
        k_candidates: int,
    ) -> list[ActionCandidate]:
        """Propose free-text follow-up candidates for a non-real node."""
        user_prompt = reasoning.build_hypothetical_user_prompt(
            subgoal=subgoal,
            predicted_observation=predicted_observation,
            k_candidates=k_candidates,
        )
        payload = self._call(reasoning.HYPOTHETICAL_SYSTEM_PROMPT, user_prompt)
        raw_candidates = _extract_candidate_list(payload)[:k_candidates]

        confidences: list[float] = []
        texts: list[str] = []
        for item in raw_candidates:
            confidence = _extract_confidence(item)
            text = str(item.get("action_text") or "").strip() if isinstance(item, dict) else ""
            if confidence is None or not text:
                continue
            confidences.append(confidence)
            texts.append(text)

        if not texts:
            raise ReasoningAgentError(
                "no usable hypothetical candidate in the model response"
            )
        priors = normalize_confidences(confidences)
        return [
            ActionCandidate(confidence=confidence, prior=prior, action_text=text)
            for confidence, prior, text in zip(confidences, priors, texts)
        ]

    def _call(self, system_prompt: str, user_prompt: str) -> dict:
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
            raise ReasoningAgentError("model response was not a JSON object")
        return response


def _extract_candidate_list(payload: dict) -> list[dict]:
    raw = payload.get("candidates")
    if not isinstance(raw, list) or not raw:
        raise ReasoningAgentError("model response is missing a non-empty 'candidates' list")
    return [item for item in raw if isinstance(item, dict)]


def _extract_confidence(item: dict) -> float | None:
    if not isinstance(item, dict):
        return None
    try:
        confidence = float(item.get("confidence"))
    except (TypeError, ValueError):
        return None
    if not (0.0 <= confidence <= 1.0):
        return None
    return confidence
