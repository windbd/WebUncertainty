"""A minimal observe-reason-act browser loop."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Callable

from WebUncertainty.react_agent.browser import BrowserSession, PageObservation
from WebUncertainty.react_agent.prompts import SYSTEM_PROMPT


SUPPORTED_ACTIONS = frozenset(
    {"click", "type", "press", "select", "scroll", "goto", "back", "wait", "finish"}
)
_ELEMENT_ACTIONS = frozenset({"click", "type", "press", "select"})
_VALUE_ACTIONS = frozenset({"type", "press", "select", "goto", "finish"})
_ALLOWED_KEYS = {
    "enter": "Enter",
    "escape": "Escape",
    "tab": "Tab",
    "arrowdown": "ArrowDown",
    "arrowup": "ArrowUp",
    "pagedown": "PageDown",
    "pageup": "PageUp",
    "space": "Space",
}


class ActionError(ValueError):
    """Raised when a model action is malformed or ungrounded."""


@dataclass(frozen=True)
class AgentAction:
    """One validated action chosen by the model."""

    action: str
    target: int | None = None
    value: str = ""
    reason: str = ""

    @classmethod
    def from_payload(
        cls,
        payload: dict,
        observation: PageObservation,
    ) -> AgentAction:
        if not isinstance(payload, dict):
            raise ActionError("model decision must be a JSON object")
        action = str(payload.get("action") or "").strip().lower()
        if action not in SUPPORTED_ACTIONS:
            raise ActionError(f"unsupported action: {action!r}")

        reason = str(payload.get("reason") or "").strip()[:500]
        value = str(payload.get("value") or "").strip()
        if len(value) > 4_000:
            raise ActionError("action value is too long")

        target = None
        if action in _ELEMENT_ACTIONS:
            try:
                target = int(payload.get("target"))
            except (TypeError, ValueError) as exc:
                raise ActionError(f"{action} requires an integer target") from exc
            if target not in observation.element_ids:
                raise ActionError(
                    f"element [{target}] is not in the current page snapshot"
                )

        if action in _VALUE_ACTIONS and not value:
            raise ActionError(f"{action} requires a non-empty value")
        if action == "scroll" and value.lower() not in {"up", "down"}:
            raise ActionError("scroll value must be 'up' or 'down'")
        if action == "scroll":
            value = value.lower()
        if action == "press":
            normalized_key = value.replace(" ", "").lower()
            if normalized_key not in _ALLOWED_KEYS:
                raise ActionError(f"unsupported key: {value!r}")
            value = _ALLOWED_KEYS[normalized_key]
        if action == "goto" and value not in observation.grounded_urls:
            raise ActionError("goto URL is not present in the current page snapshot")
        if action == "type":
            element = observation.element(target)
            if element.tag not in {"input", "textarea"} and element.role != "textbox":
                raise ActionError(f"element [{target}] is not a textbox")
        if action == "select" and observation.element(target).tag != "select":
            raise ActionError(f"element [{target}] is not a select")

        return cls(action=action, target=target, value=value, reason=reason)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RunResult:
    """Serializable result of one agent run."""

    task: str
    start_url: str
    completed: bool = False
    answer: str = ""
    final_url: str = ""
    error: str = ""
    steps: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class ReActAgent:
    """Observe the real page, choose one action, execute it, and repeat."""

    def __init__(
        self,
        *,
        task: str,
        start_url: str,
        browser: BrowserSession,
        model,
        max_steps: int = 12,
        max_text_chars: int = 8_000,
        max_elements: int = 80,
        max_consecutive_failures: int = 3,
        on_observation: Callable[[int, PageObservation], None] | None = None,
        on_step: Callable[[dict], None] | None = None,
    ) -> None:
        if not task.strip():
            raise ValueError("task must not be empty")
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self.task = task.strip()
        self.start_url = start_url.strip()
        self.browser = browser
        self.model = model
        self.max_steps = max_steps
        self.max_text_chars = max(1, max_text_chars)
        self.max_elements = max(1, max_elements)
        self.max_consecutive_failures = max(1, max_consecutive_failures)
        self.on_observation = on_observation
        self.on_step = on_step

    def run(self) -> RunResult:
        result = RunResult(task=self.task, start_url=self.start_url)
        history: list[dict] = []
        consecutive_failures = 0
        self.browser.start(self.start_url)

        for step_number in range(1, self.max_steps + 1):
            observation = self.browser.observe(
                max_text_chars=self.max_text_chars,
                max_elements=self.max_elements,
            )
            result.final_url = observation.url
            if self.on_observation is not None:
                self.on_observation(step_number, observation)

            try:
                payload = self.model.decide(
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=self._user_prompt(observation, history),
                )
                action = AgentAction.from_payload(payload, observation)
            except Exception as exc:
                record = {
                    "step": step_number,
                    "url": observation.url,
                    "action": None,
                    "success": False,
                    "error": f"decision failed: {exc}",
                }
                self._record(result, history, record)
                consecutive_failures += 1
                if consecutive_failures >= self.max_consecutive_failures:
                    result.error = record["error"]
                    return result
                continue

            record = {
                "step": step_number,
                "url": observation.url,
                "action": action.to_dict(),
                "success": False,
                "error": "",
            }
            if action.action == "finish":
                record["success"] = True
                self._record(result, history, record)
                result.completed = True
                result.answer = action.value
                return result

            try:
                self.browser.execute(action, observation)
                record["success"] = True
                consecutive_failures = 0
            except Exception as exc:
                record["error"] = f"action failed: {exc}"
                consecutive_failures += 1
            self._record(result, history, record)
            result.final_url = self.browser.current_url

            if consecutive_failures >= self.max_consecutive_failures:
                result.error = record["error"]
                return result

        result.error = f"maximum step count reached ({self.max_steps})"
        result.final_url = self.browser.current_url
        return result

    def _user_prompt(
        self,
        observation: PageObservation,
        history: list[dict],
    ) -> str:
        recent_history = history[-6:]
        return (
            f"User task:\n{self.task}\n\n"
            "Recent action results:\n"
            f"{json.dumps(recent_history, ensure_ascii=False, indent=2)}\n\n"
            "<untrusted_webpage_snapshot>\n"
            f"{observation.render()}\n"
            "</untrusted_webpage_snapshot>"
        )

    def _record(self, result: RunResult, history: list[dict], record: dict) -> None:
        result.steps.append(record)
        history.append(record)
        if self.on_step is not None:
            self.on_step(record)
