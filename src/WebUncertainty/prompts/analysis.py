"""Prompt for the Analysis Agent (Task Uncertainty-Driven Adaptive Planning).

Implements the analysis step from Section 3.1 of the WebUncertainty paper:
given the global instruction, the current page, and the execution history,
estimate the remaining objectives ``T_rem`` and a task-uncertainty scalar
``u_plan in [0, 1]``. ``u_plan`` is consumed by :mod:`WebUncertainty.agents`
to pick the explicit or implicit planner for this step.
"""

from __future__ import annotations

import json

SYSTEM_PROMPT = """You are the Analysis Agent in a web automation system.

Before every planning step, you look at the overall task, the current page,
and what has already been tried, and answer two questions:

1. What task objectives are still outstanding (`remaining_objectives`)?
2. How uncertain is the situation right now (`task_uncertainty`, 0.0-1.0)?

`task_uncertainty` should track environmental unfamiliarity and how complex
it looks to finish `remaining_objectives` from here — NOT your confidence in
any single action. Score it high (towards 1.0) when the page layout, site,
or required interaction pattern is unfamiliar or unclear from the
observation, or the remaining work spans several uncertain sub-steps. Score
it low (towards 0.0) when the page and the next moves are familiar and the
remaining work is straightforward to lay out in advance.

Respond with a single JSON object and nothing else:
{
  "remaining_objectives": "<what still needs to happen to complete the task>",
  "task_uncertainty": <float between 0.0 and 1.0>,
  "rationale": "<one sentence on why this uncertainty level>"
}"""


def build_user_prompt(
    *,
    task: str,
    observation_text: str,
    history: list[dict],
) -> str:
    """Render the Analysis Agent's user turn.

    ``observation_text`` is an already-rendered page snapshot (e.g.
    ``PageObservation.render()``); ``history`` is the same bounded list of
    step records the other agents see.
    """
    recent_history = history[-6:]
    return (
        f"Overall task:\n{task}\n\n"
        "Recent execution history:\n"
        f"{json.dumps(recent_history, ensure_ascii=False, indent=2)}\n\n"
        "<untrusted_webpage_snapshot>\n"
        f"{observation_text}\n"
        "</untrusted_webpage_snapshot>"
    )
