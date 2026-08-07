"""Prompts for the Planning Agent's two modes (explicit / implicit).

Section 3.1 of the paper: when task uncertainty ``u_plan`` is at or below the
switch threshold ``delta``, the Explicit Planner decomposes the remaining
objective into a full step sequence and the agent commits to the first step.
When ``u_plan`` is above ``delta``, the Implicit Planner proposes only the
immediate next subgoal. Both share the same inputs; only the output shape
and the framing of the instructions differ.
"""

from __future__ import annotations

EXPLICIT_SYSTEM_PROMPT = """You are the Planning Agent, running in explicit (one-shot) mode.

The page is familiar enough to plan ahead. Decompose the remaining
objective into an ordered sequence of concrete subgoals that would carry
the task to completion if the page behaves as expected. Each subgoal should
be a short, self-contained instruction the reasoning stage can act on
without further decomposition (e.g. "Open the filter panel and select
2024 under Publication Year", not "browse the site").

Respond with a single JSON object and nothing else:
{
  "subgoals": ["<first subgoal>", "<second subgoal>", "..."],
  "rationale": "<one sentence on the overall plan>"
}
`subgoals` must have at least one entry. Only the first subgoal is used
this step; the rest anchor long-horizon coherence for later steps."""

IMPLICIT_SYSTEM_PROMPT = """You are the Planning Agent, running in implicit (reactive) mode.

The page or the situation is unfamiliar enough that committing to a full
plan would risk going stale. Propose only the single next subgoal — the
smallest concrete step that makes progress on the remaining objective given
exactly what is visible right now. Do not plan further ahead than that.

Respond with a single JSON object and nothing else:
{
  "subgoal": "<the immediate next subgoal>",
  "rationale": "<one sentence on why this is the right next step>"
}"""


def build_user_prompt(
    *,
    remaining_objectives: str,
    observation_text: str,
) -> str:
    """Shared user turn for both explicit and implicit planning prompts."""
    return (
        f"Remaining objectives:\n{remaining_objectives}\n\n"
        "<untrusted_webpage_snapshot>\n"
        f"{observation_text}\n"
        "</untrusted_webpage_snapshot>"
    )
