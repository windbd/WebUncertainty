"""Prompt for the Evaluation Agent (MCTS simulation phase).

Section 3.2 of the paper: instead of a random rollout, an Evaluation Agent
assesses how promising a candidate action is and returns a base feasibility
score ``S_base`` on a 0-10 scale. A score at or above the acceptance
threshold ``tau`` is accepted outright; below it, the EU/AU quadrant from
ConActU decides how the failure is handled
(:mod:`WebUncertainty.mcts.uncertainty`).

This step never touches the real browser (see ``mcts/`` module docs): it
predicts a plausible resulting page state from the current state and the
candidate action, the same way a person would guess "clicking that filter
should show a date range picker" without actually clicking it. That
prediction is what deeper, hypothetical reasoning steps condition on; only
the search's final root action is ever actually executed.
"""

from __future__ import annotations

SYSTEM_PROMPT = """You are the Evaluation Agent in a web automation system.

You are given a subgoal, the page state the search process is reasoning
from, and one candidate action. Without executing anything, judge how
promising that action is for the subgoal and predict what would plausibly
happen next.

Score `score` from 0 to 10:
- 8-10: this action almost certainly satisfies the subgoal.
- 5-7: plausible progress, but not certain to fully satisfy the subgoal.
- 1-4: unlikely to help, or targets something that may not really be
  interactive in the way assumed.
- 0: nonsensical for this state (e.g. references something that clearly
  is not on the page, or contradicts the subgoal).

Respond with a single JSON object and nothing else:
{
  "score": <float 0-10>,
  "predicted_observation": "<short factual description of the page state you'd expect right after this action>",
  "rationale": "<one sentence on why this score>"
}"""


def build_user_prompt(
    *,
    subgoal: str,
    state_text: str,
    action_description: str,
) -> str:
    """``state_text`` is either a real rendered observation (root candidates)
    or a predicted-observation string from a shallower evaluation call
    (hypothetical, deeper nodes) — the Evaluation Agent does not need to
    know which."""
    return (
        f"Subgoal:\n{subgoal}\n\n"
        f"Candidate action:\n{action_description}\n\n"
        "Page state being reasoned from (may be predicted, not necessarily "
        "a live observation):\n"
        f"{state_text}"
    )
