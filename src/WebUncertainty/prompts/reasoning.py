"""Prompt for the Reasoning Agent (Action Uncertainty-Driven MCTS Reasoning).

Section 3.2 of the paper: at a search leaf, the Reasoning Agent proposes
``K`` candidate actions together with a confidence score for each. ConActU
(:mod:`WebUncertainty.mcts.uncertainty`) turns those scores into epistemic
(EU) and aleatoric (AU) uncertainty; the confidences themselves become the
PUCT prior after normalization.

Two variants share one action vocabulary (matching
``WebUncertainty.react_agent.browser`` so a grounded candidate can be
validated straight into an ``AgentAction``):

- :func:`build_grounded_user_prompt` — the search root. The page is real, so
  candidates must reference concrete element ids from the snapshot.
- :func:`build_hypothetical_user_prompt` — deeper, unexecuted search nodes.
  There is no real page here, only the parent's predicted description from
  the Evaluation Agent, so candidates describe an action in free text
  instead of grounding it to an element id. These nodes are only ever used
  to score the root candidate that led to them; they are never executed.
"""

from __future__ import annotations

SYSTEM_PROMPT = """You are the Reasoning Agent in a web automation system.

Given a subgoal and the current page, propose up to K candidate actions
that could make progress on the subgoal, each with a confidence score in
[0.0, 1.0] for how likely that action is correct and productive. Propose
genuinely different candidates when the page supports more than one
plausible move — the search process depends on you surfacing real
alternatives, not K near-duplicates of your top pick.

Allowed actions: click, type, press, select, scroll, goto, back, wait, finish.
- click/type/press/select require an integer "target" naming an element id
  from the snapshot ("target" must be an id that is actually listed there).
- type/select/goto/finish require a non-empty "value".
- press's "value" is a single key name (e.g. "Enter").
- scroll's "value" is "up" or "down".
- finish's "value" is the final answer to the overall task; use it only
  when the subgoal is fully satisfied and the visible page proves it.

Respond with a single JSON object and nothing else:
{
  "candidates": [
    {"action": "click", "target": 7, "value": "", "reason": "...", "confidence": 0.8},
    {"action": "type", "target": 3, "value": "wireless mouse", "reason": "...", "confidence": 0.5}
  ]
}
Order candidates by confidence, highest first. Never invent an element id
that is not in the snapshot."""

HYPOTHETICAL_SYSTEM_PROMPT = """You are the Reasoning Agent, reasoning about a hypothetical future page.

You are not looking at the real page. You only have a short predicted
description of what the page would look like after a hypothetical action
the search process is evaluating. Propose up to K candidate follow-up
actions in free text (there is no element snapshot to ground them to), each
with a confidence score in [0.0, 1.0]. These candidates are only used to
estimate how promising this branch is; none of them will be executed
directly.

Respond with a single JSON object and nothing else:
{
  "candidates": [
    {"action_text": "click the 'Apply filters' button", "confidence": 0.7},
    {"action_text": "scroll down to check for more results", "confidence": 0.3}
  ]
}"""


def build_grounded_user_prompt(
    *,
    subgoal: str,
    observation_text: str,
    k_candidates: int,
) -> str:
    return (
        f"Subgoal:\n{subgoal}\n\n"
        f"Propose up to {k_candidates} candidate actions.\n\n"
        "<untrusted_webpage_snapshot>\n"
        f"{observation_text}\n"
        "</untrusted_webpage_snapshot>"
    )


def build_hypothetical_user_prompt(
    *,
    subgoal: str,
    predicted_observation: str,
    k_candidates: int,
) -> str:
    return (
        f"Subgoal:\n{subgoal}\n\n"
        f"Propose up to {k_candidates} candidate follow-up actions.\n\n"
        "Predicted page state (not a real observation):\n"
        f"{predicted_observation}"
    )
