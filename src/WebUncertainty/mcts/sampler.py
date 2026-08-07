"""Browser-facing adapter that commits the search's chosen root action.

This is the one place a WebUncertainty search result reaches a real page.
Everything upstream (``mcts.search``, ``mcts.tree``) only ever reasons about
the observation the caller handed it and predicted continuations of it —
see the ``mcts`` package docstring.
"""

from __future__ import annotations

from dataclasses import dataclass

from WebUncertainty.mcts.tree import ActionCandidate, Node
from WebUncertainty.react_agent.agent import AgentAction
from WebUncertainty.react_agent.browser import BrowserSession, PageObservation


class SamplerError(RuntimeError):
    """Raised when the chosen root candidate cannot actually be executed."""


@dataclass(frozen=True)
class CommittedAction:
    """What was executed, plus the tree's own estimate of how good it was."""

    action: AgentAction
    reward_estimate: float


def commit_root_action(
    *,
    candidate: ActionCandidate,
    root: Node,
    browser: BrowserSession,
    observation: PageObservation,
) -> CommittedAction:
    """Execute the search's chosen root action on the real page.

    ``observation`` should be the same real snapshot the search was
    grounded in — ``AgentAction`` already re-validated the target element
    against it at candidate-construction time, so a page that changed out
    from under the search fails here rather than clicking the wrong thing.
    A ``finish`` action is never sent to the browser, matching
    ``react_agent``'s convention that finishing ends the episode instead of
    executing a primitive.
    """
    if not root.is_root:
        raise SamplerError("commit_root_action expects the search's root node")
    if candidate.grounded_action is None:
        raise SamplerError("chosen root candidate is not a grounded action")

    action = candidate.grounded_action
    matching_child = next(
        (child for child in root.children if child.candidate is candidate),
        None,
    )
    reward_estimate = matching_child.value if matching_child is not None else 0.0

    if action.action != "finish":
        browser.execute(action, observation)
    return CommittedAction(action=action, reward_estimate=reward_estimate)
