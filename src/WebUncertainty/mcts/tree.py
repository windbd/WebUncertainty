"""Search tree data structures and the PUCT selection score (Eq. 1-2).

    a_t = argmax_a [ Q(s,a) + U(s,a) ]
    U(s,a) = w_puct * P_con(s,a) * sqrt(sum_b N(s,b)) / (1 + N(s,a))

Only the root ever has a live page: it is the state the outer planner just
observed for real. Root candidates are proposed against that real page and
validated into a grounded, executable
:class:`WebUncertainty.react_agent.agent.AgentAction` — reusing that
validation reuses its element-existence and read-only-safety checks for
free, rather than re-deriving them here.

Deeper nodes exist purely to let the search reason a little further ahead
before committing; they are expanded against a *predicted* description of
the page (the Evaluation Agent's ``predicted_observation``), not a real
DOM, so their candidates are free-text action descriptions with no element
id to ground against. They score the root branch that led to them and are
never independently executable — see ``mcts/sampler.py``, which is the only
place a search result ever reaches a real ``BrowserSession``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from WebUncertainty.mcts.uncertainty import RewardCase
from WebUncertainty.react_agent.agent import AgentAction


class TreeError(ValueError):
    """Raised for structurally invalid search-tree operations."""


@dataclass(frozen=True)
class ActionCandidate:
    """One proposed action, grounded (root) or hypothetical (deeper).

    ``prior`` is the normalized confidence used as PUCT's ``P_con`` (paper
    Eq. 2); ``confidence`` is the raw score ConActU was computed from.
    Exactly one of ``grounded_action`` / ``action_text`` is set.
    """

    confidence: float
    prior: float
    reason: str = ""
    grounded_action: AgentAction | None = None
    action_text: str = ""

    def __post_init__(self) -> None:
        has_grounded = self.grounded_action is not None
        has_text = bool(self.action_text.strip())
        if has_grounded == has_text:
            raise TreeError(
                "ActionCandidate needs exactly one of grounded_action or action_text"
            )

    @property
    def is_grounded(self) -> bool:
        return self.grounded_action is not None

    def describe(self) -> str:
        """Human/LLM-readable summary, for the Evaluation Agent's prompt."""
        if self.grounded_action is not None:
            action = self.grounded_action
            target = f" [{action.target}]" if action.target is not None else ""
            value = f" {action.value!r}" if action.value else ""
            reason = f" — {self.reason}" if self.reason else ""
            return f"{action.action}{target}{value}{reason}"
        reason = f" — {self.reason}" if self.reason else ""
        return f"{self.action_text}{reason}"


@dataclass
class Node:
    """One search-tree node.

    ``state_text`` is the real rendered observation for the root, or the
    parent evaluation's ``predicted_observation`` for every other node —
    callers never need to know which when reading it back for a prompt.
    """

    parent: "Node | None"
    subgoal: str
    depth: int
    state_text: str
    candidate: ActionCandidate | None = None  # edge from parent; None at root.
    children: list["Node"] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    terminal: bool = False
    eu: float = 0.0
    au: float = 0.0
    reward_case: RewardCase | None = None

    @property
    def is_root(self) -> bool:
        return self.parent is None

    def is_leaf(self) -> bool:
        return not self.children

    def add_child(self, *, candidate: ActionCandidate, state_text: str) -> "Node":
        child = Node(
            parent=self,
            subgoal=self.subgoal,
            depth=self.depth + 1,
            state_text=state_text,
            candidate=candidate,
        )
        self.children.append(child)
        return child

    def puct_score(self, w_puct: float) -> float:
        """Eq. 1-2: exploitation (running value) plus a prior-weighted bonus."""
        if self.is_root or self.candidate is None:
            raise TreeError("PUCT score is only defined for a non-root child")
        parent_visits = self.parent.visits if self.parent is not None else 0
        exploration = (
            w_puct
            * self.candidate.prior
            * math.sqrt(parent_visits)
            / (1 + self.visits)
        )
        return self.value + exploration

    def select_child(self, w_puct: float) -> "Node":
        """Argmax-PUCT among non-terminal children (Eq. 1)."""
        live_children = [child for child in self.children if not child.terminal]
        if not live_children:
            raise TreeError("no non-terminal children to select from")
        return max(live_children, key=lambda child: child.puct_score(w_puct))

    def backpropagate(self, reward: float) -> None:
        """Eq. 5-6: incremental mean update, applied from this node to the root."""
        node: Node | None = self
        while node is not None:
            node.visits += 1
            node.value += (reward - node.value) / node.visits
            node = node.parent

    def best_root_action(self) -> ActionCandidate:
        """After the search budget is spent, pick the root's strongest child.

        Uses the same running value PUCT optimizes, without the
        now-irrelevant exploration bonus — the standard "final move" choice
        once the search itself is over.
        """
        if not self.is_root:
            raise TreeError("best_root_action is only defined on the root node")
        if not self.children:
            raise TreeError("root has no expanded children yet")
        best = max(self.children, key=lambda child: child.value)
        assert best.candidate is not None
        return best.candidate
