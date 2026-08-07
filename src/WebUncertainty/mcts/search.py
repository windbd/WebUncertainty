"""The four-phase MCTS loop: Selection, Expansion, Simulation, Backpropagation.

One :meth:`MCTSReasoner.run` call resolves one planning subgoal into exactly
one root :class:`~WebUncertainty.mcts.tree.ActionCandidate` — grounded
against the real page the caller observed. Nothing in this module executes
that action; see ``mcts/sampler.py``.

Expansion and Simulation are fused into one step here (:meth:`_expand_and_
simulate`): the paper describes the reasoning agent adding *K* candidates
at expansion and an evaluation agent then scoring "the new state" — this
reads most naturally as scoring the whole freshly expanded batch rather
than a lone child, and it sidesteps ever leaving an unsimulated, un-scored
leaf sitting in the tree between rollouts. The paper's ``max node
expansion limit`` (10 per subgoal) is charged only for
:meth:`_expand_and_simulate` calls, not for tree descent.
"""

from __future__ import annotations

from WebUncertainty.agents.evaluation_agent import EvaluationAgent, EvaluationAgentError
from WebUncertainty.agents.reasoning_agent import ReasoningAgent, ReasoningAgentError
from WebUncertainty.mcts.tree import ActionCandidate, Node
from WebUncertainty.mcts.uncertainty import RewardCase, compute_conactu, modulate_reward
from WebUncertainty.react_agent.browser import PageObservation

# Cases the paper frames as "don't come back to this specific action":
# strict/relaxed penalty explicitly prune the path, and regenerate means the
# action itself is abandoned in favor of a fresh candidate set at the parent.
_DEAD_END_CASES = frozenset(
    {RewardCase.STRICT_PENALTY, RewardCase.RELAXED_PENALTY, RewardCase.REGENERATE}
)


class SearchError(RuntimeError):
    """Raised when the root itself cannot be expanded (no usable candidate)."""


class MCTSReasoner:
    """Runs one bounded evidential-free MCTS search for a single subgoal.

    ``eu_threshold``/``au_threshold`` are forwarded to
    :func:`WebUncertainty.mcts.uncertainty.modulate_reward` — see that
    module for why they default to 0.5 rather than a paper-specified value.
    """

    def __init__(
        self,
        *,
        reasoning_agent: ReasoningAgent,
        evaluation_agent: EvaluationAgent,
        k_candidates: int = 4,
        w_puct: float = 5.0,
        tau: float = 6.0,
        max_expansions: int = 10,
        max_depth: int = 3,
        eu_threshold: float = 0.5,
        au_threshold: float = 0.5,
    ) -> None:
        if k_candidates < 1:
            raise ValueError("k_candidates must be at least 1")
        if max_expansions < 1:
            raise ValueError("max_expansions must be at least 1")
        if max_depth < 1:
            raise ValueError("max_depth must be at least 1")
        self.reasoning_agent = reasoning_agent
        self.evaluation_agent = evaluation_agent
        self.k_candidates = k_candidates
        self.w_puct = w_puct
        self.tau = tau
        self.max_expansions = max_expansions
        self.max_depth = max_depth
        self.eu_threshold = eu_threshold
        self.au_threshold = au_threshold

    def run(self, *, subgoal: str, observation: PageObservation) -> tuple[ActionCandidate, Node]:
        """Search from the real ``observation`` and return the chosen root action.

        Returns ``(candidate, root)`` — ``candidate.grounded_action`` is the
        action to execute; ``root`` is the full tree, kept for logging/tests.
        """
        root = Node(parent=None, subgoal=subgoal, depth=0, state_text=observation.render())
        fresh = self._expand_and_simulate(root, real_observation=observation)
        if not fresh:
            raise SearchError("reasoning agent produced no usable root candidate")
        expansions_used = 1
        expansions_used = self._maybe_regenerate(
            root, fresh, expansions_used, real_observation=observation
        )

        safety_cap = self.max_expansions * (self.k_candidates + 2)
        iterations = 0
        while expansions_used < self.max_expansions and iterations < safety_cap:
            iterations += 1
            leaf = self._select_leaf(root)
            if leaf is None:
                break
            fresh = self._expand_and_simulate(leaf)
            if not fresh:
                leaf.terminal = True
                continue
            expansions_used += 1
            expansions_used = self._maybe_regenerate(leaf, fresh, expansions_used)

        return root.best_root_action(), root

    def _maybe_regenerate(
        self,
        node: Node,
        fresh: list[Node],
        expansions_used: int,
        *,
        real_observation: PageObservation | None = None,
    ) -> int:
        """Re-expand ``node`` once more if any fresh child landed on REGENERATE.

        Mirrors the paper's "confident but wrong" case: the failed action is
        abandoned (already marked terminal by ``_simulate_and_backprop``)
        and the reasoning agent is asked for a new candidate set at the same
        node, budget permitting. ``real_observation`` must be supplied when
        ``node`` is the root — it is the only node that ever needs one.
        """
        needs_regeneration = any(child.reward_case is RewardCase.REGENERATE for child in fresh)
        if not needs_regeneration or expansions_used >= self.max_expansions:
            return expansions_used
        more = self._expand_and_simulate(node, real_observation=real_observation)
        return expansions_used + 1 if more else expansions_used

    def _select_leaf(self, root: Node) -> Node | None:
        """Descend via PUCT to a live leaf shallow enough to expand further."""
        node = root
        while True:
            if node.terminal:
                if node.is_root:
                    return None
                node = node.parent  # type: ignore[assignment]
                continue
            if node.is_leaf():
                if node.depth >= self.max_depth:
                    node.terminal = True
                    if node.is_root:
                        return None
                    node = node.parent  # type: ignore[assignment]
                    continue
                return node
            live = [child for child in node.children if not child.terminal]
            if not live:
                node.terminal = True
                if node.is_root:
                    return None
                node = node.parent  # type: ignore[assignment]
                continue
            node = node.select_child(self.w_puct)

    def _expand_and_simulate(
        self,
        node: Node,
        *,
        real_observation: PageObservation | None = None,
    ) -> list[Node]:
        """Expansion (propose K, seed priors) then Simulation for each new child.

        Returns the freshly created children (empty if the reasoning agent
        produced nothing usable, in which case ``node`` should be treated
        as exhausted by the caller).
        """
        if node.is_root:
            if real_observation is None:
                raise SearchError("root expansion requires the real page observation")
            try:
                candidates = self.reasoning_agent.propose_grounded(
                    subgoal=node.subgoal,
                    observation=real_observation,
                    k_candidates=self.k_candidates,
                )
            except ReasoningAgentError:
                return []
        else:
            try:
                candidates = self.reasoning_agent.propose_hypothetical(
                    subgoal=node.subgoal,
                    predicted_observation=node.state_text,
                    k_candidates=self.k_candidates,
                )
            except ReasoningAgentError:
                return []

        conactu = compute_conactu([candidate.confidence for candidate in candidates])
        fresh = []
        for candidate in candidates:
            child = node.add_child(candidate=candidate, state_text=node.state_text)
            child.eu = conactu.eu
            child.au = conactu.au
            fresh.append(child)
        for child in fresh:
            self._simulate_and_backprop(child)
        return fresh

    def _simulate_and_backprop(self, child: Node) -> None:
        assert child.candidate is not None and child.parent is not None
        try:
            result = self.evaluation_agent.evaluate(
                subgoal=child.subgoal,
                state_text=child.parent.state_text,
                candidate=child.candidate,
            )
        except EvaluationAgentError:
            # An unusable evaluation is treated as a wasted candidate, not a
            # reason to abort the whole search over one bad model response.
            child.terminal = True
            child.backpropagate(0.0)
            return

        child.state_text = result.predicted_observation
        modulated = modulate_reward(
            result.score,
            tau=self.tau,
            eu=child.eu,
            au=child.au,
            eu_threshold=self.eu_threshold,
            au_threshold=self.au_threshold,
        )
        child.reward_case = modulated.case
        child.backpropagate(modulated.reward)
        if modulated.case in _DEAD_END_CASES:
            child.terminal = True
