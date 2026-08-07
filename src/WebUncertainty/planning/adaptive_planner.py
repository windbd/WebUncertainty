"""AdaptivePlanner: the real-browser outer controller (paper Section 3.1 + 3.2).

Each step: observe the real page -> Analysis Agent estimates task
uncertainty -> Planning Agent proposes one subgoal via whichever mode that
uncertainty selects -> an MCTS search resolves the subgoal into one grounded
action -> that action is executed, and the loop repeats. This module is the
only thing in the stack that owns a live ``BrowserSession``; ``mcts/``
never touches it directly (see that package's docstring).
"""

from __future__ import annotations

from dataclasses import dataclass

from WebUncertainty.agents.analysis_agent import AnalysisAgent
from WebUncertainty.agents.evaluation_agent import EvaluationAgent
from WebUncertainty.agents.planning_agent import PlanningAgent
from WebUncertainty.agents.reasoning_agent import ReasoningAgent
from WebUncertainty.mcts.sampler import commit_root_action
from WebUncertainty.mcts.search import MCTSReasoner
from WebUncertainty.react_agent.agent import RunResult
from WebUncertainty.react_agent.browser import BrowserSession


class AdaptivePlannerError(RuntimeError):
    """Raised for configuration errors.

    A single step's model/search failure is instead recorded in that step's
    log entry and counted against ``max_consecutive_failures`` — matching
    ``react_agent``'s convention of degrading to the next step rather than
    aborting the whole run over one bad model response.
    """


@dataclass
class AdaptivePlannerConfig:
    """Hyperparameters.

    ``delta``, ``tau``, ``w_puct``, and ``max_expansions`` are the values
    the paper's own experiments settle on (Section 4.1's implementation
    details and Section 4's sensitivity sweep). ``k_candidates``,
    ``max_depth``, ``eu_threshold``, and ``au_threshold`` are this
    implementation's own defaults for choices the paper leaves open — see
    ``mcts/uncertainty.py`` and ``prompts/reasoning.py``.
    """

    delta: float = 0.4
    tau: float = 6.0
    k_candidates: int = 4
    w_puct: float = 5.0
    max_expansions: int = 10
    max_depth: int = 3
    eu_threshold: float = 0.5
    au_threshold: float = 0.5
    max_steps: int = 12
    max_consecutive_failures: int = 3


class AdaptivePlanner:
    """Owns the browser session and runs the full planning + reasoning loop."""

    def __init__(
        self,
        *,
        browser: BrowserSession,
        analysis_agent: AnalysisAgent | None = None,
        planning_agent: PlanningAgent | None = None,
        reasoning_agent: ReasoningAgent | None = None,
        evaluation_agent: EvaluationAgent | None = None,
        config: AdaptivePlannerConfig | None = None,
    ) -> None:
        self.browser = browser
        self.analysis_agent = analysis_agent or AnalysisAgent()
        self.planning_agent = planning_agent or PlanningAgent()
        self.reasoning_agent = reasoning_agent or ReasoningAgent()
        self.evaluation_agent = evaluation_agent or EvaluationAgent()
        self.config = config or AdaptivePlannerConfig()

    def run(self, *, task: str, start_url: str) -> RunResult:
        if not task.strip():
            raise AdaptivePlannerError("task must not be empty")
        cfg = self.config
        result = RunResult(task=task.strip(), start_url=start_url.strip())
        history: list[dict] = []
        consecutive_failures = 0
        self.browser.start(result.start_url)

        for step_number in range(1, cfg.max_steps + 1):
            observation = self.browser.observe()
            result.final_url = observation.url
            record: dict = {"step": step_number, "url": observation.url}

            try:
                analysis = self.analysis_agent.analyze(
                    task=task, observation=observation, history=history
                )
                record["task_uncertainty"] = analysis.task_uncertainty

                if analysis.task_uncertainty <= cfg.delta:
                    plan = self.planning_agent.plan_explicit(
                        remaining_objectives=analysis.remaining_objectives,
                        observation=observation,
                    )
                else:
                    plan = self.planning_agent.plan_implicit(
                        remaining_objectives=analysis.remaining_objectives,
                        observation=observation,
                    )
                record["planning_mode"] = plan.mode
                record["subgoal"] = plan.subgoal

                reasoner = MCTSReasoner(
                    reasoning_agent=self.reasoning_agent,
                    evaluation_agent=self.evaluation_agent,
                    k_candidates=cfg.k_candidates,
                    w_puct=cfg.w_puct,
                    tau=cfg.tau,
                    max_expansions=cfg.max_expansions,
                    max_depth=cfg.max_depth,
                    eu_threshold=cfg.eu_threshold,
                    au_threshold=cfg.au_threshold,
                )
                candidate, root = reasoner.run(subgoal=plan.subgoal, observation=observation)
                committed = commit_root_action(
                    candidate=candidate,
                    root=root,
                    browser=self.browser,
                    observation=observation,
                )
            except Exception as exc:  # noqa: BLE001 - one bad step must not abort the run
                record["success"] = False
                record["error"] = f"step failed: {exc}"
                self._record(result, history, record)
                consecutive_failures += 1
                if consecutive_failures >= cfg.max_consecutive_failures:
                    result.error = record["error"]
                    return result
                continue

            record["action"] = committed.action.to_dict()
            record["reward_estimate"] = committed.reward_estimate
            record["success"] = True
            record["error"] = ""
            self._record(result, history, record)
            consecutive_failures = 0
            result.final_url = self.browser.current_url

            if committed.action.action == "finish":
                result.completed = True
                result.answer = committed.action.value
                return result

            if consecutive_failures >= cfg.max_consecutive_failures:
                result.error = record["error"]
                return result

        result.error = f"maximum step count reached ({cfg.max_steps})"
        result.final_url = self.browser.current_url
        return result

    @staticmethod
    def _record(result: RunResult, history: list[dict], record: dict) -> None:
        result.steps.append(record)
        history.append(record)
