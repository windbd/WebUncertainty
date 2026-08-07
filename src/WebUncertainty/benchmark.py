"""Batch runner for the vendored WebVoyager task subset, using AdaptivePlanner.

Mirrors :mod:`WebUncertainty.react_agent.benchmark`'s shape (same task
loading, same :func:`WebUncertainty.webvoyager_evaluator.evaluate_webvoyager_answer`
scoring) with the dual-uncertainty agent
(:class:`WebUncertainty.planning.adaptive_planner.AdaptivePlanner`) in place
of the single-call ReAct loop. ``TaskConfig``/``load_tasks`` are reused
directly from ``react_agent.benchmark`` rather than duplicated — they are
plain WebVoyager JSON loading, not react_agent-specific logic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from WebUncertainty.planning.adaptive_planner import AdaptivePlanner, AdaptivePlannerConfig
from WebUncertainty.react_agent.agent import RunResult
from WebUncertainty.react_agent.benchmark import (
    DEFAULT_DATASET_DIR,
    BenchmarkError,
    TaskConfig,
    load_tasks,
)
from WebUncertainty.react_agent.browser import BrowserSession
from WebUncertainty.webvoyager_evaluator import evaluate_webvoyager_answer

__all__ = [
    "BenchmarkError",
    "TaskConfig",
    "TaskRunResult",
    "DEFAULT_DATASET_DIR",
    "load_tasks",
    "run_task",
    "run_benchmark",
]


@dataclass
class TaskRunResult:
    """Serializable outcome of one benchmark episode."""

    task_id: str
    result: RunResult
    score: float
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "score": self.score,
            "error": self.error,
            "run": self.result.to_dict(),
        }


def run_task(
    task: TaskConfig,
    *,
    browser_factory: Callable[[], BrowserSession],
    config: AdaptivePlannerConfig | None = None,
) -> TaskRunResult:
    """Run one AdaptivePlanner episode for ``task`` and score the final answer.

    A fresh browser is opened per task (via ``browser_factory``) so cookies
    and navigation state never leak between unrelated sites. Model
    selection is process-wide via
    :func:`WebUncertainty.model.configure_runtime`, not threaded through
    here — see ``cli.py``.
    """
    browser = browser_factory()
    result = RunResult(task=task.intent, start_url=task.start_url)
    try:
        with browser:
            planner = AdaptivePlanner(browser=browser, config=config)
            result = planner.run(task=task.intent, start_url=task.start_url)
    except Exception as exc:
        result.error = f"runtime failed: {exc}"

    if not result.completed:
        return TaskRunResult(task.task_id, result, 0.0, error=result.error)
    try:
        score = evaluate_webvoyager_answer(task.raw, result.answer)
    except Exception as exc:
        return TaskRunResult(task.task_id, result, 0.0, error=f"scoring failed: {exc}")
    return TaskRunResult(task.task_id, result, score)


def run_benchmark(
    tasks: list[TaskConfig],
    *,
    browser_factory: Callable[[], BrowserSession],
    output_dir: Path,
    config: AdaptivePlannerConfig | None = None,
    on_task_done: Callable[[TaskRunResult], None] | None = None,
) -> dict:
    """Run every task in ``tasks``, writing per-task and summary results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outcomes: list[TaskRunResult] = []
    for task in tasks:
        outcome = run_task(task, browser_factory=browser_factory, config=config)
        outcomes.append(outcome)
        (output_dir / f"{task.task_id}.json").write_text(
            json.dumps(outcome.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if on_task_done is not None:
            on_task_done(outcome)

    total = len(outcomes)
    passed = sum(1 for outcome in outcomes if outcome.score >= 1.0)
    summary = {
        "total": total,
        "passed": passed,
        "pass_rate": (passed / total) if total else 0.0,
        "tasks": [
            {"task_id": outcome.task_id, "score": outcome.score, "error": outcome.error}
            for outcome in outcomes
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary
