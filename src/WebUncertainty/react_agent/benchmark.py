"""Batch runner for the vendored WebVoyager task subset.

Loads task definitions from ``data/webvoyager/*.json`` (task_id, intent,
start_url, eval.reference_answers), runs one :class:`ReActAgent` episode per
task, and scores the final answer with
:func:`WebUncertainty.webvoyager_evaluator.evaluate_webvoyager_answer`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from WebUncertainty.paths import DATA_ROOT
from WebUncertainty.react_agent.agent import ReActAgent, RunResult
from WebUncertainty.react_agent.browser import BrowserSession
from WebUncertainty.webvoyager_evaluator import evaluate_webvoyager_answer

DEFAULT_DATASET_DIR = DATA_ROOT / "webvoyager"


class BenchmarkError(RuntimeError):
    """Raised when the WebVoyager task set cannot be loaded or run."""


@dataclass(frozen=True)
class TaskConfig:
    """One WebVoyager task definition loaded from a vendored JSON file."""

    task_id: str
    intent: str
    start_url: str
    raw: dict[str, Any]

    @classmethod
    def from_json(cls, path: Path) -> TaskConfig:
        data = json.loads(path.read_text(encoding="utf-8"))
        task_id = str(data.get("task_id") or path.stem)
        intent = str(data.get("intent") or "").strip()
        start_url = str(data.get("start_url") or "").strip()
        if not intent:
            raise BenchmarkError(f"{path}: task is missing 'intent'")
        if not start_url:
            raise BenchmarkError(f"{path}: task is missing 'start_url'")
        return cls(task_id=task_id, intent=intent, start_url=start_url, raw=data)


def load_tasks(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    task_ids: list[str] | None = None,
) -> list[TaskConfig]:
    """Load WebVoyager task configs, optionally filtered to specific ids."""
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        raise BenchmarkError(f"WebVoyager dataset directory not found: {dataset_dir}")
    paths = sorted(dataset_dir.glob("*.json"))
    if not paths:
        raise BenchmarkError(f"No task JSON files found under {dataset_dir}")
    tasks = [TaskConfig.from_json(path) for path in paths]
    if task_ids:
        wanted = set(task_ids)
        tasks = [task for task in tasks if task.task_id in wanted]
        missing = wanted - {task.task_id for task in tasks}
        if missing:
            raise BenchmarkError(
                f"Unknown WebVoyager task id(s): {', '.join(sorted(missing))}"
            )
    return tasks


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
    model: Any,
    browser_factory: Callable[[], BrowserSession],
    max_steps: int = 12,
    max_text_chars: int = 8_000,
    max_elements: int = 80,
    on_step: Callable[[dict], None] | None = None,
) -> TaskRunResult:
    """Run one ReAct episode for ``task`` and score the final answer.

    A fresh browser is opened per task (via ``browser_factory``) so cookies
    and navigation state never leak between unrelated sites.
    """
    browser = browser_factory()
    result = RunResult(task=task.intent, start_url=task.start_url)
    try:
        with browser:
            agent = ReActAgent(
                task=task.intent,
                start_url=task.start_url,
                browser=browser,
                model=model,
                max_steps=max_steps,
                max_text_chars=max_text_chars,
                max_elements=max_elements,
                on_step=on_step,
            )
            result = agent.run()
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
    model: Any,
    browser_factory: Callable[[], BrowserSession],
    output_dir: Path,
    max_steps: int = 12,
    max_text_chars: int = 8_000,
    max_elements: int = 80,
    on_task_done: Callable[[TaskRunResult], None] | None = None,
) -> dict:
    """Run every task in ``tasks``, writing per-task and summary results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outcomes: list[TaskRunResult] = []
    for task in tasks:
        outcome = run_task(
            task,
            model=model,
            browser_factory=browser_factory,
            max_steps=max_steps,
            max_text_chars=max_text_chars,
            max_elements=max_elements,
        )
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
            {
                "task_id": outcome.task_id,
                "score": outcome.score,
                "error": outcome.error,
            }
            for outcome in outcomes
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


__all__ = [
    "BenchmarkError",
    "TaskConfig",
    "TaskRunResult",
    "DEFAULT_DATASET_DIR",
    "load_tasks",
    "run_task",
    "run_benchmark",
]
