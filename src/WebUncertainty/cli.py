"""Command-line entry point for the WebUncertainty dual-uncertainty agent.

Mirrors ``react_agent.cli``'s shape (single-task mode, WebVoyager benchmark
mode, the same browser flags) with hyperparameters for the Task
Uncertainty-Driven Adaptive Planning and Action Uncertainty-Driven MCTS
Reasoning mechanisms in place of react_agent's plain step budget. Model
selection goes through
:func:`WebUncertainty.model.configure_runtime` once at startup rather than
building a model client to pass around — every agent's default
``ask_llm`` already reads that process-wide configuration.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from WebUncertainty import model as model_module
from WebUncertainty.benchmark import (
    DEFAULT_DATASET_DIR,
    BenchmarkError,
    TaskRunResult,
    load_tasks,
)
from WebUncertainty.benchmark import run_benchmark as run_webvoyager_benchmark
from WebUncertainty.paths import PROJECT_ROOT, environment_files
from WebUncertainty.planning.adaptive_planner import AdaptivePlanner, AdaptivePlannerConfig
from WebUncertainty.react_agent.agent import RunResult
from WebUncertainty.react_agent.browser import BrowserSession

_PROJECT_DIR = PROJECT_ROOT
_DEFAULT_ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the WebUncertainty dual-uncertainty planning + MCTS reasoning agent",
    )
    parser.add_argument("--task", help="Natural-language web task (single-task mode)")
    parser.add_argument("--url", help="Absolute HTTP(S) start URL (single-task mode)")
    parser.add_argument(
        "--benchmark",
        choices=["webvoyager"],
        default=None,
        help="Run the vendored WebVoyager task subset instead of one task",
    )
    parser.add_argument(
        "--task-id",
        nargs="+",
        default=None,
        help="WebVoyager task id(s) to run, e.g. Allrecipes--10 (benchmark mode)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run every vendored WebVoyager task (benchmark mode)",
    )
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help=f"WebVoyager task JSON directory (default: {DEFAULT_DATASET_DIR})",
    )

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--model", default=None, help="Override the model name")
    model_group.add_argument(
        "--endpoint",
        default=None,
        help="Override the OpenAI-compatible API endpoint",
    )
    model_group.add_argument(
        "--api-key-env",
        default=None,
        help="Environment variable containing the API key (never the key itself)",
    )
    model_group.add_argument("--model-timeout", type=float, default=60.0)

    hp = parser.add_argument_group("WebUncertainty hyperparameters")
    hp.add_argument(
        "--delta", type=float, default=0.4,
        help="Planning switch threshold: task_uncertainty <= delta uses explicit planning (default: 0.4, paper-tuned)",
    )
    hp.add_argument(
        "--tau", type=float, default=6.0,
        help="Evaluation acceptance threshold, 0-10 (default: 6.0, paper-tuned)",
    )
    hp.add_argument(
        "--k-candidates", type=int, default=4,
        help="Candidate actions proposed per MCTS expansion (default: 4; not specified by the paper)",
    )
    hp.add_argument("--w-puct", type=float, default=5.0, help="PUCT exploration weight (default: 5.0, paper-tuned)")
    hp.add_argument(
        "--max-expansions", type=int, default=10,
        help="MCTS node-expansion budget per subgoal (default: 10, paper-tuned)",
    )
    hp.add_argument(
        "--max-depth", type=int, default=3,
        help="Hypothetical search-depth cap (default: 3; not specified by the paper)",
    )
    hp.add_argument(
        "--eu-threshold", type=float, default=0.5,
        help="'High EU' cutoff for reward modulation (default: 0.5; not specified by the paper)",
    )
    hp.add_argument(
        "--au-threshold", type=float, default=0.5,
        help="'High AU' cutoff for reward modulation (default: 0.5; not specified by the paper)",
    )
    hp.add_argument("--max-steps", type=int, default=12, help="Outer planning-loop step budget")

    browser_group = parser.add_argument_group("browser")
    browser_group.add_argument(
        "--browser", choices=["chromium", "msedge"], default="chromium",
    )
    browser_group.add_argument(
        "--headed", action="store_true",
        help="Show the browser window (headless by default)",
    )
    browser_group.add_argument("--slow-mo", type=int, default=0)
    browser_group.add_argument("--action-wait-ms", type=int, default=500)
    browser_group.add_argument("--browser-timeout-ms", type=int, default=10_000)
    browser_group.add_argument(
        "--edge-cdp", default=None,
        help=(
            "Attach to an already-running Edge instead of launching a new "
            "browser, e.g. http://127.0.0.1:9222 (start Edge with "
            "--remote-debugging-port first). Overrides --browser/--headed/--slow-mo."
        ),
    )
    browser_group.add_argument(
        "--reuse-tab", action="store_true",
        help="With --edge-cdp, act in the most recently active existing tab instead of opening a new one",
    )

    parser.add_argument(
        "--output-dir", default=None,
        help="Defaults to output/webuncertainty, or output/webvoyager-webuncertainty in benchmark mode",
    )
    return parser


def run(args: argparse.Namespace) -> tuple[RunResult, Path]:
    _configure_model_runtime(args)
    config = _planner_config(args)

    run_dir = _new_run_dir(_output_dir(args), args.task)
    browser = _build_browser(args)

    result = RunResult(task=args.task, start_url=args.url)
    try:
        with browser:
            planner = AdaptivePlanner(browser=browser, config=config)
            result = planner.run(task=args.task, start_url=args.url)
            for record in result.steps:
                _print_step(record)
    except Exception as exc:
        result.error = f"runtime failed: {exc}"
        result.final_url = browser.current_url

    result_path = run_dir / "result.json"
    result_path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result, result_path


def run_benchmark(args: argparse.Namespace) -> dict:
    """Run the WebVoyager benchmark subset selected by ``--task-id``/``--all``."""
    _configure_model_runtime(args)
    config = _planner_config(args)
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else DEFAULT_DATASET_DIR
    task_ids = None if args.all else args.task_id
    tasks = load_tasks(dataset_dir, task_ids)

    def browser_factory() -> BrowserSession:
        return _build_browser(args)

    def print_task_done(outcome: TaskRunResult) -> None:
        status = "PASS" if outcome.score >= 1.0 else "FAIL"
        print(
            f"[{status}] {outcome.task_id}  score={outcome.score:g}  "
            f"answer={outcome.result.answer!r}"
        )
        if outcome.error:
            print(f"  error: {outcome.error}")

    return run_webvoyager_benchmark(
        tasks,
        browser_factory=browser_factory,
        output_dir=_output_dir(args),
        config=config,
        on_task_done=print_task_done,
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    _validate_args(parser, args)
    try:
        for path in environment_files():
            load_dotenv(path, override=False)
        if args.benchmark:
            summary = run_benchmark(args)
            print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))
            return 0 if summary["passed"] == summary["total"] else 1
        result, result_path = run(args)
    except (ValueError, OSError, BenchmarkError) as exc:
        parser.error(str(exc))

    print("\n" + json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    print(f"\nResult written to: {result_path}")
    return 0 if result.completed else 1


def _configure_model_runtime(args: argparse.Namespace) -> None:
    endpoint, model_name, api_key = _model_settings(args)
    model_module.configure_runtime(endpoint=endpoint, model=model_name, api_key=api_key)


def _planner_config(args: argparse.Namespace) -> AdaptivePlannerConfig:
    return AdaptivePlannerConfig(
        delta=args.delta,
        tau=args.tau,
        k_candidates=args.k_candidates,
        w_puct=args.w_puct,
        max_expansions=args.max_expansions,
        max_depth=args.max_depth,
        eu_threshold=args.eu_threshold,
        au_threshold=args.au_threshold,
        max_steps=args.max_steps,
    )


def _build_browser(args: argparse.Namespace) -> BrowserSession:
    return BrowserSession(
        headless=not args.headed,
        browser_name=args.browser,
        slow_mo=args.slow_mo,
        action_wait_ms=args.action_wait_ms,
        timeout_ms=args.browser_timeout_ms,
        cdp_endpoint=args.edge_cdp,
        reuse_tab=args.reuse_tab,
    )


def _output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    subdir = "webvoyager-webuncertainty" if args.benchmark else "webuncertainty"
    return _PROJECT_DIR / "output" / subdir


def _print_step(record: dict) -> None:
    mode = record.get("planning_mode")
    subgoal = record.get("subgoal")
    if mode and subgoal:
        uncertainty = record.get("task_uncertainty")
        print(f"Step {record['step']}: [{mode}] u_plan={uncertainty} subgoal={subgoal!r}")
    action = record.get("action")
    if action:
        target = f" [{action['target']}]" if action.get("target") else ""
        value = f" {action['value']}" if action.get("value") else ""
        reward = record.get("reward_estimate")
        print(f"  -> {action['action']}{target}{value}  (Q={reward})")
    if record.get("error"):
        print(f"  Error: {record['error']}")


def _model_settings(args: argparse.Namespace) -> tuple[str, str, str]:
    configured_endpoint = os.environ.get("DASHSCOPE_ENDPOINT", _DEFAULT_ENDPOINT)
    endpoint = str(args.endpoint or configured_endpoint).strip()
    model_name = str(
        args.model
        or os.environ.get("DASHSCOPE_MODEL_NAME")
        or os.environ.get("DASHSCOPE_MODEL")
        or "qwen-max"
    ).strip()

    if args.api_key_env:
        key_variable = str(args.api_key_env).strip()
    elif args.endpoint and endpoint.rstrip("/") != configured_endpoint.rstrip("/"):
        key_variable = "OPENAI_API_KEY"
    else:
        key_variable = "DASHSCOPE_API_KEY"
    api_key = str(os.environ.get(key_variable, ""))
    if not api_key.strip():
        raise ValueError(
            f"API key environment variable {key_variable!r} is missing or empty"
        )
    return endpoint, model_name, api_key


def _new_run_dir(output_root: Path, task: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", task).strip("-.")[:48]
    run_dir = output_root.resolve() / f"{timestamp}-{slug or 'task'}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.reuse_tab and not args.edge_cdp:
        parser.error("--reuse-tab requires --edge-cdp")
    if args.benchmark:
        if bool(args.task_id) == bool(args.all):
            parser.error("benchmark mode requires exactly one of --task-id or --all")
        if args.task or args.url:
            parser.error("--task/--url are single-task options; omit them with --benchmark")
    else:
        if args.task_id or args.all:
            parser.error("--task-id/--all require --benchmark")
        if not args.task or not args.task.strip():
            parser.error("--task must not be empty")
        if not args.url:
            parser.error("--url is required outside benchmark mode")
    if args.max_steps < 1:
        parser.error("--max-steps must be at least 1")
    if not (0.0 <= args.delta <= 1.0):
        parser.error("--delta must be between 0.0 and 1.0")
    if not (0.0 <= args.tau <= 10.0):
        parser.error("--tau must be between 0.0 and 10.0")
    if args.k_candidates < 1:
        parser.error("--k-candidates must be at least 1")
    if args.max_expansions < 1:
        parser.error("--max-expansions must be at least 1")
    if args.max_depth < 1:
        parser.error("--max-depth must be at least 1")
    if not (0.0 <= args.eu_threshold <= 1.0) or not (0.0 <= args.au_threshold <= 1.0):
        parser.error("--eu-threshold/--au-threshold must be between 0.0 and 1.0")
    if args.model_timeout <= 0:
        parser.error("--model-timeout must be positive")
    if args.slow_mo < 0 or args.action_wait_ms < 0:
        parser.error("browser delays must be non-negative")
    if args.browser_timeout_ms < 1:
        parser.error("--browser-timeout-ms must be positive")


if __name__ == "__main__":
    raise SystemExit(main())
