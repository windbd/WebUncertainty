"""Command-line entry point for the minimal ReAct web agent."""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from WebUncertainty.react_agent.agent import ReActAgent, RunResult
from WebUncertainty.react_agent.benchmark import (
    DEFAULT_DATASET_DIR,
    BenchmarkError,
    TaskRunResult,
    load_tasks,
)
from WebUncertainty.react_agent.benchmark import run_benchmark as run_webvoyager_benchmark
from WebUncertainty.react_agent.browser import BrowserSession
from WebUncertainty.react_agent.model import OpenAICompatibleModel
from WebUncertainty.paths import PROJECT_ROOT, environment_files


_PROJECT_DIR = PROJECT_ROOT
_DEFAULT_ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a minimal read-only ReAct agent in a real browser",
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
    parser.add_argument("--model", default=None, help="Override the model name")
    parser.add_argument(
        "--endpoint",
        default=None,
        help="Override the OpenAI-compatible API endpoint",
    )
    parser.add_argument(
        "--api-key-env",
        default=None,
        help="Environment variable containing the API key (never the key itself)",
    )
    parser.add_argument("--model-timeout", type=float, default=60.0)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--max-text-chars", type=int, default=8_000)
    parser.add_argument("--max-elements", type=int, default=80)
    parser.add_argument(
        "--browser",
        choices=["chromium", "msedge"],
        default="chromium",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Show the browser window (headless by default)",
    )
    parser.add_argument("--slow-mo", type=int, default=0)
    parser.add_argument("--action-wait-ms", type=int, default=500)
    parser.add_argument("--browser-timeout-ms", type=int, default=10_000)
    parser.add_argument(
        "--edge-cdp",
        default=None,
        help=(
            "Attach to an already-running Edge instead of launching a new "
            "browser, e.g. http://127.0.0.1:9222 (reuses its cookies/login; "
            "start Edge with --remote-debugging-port first). Overrides "
            "--browser/--headed/--slow-mo."
        ),
    )
    parser.add_argument(
        "--reuse-tab",
        action="store_true",
        help="With --edge-cdp, act in the most recently active existing tab "
             "instead of opening a new one",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to output/react-agent, or output/webvoyager in benchmark mode",
    )
    return parser


def run(args: argparse.Namespace) -> tuple[RunResult, Path]:
    model = _build_model(args)

    run_dir = _new_run_dir(_output_dir(args), args.task)
    screenshot_dir = run_dir / "screenshots"
    browser = BrowserSession(
        headless=not args.headed,
        browser_name=args.browser,
        slow_mo=args.slow_mo,
        action_wait_ms=args.action_wait_ms,
        timeout_ms=args.browser_timeout_ms,
        cdp_endpoint=args.edge_cdp,
        reuse_tab=args.reuse_tab,
    )

    def save_observation(step: int, _observation) -> None:
        try:
            browser.screenshot(screenshot_dir / f"step-{step:02d}.png")
        except Exception as exc:
            print(f"Step {step}: screenshot failed: {exc}")

    result = RunResult(task=args.task, start_url=args.url)
    try:
        with browser:
            agent = ReActAgent(
                task=args.task,
                start_url=args.url,
                browser=browser,
                model=model,
                max_steps=args.max_steps,
                max_text_chars=args.max_text_chars,
                max_elements=args.max_elements,
                on_observation=save_observation,
                on_step=_print_step,
            )
            result = agent.run()
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
    model = _build_model(args)
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else DEFAULT_DATASET_DIR
    task_ids = None if args.all else args.task_id
    tasks = load_tasks(dataset_dir, task_ids)

    def browser_factory() -> BrowserSession:
        return BrowserSession(
            headless=not args.headed,
            browser_name=args.browser,
            slow_mo=args.slow_mo,
            action_wait_ms=args.action_wait_ms,
            timeout_ms=args.browser_timeout_ms,
            cdp_endpoint=args.edge_cdp,
            reuse_tab=args.reuse_tab,
        )

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
        model=model,
        browser_factory=browser_factory,
        output_dir=_output_dir(args),
        max_steps=args.max_steps,
        max_text_chars=args.max_text_chars,
        max_elements=args.max_elements,
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


def _build_model(args: argparse.Namespace) -> OpenAICompatibleModel:
    endpoint, model_name, api_key = _model_settings(args)
    return OpenAICompatibleModel(
        api_key=api_key,
        endpoint=endpoint,
        model=model_name,
        timeout=args.model_timeout,
    )


def _output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    subdir = "webvoyager" if args.benchmark else "react-agent"
    return _PROJECT_DIR / "output" / subdir


def _print_step(record: dict) -> None:
    action = record.get("action")
    if action:
        target = f" [{action['target']}]" if action.get("target") else ""
        value = f" {action['value']}" if action.get("value") else ""
        print(f"Step {record['step']}: {action['action']}{target}{value}")
        if action.get("reason"):
            print(f"  Reason: {action['reason']}")
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
    if args.max_text_chars < 1:
        parser.error("--max-text-chars must be at least 1")
    if args.max_elements < 1:
        parser.error("--max-elements must be at least 1")
    if args.model_timeout <= 0:
        parser.error("--model-timeout must be positive")
    if args.slow_mo < 0 or args.action_wait_ms < 0:
        parser.error("browser delays must be non-negative")
    if args.browser_timeout_ms < 1:
        parser.error("--browser-timeout-ms must be positive")


if __name__ == "__main__":
    raise SystemExit(main())
