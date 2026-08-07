import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

from WebUncertainty.paths import environment_files


def _load_environment() -> None:
    """Load local configuration without overriding explicit process variables.

    Prefer the project-local file and retain the workspace location as a
    backwards-compatible fallback.
    """
    for path in environment_files():
        if path.is_file():
            load_dotenv(path, override=False)


_load_environment()


def _env_model(var: str, *, default: str = "qwen-max",
               inherit_base: bool = True) -> str:
    # Accept both the documented *_MODEL_NAME variables and the shorter names
    # commonly used in DashScope examples and existing local .env files.
    aliases = {
        "DASHSCOPE_MODEL_NAME": "DASHSCOPE_MODEL",
    }
    configured = os.environ.get(var) or os.environ.get(aliases.get(var, ""))
    if configured:
        return configured
    if inherit_base:
        base_model = (
            os.environ.get("DASHSCOPE_MODEL_NAME")
            or os.environ.get("DASHSCOPE_MODEL")
        )
        if base_model:
            return base_model
    return default


@dataclass
class Config:
    """Model settings for the ReAct WebVoyager agent, read from ``.env``."""

    MODEL: str = field(default_factory=lambda: _env_model("DASHSCOPE_MODEL_NAME"))
    # The WebVoyager answer judge is deliberately independent of MODEL so a
    # generic DASHSCOPE_MODEL setting cannot silently change the grader.
    MODEL_TASK_EVAL: str = field(
        default_factory=lambda: _env_model(
            "DASHSCOPE_MODEL_TASK_EVAL",
            default="qwen3.7-plus",
            inherit_base=False,
        )
    )
    TEMPERATURE: float = 0.0
    MAX_RETRIES: int = 3
    TIMEOUT: float = 60.0
