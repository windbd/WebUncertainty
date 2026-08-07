import json
import logging
import math
import os
import time
from urllib.parse import urlsplit

from WebUncertainty.config import _env_model

logger = logging.getLogger(__name__)


class NonRetryableLLMError(RuntimeError):
    """Raised for request errors that another retry/model cannot repair."""


class LLMDeadlineExceeded(TimeoutError):
    """Raised when one ``ask_llm`` call exhausts its total wall-clock budget."""

DEFAULT_MODEL = _env_model("DASHSCOPE_MODEL_NAME")
DEFAULT_ENDPOINT = os.environ.get(
    "DASHSCOPE_ENDPOINT", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)


def _default_timeout() -> float:
    try:
        value = float(os.environ.get("DASHSCOPE_TIMEOUT", "60"))
    except (TypeError, ValueError):
        return 60.0
    return value if math.isfinite(value) and value > 0 else 60.0


DEFAULT_TIMEOUT = _default_timeout()

# ── Lazy singleton OpenAI client ──────────────────────────────────────────────
# Avoids creating a new httpx connection pool on every ask_llm() call.
_client = None
_client_config = None
_runtime_endpoint = None
_runtime_model = None
_runtime_api_key = None


def _normalise_endpoint(endpoint: str) -> str:
    value = str(endpoint or "").strip().rstrip("/")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("LLM endpoint must be an absolute http(s) URL")
    return value


def _endpoint_origin(endpoint: str) -> tuple[str, str, int | None]:
    parsed = urlsplit(endpoint)
    return parsed.scheme.lower(), (parsed.hostname or "").lower(), parsed.port


def _discard_client() -> None:
    """Close the cached connection pool before changing its credentials/host."""
    global _client, _client_config
    old_client = _client
    _client = None
    _client_config = None
    if old_client is not None:
        try:
            old_client.close()
        except Exception:
            logger.debug("Could not close the previous LLM client", exc_info=True)


def configure_runtime(*, endpoint: str | None = None,
                      model: str | None = None,
                      api_key: str | None = None,
                      allow_env_api_key: bool = False) -> None:
    """Set process-local LLM overrides and invalidate the cached client.

    ``endpoint=None`` and ``model=None`` restore the environment-derived
    defaults.  Changing endpoint origin never forwards ``DASHSCOPE_API_KEY``
    implicitly: callers must either supply the key explicitly or acknowledge
    reuse with ``allow_env_api_key=True``.  This keeps a CLI endpoint override
    from accidentally disclosing the DashScope credential to an unrelated
    host.
    """
    global _runtime_endpoint, _runtime_model, _runtime_api_key

    resolved_endpoint = (
        _normalise_endpoint(endpoint) if endpoint is not None else None
    )
    if (
        resolved_endpoint is not None
        and _endpoint_origin(resolved_endpoint)
        != _endpoint_origin(_normalise_endpoint(DEFAULT_ENDPOINT))
        and api_key is None
        and not allow_env_api_key
    ):
        raise ValueError(
            "Refusing to reuse DASHSCOPE_API_KEY for a different endpoint; "
            "pass api_key=... or allow_env_api_key=True explicitly"
        )

    resolved_model = str(model).strip() if model is not None else None
    if model is not None and not resolved_model:
        raise ValueError("LLM model override must not be empty")

    _runtime_endpoint = resolved_endpoint
    _runtime_model = resolved_model
    _runtime_api_key = (
        str(api_key)
        if api_key is not None
        else (
            os.environ.get("DASHSCOPE_API_KEY", "")
            if resolved_endpoint is not None and allow_env_api_key
            else None
        )
    )
    _discard_client()


def get_runtime_model() -> str:
    """Return the model selected by the current process-local configuration."""
    return _runtime_model or DEFAULT_MODEL


def _get_client():
    global _client, _client_config
    endpoint = _runtime_endpoint or _normalise_endpoint(DEFAULT_ENDPOINT)
    api_key = (
        _runtime_api_key
        if _runtime_api_key is not None
        else os.environ.get("DASHSCOPE_API_KEY", "")
    )
    client_config = (endpoint, api_key, DEFAULT_TIMEOUT)
    if _client is None or _client_config != client_config:
        _discard_client()
        from openai import OpenAI
        _client = OpenAI(
            api_key=api_key,
            base_url=endpoint,
            timeout=DEFAULT_TIMEOUT,
            # ask_llm owns the one total deadline and retry budget.  The SDK's
            # default two hidden retries would otherwise exceed both.
            max_retries=0,
        )
        _client_config = client_config
    return _client


# ── Token usage tracking ──────────────────────────────────────────────────────
# ``calls`` remains the accepted-response counter for backward compatibility.
# Token totals account for every completed response carrying provider usage,
# including late or invalid-JSON responses that may still be billable.
_token_stats = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "calls": 0,
    "attempts": 0,
    "responses": 0,
    "late_responses": 0,
    "invalid_json": 0,
    "failed_attempts": 0,
}


def get_token_stats() -> dict:
    """Return billable usage plus accepted-call and request-attempt counts."""
    return dict(_token_stats)


def reset_token_stats() -> None:
    """Reset token usage counters (e.g., at the start of a new task)."""
    for key in _token_stats:
        _token_stats[key] = 0


def _record_response_usage(response) -> None:
    usage = getattr(response, "usage", None)
    if not usage:
        return
    _token_stats["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
    _token_stats["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
    _token_stats["total_tokens"] += getattr(usage, "total_tokens", 0) or 0


def ask_llm(prompt: str, system: str = None, is_json: bool = True,
            model: str | None = None,
            temperature: float = 0.3, max_retries: int = 3,
            timeout: float | None = None) -> "str | dict":
    """Call the configured model via DashScope's compatible endpoint.

    Reads DASHSCOPE_API_KEY / DASHSCOPE_ENDPOINT / DASHSCOPE_MODEL_NAME from
    WebUncertainty/.env (with the parent directory retained as a fallback).

    When ``system`` is provided it is sent as a system message before the user
    turn, which improves instruction-following on chat-optimised models.

    Returns parsed JSON when ``is_json`` is true, otherwise response text.
    """
    budget = DEFAULT_TIMEOUT if timeout is None else float(timeout)
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError("timeout must be a positive finite number")
    try:
        attempts = int(max_retries)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_retries must be a positive integer") from exc
    if attempts < 1 or attempts != max_retries:
        raise ValueError("max_retries must be a positive integer")

    deadline = time.monotonic() + budget
    client = _get_client()
    effective_model = _runtime_model or model or DEFAULT_MODEL

    messages: list = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    call_kwargs = dict(
        model=effective_model,
        messages=messages,
        temperature=temperature,
    )
    if is_json:
        call_kwargs["response_format"] = {"type": "json_object"}

    last_error = None
    attempts_made = 0
    for attempt in range(attempts):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise LLMDeadlineExceeded(
                f"LLM call exceeded {budget:g}s deadline after "
                f"{attempts_made} attempts"
            ) from last_error
        try:
            request_kwargs = dict(call_kwargs)
            # Never round this upward: even a 100 ms floor can overrun a small
            # remaining MCTS budget.  DEFAULT_TIMEOUT remains a per-attempt cap.
            request_kwargs["timeout"] = min(DEFAULT_TIMEOUT, remaining)
            logger.info("Calling %s (attempt %d/%d)",
                        effective_model, attempt + 1, attempts)
            attempts_made += 1
            _token_stats["attempts"] += 1
            response = client.chat.completions.create(**request_kwargs)
            _token_stats["responses"] += 1
            _record_response_usage(response)
            if time.monotonic() >= deadline:
                _token_stats["late_responses"] += 1
                raise LLMDeadlineExceeded(
                    f"LLM response arrived after the {budget:g}s deadline"
                )
            content = response.choices[0].message.content
            parsed = json.loads(content) if is_json else None
            if time.monotonic() >= deadline:
                _token_stats["late_responses"] += 1
                raise LLMDeadlineExceeded(
                    f"LLM response processing exceeded the {budget:g}s deadline"
                )

            _token_stats["calls"] += 1

            logger.info("LLM response received (call #%d)", _token_stats["calls"])
            return parsed if is_json else content
        except LLMDeadlineExceeded:
            raise
        except json.JSONDecodeError as exc:
            _token_stats["invalid_json"] += 1
            last_error = exc
            logger.warning("LLM returned invalid JSON (attempt %d/%d): %s",
                           attempt + 1, attempts, exc)
            if attempt < attempts - 1:
                _sleep_with_deadline(1.0, deadline)
            continue
        except Exception as exc:
            _token_stats["failed_attempts"] += 1
            last_error = exc
            logger.warning("LLM call failed (attempt %d/%d): %s",
                           attempt + 1, attempts, exc)
            if _is_non_retryable(exc):
                raise NonRetryableLLMError(str(exc)) from exc
            if attempt < attempts - 1:
                _sleep_with_deadline(float(2 ** attempt), deadline)

    if time.monotonic() >= deadline:
        raise LLMDeadlineExceeded(
            f"LLM call exceeded {budget:g}s deadline after {attempts_made} attempts"
        ) from last_error
    raise RuntimeError(f"LLM call failed after {attempts_made} attempts") from last_error


def _sleep_with_deadline(delay: float, deadline: float | None) -> None:
    remaining = deadline - time.monotonic()
    if remaining > 0:
        time.sleep(min(delay, remaining))


def _is_non_retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and 400 <= status < 500 and status not in {
        408, 409, 429,
    }
