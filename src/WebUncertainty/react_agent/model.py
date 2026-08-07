"""OpenAI-compatible model adapter for the ReAct agent."""

from __future__ import annotations

import json
import re
from typing import Any

from openai import OpenAI


class ModelError(RuntimeError):
    """Raised when the model does not return a usable action object."""


class OpenAICompatibleModel:
    """Request one JSON decision from an OpenAI-compatible endpoint."""

    def __init__(
        self,
        *,
        api_key: str,
        endpoint: str,
        model: str,
        timeout: float = 60.0,
    ) -> None:
        if not api_key.strip():
            raise ValueError("model API key is empty")
        if not endpoint.strip():
            raise ValueError("model endpoint is empty")
        if not model.strip():
            raise ValueError("model name is empty")
        if timeout <= 0:
            raise ValueError("model timeout must be positive")
        self.model = model.strip()
        self._client = OpenAI(
            api_key=api_key.strip(),
            base_url=endpoint.rstrip("/"),
            timeout=timeout,
        )

    def decide(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
        except Exception as exc:
            raise ModelError(f"model request failed: {exc}") from exc

        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise ModelError("model returned empty content")
        return parse_json_object(content)


def parse_json_object(content: str) -> dict[str, Any]:
    """Parse a JSON object, tolerating a single Markdown JSON fence."""
    value = content.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", value, re.DOTALL)
    if fence:
        value = fence.group(1).strip()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ModelError(f"model output is not valid JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise ModelError("model output must be one JSON object")
    return parsed
