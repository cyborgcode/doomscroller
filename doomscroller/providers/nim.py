"""NVIDIA NIM provider — OpenAI-compatible, default model DeepSeek V4 Flash.

Three NIM-specific things this handles, none of which are optional:

1. **`chat_template_kwargs` or it hangs.** DeepSeek V4 models on NIM require
   `chat_template_kwargs: {enable_thinking, thinking}` at the root of the
   payload. Omit it and the request doesn't error — it hangs until the timeout,
   which in a nightly cron looks like the bot silently dying. It goes on every
   request here.
2. **`guided_json`, not `response_format`.** NVIDIA recommends `nvext.guided_json`
   for schema-constrained output; it's backed by xgrammar and actually enforces
   the schema, where `response_format: {"type": "json_object"}` permits any
   valid JSON including `{}`.
3. **Reasoning arrives out-of-band.** The thinking lands on `reasoning_content`,
   not in `content`, so the content parses as clean JSON — but self-hosted
   containers vary, so the content is defensively stripped anyway.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from .base import LLMProvider, LLMResult, ProviderUnavailable, strip_reasoning

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "deepseek-ai/deepseek-v4-flash"

EFFORT_MAP = {
    "low": "none",
    "medium": "high",
    "high": "high",
    "xhigh": "max",
    "max": "max",
}
"""This project's five effort levels onto NIM's three `reasoning_effort` values.

`low` maps to `none` deliberately: triage is bulk classification against a very
explicit rubric, which is the one job that doesn't need the model to deliberate.
"""


class NIMProvider(LLMProvider):
    name = "nvidia_nim"
    supports_prompt_caching = False
    """NIM has no prompt-caching equivalent, so the triage system prompt is
    re-sent in full on every batch. That's the main cost difference against
    Claude — see the README."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        thinking: bool = True,
        timeout: float = 300.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY") or os.environ.get("NIM_API_KEY")
        self.base_url = base_url or os.environ.get("NVIDIA_BASE_URL") or DEFAULT_BASE_URL
        self.thinking = thinking
        self.timeout = timeout
        self._client: Any = None

    def _ensure(self) -> Any:
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise ProviderUnavailable(
                "NVIDIA_API_KEY is not set. Get a free key at https://build.nvidia.com "
                "(DeepSeek V4 Flash is free there, rate-limited rather than metered)."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on install
            raise ProviderUnavailable(
                "the openai package is not installed (NIM speaks the OpenAI protocol). "
                "Run: pip install openai"
            ) from exc
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        return self._client

    def complete(
        self,
        *,
        model: str,
        system: str,
        user: str,
        schema: dict[str, Any],
        effort: str,
        max_tokens: int,
        stream: bool = False,
    ) -> LLMResult:
        client = self._ensure()

        extra_body: dict[str, Any] = {
            # Required. Without this the request hangs rather than failing.
            "chat_template_kwargs": {
                "enable_thinking": self.thinking,
                "thinking": self.thinking,
            },
            "nvext": {"guided_json": schema},
        }
        reasoning_effort = EFFORT_MAP.get(effort, "high")

        try:
            response = client.chat.completions.create(
                model=model or DEFAULT_MODEL,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                reasoning_effort=reasoning_effort,
                extra_body=extra_body,
            )
        except Exception as exc:  # noqa: BLE001 - provider errors are wide and varied
            log.warning("NIM request failed: %s", exc)
            return LLMResult(error=f"{type(exc).__name__}: {exc}")

        return _to_result(response)


def _to_result(response: Any) -> LLMResult:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return LLMResult(error="NIM returned no choices")

    choice = choices[0]
    message = getattr(choice, "message", None)
    content = strip_reasoning(getattr(message, "content", "") or "")

    usage = getattr(response, "usage", None)
    result = LLMResult(
        text=content,
        input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        output_tokens=getattr(usage, "completion_tokens", 0) or 0,
    )

    finish = getattr(choice, "finish_reason", "") or ""
    if finish == "content_filter":
        result.refused = True
    elif finish == "length" and not content:
        # The whole budget went to reasoning and nothing came back. Report it as
        # an error rather than an empty answer, so the caller falls back.
        result.error = "response truncated before any content (raise max_tokens or lower effort)"
    elif not content:
        result.error = f"empty content (finish_reason={finish or 'unknown'})"
    return result
