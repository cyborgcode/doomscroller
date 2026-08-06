"""Anthropic provider — kept so you can A/B the two without changing anything else.

Named `claude.py` rather than `anthropic.py` so it can't be confused with the
SDK it imports.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from .base import LLMProvider, LLMResult, ProviderUnavailable, strip_reasoning

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5"


class ClaudeProvider(LLMProvider):
    name = "anthropic"
    supports_prompt_caching = True

    def __init__(self, *, api_key: str | None = None) -> None:
        self.api_key = api_key
        self._client: Any = None

    def _ensure(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on install
            raise ProviderUnavailable("the anthropic package is not installed") from exc
        if not (
            self.api_key
            or os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("ANTHROPIC_AUTH_TOKEN")
        ):
            raise ProviderUnavailable(
                "ANTHROPIC_API_KEY is not set — the digest needs it to read your feed for you."
            )
        self._client = anthropic.Anthropic(api_key=self.api_key) if self.api_key else anthropic.Anthropic()
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
        request: dict[str, Any] = {
            "model": model or DEFAULT_MODEL,
            "max_tokens": max_tokens,
            # Frozen across every batch and every run, so it caches.
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "thinking": {"type": "adaptive"},
            "output_config": {
                "effort": effort,
                "format": {"type": "json_schema", "schema": schema},
            },
            "messages": [{"role": "user", "content": user}],
        }

        try:
            if stream:
                # High effort can run long enough to trip the SDK's
                # non-streaming timeout guard.
                with client.messages.stream(**request) as active:
                    response = active.get_final_message()
            else:
                response = client.messages.create(**request)
        except Exception as exc:  # noqa: BLE001
            log.warning("Anthropic request failed: %s", exc)
            return LLMResult(error=f"{type(exc).__name__}: {exc}")

        usage = getattr(response, "usage", None)
        result = LLMResult(
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cached_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        )
        if response.stop_reason == "refusal":
            result.refused = True
            return result

        result.text = strip_reasoning(
            next((block.text for block in response.content if block.type == "text"), "")
        )
        if not result.text:
            result.error = f"empty content (stop_reason={response.stop_reason})"
        return result
