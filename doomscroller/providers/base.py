"""The seam between the pipeline and whichever model actually reads your feed.

Both passes need the same three things — a cached-if-possible system prompt, a
user payload, and JSON matching a schema — so that's the whole interface. Every
provider difference (how structured output is requested, how reasoning depth is
spelled, whether prompt caching exists) is a provider's problem, not the
pipeline's.

Providers do not raise for API errors. A failed call comes back as an
`LLMResult` carrying the error, because one bad batch should cost you a few
items, not the whole brief. They raise `ProviderUnavailable` only for the
things no retry can fix: a missing package or a missing API key.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


class ProviderUnavailable(RuntimeError):
    """No SDK, or no credentials. Not retryable — the caller falls back."""


@dataclass
class LLMResult:
    text: str = ""
    refused: bool = False
    error: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    """Prompt tokens served from cache. Always 0 on providers without caching."""

    @property
    def ok(self) -> bool:
        return bool(self.text) and not self.refused and not self.error


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """Remove reasoning scaffolding some deployments inline in the content.

    Reasoning models are supposed to return their thinking in a separate field,
    and the ones here do — but a self-hosted container or a template change can
    put `<think>…</think>` or a markdown fence in the content instead, and then
    a JSON parse fails on output that was otherwise perfectly good. Cheap to
    strip, so strip it rather than lose the batch.
    """
    cleaned = _THINK_BLOCK.sub("", text)
    # An unterminated <think> means the model ran out of room mid-thought;
    # everything after it is reasoning, not answer.
    if "<think>" in cleaned.lower():
        cleaned = cleaned[: cleaned.lower().index("<think>")]
    return _FENCE.sub("", cleaned).strip()


class LLMProvider(ABC):
    """One model backend."""

    name: str = "unknown"
    supports_prompt_caching: bool = False

    @abstractmethod
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
        """Return JSON text conforming to `schema`.

        `effort` is this project's vocabulary (low/medium/high/xhigh/max);
        each provider maps it onto whatever its own API calls the same idea.
        `stream` is a hint for long responses — providers that don't need it
        may ignore it.
        """

    def describe(self) -> str:
        return self.name
