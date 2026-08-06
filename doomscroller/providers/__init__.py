"""Provider registry."""

from __future__ import annotations

from typing import Any

from .base import LLMProvider, LLMResult, ProviderUnavailable, strip_reasoning
from .claude import ClaudeProvider
from .nim import NIMProvider

__all__ = [
    "LLMProvider",
    "LLMResult",
    "ProviderUnavailable",
    "build_provider",
    "strip_reasoning",
    "PROVIDERS",
    "DEFAULT_MODELS",
]

PROVIDERS: dict[str, type[LLMProvider]] = {
    "nvidia_nim": NIMProvider,
    "nim": NIMProvider,
    "nvidia": NIMProvider,
    "anthropic": ClaudeProvider,
    "claude": ClaudeProvider,
}

DEFAULT_MODELS = {
    "nvidia_nim": "deepseek-ai/deepseek-v4-flash",
    "anthropic": "claude-opus-5",
}


def build_provider(name: str, options: dict[str, Any] | None = None) -> LLMProvider:
    provider_class = PROVIDERS.get(name.lower().strip())
    if provider_class is None:
        raise ProviderUnavailable(
            f"unknown model provider {name!r}. Known: {', '.join(sorted(set(PROVIDERS)))}"
        )
    return provider_class(**(options or {}))


def default_model_for(provider: str) -> str:
    canonical = PROVIDERS.get(provider.lower().strip())
    if canonical is NIMProvider:
        return DEFAULT_MODELS["nvidia_nim"]
    return DEFAULT_MODELS["anthropic"]
