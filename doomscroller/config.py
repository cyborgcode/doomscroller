"""Config loading. One YAML file plus environment variables for secrets."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .providers import default_model_for

DEFAULT_CONFIG_PATHS = (
    Path("config.yaml"),
    Path.home() / ".config" / "doomscroller" / "config.yaml",
)


class ConfigError(RuntimeError):
    pass


@dataclass
class SourceConfig:
    id: str
    platform: str
    enabled: bool = True
    limit: int = 40
    weight: float = 1.0
    """Multiplies every item from this source. Drop below 1.0 for firehoses."""

    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class DeliveryConfig:
    kind: str
    enabled: bool = True
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    provider: str = "nvidia_nim"
    """Which backend reads your feed: `nvidia_nim` or `anthropic`."""

    triage: str = ""
    """Runs once per item. The expensive knob — this is where volume lives.
    Empty means the provider's default model."""

    synthesis: str = ""
    """Runs once per digest, over the survivors."""

    triage_effort: str = "low"
    synthesis_effort: str = "high"
    triage_batch_size: int = 12
    max_tokens: int = 16000
    provider_options: dict[str, Any] = field(default_factory=dict)
    """Passed to the provider constructor — `base_url`, `thinking`, `timeout`."""


@dataclass
class RankingConfig:
    noise_ceiling: float = 0.72
    """Items noisier than this never reach you, whatever else they have going for them."""

    substance_floor: float = 0.25
    headline_count: int = 8
    skim_count: int = 15
    half_life_hours: float = 14.0
    """Freshness decay. A story this old scores half what it would brand new."""

    weights: dict[str, float] = field(
        default_factory=lambda: {
            "interest": 1.0,
            "substance": 0.8,
            "noise": 0.9,
            "freshness": 0.5,
            "engagement": 0.25,
            "source": 0.4,
        }
    )


@dataclass
class Config:
    user_id: str = "default"
    window_hours: int = 24
    db_path: Path = Path("doomscroller.db")
    sources: list[SourceConfig] = field(default_factory=list)
    delivery: list[DeliveryConfig] = field(default_factory=list)
    models: ModelConfig = field(default_factory=ModelConfig)
    ranking: RankingConfig = field(default_factory=RankingConfig)
    interests: list[str] = field(default_factory=list)
    """Seed topics, used before the bot has any feedback to learn from."""

    mute: list[str] = field(default_factory=list)
    """Hard blocks. Any item whose text matches one of these is dropped pre-LLM."""

    @property
    def enabled_sources(self) -> list[SourceConfig]:
        return [source for source in self.sources if source.enabled]

    @property
    def enabled_delivery(self) -> list[DeliveryConfig]:
        return [target for target in self.delivery if target.enabled]


def _source_from_dict(raw: dict[str, Any]) -> SourceConfig:
    platform = raw.get("platform")
    if not platform:
        raise ConfigError(f"source {raw!r} is missing 'platform'")
    known = dict(raw)
    for key in ("id", "platform", "enabled", "limit", "weight"):
        known.pop(key, None)
    return SourceConfig(
        id=raw.get("id") or platform,
        platform=platform,
        enabled=bool(raw.get("enabled", True)),
        limit=int(raw.get("limit", 40)),
        weight=float(raw.get("weight", 1.0)),
        options=known,
    )


def load_config(path: str | Path | None = None) -> Config:
    """Read config from `path`, else the first default location that exists."""
    candidates = [Path(path)] if path else list(DEFAULT_CONFIG_PATHS)
    for candidate in candidates:
        if candidate.exists():
            return _parse(yaml.safe_load(candidate.read_text()) or {}, candidate)
    if path:
        raise ConfigError(f"config file not found: {path}")
    raise ConfigError(
        "no config file found. Copy config.example.yaml to config.yaml to get started."
    )


def _parse(raw: dict[str, Any], origin: Path) -> Config:
    if not isinstance(raw, dict):
        raise ConfigError(f"{origin}: top level must be a mapping")

    config = Config()
    config.user_id = str(raw.get("user_id", os.environ.get("DOOMSCROLLER_USER", "default")))
    config.window_hours = int(raw.get("window_hours", 24))
    config.db_path = Path(raw.get("db_path", "doomscroller.db")).expanduser()
    config.interests = [str(topic) for topic in raw.get("interests", [])]
    config.mute = [str(term).lower() for term in raw.get("mute", [])]

    config.sources = [_source_from_dict(entry) for entry in raw.get("sources", [])]

    for entry in raw.get("delivery", []):
        kind = entry.get("kind")
        if not kind:
            raise ConfigError(f"{origin}: delivery entry {entry!r} is missing 'kind'")
        options = {k: v for k, v in entry.items() if k not in ("kind", "enabled")}
        config.delivery.append(
            DeliveryConfig(kind=kind, enabled=bool(entry.get("enabled", True)), options=options)
        )
    if not config.delivery:
        config.delivery = [DeliveryConfig(kind="console")]

    models = raw.get("models") or {}
    provider = str(models.get("provider", ModelConfig.provider))
    # An unset model means "whatever this provider's default is", so switching
    # providers doesn't require also knowing both model-id spellings.
    fallback_model = default_model_for(provider)
    known_model_keys = {
        "provider",
        "triage",
        "synthesis",
        "triage_effort",
        "synthesis_effort",
        "triage_batch_size",
        "max_tokens",
    }
    config.models = ModelConfig(
        provider=provider,
        triage=str(models.get("triage") or fallback_model),
        synthesis=str(models.get("synthesis") or fallback_model),
        triage_effort=models.get("triage_effort", ModelConfig.triage_effort),
        synthesis_effort=models.get("synthesis_effort", ModelConfig.synthesis_effort),
        triage_batch_size=int(models.get("triage_batch_size", ModelConfig.triage_batch_size)),
        max_tokens=int(models.get("max_tokens", ModelConfig.max_tokens)),
        provider_options={k: v for k, v in models.items() if k not in known_model_keys},
    )

    ranking = raw.get("ranking") or {}
    defaults = RankingConfig()
    weights = dict(defaults.weights)
    weights.update(ranking.get("weights") or {})
    config.ranking = RankingConfig(
        noise_ceiling=float(ranking.get("noise_ceiling", defaults.noise_ceiling)),
        substance_floor=float(ranking.get("substance_floor", defaults.substance_floor)),
        headline_count=int(ranking.get("headline_count", defaults.headline_count)),
        skim_count=int(ranking.get("skim_count", defaults.skim_count)),
        half_life_hours=float(ranking.get("half_life_hours", defaults.half_life_hours)),
        weights={key: float(value) for key, value in weights.items()},
    )
    return config
