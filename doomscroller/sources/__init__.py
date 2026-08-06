"""Source registry and the collection loop."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from ..config import Config, SourceConfig
from ..models import Item
from .base import BaseSource, Source, SourceError, within_window
from .composio_client import ComposioClient
from .platforms import COMPOSIO_SOURCES
from .rss import RSSSource

log = logging.getLogger(__name__)

__all__ = [
    "ComposioClient",
    "Source",
    "SourceError",
    "build_source",
    "collect",
]


def build_source(config: SourceConfig, client: ComposioClient) -> BaseSource:
    platform = config.platform.lower()
    if platform == "rss":
        return RSSSource(config)
    source_class = COMPOSIO_SOURCES.get(platform)
    if source_class is None:
        raise SourceError(
            f"{config.id}: unknown platform {config.platform!r}. "
            f"Known: rss, {', '.join(sorted(COMPOSIO_SOURCES))}"
        )
    return source_class(config, client)


def collect(config: Config, client: ComposioClient) -> tuple[list[Item], list[str]]:
    """Fetch every enabled source in parallel.

    Sources are independent and mostly network-bound, so they run concurrently;
    a source that fails is reported and skipped rather than aborting the run.
    Returns the items and a list of human-readable failure messages.
    """
    sources: list[BaseSource] = []
    errors: list[str] = []

    for source_config in config.enabled_sources:
        try:
            sources.append(build_source(source_config, client))
        except SourceError as exc:
            errors.append(str(exc))

    if not sources:
        return [], errors

    items: list[Item] = []
    with ThreadPoolExecutor(max_workers=min(8, len(sources))) as pool:
        futures = {pool.submit(_fetch_one, source, config.window_hours): source for source in sources}
        for future in futures:
            source = futures[future]
            try:
                fetched = future.result()
            except SourceError as exc:
                errors.append(str(exc))
                continue
            except Exception as exc:  # noqa: BLE001 - never let one source kill the run
                errors.append(f"{source.id}: unexpected {type(exc).__name__}: {exc}")
                continue
            log.info("%s: %d items", source.id, len(fetched))
            items.extend(fetched)

    return items, errors


def _fetch_one(source: BaseSource, window_hours: int) -> list[Item]:
    # Sources filter by window where the API supports it; this enforces it for
    # the ones that don't, so `window_hours` means the same thing everywhere.
    return within_window(source.fetch(window_hours), window_hours)
