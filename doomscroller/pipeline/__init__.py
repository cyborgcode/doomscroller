"""The end-to-end run: collect → mute → triage → score → filter → cluster → write."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from ..config import Config
from ..models import Digest
from ..sources import ComposioClient, collect
from ..store import Store
from .dedup import cluster_items
from .distill import Distiller, DistillerUnavailable, apply_summaries, heuristic_verdicts, scored_from
from .rank import Ranker, apply_mutes, mute_items

log = logging.getLogger(__name__)

__all__ = ["run", "Digest"]


def run(
    config: Config,
    store: Store,
    *,
    use_llm: bool = True,
    include_seen: bool = False,
) -> Digest:
    """Build one digest.

    `include_seen` re-processes items from previous runs — useful when tuning
    ranking, since it lets you re-rank yesterday's feed without refetching it.
    """
    started = datetime.now(timezone.utc)
    stats: dict[str, object] = {}

    client = ComposioClient(user_id=config.user_id)
    items, errors = collect(config, client)
    stats["fetched"] = len(items)
    stats["tool_calls"] = client.calls
    stats["source_errors"] = errors
    for message in errors:
        log.warning("source failed: %s", message)

    items, muted = mute_items(items, config.mute)
    stats["muted"] = muted

    # Cross-source duplicates arrive as separate items with the same id only
    # when they share a platform; the rest are caught later by clustering.
    unique: dict[str, object] = {}
    deduped = []
    for item in items:
        if item.id not in unique:
            unique[item.id] = item
            deduped.append(item)
    items = deduped

    if not include_seen:
        already = store.known_ids([item.id for item in items])
        fresh = [item for item in items if item.id not in already]
        stats["already_seen"] = len(items) - len(fresh)
        # Everything gets recorded, including repeats, so engagement counts stay current.
        store.record_items(items)
        items = fresh
    else:
        store.record_items(items)
        stats["already_seen"] = 0

    stats["new"] = len(items)
    if not items:
        return Digest(window_hours=config.window_hours, stats=stats)

    # Triage, reusing anything we've already paid to classify.
    cached = store.cached_verdicts([item.id for item in items])
    needs_triage = [item for item in items if item.id not in cached]
    stats["verdicts_cached"] = len(cached)
    stats["verdicts_new"] = len(needs_triage)

    distiller = Distiller(config)
    if use_llm and needs_triage:
        try:
            fresh_verdicts = distiller.triage(needs_triage)
        except DistillerUnavailable as exc:
            log.warning("%s — falling back to heuristics", exc)
            fresh_verdicts = heuristic_verdicts(needs_triage)
            stats["llm"] = f"unavailable: {exc}"
        else:
            store.record_verdicts(list(fresh_verdicts.values()))
    else:
        fresh_verdicts = heuristic_verdicts(needs_triage)
        if not use_llm:
            stats["llm"] = "disabled (--no-llm)"

    verdicts = {**cached, **fresh_verdicts}
    entries = scored_from(items, verdicts)

    entries, topic_muted = apply_mutes(entries, config.mute)
    stats["muted"] = int(stats["muted"]) + topic_muted

    ranker = Ranker(config, store)
    entries = ranker.score(entries)

    survivors = []
    dropped: dict[str, int] = {}
    for entry in entries:
        kept, reason = ranker.survives(entry)
        if kept:
            survivors.append(entry)
        else:
            label = reason.split(" ", 1)[0]
            dropped[label] = dropped.get(label, 0) + 1
    stats["filtered_out"] = sum(dropped.values())
    stats["filter_reasons"] = dropped

    clusters = cluster_items(survivors)
    headline_clusters = clusters[: config.ranking.headline_count]

    # Skim items: best of the rest, one per cluster so the tail doesn't repeat
    # a story that already has a headline.
    skimmed = [cluster.lead for cluster in clusters[config.ranking.headline_count :]]
    skimmed = skimmed[: config.ranking.skim_count]

    overview = ""
    if use_llm and headline_clusters:
        try:
            overview, written = distiller.synthesize(headline_clusters)
            apply_summaries(headline_clusters, written)
        except DistillerUnavailable as exc:
            log.warning("synthesis unavailable: %s", exc)

    digest = Digest(
        generated_at=started,
        window_hours=config.window_hours,
        clusters=headline_clusters,
        skimmed=skimmed,
        stats=stats,
    )
    digest.stats["overview"] = overview
    digest.stats["clusters"] = len(clusters)
    digest.stats["input_tokens"] = distiller.input_tokens
    digest.stats["output_tokens"] = distiller.output_tokens
    digest.stats["cached_tokens"] = distiller.cached_tokens
    digest.stats["elapsed_seconds"] = round(
        (datetime.now(timezone.utc) - started).total_seconds(), 1
    )

    store.record_shown(
        started, [cluster.lead for cluster in headline_clusters], skimmed
    )
    return digest
