"""The feedback loop — how the bot gets better at knowing what you want.

Feedback is collected as signals against item ids (`up`, `down`, `save`, `mute`,
`open`, `skip`) and folded into three weight tables:

* `token`  — vocabulary from the item's text
* `topic`  — the tags the triage model assigned
* `source` — which feeds are earning their place

Each fold decays the existing weights slightly before adding the new evidence,
so the profile tracks what you care about *now* rather than accumulating
everything you ever clicked. Interests you stop reinforcing fade out on their
own, which is the difference between a profile that improves and one that
merely calcifies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .models import tokenize
from .store import Store

log = logging.getLogger(__name__)

DECAY = 0.97
"""Applied to every stored weight per learning pass. ~23 passes to halve."""

TOKEN_RATE = 0.10
TOPIC_RATE = 0.35
SOURCE_RATE = 0.15
MAX_TOKENS_PER_ITEM = 12
"""Only the most distinctive tokens count, so common words don't drift upward."""


@dataclass
class LearningReport:
    applied: int = 0
    tokens_changed: int = 0
    topics_changed: int = 0
    sources_changed: int = 0

    def __str__(self) -> str:
        if not self.applied:
            return "no new feedback to learn from"
        return (
            f"learned from {self.applied} signal(s): "
            f"{self.tokens_changed} tokens, {self.topics_changed} topics, "
            f"{self.sources_changed} sources updated"
        )


def learn(store: Store) -> LearningReport:
    """Fold all unapplied feedback into the interest profile."""
    pending = store.pending_feedback()
    report = LearningReport()
    if not pending:
        return report

    token_deltas: dict[str, float] = {}
    topic_deltas: dict[str, float] = {}
    source_deltas: dict[str, float] = {}

    verdicts = store.cached_verdicts([str(row["item_id"]) for row in pending])

    for row in pending:
        weight = float(row["weight"])
        item_id = str(row["item_id"])
        text = f"{row['title']} {row['body']}"

        # Frequency-weighted within the item, so a word the post is *about*
        # counts more than one it mentions once in passing.
        counts: dict[str, int] = {}
        for token in tokenize(text):
            counts[token] = counts.get(token, 0) + 1
        ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:MAX_TOKENS_PER_ITEM]
        for token, count in ranked:
            token_deltas[token] = token_deltas.get(token, 0.0) + weight * TOKEN_RATE * min(count, 3)

        verdict = verdicts.get(item_id)
        if verdict:
            for topic in verdict.topics:
                topic_deltas[topic] = topic_deltas.get(topic, 0.0) + weight * TOPIC_RATE
            # A muted item's topics are a strong negative — stronger than a
            # thumbs-down, which might just be one bad post on a good subject.
            if row["signal"] == "mute":
                for topic in verdict.topics:
                    topic_deltas[topic] = topic_deltas.get(topic, 0.0) + weight * TOPIC_RATE

        source = str(row["source"])
        source_deltas[source] = source_deltas.get(source, 0.0) + weight * SOURCE_RATE

    store.bump_profile("token", token_deltas, decay=DECAY)
    store.bump_profile("topic", topic_deltas, decay=DECAY)
    store.bump_profile("source", source_deltas, decay=DECAY)
    store.mark_feedback_applied([int(row["id"]) for row in pending])

    report.applied = len(pending)
    report.tokens_changed = len(token_deltas)
    report.topics_changed = len(topic_deltas)
    report.sources_changed = len(source_deltas)
    log.info("%s", report)
    return report


def profile_summary(store: Store, limit: int = 10) -> dict[str, list[tuple[str, float]]]:
    """What the bot currently believes you like and dislike.

    Each list is sign-filtered, so a single weakly-negative source can't show up
    under both "earning their place" and "on thin ice".
    """

    def liked(kind: str) -> list[tuple[str, float]]:
        return [row for row in store.top_profile(kind, limit, positive=True) if row[1] > 0]

    def disliked(kind: str) -> list[tuple[str, float]]:
        return [row for row in store.top_profile(kind, limit, positive=False) if row[1] < 0]

    return {
        "topics_up": liked("topic"),
        "topics_down": disliked("topic"),
        "tokens_up": liked("token"),
        "sources_up": liked("source"),
        "sources_down": disliked("source"),
    }
