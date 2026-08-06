"""Core data shapes that flow through the pipeline.

Everything a source produces is normalised into an `Item`. The pipeline then
attaches a `Verdict` (what the LLM thought of it), groups items into `Cluster`s
(the same story told by five accounts), and finally emits a `Digest`.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Item:
    """One post, email, comment, video, or article — whatever the platform calls it."""

    source: str
    """Source id from config, e.g. "hackernews" or "reddit:programming"."""

    platform: str
    """Platform family: hackernews, reddit, twitter, youtube, gmail, linkedin, rss."""

    external_id: str
    """Platform-native id. Only unique within a platform."""

    title: str
    body: str = ""
    url: str = ""
    author: str = ""
    published_at: datetime = field(default_factory=_now)

    engagement: int = 0
    """Upvotes, likes, points — whatever the platform counts. Used only as a weak prior."""

    raw: dict[str, Any] = field(default_factory=dict)
    """The untouched payload, kept for debugging and for sources that add extras."""

    @property
    def id(self) -> str:
        """Stable id across runs, so the store can remember what you've already seen."""
        return hashlib.sha1(
            f"{self.platform}\x00{self.external_id}".encode("utf-8", "replace")
        ).hexdigest()[:16]

    @property
    def text(self) -> str:
        """Title plus body, which is what every text-based stage actually wants."""
        return f"{self.title}\n{self.body}".strip()

    def age_hours(self, now: datetime | None = None) -> float:
        now = now or _now()
        published = self.published_at
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        return max(0.0, (now - published).total_seconds() / 3600.0)


@dataclass
class Verdict:
    """What the triage model concluded about a single item."""

    item_id: str
    kind: str = "unknown"
    """One of: news, analysis, opinion, promotion, drama, meme, question, unknown."""

    noise: float = 0.5
    """0 = dense with information, 1 = pure engagement bait. Drives the noise floor."""

    substance: float = 0.5
    """0 = says nothing you couldn't guess, 1 = tells you something new and checkable."""

    claims: list[str] = field(default_factory=list)
    """Standalone factual assertions, each readable without the original post."""

    topics: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    summary: str = ""
    """One line. This is what actually reaches your eyes for most items."""

    @property
    def is_factual(self) -> bool:
        return self.kind in {"news", "analysis"} and bool(self.claims)


@dataclass
class ScoredItem:
    item: Item
    verdict: Verdict
    score: float = 0.0
    reasons: dict[str, float] = field(default_factory=dict)
    """Per-component score contributions, so `why` can explain a ranking."""


@dataclass
class Cluster:
    """One story, however many times it was posted."""

    key: str
    members: list[ScoredItem] = field(default_factory=list)
    headline: str = ""
    body: str = ""
    claims: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)

    @property
    def lead(self) -> ScoredItem:
        return max(self.members, key=lambda m: m.score)

    @property
    def score(self) -> float:
        """Lead item's score, with a modest bump for corroboration across sources."""
        platforms = {m.item.platform for m in self.members}
        return self.lead.score * (1.0 + 0.12 * (len(platforms) - 1))

    @property
    def sources(self) -> list[str]:
        seen: list[str] = []
        for member in self.members:
            if member.item.source not in seen:
                seen.append(member.item.source)
        return seen


@dataclass
class Digest:
    generated_at: datetime = field(default_factory=_now)
    window_hours: int = 24
    clusters: list[Cluster] = field(default_factory=list)
    skimmed: list[ScoredItem] = field(default_factory=list)
    """Made the cut but not the headlines — one line each."""

    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.clusters and not self.skimmed


_WORD = re.compile(r"[a-z0-9][a-z0-9'’\-]+")

_STOPWORDS = frozenset(
    """
    a about after all also am an and any are as at be because been before being but by can could
    did do does doing don for from had has have he her here hers him his how i if in into is it its
    just me more most my no nor not of off on once only or other our out over own said same she
    should so some such than that the their them then there these they this those through to too
    under up very was we were what when where which while who whom why will with would you your
    new news says say get got make made like time day today year years back people first two three
    """.split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens with stopwords and one-character junk removed.

    Shared by dedup and by the interest profile so that the vocabulary the model
    learns from is the same vocabulary used to match new items.
    """
    return [
        word
        for word in _WORD.findall(text.lower())
        if word not in _STOPWORDS and len(word) > 2
    ]
