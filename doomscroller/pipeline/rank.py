"""Scoring and filtering.

The score is a weighted sum of six components, all normalised to roughly 0-1 so
the weights in config mean what they look like they mean. Each component's
contribution is kept on `ScoredItem.reasons`, which is what `doomscroller why`
prints — a ranker you can't interrogate is one you can't tune.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from ..config import Config, RankingConfig
from ..models import Item, ScoredItem, tokenize
from ..store import Store


class Ranker:
    def __init__(self, config: Config, store: Store) -> None:
        self.config = config
        self.ranking: RankingConfig = config.ranking
        self.token_profile = store.profile("token")
        self.topic_profile = store.profile("topic")
        self.source_profile = store.profile("source")

        # Seed interests act as a starter profile so the very first digest is
        # already pointed roughly the right way. They stay weaker than anything
        # learned from real feedback, and are overridden as the profile fills in.
        for topic in config.interests:
            for token in tokenize(topic):
                self.token_profile.setdefault(token, 0.6)
            self.topic_profile.setdefault(topic.lower().strip(), 0.6)

        # Scale against the profile's own weight distribution, not a constant.
        # A fixed divisor would leave a young profile — whose weights are all
        # small — permanently flat, so nothing you taught it could ever outweigh
        # freshness. Relative scaling makes the component behave the same on
        # day one as on day one hundred.
        self._token_scale = _distribution_scale(self.token_profile)
        self._topic_scale = _distribution_scale(self.topic_profile)

    # -- components ------------------------------------------------------

    def interest(self, entry: ScoredItem) -> float:
        """How well this matches what you've rewarded before.

        Returns 0.5 for an item the profile has nothing to say about, above for
        one it likes, below for one it doesn't — so a neutral item is neither
        rewarded nor punished.
        """
        tokens = set(tokenize(entry.item.text)) | set(tokenize(entry.verdict.summary))
        if not tokens:
            return 0.5

        hits = [self.token_profile[token] for token in tokens if token in self.token_profile]
        topic_hits = [
            self.topic_profile[topic]
            for topic in entry.verdict.topics
            if topic in self.topic_profile
        ]

        # Mean of the strongest matches, not the sum: a long post shouldn't
        # outrank a short one just by containing more words. The floor of 3
        # stops a single incidental match from saturating the score.
        token_score = (
            sum(sorted(hits, key=abs, reverse=True)[:8]) / max(3, len(hits)) if hits else 0.0
        )
        topic_score = sum(topic_hits) / len(topic_hits) if topic_hits else 0.0

        # Topics are the higher-signal channel — they're the model's own
        # judgement of subject, not incidental vocabulary.
        raw = 0.4 * (token_score / self._token_scale) + 0.6 * (topic_score / self._topic_scale)
        return _squash(raw)

    def freshness(self, entry: ScoredItem, now: datetime) -> float:
        half_life = max(1.0, self.ranking.half_life_hours)
        return 0.5 ** (entry.item.age_hours(now) / half_life)

    def engagement(self, entry: ScoredItem) -> float:
        """Log-scaled and capped. Popularity is a weak prior, not a verdict."""
        return min(1.0, math.log10(max(1, entry.item.engagement) ) / 4.0)

    def source_bias(self, entry: ScoredItem) -> float:
        return _squash(self.source_profile.get(entry.item.source, 0.0))

    # -- combination -----------------------------------------------------

    def score(self, entries: list[ScoredItem]) -> list[ScoredItem]:
        now = datetime.now(timezone.utc)
        weights = self.ranking.weights
        source_weights = {source.id: source.weight for source in self.config.sources}

        for entry in entries:
            components = {
                "interest": self.interest(entry),
                "substance": entry.verdict.substance,
                # Noise is a penalty: a quiet item should gain here, not merely
                # avoid losing, so that substance and quiet are both rewarded.
                "noise": 1.0 - entry.verdict.noise,
                "freshness": self.freshness(entry, now),
                "engagement": self.engagement(entry),
                "source": self.source_bias(entry),
            }
            entry.reasons = {
                name: round(weights.get(name, 0.0) * value, 4)
                for name, value in components.items()
            }
            total = sum(entry.reasons.values())
            entry.score = round(total * source_weights.get(entry.item.source, 1.0), 4)

        return sorted(entries, key=lambda e: e.score, reverse=True)

    # -- filtering -------------------------------------------------------

    def survives(self, entry: ScoredItem) -> tuple[bool, str]:
        """The noise floor. Returns (kept, reason-if-dropped)."""
        verdict = entry.verdict
        if verdict.noise > self.ranking.noise_ceiling:
            return False, f"noise {verdict.noise:.2f} > ceiling {self.ranking.noise_ceiling:.2f}"
        if verdict.substance < self.ranking.substance_floor:
            return False, f"substance {verdict.substance:.2f} < floor {self.ranking.substance_floor:.2f}"
        if verdict.kind in {"promotion", "meme"} and verdict.substance < 0.6:
            return False, f"{verdict.kind} without substance"
        if verdict.kind == "drama" and self.interest(entry) < 0.55:
            # Drama gets through only if it's about something you've shown you
            # care about — otherwise it's the exact thing you asked to be spared.
            return False, "drama outside your interests"
        return True, ""


def mute_items(items: list[Item], mute_terms: list[str]) -> tuple[list[Item], int]:
    """Drop muted items before triage, so muted noise costs no tokens at all."""
    if not mute_terms:
        return items, 0
    kept = [item for item in items if not _muted(item.text, mute_terms)]
    return kept, len(items) - len(kept)


def apply_mutes(entries: list[ScoredItem], mute_terms: list[str]) -> tuple[list[ScoredItem], int]:
    """Second mute pass, now that topics exist — catches what the text didn't say."""
    if not mute_terms:
        return entries, 0
    kept = [
        entry
        for entry in entries
        if not _muted(" ".join(entry.verdict.topics + entry.verdict.entities), mute_terms)
    ]
    return kept, len(entries) - len(kept)


def _muted(text: str, mute_terms: list[str]) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in mute_terms)


SQUASH_STEEPNESS = 3.0
"""Chosen so a topic you've clearly signalled on lands near 0.85 (or 0.15 if you
muted it) against a neutral 0.5 — wide enough that the interest weight can
actually outrank freshness, and bounded so no single weight runs away."""


def _squash(value: float) -> float:
    """Map an unbounded profile weight into 0-1 without a hard cutoff."""
    return 1.0 / (1.0 + math.exp(-SQUASH_STEEPNESS * value))


def _distribution_scale(profile: dict[str, float]) -> float:
    """The median |weight|, floored so a near-empty profile is safe.

    Median rather than max or an upper percentile: with a handful of weights an
    upper percentile *is* the outlier, so one runaway interest would divide
    everything else down to nothing. The median tracks the typical weight, which
    is what "how strong is this preference, relatively?" actually means.
    """
    magnitudes = sorted(abs(weight) for weight in profile.values())
    if not magnitudes:
        return 1.0
    return max(0.05, magnitudes[len(magnitudes) // 2])
