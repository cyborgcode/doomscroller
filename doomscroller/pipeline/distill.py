"""The part that does the actual reading for you.

Two passes, both through whichever provider is configured (see
`doomscroller/providers/`):

1. **Triage** — every new item, in batches, gets classified and compressed to a
   single line plus its checkable claims. This is the volume pass, so it runs at
   low effort, and on providers that support it the system prompt is cached.
2. **Synthesis** — the survivors get written up as a brief. One call per digest,
   at high effort.

Triage results are cached in the store keyed by item id, so a post that appears
in tomorrow's window too is never paid for twice. That matters more on a
provider without prompt caching, where the system prompt is re-sent per batch.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Iterable

from ..config import Config
from ..models import Cluster, Item, ScoredItem, Verdict
from ..providers import LLMProvider, ProviderUnavailable, build_provider

log = logging.getLogger(__name__)

KINDS = ["news", "analysis", "opinion", "promotion", "drama", "meme", "question", "unknown"]

TRIAGE_SYSTEM = """\
You are the filter between a person and their social media feeds. They are tired \
of scrolling. Your job is to read raw feed items and decide, item by item, whether \
there is anything in there worth a human's attention — and if so, to state it \
plainly enough that they never need to open the original.

For each item you receive, produce:

- `kind`: what the item fundamentally is.
  - `news` — reports something that happened, with specifics.
  - `analysis` — explains or interprets, and teaches something checkable.
  - `opinion` — a take. May be interesting, but asserts rather than establishes.
  - `promotion` — selling, launching, hiring, self-marketing, affiliate content.
  - `drama` — interpersonal conflict, dunks, pile-ons, outrage cycles.
  - `meme` — jokes, shitposts, reaction content.
  - `question` — someone asking rather than telling.
  - `unknown` — genuinely cannot tell from what you were given.

- `noise` (0.0-1.0): how much of this is engagement machinery rather than \
information. Rage bait, vague-posting, "thread 🧵", manufactured urgency, \
recycled takes, and headlines that withhold the answer all score high. A dry \
changelog entry scores near zero even if it is boring.

- `substance` (0.0-1.0): how much a reader would actually learn. Specific \
numbers, named entities, dated events, concrete technical detail, and \
first-hand accounts score high. Restatements of common knowledge, predictions \
with no reasoning, and pure sentiment score low.

- `claims`: the standalone factual assertions, at most four. Each must read \
correctly on its own, without the headline for context — write "Postgres 18 \
ships asynchronous I/O on Linux", not "it adds async I/O". Include figures and \
dates where the item gives them. If the item asserts nothing checkable, return \
an empty list. Do not invent claims the item does not make, and do not soften \
or strengthen what it claims.

- `topics`: 1-4 lowercase subject tags, reusable across items (`databases`, \
`ai-policy`, `formula-1`). Prefer an existing-sounding tag over a novel one.

- `entities`: named people, companies, products, or places that the item is \
actually about. Skip incidental mentions.

- `summary`: one sentence, under 30 words, in plain declarative English. This \
is what the reader sees instead of the post, so it must carry the point rather \
than tease it. No "this post discusses", no ellipses, no clickbait framing.

Judge each item on its own merits. Being popular is not evidence of substance; \
being unpopular is not evidence of its absence. Where an item is ambiguous, \
prefer the less flattering classification — a false `news` costs the reader more \
than a false `opinion`.
"""

TRIAGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "ref": {"type": "string", "description": "The item's ref, copied exactly."},
                    "kind": {"type": "string", "enum": KINDS},
                    "noise": {"type": "number"},
                    "substance": {"type": "number"},
                    "claims": {"type": "array", "items": {"type": "string"}},
                    "topics": {"type": "array", "items": {"type": "string"}},
                    "entities": {"type": "array", "items": {"type": "string"}},
                    "summary": {"type": "string"},
                },
                "required": [
                    "ref",
                    "kind",
                    "noise",
                    "substance",
                    "claims",
                    "topics",
                    "entities",
                    "summary",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["verdicts"],
    "additionalProperties": False,
}

SYNTHESIS_SYSTEM = """\
You write one person's daily brief. They have not looked at their feeds and do \
not intend to; this is the whole of what they will read.

You are given clusters of related items that already survived filtering, each \
with a headline, the sources that carried it, and the factual claims extracted \
from them. Write the brief.

Rules:

- Lead with what happened. The first clause of each entry carries the news.
- Merge across sources. If four accounts covered one story, that is one entry \
that mentions the corroboration, not four entries.
- Use only the claims you were given. You may connect and contextualise them, \
but do not add facts from your own knowledge, and do not resolve a contradiction \
between sources by picking a side — say the sources disagree.
- Where sources conflict or a claim rests on a single unverified account, say so \
in the entry. The reader is relying on you to flag thin sourcing.
- Two to four sentences per entry. Specifics over adjectives.
- Skip the throat-clearing. No "in today's news", no summary of the summary.
- If the day was genuinely quiet, say that plainly rather than inflating minor \
items into headlines.

Then write a two-sentence `overview` naming the through-line of the day, if \
there is one. If there isn't, say what the day was mostly about and move on.
"""

SYNTHESIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "overview": {"type": "string"},
        "entries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "The cluster key, copied exactly."},
                    "headline": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["key", "headline", "body"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["overview", "entries"],
    "additionalProperties": False,
}


# Kept as an alias so callers that catch DistillerUnavailable keep working;
# the providers raise the same condition under its own name.
DistillerUnavailable = ProviderUnavailable


class Distiller:
    """This project's two prompts, pointed at whichever provider is configured."""

    def __init__(self, config: Config, provider: LLMProvider | None = None) -> None:
        self.config = config
        self.models = config.models
        self.provider = provider or build_provider(
            config.models.provider, config.models.provider_options
        )
        self.input_tokens = 0
        self.output_tokens = 0
        self.cached_tokens = 0

    def _account(self, result: Any) -> None:
        self.input_tokens += result.input_tokens
        self.output_tokens += result.output_tokens
        self.cached_tokens += result.cached_tokens

    # -- pass 1: triage --------------------------------------------------

    def triage(self, items: list[Item]) -> dict[str, Verdict]:
        """Classify and compress items. Returns verdicts keyed by item id."""
        if not items:
            return {}
        verdicts: dict[str, Verdict] = {}

        for batch in _batches(items, self.models.triage_batch_size):
            by_ref = {item.id: item for item in batch}
            payload = json.dumps(
                [
                    {
                        "ref": item.id,
                        "source": item.source,
                        "author": item.author,
                        "age_hours": round(item.age_hours(), 1),
                        "engagement": item.engagement,
                        "title": item.title,
                        "body": item.body[:1500],
                    }
                    for item in batch
                ],
                ensure_ascii=False,
            )

            result = self.provider.complete(
                model=self.models.triage,
                system=TRIAGE_SYSTEM,
                user=f"Triage these {len(batch)} feed items.\n\n{payload}",
                schema=TRIAGE_SCHEMA,
                effort=self.models.triage_effort,
                max_tokens=self.models.max_tokens,
            )
            self._account(result)

            if not result.ok:
                log.warning(
                    "triage batch of %d degraded to heuristics: %s",
                    len(batch),
                    "refused" if result.refused else result.error,
                )
                verdicts.update({item.id: _fallback_verdict(item) for item in batch})
                continue

            parsed = _parse_json(result.text)
            for entry in (parsed or {}).get("verdicts", []):
                item = by_ref.get(str(entry.get("ref", "")))
                if item is None:
                    continue
                verdicts[item.id] = Verdict(
                    item_id=item.id,
                    kind=str(entry.get("kind", "unknown")),
                    noise=_clamp(entry.get("noise", 0.5)),
                    substance=_clamp(entry.get("substance", 0.5)),
                    claims=[str(c) for c in entry.get("claims", [])][:4],
                    topics=[str(t).lower() for t in entry.get("topics", [])][:4],
                    entities=[str(e) for e in entry.get("entities", [])][:6],
                    summary=str(entry.get("summary", "")).strip(),
                )

            # The model dropped some items from the batch; don't lose them silently.
            for item in batch:
                verdicts.setdefault(item.id, _fallback_verdict(item))

        return verdicts

    # -- pass 2: synthesis -----------------------------------------------

    def synthesize(self, clusters: list[Cluster]) -> tuple[str, dict[str, tuple[str, str]]]:
        """Write the brief. Returns (overview, {cluster_key: (headline, body)})."""
        if not clusters:
            return "", {}

        payload = json.dumps(
            [
                {
                    "key": cluster.key,
                    "headline": cluster.headline,
                    "sources": cluster.sources,
                    "source_count": len(cluster.members),
                    "topics": cluster.topics,
                    "claims": cluster.claims,
                    "summaries": [m.verdict.summary for m in cluster.members[:4] if m.verdict.summary],
                }
                for cluster in clusters
            ],
            ensure_ascii=False,
        )

        result = self.provider.complete(
            model=self.models.synthesis,
            system=SYNTHESIS_SYSTEM,
            user=(
                f"Write today's brief from these {len(clusters)} story clusters, "
                f"covering roughly the last {self.config.window_hours} hours.\n\n{payload}"
            ),
            schema=SYNTHESIS_SCHEMA,
            effort=self.models.synthesis_effort,
            max_tokens=self.models.max_tokens,
            stream=True,
        )
        self._account(result)

        if not result.ok:
            log.warning(
                "synthesis degraded to per-item summaries: %s",
                "refused" if result.refused else result.error,
            )
            return "", {}

        parsed = _parse_json(result.text) or {}
        entries = {
            str(entry.get("key", "")): (
                str(entry.get("headline", "")).strip(),
                str(entry.get("body", "")).strip(),
            )
            for entry in parsed.get("entries", [])
        }
        return str(parsed.get("overview", "")).strip(), entries


# -- helpers -------------------------------------------------------------


def _batches(items: list[Item], size: int) -> Iterable[list[Item]]:
    size = max(1, size)
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _clamp(value: Any, low: float = 0.0, high: float = 1.0) -> float:
    try:
        return max(low, min(high, float(value)))
    except (TypeError, ValueError):
        return 0.5


def _parse_json(text: str) -> dict[str, Any] | None:
    """Parse a structured-output body, tolerating a provider that wrapped it."""
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        log.warning("model returned unparseable JSON (%d chars)", len(text))
        return None
    return parsed if isinstance(parsed, dict) else None


def _fallback_verdict(item: Item) -> Verdict:
    """Heuristic stand-in when the model is unavailable or dropped an item.

    Neutral on purpose: it should let an item through to ranking without
    pretending to a judgement it hasn't made.
    """
    return Verdict(
        item_id=item.id,
        kind="unknown",
        noise=0.5,
        substance=0.4,
        summary=(item.title or item.body)[:180],
        topics=[],
    )


def heuristic_verdicts(items: list[Item]) -> dict[str, Verdict]:
    """Verdicts with no LLM at all — used by `--no-llm` and by the test suite."""
    return {item.id: _fallback_verdict(item) for item in items}


def apply_summaries(clusters: list[Cluster], overview_entries: dict[str, tuple[str, str]]) -> None:
    """Overwrite cluster headline/body with the synthesised versions where present."""
    for cluster in clusters:
        written = overview_entries.get(cluster.key)
        if not written:
            continue
        headline, body = written
        if headline:
            cluster.headline = headline
        if body:
            cluster.body = body


def scored_from(items: list[Item], verdicts: dict[str, Verdict]) -> list[ScoredItem]:
    return [
        ScoredItem(item=item, verdict=verdicts.get(item.id) or _fallback_verdict(item))
        for item in items
    ]
