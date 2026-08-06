"""Near-duplicate detection and clustering.

The single biggest source of doomscroll fatigue is the same story arriving five
times. This module collapses those into one cluster before the LLM ever sees
them, which cuts token spend and makes the digest read like a briefing instead
of a feed.

Two passes: exact URL match (cheap, catches most cross-posts), then token
overlap on titles (catches the same story reported by different outlets).
"""

from __future__ import annotations

import re
from urllib.parse import urlparse, urlunparse

from ..models import Cluster, ScoredItem, tokenize

_TRACKING_PARAMS = re.compile(
    r"^(utm_|ref_?|fbclid|gclid|mc_[ce]id|igshid|si|s|source|campaign)", re.IGNORECASE
)


def canonical_url(url: str) -> str:
    """Strip tracking noise so two links to the same article compare equal."""
    if not url:
        return ""
    try:
        parsed = urlparse(url.strip())
    except ValueError:
        return url.strip().lower()
    if not parsed.netloc:
        return url.strip().lower()

    host = parsed.netloc.lower().removeprefix("www.").removeprefix("m.")
    query = "&".join(
        part
        for part in parsed.query.split("&")
        if part and not _TRACKING_PARAMS.match(part.split("=", 1)[0])
    )
    path = parsed.path.rstrip("/") or "/"
    return urlunparse(("https", host, path, "", query, ""))


def similarity(left: set[str], right: set[str]) -> float:
    """Jaccard overlap. Cheap, order-independent, and good enough for headlines."""
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    if not intersection:
        return 0.0
    return intersection / len(left | right)


def trigrams(tokens: set[str]) -> set[str]:
    """Character trigrams over the sorted token string.

    Catches what whole-token matching misses: "stabilises"/"stabilizes",
    reordered headlines, and morphological variants. Measured on sample
    headline pairs, genuine restatements score 0.8+ here while unrelated
    headlines stay under 0.25 — hence the deliberately high threshold at the
    call site.
    """
    joined = " ".join(sorted(tokens))
    return {joined[i : i + 3] for i in range(len(joined) - 2)}


TOKEN_THRESHOLD = 0.42
TRIGRAM_THRESHOLD = 0.62
"""Higher on purpose. Trigrams are a fuzzier signal, so they only get to merge
two items when the evidence is overwhelming.

A note on what this does *not* catch: two headlines that mean the same thing
while sharing almost no surface form ("Postgres 18 ships async I/O" vs
"PostgreSQL 18 released with asynchronous I/O support") stay separate, because
every threshold low enough to merge them also merges genuinely different
stories about the same subject. That trade is deliberate — a duplicate costs
you one extra line, a bad merge hides a story entirely.
"""


def cluster_items(
    scored: list[ScoredItem],
    threshold: float = TOKEN_THRESHOLD,
    trigram_threshold: float = TRIGRAM_THRESHOLD,
) -> list[Cluster]:
    """Group items that tell the same story.

    Three signals, cheapest first: canonical URL match, token overlap, then
    character trigrams for spelling and word-order variants.

    Greedy single-pass assignment against cluster centroids. O(n·k) where k is
    the cluster count — fine for the few hundred items a personal feed produces,
    and it keeps the "best item anchors the cluster" behaviour predictable.
    """
    clusters: list[Cluster] = []
    signatures: list[set[str]] = []
    trigram_signatures: list[set[str]] = []
    urls: list[set[str]] = []

    # Strongest signal first, so the best item anchors each cluster.
    ordered = sorted(scored, key=lambda s: s.score, reverse=True)

    for entry in ordered:
        item = entry.item
        # Title plus the model's one-line summary. Bodies drift into commentary,
        # but the summary is canonicalised phrasing, which helps two outlets
        # describing one event look alike.
        tokens = set(tokenize(f"{item.title} {entry.verdict.summary}"))
        item_trigrams = trigrams(tokens)
        url = canonical_url(item.url)

        placed = False
        for index, cluster in enumerate(clusters):
            matched = (
                (url and url in urls[index])
                or similarity(tokens, signatures[index]) >= threshold
                or similarity(item_trigrams, trigram_signatures[index]) >= trigram_threshold
            )
            if matched:
                cluster.members.append(entry)
                signatures[index] |= tokens
                trigram_signatures[index] |= item_trigrams
                if url:
                    urls[index].add(url)
                placed = True
                break

        if not placed:
            cluster = Cluster(key=item.id, members=[entry], headline=item.title)
            clusters.append(cluster)
            signatures.append(tokens)
            trigram_signatures.append(item_trigrams)
            urls.append({url} if url else set())

    for cluster in clusters:
        _fill_cluster(cluster)

    return sorted(clusters, key=lambda c: c.score, reverse=True)


def _fill_cluster(cluster: Cluster) -> None:
    lead = cluster.lead
    cluster.headline = lead.item.title or lead.verdict.summary
    cluster.body = lead.verdict.summary

    claims: list[str] = []
    topics: list[str] = []
    for member in sorted(cluster.members, key=lambda m: m.score, reverse=True):
        for claim in member.verdict.claims:
            # De-dup claims across members by their token set, not exact text —
            # two outlets rarely phrase the same fact identically.
            key = frozenset(tokenize(claim))
            if key and not any(similarity(key, frozenset(tokenize(seen))) > 0.7 for seen in claims):
                claims.append(claim)
        for topic in member.verdict.topics:
            if topic not in topics:
                topics.append(topic)

    cluster.claims = claims[:6]
    cluster.topics = topics[:6]
