"""Tests for the distillation layer, with the provider stubbed out.

No API can be called here, but the risky parts are ours: batching, parsing
structured output, and every degradation path. A digest that silently loses
items because the model dropped one from a batch is the failure that matters,
so these cover the seams rather than the happy path alone.

Provider-specific wire-format tests live in `test_providers.py`.
"""

from __future__ import annotations

import json

import pytest

from doomscroller.config import Config, ModelConfig
from doomscroller.models import Cluster
from doomscroller.pipeline.dedup import cluster_items
from doomscroller.pipeline.distill import (
    Distiller,
    DistillerUnavailable,
    apply_summaries,
    heuristic_verdicts,
    scored_from,
)
from doomscroller.providers import LLMResult
from doomscroller.providers.base import LLMProvider
from tests.conftest import make_item, make_scored


class FakeProvider(LLMProvider):
    """Records what it was asked and returns canned results in order."""

    name = "fake"

    def __init__(self, results: list[LLMResult] | None = None) -> None:
        self.results = list(results or [])
        self.calls: list[dict] = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        return self.results.pop(0) if self.results else LLMResult(error="no canned result")


def ok(payload) -> LLMResult:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return LLMResult(text=text, input_tokens=100, output_tokens=50)


def make_distiller(results: list[LLMResult], batch_size: int = 12) -> Distiller:
    config = Config()
    config.models = ModelConfig(
        provider="fake", triage="m", synthesis="m", triage_batch_size=batch_size
    )
    return Distiller(config, provider=FakeProvider(results))


def verdict_payload(items, **overrides):
    return {
        "verdicts": [
            {
                "ref": item.id,
                "kind": overrides.get("kind", "news"),
                "noise": overrides.get("noise", 0.2),
                "substance": overrides.get("substance", 0.8),
                "claims": ["A concrete claim."],
                "topics": ["databases"],
                "entities": ["Postgres"],
                "summary": f"Summary of {item.title}",
            }
            for item in items
        ]
    }


# -- triage --------------------------------------------------------------


def test_triage_parses_verdicts():
    items = [make_item(external_id="1"), make_item(external_id="2", title="Another story")]
    verdicts = make_distiller([ok(verdict_payload(items))]).triage(items)

    assert set(verdicts) == {item.id for item in items}
    assert verdicts[items[0].id].kind == "news"
    assert verdicts[items[0].id].topics == ["databases"]


def test_triage_of_nothing_makes_no_call():
    distiller = make_distiller([])
    assert distiller.triage([]) == {}
    assert distiller.provider.calls == []


def test_triage_batches_by_configured_size():
    items = [make_item(external_id=str(n)) for n in range(7)]
    distiller = make_distiller(
        [ok(verdict_payload(items[:3])), ok(verdict_payload(items[3:6])), ok(verdict_payload(items[6:]))],
        batch_size=3,
    )
    verdicts = distiller.triage(items)

    assert len(distiller.provider.calls) == 3
    assert len(verdicts) == 7


def test_items_the_model_omits_still_get_a_verdict():
    """A dropped item must not vanish from the digest."""
    items = [make_item(external_id="1"), make_item(external_id="2", title="Forgotten story")]
    verdicts = make_distiller([ok(verdict_payload(items[:1]))]).triage(items)

    assert set(verdicts) == {item.id for item in items}
    assert verdicts[items[1].id].kind == "unknown"  # neutral fallback, not a guess


def test_a_failed_batch_falls_back_instead_of_raising():
    items = [make_item(external_id="1")]
    verdicts = make_distiller([LLMResult(error="503 overloaded")]).triage(items)
    assert verdicts[items[0].id].kind == "unknown"


def test_a_refusal_falls_back_rather_than_reading_empty_content():
    items = [make_item(external_id="1")]
    assert make_distiller([LLMResult(refused=True)]).triage(items)[items[0].id].kind == "unknown"


def test_unparseable_output_falls_back():
    items = [make_item(external_id="1")]
    assert make_distiller([ok("not json at all")]).triage(items)[items[0].id].kind == "unknown"


def test_out_of_range_scores_are_clamped():
    items = [make_item(external_id="1")]
    payload = verdict_payload(items, noise=5.0, substance=-3.0)
    verdict = make_distiller([ok(payload)]).triage(items)[items[0].id]
    assert verdict.noise == 1.0
    assert verdict.substance == 0.0


def test_unknown_refs_are_ignored():
    items = [make_item(external_id="1")]
    payload = verdict_payload(items)
    payload["verdicts"].append({**payload["verdicts"][0], "ref": "hallucinated"})
    assert len(make_distiller([ok(payload)]).triage(items)) == 1


def test_token_usage_is_accounted_across_batches():
    items = [make_item(external_id=str(n)) for n in range(4)]
    distiller = make_distiller([ok(verdict_payload(items[:2])), ok(verdict_payload(items[2:]))], batch_size=2)
    distiller.triage(items)
    assert distiller.input_tokens == 200
    assert distiller.output_tokens == 100


def test_failed_batches_still_account_their_tokens():
    """A refusal that burned reasoning tokens must not report as free."""
    items = [make_item(external_id="1")]
    distiller = make_distiller([LLMResult(refused=True, input_tokens=90, output_tokens=400)])
    distiller.triage(items)
    assert distiller.input_tokens == 90
    assert distiller.output_tokens == 400


def test_triage_passes_the_schema_and_low_effort():
    items = [make_item(external_id="1")]
    distiller = make_distiller([ok(verdict_payload(items))])
    distiller.triage(items)
    call = distiller.provider.calls[0]

    assert call["effort"] == "low"
    assert "verdicts" in call["schema"]["properties"]
    assert call["schema"]["properties"]["verdicts"]["items"]["additionalProperties"] is False


def test_an_unknown_provider_is_rejected_clearly():
    config = Config()
    config.models = ModelConfig(provider="mystery-llm")
    with pytest.raises(DistillerUnavailable, match="unknown model provider"):
        Distiller(config)


# -- synthesis -----------------------------------------------------------


def _clusters() -> list[Cluster]:
    return cluster_items([make_scored(make_item(external_id="1", url="https://a.test/1"))])


def test_synthesis_returns_overview_and_rewritten_entries():
    clusters = _clusters()
    payload = {
        "overview": "A quiet day in databases.",
        "entries": [
            {"key": clusters[0].key, "headline": "Postgres 18 lands async I/O", "body": "Two sources agree."}
        ],
    }
    overview, entries = make_distiller([ok(payload)]).synthesize(clusters)

    assert overview == "A quiet day in databases."
    apply_summaries(clusters, entries)
    assert clusters[0].headline == "Postgres 18 lands async I/O"
    assert clusters[0].body == "Two sources agree."


def test_synthesis_asks_for_high_effort_and_streaming():
    clusters = _clusters()
    distiller = make_distiller([ok({"overview": "x", "entries": []})])
    distiller.synthesize(clusters)
    call = distiller.provider.calls[0]
    assert call["effort"] == "high"
    assert call["stream"] is True


def test_synthesis_failure_leaves_the_digest_usable():
    clusters = _clusters()
    original = clusters[0].headline
    overview, entries = make_distiller([LLMResult(error="timeout")]).synthesize(clusters)

    assert overview == "" and entries == {}
    apply_summaries(clusters, entries)
    assert clusters[0].headline == original  # falls back to the item's own title


def test_synthesis_refusal_degrades_quietly():
    assert make_distiller([LLMResult(refused=True)]).synthesize(_clusters()) == ("", {})


def test_synthesis_of_nothing_makes_no_call():
    distiller = make_distiller([])
    assert distiller.synthesize([]) == ("", {})
    assert distiller.provider.calls == []


def test_blank_rewrites_do_not_erase_a_good_headline():
    clusters = _clusters()
    original = clusters[0].headline
    apply_summaries(clusters, {clusters[0].key: ("", "")})
    assert clusters[0].headline == original


# -- heuristic fallback --------------------------------------------------


def test_heuristic_verdicts_are_neutral_not_confident():
    verdict = heuristic_verdicts([make_item()])[make_item().id]
    assert verdict.kind == "unknown"
    assert verdict.claims == []
    assert 0.3 < verdict.noise < 0.7  # asserts no judgement it hasn't made


def test_scored_from_covers_items_with_no_verdict():
    items = [make_item(external_id="1")]
    assert scored_from(items, {})[0].verdict.item_id == items[0].id
