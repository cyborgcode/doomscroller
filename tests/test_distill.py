"""Tests for the Claude-facing layer, with the SDK stubbed out.

The real API can't be called here, but the risky parts are ours: batching,
parsing structured output, and every degradation path. A digest that silently
loses items because the model dropped one from a batch is the failure that
matters, so these cover the seams rather than the happy path alone.
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
from tests.conftest import make_item, make_scored


class FakeBlock:
    type = "text"

    def __init__(self, text: str) -> None:
        self.text = text


class FakeUsage:
    def __init__(self, cached: int = 0) -> None:
        self.input_tokens = 100
        self.output_tokens = 50
        self.cache_read_input_tokens = cached


class FakeResponse:
    def __init__(self, payload, stop_reason: str = "end_turn", cached: int = 0) -> None:
        text = payload if isinstance(payload, str) else json.dumps(payload)
        self.content = [FakeBlock(text)]
        self.stop_reason = stop_reason
        self.usage = FakeUsage(cached)


class FakeStream:
    def __init__(self, response: FakeResponse) -> None:
        self._response = response

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self._response


class FakeMessages:
    def __init__(self, responses: list, error: Exception | None = None) -> None:
        self.responses = list(responses)
        self.error = error
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.responses.pop(0)

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return FakeStream(self.responses.pop(0))


class FakeClient:
    def __init__(self, responses: list, error: Exception | None = None) -> None:
        self.messages = FakeMessages(responses, error)


def make_distiller(responses: list, error: Exception | None = None, batch_size: int = 12):
    config = Config()
    config.models = ModelConfig(triage_batch_size=batch_size)
    distiller = Distiller(config)
    distiller._client = FakeClient(responses, error)
    return distiller


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
    distiller = make_distiller([FakeResponse(verdict_payload(items))])
    verdicts = distiller.triage(items)

    assert set(verdicts) == {item.id for item in items}
    assert verdicts[items[0].id].kind == "news"
    assert verdicts[items[0].id].topics == ["databases"]


def test_triage_of_nothing_makes_no_api_call():
    distiller = make_distiller([])
    assert distiller.triage([]) == {}
    assert distiller._client.messages.calls == []


def test_triage_batches_by_configured_size():
    items = [make_item(external_id=str(n)) for n in range(7)]
    responses = [
        FakeResponse(verdict_payload(items[:3])),
        FakeResponse(verdict_payload(items[3:6])),
        FakeResponse(verdict_payload(items[6:])),
    ]
    distiller = make_distiller(responses, batch_size=3)
    verdicts = distiller.triage(items)

    assert len(distiller._client.messages.calls) == 3
    assert len(verdicts) == 7


def test_items_the_model_omits_still_get_a_verdict():
    """A dropped item must not vanish from the digest."""
    items = [make_item(external_id="1"), make_item(external_id="2", title="Forgotten story")]
    distiller = make_distiller([FakeResponse(verdict_payload(items[:1]))])
    verdicts = distiller.triage(items)

    assert set(verdicts) == {item.id for item in items}
    assert verdicts[items[1].id].kind == "unknown"  # neutral fallback, not a guess


def test_a_failed_batch_falls_back_instead_of_raising():
    items = [make_item(external_id="1")]
    distiller = make_distiller([], error=RuntimeError("503 overloaded"))
    verdicts = distiller.triage(items)
    assert verdicts[items[0].id].kind == "unknown"


def test_a_refusal_falls_back_rather_than_reading_empty_content():
    items = [make_item(external_id="1")]
    distiller = make_distiller([FakeResponse({"verdicts": []}, stop_reason="refusal")])
    assert distiller.triage(items)[items[0].id].kind == "unknown"


def test_unparseable_output_falls_back():
    items = [make_item(external_id="1")]
    distiller = make_distiller([FakeResponse("not json at all")])
    assert distiller.triage(items)[items[0].id].kind == "unknown"


def test_out_of_range_scores_are_clamped():
    items = [make_item(external_id="1")]
    payload = verdict_payload(items, noise=5.0, substance=-3.0)
    verdict = make_distiller([FakeResponse(payload)]).triage(items)[items[0].id]
    assert verdict.noise == 1.0
    assert verdict.substance == 0.0


def test_unknown_refs_are_ignored():
    items = [make_item(external_id="1")]
    payload = verdict_payload(items)
    payload["verdicts"].append({**payload["verdicts"][0], "ref": "hallucinated"})
    assert len(make_distiller([FakeResponse(payload)]).triage(items)) == 1


def test_token_usage_is_accounted():
    items = [make_item(external_id="1")]
    distiller = make_distiller([FakeResponse(verdict_payload(items), cached=80)])
    distiller.triage(items)
    assert distiller.input_tokens == 100
    assert distiller.output_tokens == 50
    assert distiller.cached_tokens == 80


def test_triage_request_is_shaped_for_caching_and_structured_output():
    items = [make_item(external_id="1")]
    distiller = make_distiller([FakeResponse(verdict_payload(items))])
    distiller.triage(items)
    call = distiller._client.messages.calls[0]

    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert call["output_config"]["effort"] == "low"
    assert call["thinking"] == {"type": "adaptive"}


def test_missing_credentials_raise_a_clear_error(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    with pytest.raises(DistillerUnavailable, match="ANTHROPIC_API_KEY"):
        Distiller(Config()).triage([make_item()])


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
    overview, entries = make_distiller([FakeResponse(payload)]).synthesize(clusters)

    assert overview == "A quiet day in databases."
    apply_summaries(clusters, entries)
    assert clusters[0].headline == "Postgres 18 lands async I/O"
    assert clusters[0].body == "Two sources agree."


def test_synthesis_uses_streaming():
    clusters = _clusters()
    distiller = make_distiller([FakeResponse({"overview": "x", "entries": []})])
    distiller.synthesize(clusters)
    assert distiller._client.messages.calls[0]["output_config"]["effort"] == "high"


def test_synthesis_failure_leaves_the_digest_usable():
    clusters = _clusters()
    original = clusters[0].headline
    overview, entries = make_distiller([], error=RuntimeError("timeout")).synthesize(clusters)

    assert overview == "" and entries == {}
    apply_summaries(clusters, entries)
    assert clusters[0].headline == original  # falls back to the item's own title


def test_synthesis_refusal_degrades_quietly():
    clusters = _clusters()
    distiller = make_distiller([FakeResponse({"overview": "x", "entries": []}, stop_reason="refusal")])
    assert distiller.synthesize(clusters) == ("", {})


def test_synthesis_of_nothing_makes_no_api_call():
    distiller = make_distiller([])
    assert distiller.synthesize([]) == ("", {})
    assert distiller._client.messages.calls == []


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
