from __future__ import annotations

from datetime import timezone

import pytest

from doomscroller.models import Item, tokenize
from doomscroller.pipeline.dedup import canonical_url, cluster_items, similarity
from doomscroller.pipeline.rank import Ranker, apply_mutes, mute_items
from doomscroller.sources.base import parse_timestamp, pick, unwrap_records, within_window
from tests.conftest import make_item, make_scored


# -- tokenizing ----------------------------------------------------------


def test_tokenize_drops_stopwords_and_short_words():
    assert tokenize("The new Postgres is a database") == ["postgres", "database"]


def test_tokenize_is_case_insensitive():
    assert tokenize("KAFKA Kafka kafka") == ["kafka"] * 3


# -- url canonicalisation ------------------------------------------------


@pytest.mark.parametrize(
    "left,right",
    [
        ("https://example.com/a?utm_source=x", "http://www.example.com/a"),
        ("https://example.com/a/", "https://example.com/a"),
        ("https://m.example.com/a?fbclid=abc", "https://example.com/a"),
    ],
)
def test_canonical_url_collapses_equivalent_links(left, right):
    assert canonical_url(left) == canonical_url(right)


def test_canonical_url_keeps_meaningful_query_params():
    assert "v=abc" in canonical_url("https://youtube.com/watch?v=abc&utm_medium=social")


def test_canonical_url_survives_garbage():
    assert canonical_url("") == ""
    assert canonical_url("not a url") == "not a url"


# -- similarity ----------------------------------------------------------


def test_similarity_bounds():
    assert similarity(set(), {"a"}) == 0.0
    assert similarity({"a", "b"}, {"a", "b"}) == 1.0
    assert 0 < similarity({"a", "b"}, {"b", "c"}) < 1


# -- clustering ----------------------------------------------------------


def test_same_url_across_platforms_becomes_one_cluster():
    scored = [
        make_scored(make_item(external_id="1", platform="hackernews", url="https://x.com/a?utm_source=hn")),
        make_scored(
            make_item(
                title="Async I/O lands in Postgres 18",
                external_id="2",
                platform="reddit",
                source="reddit:rust",
                url="https://www.x.com/a",
            ),
            score=0.8,
        ),
    ]
    clusters = cluster_items(scored)
    assert len(clusters) == 1
    assert len(clusters[0].members) == 2
    assert len(clusters[0].sources) == 2


def test_similar_headlines_cluster_without_a_shared_url():
    scored = [
        make_scored(make_item(title="Postgres 18 ships asynchronous I/O", external_id="1", url="https://a.test/1")),
        make_scored(
            make_item(
                title="Postgres 18 ships asynchronous I/O support",
                external_id="2",
                platform="reddit",
                url="https://b.test/2",
            ),
            score=0.5,
        ),
    ]
    assert len(cluster_items(scored)) == 1


def test_spelling_and_word_order_variants_cluster_via_trigrams():
    scored = [
        make_scored(
            make_item(title="Rust 1.90 stabilises async closures", external_id="1", url="https://a.test/1"),
            summary="Rust 1.90 stabilises async closures.",
        ),
        make_scored(
            make_item(title="Rust 1.90 stabilizes async closures", external_id="2", url="https://b.test/2"),
            summary="Rust 1.90 stabilizes async closures.",
            score=0.5,
        ),
    ]
    assert len(cluster_items(scored)) == 1


@pytest.mark.parametrize(
    "left,right",
    [
        # Same subject, different story. Merging these would hide one of them.
        ("Postgres 18 released", "Postgres security advisory CVE-2026-1"),
        ("A deep dive into Kafka partitioning", "A deep dive into Redis persistence"),
        ("Ferrari signs a driver for 2027", "Mercedes announces a power unit for 2027"),
    ],
)
def test_same_subject_different_story_is_not_merged(left, right):
    scored = [
        make_scored(make_item(title=left, external_id="1", url="https://a.test/1"), summary=left),
        make_scored(
            make_item(title=right, external_id="2", url="https://b.test/2"), summary=right, score=0.5
        ),
    ]
    assert len(cluster_items(scored)) == 2


def test_unrelated_stories_stay_apart():
    scored = [
        make_scored(make_item(title="Postgres 18 ships asynchronous I/O", external_id="1", url="https://a.test/1")),
        make_scored(
            make_item(
                title="Ferrari signs a new driver for next season",
                external_id="2",
                url="https://b.test/2",
            ),
            score=0.5,
        ),
    ]
    assert len(cluster_items(scored)) == 2


def test_cluster_lead_is_the_highest_scoring_member():
    low = make_scored(make_item(external_id="1", url="https://a.test/x"), score=0.2)
    high = make_scored(make_item(external_id="2", url="https://a.test/x"), score=0.9)
    cluster = cluster_items([low, high])[0]
    assert cluster.lead is high


def test_corroboration_across_platforms_lifts_cluster_score():
    solo = cluster_items([make_scored(make_item(external_id="1", url="https://a.test/x"), score=1.0)])[0]
    pair = cluster_items(
        [
            make_scored(make_item(external_id="1", url="https://a.test/x"), score=1.0),
            make_scored(
                make_item(external_id="2", platform="reddit", source="r/db", url="https://a.test/x"),
                score=0.4,
            ),
        ]
    )[0]
    assert pair.score > solo.score


def test_cluster_deduplicates_near_identical_claims():
    claim = "Postgres 18 ships asynchronous I/O on Linux."
    reworded = "Postgres 18 ships asynchronous I/O on Linux systems."
    cluster = cluster_items(
        [
            make_scored(make_item(external_id="1", url="https://a.test/x"), claims=[claim], score=1.0),
            make_scored(
                make_item(external_id="2", platform="reddit", url="https://a.test/x"),
                claims=[reworded],
                score=0.5,
            ),
        ]
    )[0]
    assert len(cluster.claims) == 1


# -- ranking -------------------------------------------------------------


def test_noisy_items_are_filtered_out(config, store):
    ranker = Ranker(config, store)
    kept, reason = ranker.survives(make_scored(noise=0.95))
    assert not kept and "noise" in reason


def test_thin_items_are_filtered_out(config, store):
    ranker = Ranker(config, store)
    kept, reason = ranker.survives(make_scored(substance=0.05))
    assert not kept and "substance" in reason


def test_promotion_needs_real_substance(config, store):
    ranker = Ranker(config, store)
    assert not ranker.survives(make_scored(kind="promotion", substance=0.4))[0]
    assert ranker.survives(make_scored(kind="promotion", substance=0.8))[0]


def test_substantive_news_survives(config, store):
    assert Ranker(config, store).survives(make_scored())[0]


def test_fresher_items_outrank_stale_ones(config, store):
    fresh = make_scored(make_item(external_id="1", age_hours=1))
    stale = make_scored(make_item(external_id="2", age_hours=72))
    ranked = Ranker(config, store).score([stale, fresh])
    assert ranked[0].item.external_id == "1"


def test_learned_topic_interest_changes_the_order(config, store):
    store.bump_profile("topic", {"formula-1": 3.0})
    liked = make_scored(make_item(title="Ferrari upgrade package", external_id="1"), topics=["formula-1"])
    other = make_scored(make_item(title="Postgres release notes", external_id="2"), topics=["databases"])
    ranked = Ranker(config, store).score([other, liked])
    assert ranked[0].item.external_id == "1"


def test_source_weight_scales_the_final_score(config, store):
    config.sources[0].weight = 0.5
    entry = make_scored(make_item(source="hn"))
    full = Ranker(config, store).score([make_scored(make_item(source="other"))])[0].score
    halved = Ranker(config, store).score([entry])[0].score
    assert halved == pytest.approx(full * 0.5, rel=0.02)


def test_reasons_explain_the_score(config, store):
    entry = Ranker(config, store).score([make_scored()])[0]
    assert set(entry.reasons) == {
        "interest",
        "substance",
        "noise",
        "freshness",
        "engagement",
        "source",
    }
    assert entry.score == pytest.approx(sum(entry.reasons.values()), rel=1e-6)


def test_an_unknown_item_scores_neutral_interest(config, store):
    config.interests = []
    entry = make_scored(make_item(title="Something the profile has never seen"), topics=["novel"])
    assert Ranker(config, store).interest(entry) == pytest.approx(0.5)


def test_interest_spread_is_wide_enough_to_outrank_freshness(config, store):
    """A learned preference has to actually move things, or personalisation is decorative."""
    config.interests = []
    store.bump_profile("topic", {"databases": 1.0, "drama": -1.0})
    ranker = Ranker(config, store)

    liked = ranker.interest(make_scored(topics=["databases"]))
    disliked = ranker.interest(make_scored(topics=["drama"]))
    assert liked > 0.6 and disliked < 0.4

    # The gap must exceed what the freshness component can swing (weight 0.5),
    # otherwise a slightly newer disliked item always wins.
    freshness_swing = config.ranking.weights["freshness"]
    assert (liked - disliked) * config.ranking.weights["interest"] > freshness_swing * 0.5


def test_a_young_profile_still_discriminates(config, store):
    """Small absolute weights must not flatten the signal — the scale is relative."""
    config.interests = []
    store.bump_profile("topic", {"databases": 0.05, "drama": -0.05})
    ranker = Ranker(config, store)
    assert ranker.interest(make_scored(topics=["databases"])) > ranker.interest(
        make_scored(topics=["drama"])
    ) + 0.15


def test_one_runaway_weight_does_not_flatten_the_rest(config, store):
    config.interests = []
    store.bump_profile("topic", {"obsession": 50.0, "databases": 1.0, "drama": -1.0})
    ranker = Ranker(config, store)
    assert ranker.interest(make_scored(topics=["databases"])) > ranker.interest(
        make_scored(topics=["drama"])
    ) + 0.1


def test_quiet_items_are_rewarded_not_merely_unpunished(config, store):
    ranker = Ranker(config, store)
    quiet = ranker.score([make_scored(noise=0.05)])[0]
    loud = ranker.score([make_scored(noise=0.65)])[0]
    assert quiet.reasons["noise"] > loud.reasons["noise"] > 0


# -- mutes ---------------------------------------------------------------


def test_mute_items_runs_before_triage():
    items = [make_item(title="The 10x engineer myth"), make_item(title="Postgres 18", external_id="2")]
    kept, dropped = mute_items(items, ["10x engineer"])
    assert dropped == 1 and len(kept) == 1


def test_mute_also_matches_assigned_topics():
    kept, dropped = apply_mutes([make_scored(topics=["crypto pump"])], ["crypto pump"])
    assert dropped == 1 and kept == []


def test_no_mute_terms_is_a_no_op():
    items = [make_item()]
    assert mute_items(items, []) == (items, 0)


# -- source helpers ------------------------------------------------------


def test_parse_timestamp_handles_the_formats_providers_actually_send():
    assert parse_timestamp(1700000000).year == 2023
    assert parse_timestamp(1700000000000).year == 2023  # milliseconds
    assert parse_timestamp("2024-03-01T12:00:00Z").month == 3
    assert parse_timestamp("Fri, 01 Mar 2024 12:00:00 GMT").month == 3


def test_parse_timestamp_falls_back_to_now_rather_than_raising():
    assert parse_timestamp("nonsense").tzinfo is timezone.utc


def test_unwrap_records_digs_through_envelopes():
    assert unwrap_records({"items": [{"id": 1}]}) == [{"id": 1}]
    assert unwrap_records({"data": {"results": [{"id": 2}]}}) == [{"id": 2}]
    assert unwrap_records([{"id": 3}]) == [{"id": 3}]
    assert unwrap_records({"title": "solo"}) == [{"title": "solo"}]
    assert unwrap_records(None) == []
    assert unwrap_records({"nothing": "here"}) == []


def test_pick_supports_fallbacks_and_dotted_paths():
    payload = {"a": "", "b": "found", "nested": {"deep": "value"}}
    assert pick(payload, "a", "b") == "found"
    assert pick(payload, "nested.deep") == "value"
    assert pick(payload, "missing", default="fallback") == "fallback"


def test_within_window_drops_stale_items():
    fresh = make_item(external_id="1", age_hours=1)
    stale = make_item(external_id="2", age_hours=50)
    assert within_window([fresh, stale], 24) == [fresh]


def test_item_id_is_stable_and_platform_scoped():
    one = Item(source="a", platform="hackernews", external_id="42", title="t")
    two = Item(source="b", platform="hackernews", external_id="42", title="different")
    three = Item(source="a", platform="reddit", external_id="42", title="t")
    assert one.id == two.id  # same platform + external id = same thing
    assert one.id != three.id  # different platform = different thing
