from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from doomscroller.learn import learn, profile_summary
from doomscroller.models import Verdict
from tests.conftest import make_item, make_scored


# -- store ---------------------------------------------------------------


def test_items_round_trip(store):
    item = make_item()
    store.record_items([item])
    loaded = store.get_item(item.id)
    assert loaded is not None
    assert loaded.title == item.title
    assert loaded.platform == item.platform


def test_known_ids_reports_only_what_was_stored(store):
    item = make_item()
    store.record_items([item])
    assert store.known_ids([item.id, "deadbeef"]) == {item.id}


def test_known_ids_handles_more_than_one_chunk(store):
    items = [make_item(external_id=str(n)) for n in range(600)]
    store.record_items(items)
    assert len(store.known_ids([item.id for item in items])) == 600


def test_recording_the_same_item_twice_updates_engagement(store):
    item = make_item(engagement=10)
    store.record_items([item])
    item.engagement = 250
    store.record_items([item])
    assert store.get_item(item.id).engagement == 250


def test_verdicts_round_trip(store):
    item = make_item()
    store.record_items([item])
    verdict = Verdict(
        item_id=item.id,
        kind="news",
        noise=0.1,
        substance=0.9,
        claims=["a claim"],
        topics=["databases"],
        entities=["Postgres"],
        summary="one line",
    )
    store.record_verdicts([verdict])
    loaded = store.cached_verdicts([item.id])[item.id]
    assert loaded.claims == ["a claim"]
    assert loaded.topics == ["databases"]
    assert loaded.substance == pytest.approx(0.9)


def test_cached_verdicts_is_empty_for_unknown_items(store):
    assert store.cached_verdicts([]) == {}
    assert store.cached_verdicts(["nope"]) == {}


def test_feedback_rejects_unknown_signals(store):
    item = make_item()
    store.record_items([item])
    with pytest.raises(ValueError):
        store.add_feedback(item.id, "shrug")


def test_shown_history_is_queryable(store):
    entry = make_scored()
    store.record_items([entry.item])
    now = datetime.now(timezone.utc)
    store.record_shown(now, [entry], [])
    rows = store.shown_since(now - timedelta(hours=1))
    assert len(rows) == 1 and rows[0]["slot"] == "headline"
    assert store.last_digest_at() is not None


def test_prune_removes_old_items_and_cascades(store):
    item = make_item()
    store.record_items([item])
    store.record_verdicts([Verdict(item_id=item.id)])
    assert store.prune(older_than_days=0) == 1
    assert store.get_item(item.id) is None
    assert store.cached_verdicts([item.id]) == {}


# -- learning ------------------------------------------------------------


def _seed(store, *, title: str, topics: list[str], source: str = "hn", external_id: str = "1"):
    item = make_item(title=title, source=source, external_id=external_id)
    store.record_items([item])
    store.record_verdicts([Verdict(item_id=item.id, topics=topics)])
    return item


def test_learning_is_a_no_op_without_feedback(store):
    assert learn(store).applied == 0


def test_thumbs_up_raises_the_topic_weight(store):
    item = _seed(store, title="Ferrari upgrade package", topics=["formula-1"])
    store.add_feedback(item.id, "up")
    report = learn(store)
    assert report.applied == 1
    assert store.profile("topic")["formula-1"] > 0


def test_thumbs_down_lowers_the_topic_weight(store):
    item = _seed(store, title="Crypto is back", topics=["crypto"])
    store.add_feedback(item.id, "down")
    learn(store)
    assert store.profile("topic")["crypto"] < 0


def test_mute_is_a_stronger_negative_than_a_thumbs_down(store):
    muted = _seed(store, title="Airdrop szn", topics=["crypto"], external_id="1")
    disliked = _seed(store, title="Sportsball recap", topics=["sports"], external_id="2")
    store.add_feedback(muted.id, "mute")
    store.add_feedback(disliked.id, "down")
    learn(store)
    profile = store.profile("topic")
    assert profile["crypto"] < profile["sports"] < 0


def test_feedback_is_applied_exactly_once(store):
    item = _seed(store, title="Postgres 18", topics=["databases"])
    store.add_feedback(item.id, "up")
    learn(store)
    first = store.profile("topic")["databases"]
    assert learn(store).applied == 0
    assert store.profile("topic")["databases"] == pytest.approx(first)


def test_repeated_signals_compound(store):
    for index in range(3):
        item = _seed(store, title="Postgres internals", topics=["databases"], external_id=str(index))
        store.add_feedback(item.id, "up")
        learn(store)
    assert store.profile("topic")["databases"] > 0.5


def test_unreinforced_interests_decay(store):
    liked = _seed(store, title="Postgres internals", topics=["databases"], external_id="1")
    store.add_feedback(liked.id, "up")
    learn(store)
    before = store.profile("topic")["databases"]

    # Later passes about something else should erode the stale weight.
    for index in range(10):
        other = _seed(store, title="Ferrari news", topics=["formula-1"], external_id=f"f{index}")
        store.add_feedback(other.id, "up")
        learn(store)

    assert store.profile("topic")["databases"] < before


def test_source_trust_is_learned(store):
    good = _seed(store, title="Deep dive", topics=["databases"], source="lobsters", external_id="1")
    bad = _seed(store, title="Ragebait", topics=["drama"], source="x:search", external_id="2")
    store.add_feedback(good.id, "up")
    store.add_feedback(bad.id, "down")
    learn(store)
    sources = store.profile("source")
    assert sources["lobsters"] > 0 > sources["x:search"]


def test_profile_summary_splits_liked_and_disliked(store):
    good = _seed(store, title="Postgres internals", topics=["databases"], external_id="1")
    bad = _seed(store, title="Airdrop szn", topics=["crypto"], external_id="2")
    store.add_feedback(good.id, "up")
    store.add_feedback(bad.id, "mute")
    learn(store)

    summary = profile_summary(store)
    assert "databases" in dict(summary["topics_up"])
    assert "crypto" in dict(summary["topics_down"])


def test_a_single_negative_source_appears_in_only_one_list(store):
    item = _seed(store, title="Ragebait", topics=["drama"], source="x:search")
    store.add_feedback(item.id, "down")
    learn(store)

    summary = profile_summary(store)
    assert "x:search" in dict(summary["sources_down"])
    assert "x:search" not in dict(summary["sources_up"])


def test_every_profile_list_is_sign_correct(store):
    good = _seed(store, title="Postgres internals", topics=["databases"], source="lobsters", external_id="1")
    bad = _seed(store, title="Airdrop szn", topics=["crypto"], source="x:search", external_id="2")
    store.add_feedback(good.id, "up")
    store.add_feedback(bad.id, "mute")
    learn(store)

    summary = profile_summary(store)
    for key in ("topics_up", "tokens_up", "sources_up"):
        assert all(weight > 0 for _, weight in summary[key]), key
    for key in ("topics_down", "sources_down"):
        assert all(weight < 0 for _, weight in summary[key]), key
