from __future__ import annotations

import pytest

from doomscroller.config import ConfigError, load_config
from doomscroller.models import Digest
from doomscroller.pipeline.dedup import cluster_items
from doomscroller.render import to_html, to_markdown, to_terminal
from doomscroller.sources import build_source
from doomscroller.sources.composio_client import ComposioClient, _as_dict
from doomscroller.sources.platforms import HackerNewsSource, RedditSource
from tests.conftest import make_item, make_scored


# -- config --------------------------------------------------------------


def _write(tmp_path, body: str):
    path = tmp_path / "config.yaml"
    path.write_text(body)
    return path


def test_minimal_config_gets_sensible_defaults(tmp_path):
    config = load_config(_write(tmp_path, "sources:\n  - platform: hackernews\n"))
    assert config.window_hours == 24
    assert config.sources[0].id == "hackernews"
    assert config.delivery[0].kind == "console"  # nothing configured -> console
    assert config.models.provider == "nvidia_nim"
    assert config.models.triage == "deepseek-ai/deepseek-v4-flash"


def test_switching_provider_switches_the_default_models(tmp_path):
    """You shouldn't have to know both providers' model-id spellings to switch."""
    config = load_config(_write(tmp_path, "models:\n  provider: anthropic\nsources: []\n"))
    assert config.models.triage == "claude-opus-5"
    assert config.models.synthesis == "claude-opus-5"


def test_an_explicit_model_overrides_the_provider_default(tmp_path):
    config = load_config(
        _write(tmp_path, "models:\n  provider: nvidia_nim\n  triage: deepseek-ai/deepseek-v4-pro\nsources: []\n")
    )
    assert config.models.triage == "deepseek-ai/deepseek-v4-pro"
    assert config.models.synthesis == "deepseek-ai/deepseek-v4-flash"  # untouched default


def test_unknown_model_keys_become_provider_options(tmp_path):
    """So a self-hosted NIM container is a config change, not a code change."""
    config = load_config(
        _write(tmp_path, "models:\n  provider: nim\n  base_url: http://localhost:8000/v1\nsources: []\n")
    )
    assert config.models.provider_options == {"base_url": "http://localhost:8000/v1"}


def test_unknown_source_keys_become_options(tmp_path):
    config = load_config(
        _write(tmp_path, "sources:\n  - platform: reddit\n    subreddit: rust\n    sort: top\n")
    )
    assert config.sources[0].options == {"subreddit": "rust", "sort": "top"}


def test_ranking_weights_merge_over_defaults(tmp_path):
    config = load_config(
        _write(tmp_path, "ranking:\n  weights:\n    interest: 2.5\nsources: []\n")
    )
    assert config.ranking.weights["interest"] == 2.5
    assert config.ranking.weights["noise"] == 0.9  # untouched default survives


def test_source_without_platform_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="platform"):
        load_config(_write(tmp_path, "sources:\n  - id: broken\n"))


def test_missing_config_file_is_reported_clearly(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_disabled_sources_are_excluded(tmp_path):
    config = load_config(
        _write(
            tmp_path,
            "sources:\n  - platform: hackernews\n  - platform: reddit\n    subreddit: x\n    enabled: false\n",
        )
    )
    assert [source.id for source in config.enabled_sources] == ["hackernews"]


# -- sources -------------------------------------------------------------


def test_reddit_source_requires_a_subreddit(config):
    from doomscroller.config import SourceConfig
    from doomscroller.sources.base import SourceError

    with pytest.raises(SourceError, match="subreddit"):
        build_source(SourceConfig(id="r", platform="reddit"), ComposioClient())


def test_unknown_platform_is_rejected():
    from doomscroller.config import SourceConfig
    from doomscroller.sources.base import SourceError

    with pytest.raises(SourceError, match="unknown platform"):
        build_source(SourceConfig(id="x", platform="myspace"), ComposioClient())


def test_config_can_override_the_tool_slug():
    from doomscroller.config import SourceConfig

    source = build_source(
        SourceConfig(id="hn", platform="hackernews", options={"slug": "CUSTOM_SLUG"}),
        ComposioClient(),
    )
    assert source.slug == "CUSTOM_SLUG"


def test_hackernews_record_maps_onto_an_item():
    from doomscroller.config import SourceConfig

    source = HackerNewsSource(SourceConfig(id="hn", platform="hackernews"), ComposioClient())
    item = source.to_item(
        {"id": 42, "title": "A title", "url": "https://x.test", "by": "pg", "score": 300, "time": 1700000000}
    )
    assert item.external_id == "42"
    assert item.engagement == 300
    assert item.author == "pg"


def test_hackernews_falls_back_to_the_discussion_url():
    from doomscroller.config import SourceConfig

    source = HackerNewsSource(SourceConfig(id="hn", platform="hackernews"), ComposioClient())
    item = source.to_item({"id": 7, "title": "Ask HN: anything"})
    assert item.url.endswith("item?id=7")


def test_reddit_unwraps_the_nested_data_envelope():
    from doomscroller.config import SourceConfig

    source = RedditSource(
        SourceConfig(id="r", platform="reddit", options={"subreddit": "rust"}), ComposioClient()
    )
    item = source.to_item(
        {"kind": "t3", "data": {"id": "abc", "title": "Rust 2.0", "permalink": "/r/rust/abc", "ups": 90}}
    )
    assert item.external_id == "abc"
    assert item.url == "https://reddit.com/r/rust/abc"
    assert item.engagement == 90


def test_records_without_an_id_are_skipped():
    from doomscroller.config import SourceConfig

    source = HackerNewsSource(SourceConfig(id="hn", platform="hackernews"), ComposioClient())
    assert source.to_item({"title": "no id here"}) is None


def test_composio_client_reports_missing_credentials_instead_of_raising():
    result = ComposioClient(api_key=None).execute("ANY_SLUG")
    assert not result.ok
    assert "COMPOSIO_API_KEY" in result.error


def test_as_dict_normalises_sdk_objects():
    class Response:
        def model_dump(self):
            return {"successful": True, "data": {"items": []}}

    assert _as_dict(Response())["successful"] is True
    assert _as_dict({"a": 1}) == {"a": 1}


# -- rss -----------------------------------------------------------------

RSS_FEED = """<?xml version="1.0"?>
<rss version="2.0"><channel>
  <item>
    <title>Postgres 18 released</title>
    <link>https://example.com/pg18</link>
    <guid>pg18</guid>
    <pubDate>Fri, 01 Mar 2024 12:00:00 GMT</pubDate>
    <description>&lt;p&gt;Async I/O and more.&lt;/p&gt;</description>
  </item>
</channel></rss>
"""


def test_rss_parses_a_feed_and_strips_markup(tmp_path, monkeypatch):
    import httpx

    from doomscroller.config import SourceConfig
    from doomscroller.sources.rss import RSSSource

    class FakeResponse:
        content = RSS_FEED.encode()

        def raise_for_status(self):
            return None

    monkeypatch.setattr(httpx, "get", lambda *a, **k: FakeResponse())
    source = RSSSource(SourceConfig(id="feed", platform="rss", options={"url": "https://x.test/rss"}))
    items = source.fetch(24 * 365 * 10)

    assert len(items) == 1
    assert items[0].title == "Postgres 18 released"
    assert items[0].body == "Async I/O and more."  # tags stripped, entities decoded
    assert items[0].published_at.year == 2024


def test_rss_requires_a_url():
    from doomscroller.config import SourceConfig
    from doomscroller.sources.base import SourceError
    from doomscroller.sources.rss import RSSSource

    with pytest.raises(SourceError, match="url"):
        RSSSource(SourceConfig(id="feed", platform="rss"))


# -- rendering -----------------------------------------------------------


def _digest() -> Digest:
    clusters = cluster_items(
        [
            make_scored(make_item(external_id="1", url="https://a.test/1")),
            make_scored(
                make_item(title="Ferrari signs a driver", external_id="2", url="https://b.test/2"),
                topics=["formula-1"],
                summary="Ferrari signed a driver for next season.",
                score=0.6,
            ),
        ]
    )
    digest = Digest(clusters=clusters[:1], skimmed=[clusters[1].lead])
    digest.stats = {"fetched": 20, "new": 8, "filtered_out": 5, "muted": 1, "tool_calls": 3,
                    "overview": "A quiet day in databases."}
    return digest


def test_terminal_output_includes_ids_for_feedback():
    text = to_terminal(_digest(), color=False)
    assert "Your brief" in text
    assert "A quiet day in databases." in text
    assert "doomscroller feedback" in text
    assert "\033[" not in text  # color disabled means no escape codes


def test_markdown_links_the_lead_url():
    body = to_markdown(_digest())
    assert "(https://a.test/1)" in body
    assert "Also, briefly" in body


def test_html_is_self_contained_and_escaped():
    digest = _digest()
    digest.clusters[0].headline = "<script>alert(1)</script>"
    html = to_html(digest)
    assert html.startswith("<!doctype html>")
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
    assert "prefers-color-scheme" in html  # renders in both themes


def test_empty_digest_says_so_in_every_format():
    empty = Digest(stats={"fetched": 0, "new": 0})
    assert "Nothing worth your time" in to_terminal(empty, color=False)
    assert "Nothing worth your time" in to_markdown(empty)
    assert "Nothing worth your time" in to_html(empty)
