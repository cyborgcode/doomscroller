"""Composio-backed sources, one class per platform.

Each class declares a default tool slug, builds that tool's arguments from
config, and maps the returned records onto `Item`. Both the slug and the
arguments are overridable per source in config.yaml:

    - id: reddit:rust
      platform: reddit
      subreddit: rust
      slug: REDDIT_RETRIEVE_REDDIT_POST     # override if yours differs
      arguments: {size: 50}                 # merged over the computed args

Run `doomscroller tools reddit` to see the slugs your account actually has;
providers rename them from time to time and the defaults here are only defaults.
"""

from __future__ import annotations

import logging
from typing import Any

from ..config import SourceConfig
from ..models import Item
from .base import BaseSource, SourceError, clean_text, parse_timestamp, pick, pick_int
from .composio_client import ComposioClient

log = logging.getLogger(__name__)


class ComposioSource(BaseSource):
    """Base for anything that reaches a platform through Composio."""

    platform = "composio"
    default_slug = ""

    def __init__(self, config: SourceConfig, client: ComposioClient) -> None:
        super().__init__(config)
        self.client = client
        self.slug = str(self.option("slug") or self.default_slug)

    def build_arguments(self, window_hours: int) -> dict[str, Any]:
        return {}

    def to_item(self, record: dict[str, Any]) -> Item | None:  # pragma: no cover - interface
        raise NotImplementedError

    def fetch(self, window_hours: int) -> list[Item]:
        if not self.slug:
            raise SourceError(f"{self.id}: no tool slug configured")

        arguments = self.build_arguments(window_hours)
        arguments.update(self.option("arguments") or {})

        result = self.client.execute(self.slug, arguments)
        if not result.ok:
            raise SourceError(f"{self.id}: {result.error}")

        items: list[Item] = []
        for record in result.records()[: self.limit]:
            try:
                item = self.to_item(record)
            except Exception as exc:  # noqa: BLE001 - one bad record shouldn't kill the source
                log.debug("%s: skipped a record: %s", self.id, exc)
                continue
            if item and item.title:
                items.append(item)
        return items


class HackerNewsSource(ComposioSource):
    platform = "hackernews"
    default_slug = "HACKERNEWS_GET_FRONTPAGE"

    def build_arguments(self, window_hours: int) -> dict[str, Any]:
        return {"size": self.limit}

    def to_item(self, record: dict[str, Any]) -> Item | None:
        external_id = str(pick(record, "id", "objectID", "story_id", default=""))
        if not external_id:
            return None
        title = clean_text(pick(record, "title", "story_title"), 300)
        url = str(pick(record, "url", "story_url", default="") or f"https://news.ycombinator.com/item?id={external_id}")
        return Item(
            source=self.id,
            platform=self.platform,
            external_id=external_id,
            title=title,
            body=clean_text(pick(record, "text", "story_text", "comment_text")),
            url=url,
            author=str(pick(record, "by", "author", default="")),
            published_at=parse_timestamp(pick(record, "time", "created_at", "created_at_i")),
            engagement=pick_int(record, "score", "points"),
            raw=record,
        )


class RedditSource(ComposioSource):
    platform = "reddit"
    default_slug = "REDDIT_RETRIEVE_REDDIT_POST"

    def __init__(self, config: SourceConfig, client: ComposioClient) -> None:
        super().__init__(config, client)
        self.subreddit = str(self.option("subreddit", "") or "")
        if not self.subreddit:
            raise SourceError(f"{config.id}: reddit sources need a 'subreddit'")

    def build_arguments(self, window_hours: int) -> dict[str, Any]:
        return {
            "subreddit": self.subreddit,
            "size": self.limit,
            "sort": self.option("sort", "hot"),
            "time_filter": self.option("time_filter", "day"),
        }

    def to_item(self, record: dict[str, Any]) -> Item | None:
        # Reddit's API nests the real post under "data" inside each "child".
        payload = record.get("data") if isinstance(record.get("data"), dict) else record
        external_id = str(pick(payload, "id", "name", default=""))
        if not external_id:
            return None
        permalink = str(pick(payload, "permalink", default=""))
        url = f"https://reddit.com{permalink}" if permalink.startswith("/") else str(
            pick(payload, "url", "link", default=permalink)
        )
        return Item(
            source=self.id,
            platform=self.platform,
            external_id=external_id,
            title=clean_text(pick(payload, "title"), 300),
            body=clean_text(pick(payload, "selftext", "body", "description")),
            url=url,
            author=str(pick(payload, "author", default="")),
            published_at=parse_timestamp(pick(payload, "created_utc", "created", "created_at")),
            engagement=pick_int(payload, "score", "ups", "upvotes"),
            raw=payload,
        )


class TwitterSource(ComposioSource):
    """X/Twitter. Defaults to a recent search; set `list_id` or `user_id` to
    pull a list or home timeline instead (with the matching slug)."""

    platform = "twitter"
    default_slug = "TWITTER_RECENT_SEARCH"

    def build_arguments(self, window_hours: int) -> dict[str, Any]:
        arguments: dict[str, Any] = {"max_results": min(self.limit, 100)}
        query = self.option("query")
        if query:
            arguments["query"] = query
        for key in ("user_id", "list_id"):
            if self.option(key):
                arguments[key] = self.option(key)
        return arguments

    def to_item(self, record: dict[str, Any]) -> Item | None:
        external_id = str(pick(record, "id", "id_str", "tweet_id", default=""))
        if not external_id:
            return None
        text = clean_text(pick(record, "text", "full_text", "content"), 800)
        author = str(pick(record, "author_id", "username", "user.screen_name", default=""))
        metrics = record.get("public_metrics") if isinstance(record.get("public_metrics"), dict) else {}
        engagement = pick_int(metrics, "like_count", "retweet_count") or pick_int(
            record, "favorite_count", "like_count"
        )
        return Item(
            source=self.id,
            platform=self.platform,
            external_id=external_id,
            title=text.split("\n", 1)[0][:280] or f"post by {author}",
            body=text,
            url=f"https://x.com/{author or 'i'}/status/{external_id}",
            author=author,
            published_at=parse_timestamp(pick(record, "created_at", "timestamp")),
            engagement=engagement,
            raw=record,
        )


class YouTubeSource(ComposioSource):
    platform = "youtube"
    default_slug = "YOUTUBE_SEARCH_YOU_TUBE"

    def build_arguments(self, window_hours: int) -> dict[str, Any]:
        arguments: dict[str, Any] = {"maxResults": min(self.limit, 50)}
        if self.option("query"):
            arguments["q"] = self.option("query")
        if self.option("channel_id"):
            arguments["channelId"] = self.option("channel_id")
        arguments.setdefault("order", self.option("order", "date"))
        return arguments

    def to_item(self, record: dict[str, Any]) -> Item | None:
        external_id = str(
            pick(record, "id.videoId", "videoId", "id", "contentDetails.upload.videoId", default="")
        )
        if not external_id:
            return None
        snippet = record.get("snippet") if isinstance(record.get("snippet"), dict) else record
        return Item(
            source=self.id,
            platform=self.platform,
            external_id=external_id,
            title=clean_text(pick(snippet, "title"), 300),
            body=clean_text(pick(snippet, "description")),
            url=f"https://www.youtube.com/watch?v={external_id}",
            author=str(pick(snippet, "channelTitle", "channelId", default="")),
            published_at=parse_timestamp(pick(snippet, "publishedAt", "publishTime")),
            engagement=pick_int(record, "statistics.viewCount"),
            raw=record,
        )


class GmailSource(ComposioSource):
    """Newsletters, mostly. Point `query` at wherever yours land."""

    platform = "gmail"
    default_slug = "GMAIL_FETCH_EMAILS"

    def build_arguments(self, window_hours: int) -> dict[str, Any]:
        days = max(1, round(window_hours / 24))
        query = self.option("query") or f"category:updates newer_than:{days}d"
        return {
            "query": query,
            "max_results": self.limit,
            "user_id": "me",
        }

    def to_item(self, record: dict[str, Any]) -> Item | None:
        external_id = str(pick(record, "id", "messageId", "message_id", "threadId", default=""))
        if not external_id:
            return None
        subject = clean_text(
            pick(record, "subject", "payload.headers.Subject", "messageText"), 300
        )
        body = clean_text(
            pick(record, "messageText", "snippet", "body", "preview", "payload.body.data"), 3000
        )
        return Item(
            source=self.id,
            platform=self.platform,
            external_id=external_id,
            title=subject or body.split("\n", 1)[0][:200],
            body=body,
            url=f"https://mail.google.com/mail/u/0/#inbox/{external_id}",
            author=str(pick(record, "sender", "from", "payload.headers.From", default="")),
            published_at=parse_timestamp(pick(record, "messageTimestamp", "internalDate", "date")),
            raw=record,
        )


class LinkedInSource(ComposioSource):
    platform = "linkedin"
    default_slug = "LINKEDIN_GET_MY_FEED"

    def build_arguments(self, window_hours: int) -> dict[str, Any]:
        return {"count": self.limit}

    def to_item(self, record: dict[str, Any]) -> Item | None:
        external_id = str(pick(record, "id", "urn", "activityUrn", default=""))
        if not external_id:
            return None
        text = clean_text(pick(record, "text", "commentary", "content", "description"), 1200)
        return Item(
            source=self.id,
            platform=self.platform,
            external_id=external_id,
            title=text.split("\n", 1)[0][:280] or "LinkedIn post",
            body=text,
            url=str(pick(record, "url", "permalink", default="")),
            author=str(pick(record, "author", "actor.name", "authorName", default="")),
            published_at=parse_timestamp(pick(record, "created_at", "publishedAt", "createdAt")),
            engagement=pick_int(record, "numLikes", "likes", "reactions"),
            raw=record,
        )


class GenericComposioSource(ComposioSource):
    """Escape hatch for any toolkit that isn't modelled above.

        - id: notion-digest
          platform: composio
          slug: NOTION_SEARCH_NOTION_PAGE
          arguments: {query: "weekly"}
          title_keys: [title, name]
          body_keys: [content]
    """

    platform = "composio"

    def to_item(self, record: dict[str, Any]) -> Item | None:
        title_keys = self.option("title_keys") or ["title", "name", "subject", "headline"]
        body_keys = self.option("body_keys") or ["body", "text", "content", "description", "snippet"]
        id_keys = self.option("id_keys") or ["id", "uuid", "key", "url"]

        external_id = str(pick(record, *id_keys, default=""))
        if not external_id:
            return None
        return Item(
            source=self.id,
            platform=self.option("platform_label", self.id),
            external_id=external_id,
            title=clean_text(pick(record, *title_keys), 300),
            body=clean_text(pick(record, *body_keys)),
            url=str(pick(record, "url", "link", "permalink", default="")),
            author=str(pick(record, "author", "user", "sender", default="")),
            published_at=parse_timestamp(
                pick(record, "created_at", "published_at", "timestamp", "date")
            ),
            engagement=pick_int(record, "score", "likes", "count"),
            raw=record,
        )


COMPOSIO_SOURCES: dict[str, type[ComposioSource]] = {
    "hackernews": HackerNewsSource,
    "reddit": RedditSource,
    "twitter": TwitterSource,
    "x": TwitterSource,
    "youtube": YouTubeSource,
    "gmail": GmailSource,
    "linkedin": LinkedInSource,
    "composio": GenericComposioSource,
}
