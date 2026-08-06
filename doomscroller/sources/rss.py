"""RSS/Atom source.

Deliberately dependency-light and Composio-free: every blog and most news sites
still publish a feed, it costs no tool calls against your free-tier quota, and
it means the bot does something useful before you've connected a single account.
"""

from __future__ import annotations

from typing import Any
from xml.etree import ElementTree

import httpx

from ..config import SourceConfig
from ..models import Item
from .base import BaseSource, SourceError, clean_text, parse_timestamp

_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "content": "http://purl.org/rss/1.0/modules/content/",
    "dc": "http://purl.org/dc/elements/1.1/",
}


class RSSSource(BaseSource):
    platform = "rss"

    def __init__(self, config: SourceConfig) -> None:
        super().__init__(config)
        self.url = str(self.option("url", "") or "")
        if not self.url:
            raise SourceError(f"{config.id}: rss sources need a 'url'")
        self.timeout = float(self.option("timeout", 20.0))

    def fetch(self, window_hours: int) -> list[Item]:
        try:
            response = httpx.get(
                self.url,
                timeout=self.timeout,
                follow_redirects=True,
                headers={"User-Agent": "doomscroller/0.1 (+personal feed digest)"},
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise SourceError(f"{self.id}: {exc}") from exc

        try:
            root = ElementTree.fromstring(response.content)
        except ElementTree.ParseError as exc:
            raise SourceError(f"{self.id}: malformed feed ({exc})") from exc

        entries = root.findall(".//item") or root.findall(".//atom:entry", _NS)
        items: list[Item] = []
        for entry in entries[: self.limit]:
            item = self._to_item(entry)
            if item and item.title:
                items.append(item)
        return items

    def _to_item(self, entry: ElementTree.Element) -> Item | None:
        title = clean_text(_text(entry, "title", "atom:title"), 300)
        link = _text(entry, "link", "atom:link") or _attr(entry, "atom:link", "href") or ""
        guid = _text(entry, "guid", "atom:id") or link or title
        if not guid:
            return None
        body = clean_text(
            _text(entry, "content:encoded", "description", "atom:summary", "atom:content")
        )
        return Item(
            source=self.id,
            platform=self.platform,
            external_id=guid,
            title=title,
            body=body,
            url=link,
            author=_text(entry, "author", "dc:creator", "atom:author/atom:name") or "",
            published_at=parse_timestamp(
                _text(entry, "pubDate", "atom:published", "atom:updated", "dc:date")
            ),
            raw={"feed": self.url},
        )


def _text(element: ElementTree.Element, *paths: str) -> str:
    for path in paths:
        found = element.find(path, _NS)
        if found is not None and (found.text or "").strip():
            return (found.text or "").strip()
    return ""


def _attr(element: ElementTree.Element, path: str, attribute: str) -> str:
    found = element.find(path, _NS)
    value: Any = found.get(attribute) if found is not None else None
    return str(value) if value else ""
