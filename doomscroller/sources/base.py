"""Shared plumbing for feed sources.

Every platform hands back a differently-shaped blob. Rather than pin exact
schemas — which drift whenever a provider changes an endpoint — sources use the
tolerant pickers below: try a list of likely keys, fall back to a default, and
keep the untouched payload on `Item.raw` so nothing is silently lost.
"""

from __future__ import annotations

import html
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Protocol

from ..config import SourceConfig
from ..models import Item


class Source(Protocol):
    """A thing that can produce items for a time window."""

    id: str
    platform: str

    def fetch(self, window_hours: int) -> list[Item]: ...


class SourceError(RuntimeError):
    """Raised when a source can't fetch. The runner logs it and carries on with the rest."""


_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"[ \t\r\f\v]+")
_BLANKS = re.compile(r"\n{3,}")


def clean_text(value: Any, limit: int = 2000) -> str:
    """Strip markup and collapse whitespace, then truncate on a word boundary."""
    if not value:
        return ""
    text = html.unescape(_TAG.sub(" ", str(value)))
    text = _BLANKS.sub("\n\n", _WS.sub(" ", text)).strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    space = cut.rfind(" ")
    return (cut[:space] if space > limit * 0.6 else cut).rstrip() + "…"


def pick(payload: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """First non-empty value among `keys`, supporting `a.b` dotted paths."""
    for key in keys:
        cursor: Any = payload
        for part in key.split("."):
            if not isinstance(cursor, dict):
                cursor = None
                break
            cursor = cursor.get(part)
        if cursor not in (None, "", [], {}):
            return cursor
    return default


def pick_int(payload: dict[str, Any], *keys: str, default: int = 0) -> int:
    value = pick(payload, *keys)
    try:
        return int(float(value))  # tolerates "42" and 42.0
    except (TypeError, ValueError):
        return default


def parse_timestamp(value: Any) -> datetime:
    """Best-effort timestamp parse. Unknown formats fall back to 'now'.

    Falling back to now is deliberate: an item with an unparseable date should
    still surface, just without a freshness bonus it hasn't earned. It gets one
    it hasn't earned instead — the alternative, dropping it, is worse.
    """
    if value in (None, ""):
        return datetime.now(timezone.utc)
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)) or (isinstance(value, str) and value.isdigit()):
        seconds = float(value)
        if seconds > 1e11:  # milliseconds
            seconds /= 1000.0
        try:
            return datetime.fromtimestamp(seconds, tz=timezone.utc)
        except (OSError, OverflowError, ValueError):
            return datetime.now(timezone.utc)
    text = str(value).strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    try:
        from email.utils import parsedate_to_datetime

        parsed = parsedate_to_datetime(str(value))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return datetime.now(timezone.utc)


def unwrap_records(payload: Any, *keys: str) -> list[dict[str, Any]]:
    """Dig the list of records out of a response envelope.

    Providers wrap results as `{"data": {"items": [...]}}`, `{"results": [...]}`,
    a bare list, or occasionally a single object. All of those end up here.
    """
    if payload is None:
        return []
    if isinstance(payload, list):
        return [record for record in payload if isinstance(record, dict)]
    if not isinstance(payload, dict):
        return []

    candidates = (*keys, "items", "results", "data", "posts", "messages", "response", "children")
    for key in candidates:
        found = pick(payload, key)
        if isinstance(found, list):
            records = [record for record in found if isinstance(record, dict)]
            if records:
                return records
        if isinstance(found, dict):
            nested = unwrap_records(found, *keys)
            if nested:
                return nested

    # A single record with no envelope at all.
    if any(key in payload for key in ("title", "id", "text", "name", "subject")):
        return [payload]
    return []


def within_window(items: Iterable[Item], window_hours: int) -> list[Item]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
    kept = []
    for item in items:
        published = item.published_at
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        if published >= cutoff:
            kept.append(item)
    return kept


class BaseSource:
    """Small base that holds config and gives sources a consistent identity."""

    platform = "unknown"

    def __init__(self, config: SourceConfig) -> None:
        self.config = config
        self.id = config.id
        self.limit = config.limit
        self.options = config.options

    def option(self, key: str, default: Any = None) -> Any:
        return self.options.get(key, default)

    def fetch(self, window_hours: int) -> list[Item]:  # pragma: no cover - interface
        raise NotImplementedError
