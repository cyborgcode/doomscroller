from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from doomscroller.config import Config, SourceConfig
from doomscroller.models import Item, ScoredItem, Verdict
from doomscroller.store import Store


@pytest.fixture(params=["sqlite", "libsql"])
def store(request, tmp_path, monkeypatch):
    """Every store test runs against both drivers.

    The remote path can't reach Turso from here, but the code that would break
    against it is ours — statement splitting, batching, row conversion — so it
    runs against a fake client with the real `ClientSync` surface. Deploying to
    a host with no disk shouldn't be the first time that code executes.
    """
    if request.param == "sqlite":
        with Store(tmp_path / "test.db") as db:
            yield db
        return

    import sys

    from tests import fake_libsql

    fake_libsql.create_client_sync.path = str(tmp_path / "remote.db")
    monkeypatch.setitem(sys.modules, "libsql_client", fake_libsql)
    monkeypatch.delenv("DOOMSCROLLER_DB_URL", raising=False)
    monkeypatch.delenv("TURSO_DATABASE_URL", raising=False)
    with Store("libsql://fake-db.turso.io", auth_token="test-token") as db:
        yield db


@pytest.fixture
def config(tmp_path):
    cfg = Config(db_path=tmp_path / "test.db")
    cfg.sources = [SourceConfig(id="hn", platform="hackernews")]
    return cfg


def make_item(
    title: str = "Postgres 18 ships asynchronous I/O",
    *,
    source: str = "hn",
    platform: str = "hackernews",
    external_id: str = "1",
    body: str = "",
    url: str = "https://example.com/pg18",
    age_hours: float = 2.0,
    engagement: int = 100,
) -> Item:
    return Item(
        source=source,
        platform=platform,
        external_id=external_id,
        title=title,
        body=body,
        url=url,
        published_at=datetime.now(timezone.utc) - timedelta(hours=age_hours),
        engagement=engagement,
    )


def make_scored(
    item: Item | None = None,
    *,
    kind: str = "news",
    noise: float = 0.2,
    substance: float = 0.8,
    topics: list[str] | None = None,
    claims: list[str] | None = None,
    summary: str = "Postgres 18 adds async I/O on Linux.",
    score: float = 1.0,
) -> ScoredItem:
    item = item or make_item()
    return ScoredItem(
        item=item,
        verdict=Verdict(
            item_id=item.id,
            kind=kind,
            noise=noise,
            substance=substance,
            topics=topics or ["databases"],
            claims=claims or ["Postgres 18 ships asynchronous I/O on Linux."],
            summary=summary,
        ),
        score=score,
    )
