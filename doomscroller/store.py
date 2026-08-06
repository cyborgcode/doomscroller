"""SQLite persistence: what you've been shown, what you thought of it, and the
interest profile learned from those two things.

The store is deliberately boring. All the learning logic lives in `learn.py`;
this module only reads and writes.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

from .models import Item, ScoredItem, Verdict

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id            TEXT PRIMARY KEY,
    source        TEXT NOT NULL,
    platform      TEXT NOT NULL,
    external_id   TEXT NOT NULL,
    title         TEXT NOT NULL,
    body          TEXT NOT NULL DEFAULT '',
    url           TEXT NOT NULL DEFAULT '',
    author        TEXT NOT NULL DEFAULT '',
    published_at  TEXT NOT NULL,
    engagement    INTEGER NOT NULL DEFAULT 0,
    first_seen_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS verdicts (
    item_id   TEXT PRIMARY KEY REFERENCES items(id) ON DELETE CASCADE,
    kind      TEXT NOT NULL,
    noise     REAL NOT NULL,
    substance REAL NOT NULL,
    summary   TEXT NOT NULL DEFAULT '',
    claims    TEXT NOT NULL DEFAULT '[]',
    topics    TEXT NOT NULL DEFAULT '[]',
    entities  TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS shown (
    item_id   TEXT NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    digest_at TEXT NOT NULL,
    slot      TEXT NOT NULL,
    score     REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (item_id, digest_at)
);

CREATE TABLE IF NOT EXISTS feedback (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id   TEXT NOT NULL,
    signal    TEXT NOT NULL,
    weight    REAL NOT NULL DEFAULT 1.0,
    note      TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    applied   INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS profile (
    kind   TEXT NOT NULL,
    key    TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 0,
    count  INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (kind, key)
);

CREATE INDEX IF NOT EXISTS idx_items_published ON items(published_at);
CREATE INDEX IF NOT EXISTS idx_feedback_applied ON feedback(applied);
"""

POSITIVE_SIGNALS = {"up": 1.0, "open": 0.4, "save": 1.2}
NEGATIVE_SIGNALS = {"down": -1.0, "skip": -0.3, "mute": -2.0}
VALID_SIGNALS = set(POSITIVE_SIGNALS) | set(NEGATIVE_SIGNALS)


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _parse_dt(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


class Store:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._conn:
            yield self._conn

    # -- items -----------------------------------------------------------

    def record_items(self, items: list[Item]) -> None:
        now = _iso(datetime.now(timezone.utc))
        with self._tx() as conn:
            conn.executemany(
                """
                INSERT INTO items (id, source, platform, external_id, title, body, url,
                                   author, published_at, engagement, first_seen_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    engagement = excluded.engagement,
                    title = excluded.title
                """,
                [
                    (
                        item.id,
                        item.source,
                        item.platform,
                        item.external_id,
                        item.title,
                        item.body,
                        item.url,
                        item.author,
                        _iso(item.published_at),
                        item.engagement,
                        now,
                    )
                    for item in items
                ],
            )

    def known_ids(self, ids: list[str]) -> set[str]:
        """Which of these have we already ingested on a previous run?"""
        if not ids:
            return set()
        found: set[str] = set()
        for start in range(0, len(ids), 500):
            chunk = ids[start : start + 500]
            placeholders = ",".join("?" * len(chunk))
            rows = self._conn.execute(
                f"SELECT id FROM items WHERE id IN ({placeholders})", chunk
            ).fetchall()
            found.update(row["id"] for row in rows)
        return found

    def get_item(self, item_id: str) -> Item | None:
        row = self._conn.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
        return _row_to_item(row) if row else None

    def prune(self, older_than_days: int = 45) -> int:
        cutoff = _iso(datetime.now(timezone.utc) - timedelta(days=older_than_days))
        with self._tx() as conn:
            cursor = conn.execute("DELETE FROM items WHERE first_seen_at < ?", (cutoff,))
        return cursor.rowcount

    # -- verdicts --------------------------------------------------------

    def record_verdicts(self, verdicts: list[Verdict]) -> None:
        with self._tx() as conn:
            conn.executemany(
                """
                INSERT INTO verdicts (item_id, kind, noise, substance, summary, claims, topics, entities)
                VALUES (?,?,?,?,?,?,?,?)
                ON CONFLICT(item_id) DO UPDATE SET
                    kind = excluded.kind, noise = excluded.noise,
                    substance = excluded.substance, summary = excluded.summary,
                    claims = excluded.claims, topics = excluded.topics,
                    entities = excluded.entities
                """,
                [
                    (
                        verdict.item_id,
                        verdict.kind,
                        verdict.noise,
                        verdict.substance,
                        verdict.summary,
                        json.dumps(verdict.claims),
                        json.dumps(verdict.topics),
                        json.dumps(verdict.entities),
                    )
                    for verdict in verdicts
                ],
            )

    def cached_verdicts(self, item_ids: list[str]) -> dict[str, Verdict]:
        """Verdicts we already paid for. Re-triaging the same post is pure waste."""
        if not item_ids:
            return {}
        out: dict[str, Verdict] = {}
        for start in range(0, len(item_ids), 500):
            chunk = item_ids[start : start + 500]
            placeholders = ",".join("?" * len(chunk))
            rows = self._conn.execute(
                f"SELECT * FROM verdicts WHERE item_id IN ({placeholders})", chunk
            ).fetchall()
            for row in rows:
                out[row["item_id"]] = Verdict(
                    item_id=row["item_id"],
                    kind=row["kind"],
                    noise=row["noise"],
                    substance=row["substance"],
                    summary=row["summary"],
                    claims=json.loads(row["claims"]),
                    topics=json.loads(row["topics"]),
                    entities=json.loads(row["entities"]),
                )
        return out

    # -- digest history --------------------------------------------------

    def record_shown(self, digest_at: datetime, headlines: list[ScoredItem], skimmed: list[ScoredItem]) -> None:
        stamp = _iso(digest_at)
        rows = [(s.item.id, stamp, "headline", s.score) for s in headlines]
        rows += [(s.item.id, stamp, "skim", s.score) for s in skimmed]
        with self._tx() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO shown (item_id, digest_at, slot, score) VALUES (?,?,?,?)",
                rows,
            )

    def shown_since(self, since: datetime) -> list[dict[str, object]]:
        rows = self._conn.execute(
            """
            SELECT s.item_id, s.slot, s.score, s.digest_at, i.title, i.url, i.source
            FROM shown s JOIN items i ON i.id = s.item_id
            WHERE s.digest_at >= ?
            ORDER BY s.digest_at DESC, s.score DESC
            """,
            (_iso(since),),
        ).fetchall()
        return [dict(row) for row in rows]

    def last_digest_at(self) -> datetime | None:
        row = self._conn.execute("SELECT MAX(digest_at) AS latest FROM shown").fetchone()
        return _parse_dt(row["latest"]) if row and row["latest"] else None

    # -- feedback --------------------------------------------------------

    def add_feedback(self, item_id: str, signal: str, note: str = "") -> None:
        if signal not in VALID_SIGNALS:
            raise ValueError(
                f"unknown signal {signal!r}; expected one of {sorted(VALID_SIGNALS)}"
            )
        weight = POSITIVE_SIGNALS.get(signal) or NEGATIVE_SIGNALS[signal]
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO feedback (item_id, signal, weight, note, created_at) VALUES (?,?,?,?,?)",
                (item_id, signal, weight, note, _iso(datetime.now(timezone.utc))),
            )

    def pending_feedback(self) -> list[dict[str, object]]:
        """Feedback that hasn't been folded into the profile yet."""
        rows = self._conn.execute(
            """
            SELECT f.id, f.item_id, f.signal, f.weight, i.title, i.body, i.source, i.platform
            FROM feedback f JOIN items i ON i.id = f.item_id
            WHERE f.applied = 0
            ORDER BY f.id
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def mark_feedback_applied(self, feedback_ids: list[int]) -> None:
        if not feedback_ids:
            return
        with self._tx() as conn:
            conn.executemany(
                "UPDATE feedback SET applied = 1 WHERE id = ?", [(fid,) for fid in feedback_ids]
            )

    def feedback_counts(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT signal, COUNT(*) AS n FROM feedback GROUP BY signal"
        ).fetchall()
        return {row["signal"]: row["n"] for row in rows}

    # -- profile ---------------------------------------------------------

    def profile(self, kind: str) -> dict[str, float]:
        rows = self._conn.execute(
            "SELECT key, weight FROM profile WHERE kind = ?", (kind,)
        ).fetchall()
        return {row["key"]: row["weight"] for row in rows}

    def bump_profile(self, kind: str, deltas: dict[str, float], decay: float = 1.0) -> None:
        """Add `deltas` to the stored weights, optionally decaying what's there first.

        Decay is what keeps the profile from ossifying: interests you stop
        reinforcing fade instead of ranking forever on a click from March.
        """
        with self._tx() as conn:
            if decay != 1.0:
                conn.execute(
                    "UPDATE profile SET weight = weight * ? WHERE kind = ?", (decay, kind)
                )
            for key, delta in deltas.items():
                conn.execute(
                    """
                    INSERT INTO profile (kind, key, weight, count) VALUES (?,?,?,1)
                    ON CONFLICT(kind, key) DO UPDATE SET
                        weight = profile.weight + excluded.weight,
                        count = profile.count + 1
                    """,
                    (kind, key, delta),
                )
            # Keep the token table from growing without bound.
            conn.execute(
                """
                DELETE FROM profile
                WHERE kind = ? AND ABS(weight) < 0.02 AND count < 3
                """,
                (kind,),
            )

    def top_profile(self, kind: str, limit: int = 20, positive: bool = True) -> list[tuple[str, float]]:
        order = "DESC" if positive else "ASC"
        rows = self._conn.execute(
            f"SELECT key, weight FROM profile WHERE kind = ? ORDER BY weight {order} LIMIT ?",
            (kind, limit),
        ).fetchall()
        return [(row["key"], row["weight"]) for row in rows]


def _row_to_item(row: sqlite3.Row) -> Item:
    return Item(
        source=row["source"],
        platform=row["platform"],
        external_id=row["external_id"],
        title=row["title"],
        body=row["body"],
        url=row["url"],
        author=row["author"],
        published_at=_parse_dt(row["published_at"]),
        engagement=row["engagement"],
    )
