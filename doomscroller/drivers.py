"""Storage drivers.

The store is SQLite either way — the only question is whether the file is on
this machine. Locally it is. On a serverless host there is no disk that
survives between invocations, so the same SQLite lives in Turso and is reached
over HTTP.

Both drivers speak the same SQL, so every query in `store.py` is written once.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Protocol, Sequence

log = logging.getLogger(__name__)

Params = Sequence[Any]
Statement = tuple[str, Params]

REMOTE_SCHEMES = ("libsql://", "http://", "https://", "wss://", "ws://")


class Driver(Protocol):
    """The four shapes of database access this project needs."""

    def script(self, sql: str) -> None:
        """Run several statements separated by semicolons (schema setup)."""

    def query(self, sql: str, params: Params = ()) -> list[dict[str, Any]]:
        """Rows as plain dicts, so callers never touch a driver-specific row type."""

    def write(self, sql: str, params: Params = ()) -> int:
        """Run one statement; return rows affected."""

    def write_many(self, sql: str, rows: Sequence[Params]) -> None:
        """Run one statement over many parameter sets, atomically."""

    def batch(self, statements: Sequence[Statement]) -> None:
        """Run several different statements atomically."""

    def close(self) -> None: ...


def split_script(sql: str) -> list[str]:
    """Split a schema script into statements.

    Naive on purpose — it only has to handle this project's schema, which has
    no semicolons inside string literals or triggers.
    """
    return [part.strip() for part in sql.split(";") if part.strip()]


class SQLiteDriver:
    """Local file. What you get when running on your own machine."""

    kind = "sqlite"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if self.path.parent and str(self.path.parent) not in ("", "."):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")

    def script(self, sql: str) -> None:
        with self._conn:
            self._conn.executescript(sql)

    def query(self, sql: str, params: Params = ()) -> list[dict[str, Any]]:
        return [dict(row) for row in self._conn.execute(sql, tuple(params)).fetchall()]

    def write(self, sql: str, params: Params = ()) -> int:
        with self._conn:
            return self._conn.execute(sql, tuple(params)).rowcount

    def write_many(self, sql: str, rows: Sequence[Params]) -> None:
        if not rows:
            return
        with self._conn:
            self._conn.executemany(sql, [tuple(row) for row in rows])

    def batch(self, statements: Sequence[Statement]) -> None:
        if not statements:
            return
        with self._conn:
            for sql, params in statements:
                self._conn.execute(sql, tuple(params))

    def close(self) -> None:
        self._conn.close()

    def describe(self) -> str:
        return f"sqlite {self.path}"


class LibSQLDriver:
    """Turso / libSQL over HTTP. What you get on a host without a disk.

    Every call is a network round trip, so the store batches deliberately:
    `write_many` and `batch` are one request each, not one per row.
    """

    kind = "libsql"

    def __init__(self, url: str, auth_token: str | None = None) -> None:
        try:
            import libsql_client
        except ImportError as exc:  # pragma: no cover - depends on install
            raise RuntimeError(
                "a remote database URL was configured but libsql-client is not "
                "installed. Run: pip install libsql-client"
            ) from exc

        self.url = url
        self._libsql = libsql_client
        # The HTTP transport is the one that works on serverless; a libsql://
        # URL selects a websocket transport that a short-lived function can't
        # keep open, so normalise it.
        self._client = libsql_client.create_client_sync(
            url=url.replace("libsql://", "https://", 1),
            auth_token=auth_token or os.environ.get("TURSO_AUTH_TOKEN"),
        )

    def script(self, sql: str) -> None:
        self._client.batch(split_script(sql))

    def query(self, sql: str, params: Params = ()) -> list[dict[str, Any]]:
        result = self._client.execute(sql, list(params))
        return [row.asdict() for row in result.rows]

    def write(self, sql: str, params: Params = ()) -> int:
        return self._client.execute(sql, list(params)).rows_affected

    def write_many(self, sql: str, rows: Sequence[Params]) -> None:
        if not rows:
            return
        self._client.batch([self._libsql.Statement(sql, list(row)) for row in rows])

    def batch(self, statements: Sequence[Statement]) -> None:
        if not statements:
            return
        self._client.batch(
            [self._libsql.Statement(sql, list(params)) for sql, params in statements]
        )

    def close(self) -> None:
        self._client.close()

    def describe(self) -> str:
        return f"libsql {self.url}"


def is_remote(target: str) -> bool:
    return str(target).startswith(REMOTE_SCHEMES)


def open_driver(target: str | Path, auth_token: str | None = None) -> Driver:
    """Pick a driver from the target.

    Environment wins over config, because on a serverless host the config file
    is baked into the deployment and the database URL is not.
    """
    env_url = os.environ.get("DOOMSCROLLER_DB_URL") or os.environ.get("TURSO_DATABASE_URL")
    resolved = env_url or str(target)

    if is_remote(resolved):
        return LibSQLDriver(resolved, auth_token)
    return SQLiteDriver(resolved)
