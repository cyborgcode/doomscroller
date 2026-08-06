"""A stand-in for `libsql_client`, backed by local SQLite.

Turso can't be reached from the test suite, but the code that would break
against it is ours: `split_script`, the batch translation, `rows_affected`, and
the `Row.asdict()` conversion. This exposes the same surface the real
`ClientSync` does — verified against the installed package — so `LibSQLDriver`
runs unmodified and every store test can be executed twice, once per driver.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass
class Statement:
    sql: str
    args: Sequence[Any] | None = None


class Row(dict):
    """The real Row exposes `asdict()`; dict gives us the rest for free."""

    def asdict(self) -> dict[str, Any]:
        return dict(self)


@dataclass
class ResultSet:
    columns: tuple[str, ...]
    rows: list[Row]
    rows_affected: int = 0
    last_insert_rowid: int | None = None


class ClientSync:
    def __init__(self, path: str = ":memory:") -> None:
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self.closed = False
        self.statements: list[str] = []
        """Every SQL string seen."""

        self.requests = 0
        """Network round trips. One per execute() or batch(), regardless of how
        many statements the batch carries — this is the number that matters on a
        remote database, and what the batching tests assert on."""

    def _run(self, stmt: Any, args: Sequence[Any] | None = None) -> ResultSet:
        if isinstance(stmt, Statement):
            sql, params = stmt.sql, stmt.args or []
        else:
            sql, params = stmt, args or []
        self.statements.append(sql)
        cursor = self._conn.execute(sql, tuple(params))
        rows = [Row(dict(row)) for row in cursor.fetchall()]
        return ResultSet(
            columns=tuple(d[0] for d in cursor.description or ()),
            rows=rows,
            rows_affected=cursor.rowcount if cursor.rowcount and cursor.rowcount > 0 else 0,
            last_insert_rowid=cursor.lastrowid,
        )

    def execute(self, stmt: Any, args: Sequence[Any] | None = None) -> ResultSet:
        self.requests += 1
        result = self._run(stmt, args)
        self._conn.commit()
        return result

    def batch(self, stmts: Sequence[Any]) -> list[ResultSet]:
        """All-or-nothing, like the real one."""
        self.requests += 1
        try:
            results = [self._run(stmt) for stmt in stmts]
        except Exception:
            self._conn.rollback()
            raise
        self._conn.commit()
        return results

    def close(self) -> None:
        self._conn.close()
        self.closed = True


def create_client_sync(url: str = "", auth_token: str | None = None, **_: Any) -> ClientSync:
    create_client_sync.last_url = url  # type: ignore[attr-defined]
    create_client_sync.last_token = auth_token  # type: ignore[attr-defined]
    return ClientSync(create_client_sync.path)  # type: ignore[attr-defined]


create_client_sync.path = ":memory:"  # type: ignore[attr-defined]
create_client_sync.last_url = ""  # type: ignore[attr-defined]
create_client_sync.last_token = None  # type: ignore[attr-defined]
