"""Shared helpers for the Vercel functions.

Serverless changes three assumptions the CLI makes: the working directory isn't
the repo root, there is no `.env` file, and there is no disk that survives the
invocation. Each is handled once here.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

# The function's working directory isn't the repo root, so make the package
# importable from wherever the bundle unpacked.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from doomscroller.config import Config, ConfigError, load_config  # noqa: E402
from doomscroller.drivers import is_remote  # noqa: E402


def load() -> Config:
    """Config from the deployment, with the database pointed at the remote.

    `DOOMSCROLLER_DB_URL` is read by the driver itself; this only makes the
    failure obvious if it's missing, since a serverless SQLite file would appear
    to work and then silently forget everything between runs.
    """
    path = os.environ.get("DOOMSCROLLER_CONFIG") or str(ROOT / "config.yaml")
    try:
        config = load_config(path)
    except ConfigError as exc:
        raise RuntimeError(
            f"{exc}. Commit a config.yaml, or set DOOMSCROLLER_CONFIG to its path."
        ) from exc

    if os.environ.get("DOOMSCROLLER_ALLOW_LOCAL_DB"):
        return config  # escape hatch for running these handlers locally

    target = (
        os.environ.get("DOOMSCROLLER_DB_URL")
        or os.environ.get("TURSO_DATABASE_URL")
        or str(config.db_path)
    )
    # Checking the value is a remote URL, not merely that a variable is set:
    # a file path here would appear to work and then forget every item and all
    # your feedback between invocations, which is worse than failing outright.
    if not is_remote(target):
        raise RuntimeError(
            f"Database target {target!r} is a local file, and serverless functions have "
            "no persistent disk — every item and all your feedback would be lost between "
            "runs. Set TURSO_DATABASE_URL (libsql://…) and TURSO_AUTH_TOKEN, or set "
            "DOOMSCROLLER_ALLOW_LOCAL_DB=1 if you are running this handler locally."
        )
    return config


def authorized(headers: Any, secret_env: str = "CRON_SECRET") -> bool:
    """Vercel sends `Authorization: Bearer $CRON_SECRET` on scheduled requests.

    An unset secret means open — deliberately, so a first deploy works — but
    `check` and the docs both push you to set one, because this endpoint spends
    API quota and anyone can find a deployment URL.
    """
    secret = os.environ.get(secret_env)
    if not secret:
        return True
    supplied = headers.get("authorization") or headers.get("Authorization") or ""
    return supplied == f"Bearer {secret}"


def respond(handler: Any, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def read_json(handler: Any) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length") or 0)
    if not length:
        return {}
    try:
        return json.loads(handler.rfile.read(length) or b"{}")
    except (ValueError, TypeError):
        return {}
