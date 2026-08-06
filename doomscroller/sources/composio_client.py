"""Thin wrapper over the Composio SDK.

Two jobs: hold one lazily-created client for the whole process, and give the
rest of the codebase a single `execute()` that always returns a plain dict —
never an exception you have to guess the shape of, never an SDK object.

Free tier is 20,000 tool calls/month, so the wrapper also counts calls. One
digest run costs roughly one call per configured source.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


class ComposioUnavailable(RuntimeError):
    """The SDK isn't installed or no API key is configured."""


@dataclass
class CallResult:
    ok: bool
    data: Any = None
    error: str = ""
    slug: str = ""

    def records(self) -> list[dict[str, Any]]:
        from .base import unwrap_records

        return unwrap_records(self.data) if self.ok else []


@dataclass
class ComposioClient:
    user_id: str = "default"
    api_key: str | None = None
    _client: Any = field(default=None, init=False, repr=False)
    calls: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.environ.get("COMPOSIO_API_KEY")

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _ensure(self) -> Any:
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise ComposioUnavailable(
                "COMPOSIO_API_KEY is not set. Get a free key at https://app.composio.dev "
                "and put it in your .env — the free tier covers 20k tool calls a month."
            )
        try:
            from composio import Composio
        except ImportError as exc:  # pragma: no cover - depends on install
            raise ComposioUnavailable(
                "the composio package is not installed. Run: pip install composio"
            ) from exc
        self._client = Composio(api_key=self.api_key)
        return self._client

    def execute(self, slug: str, arguments: dict[str, Any] | None = None) -> CallResult:
        """Run one Composio tool. Never raises — failures come back as `CallResult`.

        Sources call this in a loop, and one dead integration shouldn't take the
        whole digest down with it.
        """
        try:
            client = self._ensure()
        except ComposioUnavailable as exc:
            return CallResult(ok=False, error=str(exc), slug=slug)

        try:
            self.calls += 1
            response = client.tools.execute(
                slug=slug,
                user_id=self.user_id,
                arguments=arguments or {},
            )
        except Exception as exc:  # noqa: BLE001 - provider errors are wide and varied
            log.warning("composio %s failed: %s", slug, exc)
            return CallResult(ok=False, error=f"{type(exc).__name__}: {exc}", slug=slug)

        payload = _as_dict(response)
        # The SDK reports tool-level failure in the body, not by raising.
        successful = payload.get("successful", payload.get("success", True))
        if not successful:
            error = str(payload.get("error") or payload.get("message") or "tool reported failure")
            log.warning("composio %s returned an error: %s", slug, error)
            return CallResult(ok=False, error=error, slug=slug)

        return CallResult(ok=True, data=payload.get("data", payload), slug=slug)

    # -- discovery -------------------------------------------------------

    def list_tools(self, toolkit: str, limit: int = 100) -> list[dict[str, Any]]:
        """Real tool slugs for a toolkit, so you can correct the config defaults.

        Slugs move around as providers change. `doomscroller tools reddit` is
        how you find out what yours actually are today.
        """
        client = self._ensure()
        try:
            tools = client.tools.get_raw_composio_tools(toolkits=[toolkit.upper()], limit=limit)
        except AttributeError:
            tools = client.tools.get(user_id=self.user_id, toolkits=[toolkit.upper()])
        except Exception as exc:  # noqa: BLE001
            raise ComposioUnavailable(f"could not list tools for {toolkit}: {exc}") from exc

        listed: list[dict[str, Any]] = []
        for tool in tools or []:
            entry = _as_dict(tool)
            function = entry.get("function") if isinstance(entry.get("function"), dict) else {}
            slug = entry.get("slug") or entry.get("name") or function.get("name")
            if slug:
                listed.append(
                    {
                        "slug": slug,
                        "description": (entry.get("description") or function.get("description") or "")[:160],
                    }
                )
        return listed

    def authorize(self, toolkit: str) -> str:
        """Start an OAuth flow and return the URL to open in a browser."""
        client = self._ensure()
        session = client.create(user_id=self.user_id)
        request = session.authorize(toolkit.lower())
        return getattr(request, "redirect_url", "") or str(request)


def _as_dict(value: Any) -> dict[str, Any]:
    """Coerce whatever the SDK returned into a plain dict."""
    if isinstance(value, dict):
        return value
    for attr in ("model_dump", "dict", "to_dict"):
        method = getattr(value, attr, None)
        if callable(method):
            try:
                result = method()
                if isinstance(result, dict):
                    return result
            except TypeError:
                continue
    if hasattr(value, "__dict__"):
        return {k: v for k, v in vars(value).items() if not k.startswith("_")}
    return {"data": value}
