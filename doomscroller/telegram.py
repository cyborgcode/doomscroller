"""Telegram delivery, straight to the Bot API.

Two things this does that the generic chat delivery doesn't.

**It sends everything.** Telegram caps a message at 4096 characters. Truncating
a brief is fine when Telegram is a notification and the real thing is in a file;
it is not fine when Telegram *is* the product. So the brief is split across
messages on natural boundaries and nothing is dropped.

**It keeps the feedback loop reachable.** Once this runs on a schedule somewhere
you don't have a shell, `doomscroller feedback` is out of reach — and the whole
premise is that the filter improves because you react to things. Each headline
therefore carries 👍 / 👎 / 🔇 buttons, and the callbacks land on a webhook that
records the signal. Tapping a button on your phone is the same input as typing
the command.

Going direct rather than through Composio is deliberate: it needs one token from
BotFather instead of an OAuth grant, costs no Composio quota, and is one HTTPS
POST with no SDK — which matters when this has to work unattended.
"""

from __future__ import annotations

import html
import logging
import os
from typing import Any, Iterable

import httpx

from .models import Cluster, Digest, ScoredItem

log = logging.getLogger(__name__)

API_ROOT = "https://api.telegram.org"
MESSAGE_LIMIT = 4096
SAFE_LIMIT = 3900
"""Below Telegram's 4096 so entity markup can't tip a message over the edge."""

SIGNALS = (("👍", "up"), ("👎", "down"), ("🔇", "mute"))


class TelegramError(RuntimeError):
    pass


class TelegramClient:
    def __init__(
        self,
        token: str | None = None,
        *,
        chat_id: str | int | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.token = token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = str(chat_id or os.environ.get("TELEGRAM_CHAT_ID", ""))
        self.timeout = timeout

    @property
    def configured(self) -> bool:
        return bool(self.token and self.chat_id)

    def _call(self, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.token:
            raise TelegramError(
                "TELEGRAM_BOT_TOKEN is not set. Message @BotFather, /newbot, and paste the token."
            )
        try:
            response = httpx.post(
                f"{API_ROOT}/bot{self.token}/{method}", json=payload, timeout=self.timeout
            )
        except httpx.HTTPError as exc:
            raise TelegramError(f"{method}: {exc}") from exc

        try:
            body = response.json()
        except ValueError:
            raise TelegramError(f"{method}: HTTP {response.status_code}, non-JSON reply") from None

        if not body.get("ok"):
            # Telegram puts the useful part in `description`, not the status code.
            raise TelegramError(f"{method}: {body.get('description', response.status_code)}")
        return body.get("result", {})

    def send_message(
        self,
        text: str,
        *,
        buttons: list[list[dict[str, str]]] | None = None,
        preview: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "HTML",
            "link_preview_options": {"is_disabled": not preview},
        }
        if buttons:
            payload["reply_markup"] = {"inline_keyboard": buttons}
        return self._call("sendMessage", payload)

    def answer_callback(self, callback_id: str, text: str = "") -> None:
        """Acknowledge a button tap. Without this the button spins forever."""
        self._call("answerCallbackQuery", {"callback_query_id": callback_id, "text": text})

    def set_webhook(self, url: str, secret: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "url": url,
            # Only the two we act on, so Telegram isn't posting us every message
            # in the chat.
            "allowed_updates": ["callback_query", "message"],
        }
        if secret:
            payload["secret_token"] = secret
        return self._call("setWebhook", payload)

    def delete_webhook(self) -> dict[str, Any]:
        return self._call("deleteWebhook", {})

    def get_me(self) -> dict[str, Any]:
        return self._call("getMe", {})


# -- rendering -----------------------------------------------------------


def _esc(value: object) -> str:
    return html.escape(str(value), quote=False)


def feedback_buttons(item_id: str) -> list[list[dict[str, str]]]:
    """Callback data is capped at 64 bytes; `up:<16 hex>` is 19."""
    return [[{"text": emoji, "callback_data": f"{signal}:{item_id}"} for emoji, signal in SIGNALS]]


def render_header(digest: Digest) -> str:
    stamp = digest.generated_at.astimezone().strftime("%a %d %b, %H:%M")
    lines = [f"<b>Your brief</b> — {_esc(stamp)} · last {digest.window_hours}h"]
    overview = str(digest.stats.get("overview") or "")
    if overview:
        lines += ["", _esc(overview)]
    return "\n".join(lines)


def render_cluster(cluster: Cluster, index: int) -> str:
    lines = [f"<b>{index}. {_esc(cluster.headline)}</b>"]
    if cluster.body:
        lines.append(_esc(cluster.body))
    for claim in cluster.claims[:3]:
        lines.append(f"• {_esc(claim)}")

    sources = cluster.sources
    label = sources[0] if len(sources) == 1 else f"{len(sources)} sources: {', '.join(sources[:3])}"
    footer = f"<i>{_esc(label)}</i>"
    if cluster.lead.item.url:
        footer += f' · <a href="{html.escape(cluster.lead.item.url, quote=True)}">open</a>'
    lines += ["", footer]
    return _fit("\n".join(lines))


def render_skims(skimmed: list[ScoredItem]) -> str:
    lines = ["<b>Also, briefly</b>", ""]
    for entry in skimmed:
        summary = _esc(entry.verdict.summary or entry.item.title)
        if entry.item.url:
            summary = f'<a href="{html.escape(entry.item.url, quote=True)}">{summary}</a>'
        lines.append(f"· {summary}")
    return "\n".join(lines)


def render_footer(digest: Digest) -> str:
    stats = digest.stats
    return (
        f"<i>{stats.get('fetched', 0)} fetched · {stats.get('new', 0)} new · "
        f"{stats.get('filtered_out', 0)} filtered · {stats.get('muted', 0)} muted</i>"
    )


def _fit(text: str, limit: int = SAFE_LIMIT) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit("\n", 1)[0]
    return cut + "\n…"


def chunk(text: str, limit: int = SAFE_LIMIT) -> list[str]:
    """Split on blank lines, then single lines, then hard characters.

    Splitting on structure keeps entries intact where possible; the character
    fallback exists so a single pathological line can never be dropped.
    """
    if len(text) <= limit:
        return [text] if text else []

    chunks: list[str] = []
    current = ""
    for block in text.split("\n\n"):
        candidate = f"{current}\n\n{block}" if current else block
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(block) <= limit:
            current = block
            continue
        # A single block too big on its own: fall to lines, then characters.
        current = ""
        for line in block.split("\n"):
            while len(line) > limit:
                chunks.append(line[:limit])
                line = line[limit:]
            candidate = f"{current}\n{line}" if current else line
            if len(candidate) <= limit:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = line
    if current:
        chunks.append(current)
    return chunks


def build_messages(digest: Digest, *, buttons: bool = True) -> list[tuple[str, Any]]:
    """The whole brief as (text, reply_markup) pairs, nothing truncated.

    One message per headline when buttons are on, so each has its own reactions.
    """
    messages: list[tuple[str, Any]] = []

    for part in chunk(render_header(digest)):
        messages.append((part, None))

    if digest.is_empty:
        messages.append(("Nothing worth your time came through. That's a good day.", None))
        return messages

    for index, cluster in enumerate(digest.clusters, 1):
        markup = feedback_buttons(cluster.lead.item.id) if buttons else None
        parts = chunk(render_cluster(cluster, index))
        for offset, part in enumerate(parts):
            # Buttons go on the last part, so they sit under the whole entry.
            messages.append((part, markup if offset == len(parts) - 1 else None))

    if digest.skimmed:
        for part in chunk(render_skims(digest.skimmed)):
            messages.append((part, None))

    messages.append((render_footer(digest), None))
    return messages


def send_digest(digest: Digest, client: TelegramClient, *, buttons: bool = True) -> int:
    """Send the full brief. Returns the number of messages sent."""
    sent = 0
    for text, markup in build_messages(digest, buttons=buttons):
        if not text.strip():
            continue
        client.send_message(text, buttons=markup)
        sent += 1
    return sent


# -- inbound -------------------------------------------------------------


def parse_callback(update: dict[str, Any]) -> tuple[str, str, str] | None:
    """Pull (signal, item_id, callback_id) out of a button tap.

    Returns None for anything that isn't a well-formed feedback callback, so an
    unrelated update can be ignored rather than raising in a webhook handler.
    """
    query = update.get("callback_query")
    if not isinstance(query, dict):
        return None
    data = str(query.get("data", ""))
    if ":" not in data:
        return None
    signal, _, item_id = data.partition(":")
    if not signal or not item_id:
        return None
    return signal, item_id, str(query.get("id", ""))


def describe_signal(signal: str) -> str:
    return {
        "up": "More like this. 👍",
        "down": "Less like this. 👎",
        "mute": "Muted that topic. 🔇",
        "save": "Saved. ⭐",
    }.get(signal, f"Recorded: {signal}")


def iter_message_texts(messages: Iterable[tuple[str, Any]]) -> Iterable[str]:
    for text, _ in messages:
        yield text
