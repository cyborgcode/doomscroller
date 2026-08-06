"""Where the finished brief goes.

`console` and `file` need nothing. The rest go back out through Composio, which
means the same connected account that read your feed can also deliver the
summary — one integration, both directions.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .config import DeliveryConfig
from .models import Digest
from .render import to_html, to_markdown, to_terminal
from .sources.composio_client import ComposioClient
from .telegram import TelegramClient, TelegramError, send_digest

log = logging.getLogger(__name__)


class DeliveryError(RuntimeError):
    pass


def deliver(digest: Digest, targets: list[DeliveryConfig], client: ComposioClient) -> list[str]:
    """Send the digest everywhere configured. Returns human-readable results."""
    results: list[str] = []
    for target in targets:
        handler = _HANDLERS.get(target.kind)
        if handler is None:
            results.append(f"{target.kind}: unknown delivery kind, skipped")
            continue
        try:
            results.append(handler(digest, target, client))
        except Exception as exc:  # noqa: BLE001 - one broken channel shouldn't lose the digest
            log.warning("delivery to %s failed: %s", target.kind, exc)
            results.append(f"{target.kind}: failed — {exc}")
    return results


def _console(digest: Digest, target: DeliveryConfig, _client: ComposioClient) -> str:
    print(to_terminal(digest, color=bool(target.options.get("color", True))))
    return "console: printed"


def _file(digest: Digest, target: DeliveryConfig, _client: ComposioClient) -> str:
    fmt = str(target.options.get("format", "markdown")).lower()
    body = to_html(digest) if fmt == "html" else to_markdown(digest)
    suffix = "html" if fmt == "html" else "md"

    raw_path = str(target.options.get("path", "briefs/{date}.{ext}"))
    path = Path(
        raw_path.format(
            date=digest.generated_at.astimezone().strftime("%Y-%m-%d"),
            datetime=digest.generated_at.astimezone().strftime("%Y-%m-%d-%H%M"),
            ext=suffix,
        )
    ).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return f"file: wrote {path}"


def _gmail(digest: Digest, target: DeliveryConfig, client: ComposioClient) -> str:
    recipient = target.options.get("to")
    if not recipient:
        raise DeliveryError("gmail delivery needs a 'to' address")
    subject = str(target.options.get("subject", "Your brief — {date}")).format(
        date=digest.generated_at.astimezone().strftime("%a %d %b")
    )
    result = client.execute(
        str(target.options.get("slug", "GMAIL_SEND_EMAIL")),
        {
            "recipient_email": recipient,
            "subject": subject,
            "body": to_html(digest),
            "is_html": True,
        },
    )
    if not result.ok:
        raise DeliveryError(result.error)
    return f"gmail: sent to {recipient}"


def _slack(digest: Digest, target: DeliveryConfig, client: ComposioClient) -> str:
    channel = target.options.get("channel")
    if not channel:
        raise DeliveryError("slack delivery needs a 'channel'")
    result = client.execute(
        str(target.options.get("slug", "SLACK_SENDS_A_MESSAGE_TO_A_SLACK_CHANNEL")),
        {"channel": channel, "text": _chat_text(digest)},
    )
    if not result.ok:
        raise DeliveryError(result.error)
    return f"slack: posted to {channel}"


def _telegram(digest: Digest, target: DeliveryConfig, client: ComposioClient) -> str:
    """Direct Bot API by default; `via: composio` for the toolkit path.

    Direct is the default because it sends the *whole* brief across as many
    messages as it takes, and can attach the feedback buttons that keep the
    learning loop reachable when there's no shell to run the CLI in. The
    Composio path truncates to one message and can't do either.
    """
    if str(target.options.get("via", "direct")).lower() == "composio":
        chat_id = target.options.get("chat_id")
        if not chat_id:
            raise DeliveryError("telegram delivery needs a 'chat_id'")
        result = client.execute(
            str(target.options.get("slug", "TELEGRAM_SEND_MESSAGE")),
            {"chat_id": chat_id, "text": _chat_text(digest), "parse_mode": "Markdown"},
        )
        if not result.ok:
            raise DeliveryError(result.error)
        return f"telegram: sent to {chat_id} (via composio, truncated to one message)"

    telegram = TelegramClient(
        token=target.options.get("bot_token"),
        chat_id=target.options.get("chat_id"),
    )
    if not telegram.configured:
        raise DeliveryError(
            "telegram needs TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID (or bot_token/chat_id in config)"
        )
    try:
        sent = send_digest(digest, telegram, buttons=bool(target.options.get("buttons", True)))
    except TelegramError as exc:
        raise DeliveryError(str(exc)) from exc
    return f"telegram: sent {sent} message(s) to {telegram.chat_id}"


def _discord(digest: Digest, target: DeliveryConfig, client: ComposioClient) -> str:
    channel_id = target.options.get("channel_id")
    if not channel_id:
        raise DeliveryError("discord delivery needs a 'channel_id'")
    result = client.execute(
        str(target.options.get("slug", "DISCORD_CREATE_MESSAGE")),
        {"channel_id": channel_id, "content": _chat_text(digest, limit=1900)},
    )
    if not result.ok:
        raise DeliveryError(result.error)
    return f"discord: posted to {channel_id}"


def _chat_text(digest: Digest, limit: int = 3500) -> str:
    """Markdown trimmed to fit a chat message, cut on a line boundary."""
    body = to_markdown(digest)
    if len(body) <= limit:
        return body
    cut = body[:limit].rsplit("\n", 1)[0]
    return cut + "\n\n_(truncated — see the full brief in your saved file)_"


_HANDLERS = {
    "console": _console,
    "file": _file,
    "gmail": _gmail,
    "slack": _slack,
    "telegram": _telegram,
    "discord": _discord,
}

KNOWN_KINDS = tuple(_HANDLERS)
