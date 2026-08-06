"""Telegram webhook — this is what keeps the bot learning.

Hosted on a schedule with no shell, `doomscroller feedback` is unreachable, so
the 👍 / 👎 / 🔇 buttons under each headline post here instead. A tap records
exactly the same signal the CLI would, and it's applied to the profile at the
start of the next brief.

Register it once, after deploying:

    curl "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/setWebhook" \\
      -d url="https://<your-project>.vercel.app/api/telegram" \\
      -d secret_token="$TELEGRAM_WEBHOOK_SECRET"

or `doomscroller telegram setup --url https://<your-project>.vercel.app`.
"""

from __future__ import annotations

import logging
import os
import traceback
from http.server import BaseHTTPRequestHandler

from _runtime import load, read_json, respond

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("doomscroller.telegram")


def _secret_ok(headers) -> bool:
    """Telegram echoes the secret you registered on every update.

    Without this the endpoint is a public write to your profile — anyone who
    finds the URL could POST fabricated feedback and steer what you see.
    """
    expected = os.environ.get("TELEGRAM_WEBHOOK_SECRET")
    if not expected:
        return True
    supplied = headers.get("X-Telegram-Bot-Api-Secret-Token") or headers.get(
        "x-telegram-bot-api-secret-token"
    )
    return supplied == expected


def _ack(telegram, callback_id: str, note: str) -> None:
    """Best-effort acknowledgement.

    By the time this runs the feedback is already written. A failure here means
    a button that spins a little longer, not lost input — so it must never turn
    a recorded signal into a reported failure.
    """
    if not (callback_id and telegram.token):
        return
    try:
        telegram.answer_callback(callback_id, note)
    except Exception as exc:  # noqa: BLE001
        log.warning("could not acknowledge %s: %s", callback_id, exc)


def handle_update(update: dict) -> dict:
    from doomscroller.store import VALID_SIGNALS, Store
    from doomscroller.telegram import TelegramClient, describe_signal, parse_callback

    parsed = parse_callback(update)
    if parsed is None:
        # Anything that isn't a feedback tap — a plain message, a channel post —
        # is acknowledged and ignored, so Telegram stops retrying it.
        return {"ok": True, "ignored": True}

    signal, item_id, callback_id = parsed
    telegram = TelegramClient()

    if signal not in VALID_SIGNALS:
        _ack(telegram, callback_id, f"Unknown action: {signal}")
        return {"ok": False, "error": f"unknown signal {signal!r}"}

    config = load()
    with Store(config.db_path) as store:
        if store.get_item(item_id) is None:
            note = "That item has aged out of the store."
        else:
            store.add_feedback(item_id, signal, note="telegram")
            note = describe_signal(signal)

    # Acknowledge last: the button spins until this lands, so the confirmation
    # only appears once the write has actually happened.
    _ack(telegram, callback_id, note)
    return {"ok": True, "signal": signal, "item": item_id, "note": note}


class handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        if not _secret_ok(self.headers):
            return respond(self, 401, {"ok": False, "error": "bad secret token"})
        try:
            result = handle_update(read_json(self))
        except Exception as exc:  # noqa: BLE001
            log.error("webhook failed: %s\n%s", exc, traceback.format_exc())
            # 200 on purpose: a non-2xx makes Telegram retry the same update for
            # hours, and a failure here is ours to fix, not theirs to repeat.
            return respond(self, 200, {"ok": False, "error": str(exc)})
        respond(self, 200, result)

    def do_GET(self) -> None:
        """A health check, so you can confirm the route deployed at all."""
        respond(self, 200, {"ok": True, "endpoint": "telegram webhook", "method": "POST"})
