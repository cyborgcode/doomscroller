"""Telegram delivery and the feedback webhook.

The two properties that matter: nothing is silently dropped on the way out
(Telegram's 4096-character cap is a real ceiling, and this is the only channel
when it's hosted), and a button tap records exactly what the CLI would.
"""

from __future__ import annotations

import json

import pytest

from doomscroller.models import Digest
from doomscroller.pipeline.dedup import cluster_items
from doomscroller.telegram import (
    SAFE_LIMIT,
    TelegramClient,
    TelegramError,
    build_messages,
    chunk,
    feedback_buttons,
    parse_callback,
    render_cluster,
    send_digest,
)
from tests.conftest import make_item, make_scored


class FakeTelegram(TelegramClient):
    """Records what would have been sent instead of sending it."""

    def __init__(self, fail_on: int | None = None) -> None:
        super().__init__(token="fake-token", chat_id="123")
        self.sent: list[tuple[str, object]] = []
        self.acks: list[tuple[str, str]] = []
        self.fail_on = fail_on

    def send_message(self, text, *, buttons=None, preview=False):
        if self.fail_on is not None and len(self.sent) == self.fail_on:
            raise TelegramError("Bad Request: message is too long")
        self.sent.append((text, buttons))
        return {"message_id": len(self.sent)}

    def answer_callback(self, callback_id, text=""):
        self.acks.append((callback_id, text))


SUBJECTS = [
    "Postgres 18 ships asynchronous I/O",
    "Ferrari signs a driver for next season",
    "Rust stabilises async closures",
    "Kubernetes drops dockershim support",
    "The EU passes new AI transparency rules",
    "SQLite adds a native JSONB format",
    "Chrome ships baseline WebGPU",
    "A major CDN outage takes down banking",
]


def digest_with(headlines: int = 2, skims: int = 2, body: str = "A short body.") -> Digest:
    # Distinct subjects on purpose: near-identical titles would (correctly)
    # collapse into one cluster and the fixture wouldn't test what it claims to.
    scored = [
        make_scored(
            make_item(
                title=SUBJECTS[n % len(SUBJECTS)] + f" ({n})",
                external_id=str(n),
                url=f"https://a.test/{n}",
            ),
            summary=SUBJECTS[n % len(SUBJECTS)] + ".",
            topics=[f"topic-{n}"],
            score=10 - n,
        )
        for n in range(headlines + skims)
    ]
    clusters = cluster_items(scored)
    for cluster in clusters:
        cluster.body = body
    digest = Digest(clusters=clusters[:headlines], skimmed=[c.lead for c in clusters[headlines:]])
    digest.stats = {"fetched": 40, "new": 12, "filtered_out": 6, "muted": 2,
                    "overview": "A quiet day."}
    return digest


# -- chunking ------------------------------------------------------------


def test_short_text_is_one_chunk():
    assert chunk("hello") == ["hello"]


def test_empty_text_produces_nothing():
    assert chunk("") == []


def test_chunks_respect_the_limit():
    text = "\n\n".join(f"paragraph {n} " + "x" * 200 for n in range(60))
    parts = chunk(text)
    assert len(parts) > 1
    assert all(len(part) <= SAFE_LIMIT for part in parts)


def test_chunking_loses_no_content():
    """The whole point of splitting rather than truncating."""
    text = "\n\n".join(f"paragraph {n} " + "y" * 300 for n in range(40))
    assert sum(len(p) for p in chunk(text)) >= len(text) - 2 * len(chunk(text))
    assert "paragraph 39" in "".join(chunk(text))


def test_a_single_oversized_line_is_split_not_dropped():
    parts = chunk("z" * (SAFE_LIMIT * 3))
    assert all(len(part) <= SAFE_LIMIT for part in parts)
    assert sum(len(part) for part in parts) == SAFE_LIMIT * 3


def test_safe_limit_is_under_telegrams_hard_cap():
    assert SAFE_LIMIT < 4096


# -- message building ----------------------------------------------------


def test_every_headline_gets_its_own_feedback_buttons():
    messages = build_messages(digest_with(headlines=3, skims=0))
    with_buttons = [(t, b) for t, b in messages if b]
    assert len(with_buttons) == 3
    for _, markup in with_buttons:
        labels = [button["text"] for button in markup[0]]
        assert labels == ["👍", "👎", "🔇"]


def test_buttons_carry_the_item_id_the_webhook_needs():
    digest = digest_with(headlines=1, skims=0)
    item_id = digest.clusters[0].lead.item.id
    markup = feedback_buttons(item_id)
    assert [b["callback_data"] for b in markup[0]] == [
        f"up:{item_id}",
        f"down:{item_id}",
        f"mute:{item_id}",
    ]


def test_callback_data_fits_telegrams_64_byte_cap():
    for button in feedback_buttons("f" * 16)[0]:
        assert len(button["callback_data"].encode()) <= 64


def test_buttons_can_be_disabled():
    messages = build_messages(digest_with(), buttons=False)
    assert all(markup is None for _, markup in messages)


def test_every_message_is_within_the_limit():
    digest = digest_with(headlines=8, skims=8, body="A much longer body. " * 120)
    for text, _ in build_messages(digest):
        assert len(text) <= SAFE_LIMIT


def test_the_overview_and_footer_both_appear():
    texts = "\n".join(text for text, _ in build_messages(digest_with()))
    assert "A quiet day." in texts
    assert "40 fetched" in texts


def test_an_empty_digest_still_says_something():
    empty = Digest(stats={"fetched": 0, "new": 0})
    texts = "\n".join(text for text, _ in build_messages(empty))
    assert "Nothing worth your time" in texts


def test_html_in_a_headline_is_escaped():
    digest = digest_with(headlines=1, skims=0)
    digest.clusters[0].headline = "<script>alert(1)</script> & more"
    rendered = render_cluster(digest.clusters[0], 1)
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered and "&amp;" in rendered


def test_a_long_entry_keeps_its_buttons_on_the_final_part():
    digest = digest_with(headlines=1, skims=0, body="Very long body. " * 400)
    messages = build_messages(digest)
    entry_parts = [m for m in messages if "Postgres" in m[0] or m[1]]
    assert entry_parts[-1][1] is not None, "buttons must sit under the whole entry"


# -- sending -------------------------------------------------------------


def test_send_digest_sends_every_message():
    client = FakeTelegram()
    digest = digest_with(headlines=3, skims=4)
    assert send_digest(digest, client) == len(client.sent)
    assert len(client.sent) >= 5  # header + 3 entries + skims + footer


def test_a_send_failure_surfaces_rather_than_silently_half_delivering():
    client = FakeTelegram(fail_on=2)
    with pytest.raises(TelegramError):
        send_digest(digest_with(headlines=3), client)


def test_client_is_unconfigured_without_both_token_and_chat(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    assert not TelegramClient().configured
    assert not TelegramClient(token="t").configured
    assert TelegramClient(token="t", chat_id="1").configured


def test_a_missing_token_explains_where_to_get_one(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    with pytest.raises(TelegramError, match="BotFather"):
        TelegramClient(chat_id="1").send_message("hi")


# -- inbound callbacks ---------------------------------------------------


def test_parse_callback_extracts_signal_and_item():
    update = {"callback_query": {"id": "cb1", "data": "up:abc123"}}
    assert parse_callback(update) == ("up", "abc123", "cb1")


@pytest.mark.parametrize(
    "update",
    [
        {},
        {"message": {"text": "hello"}},
        {"callback_query": {"id": "x", "data": "malformed"}},
        {"callback_query": {"id": "x", "data": ":abc"}},
        {"callback_query": {"id": "x", "data": "up:"}},
        {"callback_query": "not a dict"},
    ],
)
def test_parse_callback_returns_none_for_anything_else(update):
    """A webhook must not raise on updates it wasn't expecting."""
    assert parse_callback(update) is None


# -- the webhook handler -------------------------------------------------


def _load_handler(monkeypatch, db_path):
    """Import the Vercel handler and point it at a scratch database."""
    import importlib
    import sys
    from pathlib import Path

    from doomscroller.config import Config

    api = Path(__file__).resolve().parent.parent / "api"
    if str(api) not in sys.path:
        sys.path.insert(0, str(api))
    monkeypatch.delenv("DOOMSCROLLER_DB_URL", raising=False)
    monkeypatch.delenv("TURSO_DATABASE_URL", raising=False)

    module = importlib.import_module("telegram")
    config = Config()
    config.db_path = db_path
    monkeypatch.setattr(module, "load", lambda: config)
    return module


def test_a_button_tap_records_feedback(monkeypatch, tmp_path):
    """The whole reason the webhook exists: hosted, this replaces the CLI."""
    from doomscroller import telegram as tg
    from doomscroller.store import Store

    db_path = tmp_path / "hook.db"
    item = make_item()
    with Store(db_path) as store:
        store.record_items([item])

    acks: list[str] = []
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake-token")
    monkeypatch.setattr(
        tg.TelegramClient, "answer_callback", lambda self, cid, text="": acks.append(text)
    )
    module = _load_handler(monkeypatch, db_path)

    result = module.handle_update({"callback_query": {"id": "cb", "data": f"up:{item.id}"}})

    assert result["ok"] and result["signal"] == "up"
    with Store(db_path) as store:
        pending = store.pending_feedback()
    assert len(pending) == 1 and pending[0]["signal"] == "up"
    assert acks and "More like this" in acks[0]


def test_a_tap_on_an_expired_item_says_so_instead_of_failing(monkeypatch, tmp_path):
    from doomscroller import telegram as tg

    acks: list[str] = []
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake-token")
    monkeypatch.setattr(
        tg.TelegramClient, "answer_callback", lambda self, cid, text="": acks.append(text)
    )
    module = _load_handler(monkeypatch, tmp_path / "empty.db")

    result = module.handle_update({"callback_query": {"id": "cb", "data": "up:deadbeef"}})
    assert result["ok"] and "aged out" in result["note"]
    assert acks and "aged out" in acks[0]


def test_an_unknown_signal_is_rejected(monkeypatch, tmp_path):
    from doomscroller import telegram as tg

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake-token")
    monkeypatch.setattr(tg.TelegramClient, "answer_callback", lambda self, cid, text="": None)
    module = _load_handler(monkeypatch, tmp_path / "x.db")
    result = module.handle_update({"callback_query": {"id": "cb", "data": "explode:abc"}})
    assert not result["ok"]


def test_a_failed_acknowledgement_does_not_lose_recorded_feedback(monkeypatch, tmp_path):
    """The write already happened; a Telegram hiccup must not report failure."""
    from doomscroller import telegram as tg
    from doomscroller.store import Store

    db_path = tmp_path / "ack.db"
    item = make_item()
    with Store(db_path) as store:
        store.record_items([item])

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake-token")
    monkeypatch.setattr(
        tg.TelegramClient,
        "answer_callback",
        lambda self, cid, text="": (_ for _ in ()).throw(TelegramError("network down")),
    )
    module = _load_handler(monkeypatch, db_path)

    result = module.handle_update({"callback_query": {"id": "cb", "data": f"up:{item.id}"}})
    assert result["ok"]
    with Store(db_path) as store:
        assert len(store.pending_feedback()) == 1


def test_the_webhook_ignores_updates_that_are_not_feedback(monkeypatch, tmp_path):
    module = _load_handler(monkeypatch, tmp_path / "x.db")
    assert module.handle_update({"message": {"text": "hi"}}) == {"ok": True, "ignored": True}


def test_the_webhook_secret_is_enforced(monkeypatch, tmp_path):
    module = _load_handler(monkeypatch, tmp_path / "x.db")
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "s3cret")

    assert not module._secret_ok({"X-Telegram-Bot-Api-Secret-Token": "wrong"})
    assert module._secret_ok({"X-Telegram-Bot-Api-Secret-Token": "s3cret"})
    assert module._secret_ok({"x-telegram-bot-api-secret-token": "s3cret"})


def test_no_secret_configured_means_open(monkeypatch, tmp_path):
    module = _load_handler(monkeypatch, tmp_path / "x.db")
    monkeypatch.delenv("TELEGRAM_WEBHOOK_SECRET", raising=False)
    assert module._secret_ok({})


# -- the cron handler ----------------------------------------------------


def test_cron_requires_the_secret_when_one_is_set(monkeypatch, tmp_path):
    import importlib
    import sys
    from pathlib import Path

    api = Path(__file__).resolve().parent.parent / "api"
    if str(api) not in sys.path:
        sys.path.insert(0, str(api))
    runtime = importlib.import_module("_runtime")

    monkeypatch.setenv("CRON_SECRET", "topsecret")
    assert runtime.authorized({"Authorization": "Bearer topsecret"})
    assert runtime.authorized({"authorization": "Bearer topsecret"})
    assert not runtime.authorized({"Authorization": "Bearer wrong"})
    assert not runtime.authorized({})

    monkeypatch.delenv("CRON_SECRET", raising=False)
    assert runtime.authorized({}), "unset secret means open, so a first deploy works"


def test_serverless_refuses_to_run_against_a_local_sqlite_file(monkeypatch, tmp_path):
    """Silently forgetting everything between runs is worse than failing loudly."""
    import importlib
    import sys
    from pathlib import Path

    api = Path(__file__).resolve().parent.parent / "api"
    if str(api) not in sys.path:
        sys.path.insert(0, str(api))
    runtime = importlib.import_module("_runtime")

    (tmp_path / "config.yaml").write_text("sources: []\n")
    monkeypatch.setenv("DOOMSCROLLER_CONFIG", str(tmp_path / "config.yaml"))
    monkeypatch.delenv("DOOMSCROLLER_DB_URL", raising=False)
    monkeypatch.delenv("TURSO_DATABASE_URL", raising=False)

    monkeypatch.delenv("DOOMSCROLLER_ALLOW_LOCAL_DB", raising=False)
    with pytest.raises(RuntimeError, match="no persistent disk"):
        runtime.load()

    # A file path in the env var is still a file path — setting the variable
    # isn't the same as configuring a remote database.
    monkeypatch.setenv("DOOMSCROLLER_DB_URL", "/tmp/not-remote.db")
    with pytest.raises(RuntimeError, match="local file"):
        runtime.load()

    monkeypatch.setenv("DOOMSCROLLER_DB_URL", "libsql://x.turso.io")
    assert runtime.load().window_hours == 24

    # ...and the local escape hatch still works for running handlers by hand.
    monkeypatch.setenv("DOOMSCROLLER_DB_URL", "/tmp/not-remote.db")
    monkeypatch.setenv("DOOMSCROLLER_ALLOW_LOCAL_DB", "1")
    assert runtime.load().window_hours == 24


def test_vercel_json_is_valid_and_wires_both_functions():
    from pathlib import Path

    config = json.loads((Path(__file__).resolve().parent.parent / "vercel.json").read_text())
    assert config["crons"][0]["path"] == "/api/cron"
    # Hobby plan allows one run per day; anything more frequent fails at deploy.
    assert config["crons"][0]["schedule"].split()[1] != "*"
    assert set(config["functions"]) == {"api/cron.py", "api/telegram.py"}
    assert config["functions"]["api/cron.py"]["maxDuration"] >= 60
