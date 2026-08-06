"""Command line entry point."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import __version__
from .config import Config, ConfigError, load_config
from .delivery import deliver
from .learn import learn, profile_summary
from .pipeline import run as run_pipeline
from .providers import ProviderUnavailable, build_provider
from .render import RENDERERS
from .sources import build_source
from .sources.composio_client import ComposioClient, ComposioUnavailable
from .store import VALID_SIGNALS, Store


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env support so secrets don't have to live in the shell profile."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="doomscroller",
        description="Read your feeds so you don't have to.",
    )
    parser.add_argument("--config", help="path to config.yaml")
    parser.add_argument("-v", "--verbose", action="store_true", help="log what each stage is doing")
    parser.add_argument("--version", action="version", version=f"doomscroller {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    brief = subparsers.add_parser("brief", help="build and deliver a digest (default)")
    brief.add_argument("--hours", type=int, help="override the config time window")
    brief.add_argument("--no-llm", action="store_true", help="skip the model entirely; heuristics only")
    brief.add_argument("--dry-run", action="store_true", help="print to the terminal, deliver nowhere")
    brief.add_argument("--include-seen", action="store_true", help="re-rank items from earlier runs")
    brief.add_argument("--format", choices=sorted(RENDERERS), help="override the dry-run format")

    feedback = subparsers.add_parser("feedback", help="teach the ranker what you liked")
    feedback.add_argument("item_id", help="the short id printed beside each entry")
    feedback.add_argument("signal", choices=sorted(VALID_SIGNALS))
    feedback.add_argument("--note", default="", help="optional note to yourself")

    subparsers.add_parser("learn", help="fold pending feedback into the interest profile")
    profile = subparsers.add_parser("profile", help="show what the bot thinks you care about")
    profile.add_argument("--limit", type=int, default=12)

    tools = subparsers.add_parser("tools", help="list the real Composio tool slugs for a toolkit")
    tools.add_argument("toolkit", help="e.g. reddit, gmail, hackernews")

    auth = subparsers.add_parser("auth", help="connect an account through Composio")
    auth.add_argument("toolkit", help="e.g. reddit, gmail, twitter")

    check = subparsers.add_parser("check", help="verify config, credentials, and each source")
    check.add_argument("--probe", action="store_true", help="actually fetch one item per source")

    history = subparsers.add_parser("history", help="what you were shown recently")
    history.add_argument("--days", type=int, default=3)

    telegram = subparsers.add_parser("telegram", help="set up Telegram delivery")
    telegram.add_argument(
        "action", choices=["check", "setup", "test", "unhook"],
        help="check: verify token+chat. setup: register the webhook. "
             "test: send a message. unhook: remove the webhook.",
    )
    telegram.add_argument("--url", help="deployment base URL, for `setup`")

    return parser


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    command = args.command or "brief"
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    handlers = {
        "brief": cmd_brief,
        "feedback": cmd_feedback,
        "learn": cmd_learn,
        "profile": cmd_profile,
        "tools": cmd_tools,
        "auth": cmd_auth,
        "check": cmd_check,
        "history": cmd_history,
        "telegram": cmd_telegram,
    }
    return handlers[command](args, config)


# -- commands ------------------------------------------------------------


def cmd_brief(args: argparse.Namespace, config: Config) -> int:
    if getattr(args, "hours", None):
        config.window_hours = args.hours

    with Store(config.db_path) as store:
        # Learn from anything you reacted to since the last run, so this brief
        # already reflects it.
        report = learn(store)
        if report.applied and args.verbose:
            print(f"[{report}]", file=sys.stderr)

        digest = run_pipeline(
            config,
            store,
            use_llm=not args.no_llm,
            include_seen=args.include_seen,
        )

        if args.dry_run:
            renderer = RENDERERS[args.format or "terminal"]
            print(renderer(digest))
            return 0

        client = ComposioClient(user_id=config.user_id)
        for line in deliver(digest, config.enabled_delivery, client):
            if not line.startswith("console:"):
                print(line, file=sys.stderr)
    return 0


def cmd_feedback(args: argparse.Namespace, config: Config) -> int:
    with Store(config.db_path) as store:
        if store.get_item(args.item_id) is None:
            print(
                f"no item {args.item_id!r} on record — ids come from the brief and "
                "expire when the store is pruned",
                file=sys.stderr,
            )
            return 1
        store.add_feedback(args.item_id, args.signal, args.note)
        print(f"recorded {args.signal} for {args.item_id}")
    return 0


def cmd_learn(args: argparse.Namespace, config: Config) -> int:
    with Store(config.db_path) as store:
        print(learn(store))
    return 0


def cmd_profile(args: argparse.Namespace, config: Config) -> int:
    with Store(config.db_path) as store:
        summary = profile_summary(store, args.limit)
        counts = store.feedback_counts()

    if not any(summary.values()):
        print(
            "The profile is empty. React to a few items with "
            "`doomscroller feedback <id> up|down` and it will start to take shape."
        )
        return 0

    if counts:
        total = sum(counts.values())
        detail = ", ".join(f"{signal} {count}" for signal, count in sorted(counts.items()))
        print(f"Learned from {total} signal(s): {detail}\n")

    labels = {
        "topics_up": "Topics you want",
        "topics_down": "Topics you don't",
        "tokens_up": "Recurring vocabulary",
        "sources_up": "Sources earning their place",
        "sources_down": "Sources on thin ice",
    }
    for key, label in labels.items():
        rows = summary.get(key) or []
        if not rows:
            continue
        print(label)
        for name, weight in rows:
            print(f"  {weight:+7.2f}  {name}")
        print()
    return 0


def cmd_tools(args: argparse.Namespace, config: Config) -> int:
    client = ComposioClient(user_id=config.user_id)
    try:
        tools = client.list_tools(args.toolkit)
    except ComposioUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not tools:
        print(f"no tools found for {args.toolkit!r}")
        return 1
    print(f"{len(tools)} tool(s) in {args.toolkit}:\n")
    for tool in sorted(tools, key=lambda t: str(t["slug"])):
        print(f"  {tool['slug']}")
        if tool["description"]:
            print(f"      {tool['description']}")
    print("\nPut the one you want in config.yaml as `slug:` on the source.")
    return 0


def cmd_auth(args: argparse.Namespace, config: Config) -> int:
    client = ComposioClient(user_id=config.user_id)
    try:
        url = client.authorize(args.toolkit)
    except ComposioUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Open this to connect {args.toolkit}:\n\n  {url}\n")
    print("Once you've approved it, `doomscroller check --probe` should show the source as live.")
    return 0


def cmd_check(args: argparse.Namespace, config: Config) -> int:
    problems = 0

    print(f"config      user_id={config.user_id} window={config.window_hours}h db={config.db_path}")

    provider_keys = {
        "nvidia_nim": ("NVIDIA_API_KEY", "NIM_API_KEY"),
        "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
    }
    try:
        provider = build_provider(config.models.provider, config.models.provider_options)
        wanted = provider_keys.get(provider.name, ())
        has_key = any(os.environ.get(name) for name in wanted)
        detail = f"{provider.name} ({config.models.triage})"
        if has_key:
            note = "ok"
            if not provider.supports_prompt_caching:
                note += " — no prompt caching, so the triage prompt is re-sent per batch"
        else:
            note = f"MISSING {wanted[0] if wanted else 'credentials'} — digests fall back to heuristics"
        print(f"model       {detail}: {note}")
        problems += 0 if has_key else 1
    except ProviderUnavailable as exc:
        print(f"model       {exc}")
        problems += 1

    client = ComposioClient(user_id=config.user_id)
    print(f"composio    {'ok' if client.available else 'MISSING COMPOSIO_API_KEY'}")
    problems += 0 if client.available else 1

    if not config.enabled_sources:
        print("sources     none enabled")
        problems += 1

    for source_config in config.enabled_sources:
        label = f"  {source_config.id:<22}"
        try:
            source = build_source(source_config, client)
        except Exception as exc:  # noqa: BLE001
            print(f"{label} config error: {exc}")
            problems += 1
            continue
        if not args.probe:
            slug = getattr(source, "slug", "") or getattr(source, "url", "")
            print(f"{label} ok ({source.platform}{' ' + slug if slug else ''})")
            continue
        try:
            items = source.fetch(config.window_hours)
        except Exception as exc:  # noqa: BLE001
            print(f"{label} FETCH FAILED: {exc}")
            problems += 1
            continue
        sample = items[0].title[:60] if items else "(no items in window)"
        print(f"{label} {len(items):>3} items — {sample}")

    for target in config.enabled_delivery:
        print(f"  delivery:{target.kind:<13} configured")

    print("\n" + ("all good" if problems == 0 else f"{problems} problem(s) to fix"))
    return 0 if problems == 0 else 1


def cmd_history(args: argparse.Namespace, config: Config) -> int:
    since = datetime.now(timezone.utc) - timedelta(days=args.days)
    with Store(config.db_path) as store:
        rows = store.shown_since(since)
    if not rows:
        print(f"nothing shown in the last {args.days} day(s)")
        return 0
    current = ""
    for row in rows:
        day = str(row["digest_at"])[:10]
        if day != current:
            current = day
            print(f"\n{day}")
        marker = "*" if row["slot"] == "headline" else " "
        print(f"  {marker} [{row['item_id']}] {str(row['title'])[:66]}")
    print("\n* = headline. React with: doomscroller feedback <id> up|down|save|mute")
    return 0


def cmd_telegram(args: argparse.Namespace, config: Config) -> int:
    from .telegram import TelegramClient, TelegramError

    client = TelegramClient()
    if not client.token:
        print(
            "TELEGRAM_BOT_TOKEN is not set.\n"
            "  1. Message @BotFather on Telegram, send /newbot, follow the prompts.\n"
            "  2. Put the token it gives you in .env as TELEGRAM_BOT_TOKEN.",
            file=sys.stderr,
        )
        return 1

    try:
        if args.action == "check":
            me = client.get_me()
            print(f"bot      @{me.get('username')} ({me.get('first_name')})")
            if not client.chat_id:
                print(
                    "chat_id  NOT SET — send your bot any message, then open\n"
                    f"         https://api.telegram.org/bot{client.token}/getUpdates\n"
                    "         and copy result[0].message.chat.id into TELEGRAM_CHAT_ID"
                )
                return 1
            print(f"chat_id  {client.chat_id}")
            return 0

        if args.action == "test":
            client.send_message("<b>doomscroller</b> is wired up correctly. ✅")
            print(f"sent a test message to {client.chat_id}")
            return 0

        if args.action == "unhook":
            client.delete_webhook()
            print("webhook removed — button taps will no longer be recorded")
            return 0

        if not args.url:
            print("setup needs --url https://<your-project>.vercel.app", file=sys.stderr)
            return 1
        endpoint = args.url.rstrip("/") + "/api/telegram"
        secret = os.environ.get("TELEGRAM_WEBHOOK_SECRET")
        client.set_webhook(endpoint, secret)
        print(f"webhook registered: {endpoint}")
        if not secret:
            print(
                "warning: TELEGRAM_WEBHOOK_SECRET is not set, so anyone who finds that "
                "URL can post feedback into your profile. Set it in both .env and Vercel.",
                file=sys.stderr,
            )
        return 0
    except TelegramError as exc:
        print(f"telegram error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
