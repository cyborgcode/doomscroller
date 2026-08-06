"""Daily brief, triggered by Vercel Cron.

Wired up in `vercel.json`. Also callable by hand — `vercel crons run /api/cron`,
or a plain GET with the `Authorization: Bearer $CRON_SECRET` header — which is
how you test without waiting until tomorrow.
"""

from __future__ import annotations

import logging
import traceback
from http.server import BaseHTTPRequestHandler

from _runtime import authorized, load, respond

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("doomscroller.cron")


def run_brief() -> dict[str, object]:
    from doomscroller.delivery import deliver
    from doomscroller.learn import learn
    from doomscroller.pipeline import run as run_pipeline
    from doomscroller.sources import ComposioClient
    from doomscroller.store import Store

    config = load()
    with Store(config.db_path) as store:
        # Fold in anything tapped in Telegram since the last run, so today's
        # brief already reflects it.
        report = learn(store)
        digest = run_pipeline(config, store)
        results = deliver(digest, config.enabled_delivery, ComposioClient(user_id=config.user_id))

        return {
            "ok": True,
            "backend": store.backend,
            "learned": report.applied,
            "headlines": len(digest.clusters),
            "skimmed": len(digest.skimmed),
            "stats": {
                key: value
                for key, value in digest.stats.items()
                # The overview is prose and the errors are already logged;
                # neither belongs in a cron response body.
                if key not in ("overview", "source_errors")
            },
            "delivery": results,
        }


class handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if not authorized(self.headers):
            return respond(self, 401, {"ok": False, "error": "unauthorized"})
        try:
            result = run_brief()
        except Exception as exc:  # noqa: BLE001 - the response is the only log you'll read
            log.error("brief failed: %s\n%s", exc, traceback.format_exc())
            return respond(self, 500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

        log.info("brief delivered: %s", result.get("delivery"))
        respond(self, 200, result)

    # Vercel Cron uses GET; POST is here so `curl -X POST` also works.
    do_POST = do_GET
