"""Turning a `Digest` into something readable.

Three renderers: terminal (what you get by default), Markdown (files, commits,
chat), and a self-contained HTML page (email, browser). All three show the
short item id next to each entry, because that id is what you pass back to
`doomscroller feedback` to teach the ranker.
"""

from __future__ import annotations

import html

from .models import Cluster, Digest

# -- terminal ------------------------------------------------------------

_BOLD = "\033[1m"
_DIM = "\033[2m"
_CYAN = "\033[36m"
_RESET = "\033[0m"


def to_terminal(digest: Digest, color: bool = True) -> str:
    def style(text: str, code: str) -> str:
        return f"{code}{text}{_RESET}" if color else text

    lines: list[str] = []
    stamp = digest.generated_at.astimezone().strftime("%a %d %b, %H:%M")
    lines.append(style(f"Your brief — {stamp} (last {digest.window_hours}h)", _BOLD))

    overview = str(digest.stats.get("overview") or "")
    if overview:
        lines += ["", _wrap(overview, indent="")]

    if digest.is_empty:
        lines += ["", "Nothing worth your time came through. That's a good day."]
        return "\n".join(lines) + "\n" + _terminal_stats(digest, style)

    if digest.clusters:
        lines.append("")
        for index, cluster in enumerate(digest.clusters, 1):
            lines.append(f"{style(f'{index}.', _BOLD)} {style(cluster.headline, _BOLD)}")
            if cluster.body:
                lines.append(_wrap(cluster.body, indent="   "))
            for claim in cluster.claims[:3]:
                lines.append(_wrap(f"• {claim}", indent="   ", hanging="     "))
            lines.append(
                style(
                    f"   {_source_line(cluster)}  [{cluster.lead.item.id}]",
                    _DIM,
                )
            )
            if cluster.lead.item.url:
                lines.append(style(f"   {cluster.lead.item.url}", _CYAN))
            lines.append("")

    if digest.skimmed:
        lines.append(style("Also, briefly", _BOLD))
        for entry in digest.skimmed:
            summary = entry.verdict.summary or entry.item.title
            lines.append(_wrap(f"· {summary}", indent="  ", hanging="    "))
            lines.append(style(f"    {entry.item.source}  [{entry.item.id}]", _DIM))
        lines.append("")

    lines.append(_terminal_stats(digest, style))
    return "\n".join(lines)


def _terminal_stats(digest: Digest, style) -> str:
    stats = digest.stats
    parts = [
        f"{stats.get('fetched', 0)} fetched",
        f"{stats.get('new', 0)} new",
        f"{stats.get('filtered_out', 0)} filtered out",
        f"{stats.get('muted', 0)} muted",
        f"{stats.get('tool_calls', 0)} tool calls",
    ]
    tokens = int(stats.get("input_tokens", 0) or 0) + int(stats.get("output_tokens", 0) or 0)
    if tokens:
        cached = int(stats.get("cached_tokens", 0) or 0)
        parts.append(f"{tokens:,} tokens" + (f" ({cached:,} cached)" if cached else ""))
    line = style("  ·  ".join(parts), _DIM)

    errors = stats.get("source_errors") or []
    if errors:
        line += "\n" + style(f"warnings: {'; '.join(str(e) for e in errors)}", _DIM)
    line += "\n" + style("react with: doomscroller feedback <id> up|down|save|mute", _DIM)
    return line


# -- markdown ------------------------------------------------------------


def to_markdown(digest: Digest) -> str:
    stamp = digest.generated_at.astimezone().strftime("%A %d %B %Y, %H:%M")
    lines = ["# Your brief", "", f"*{stamp} — last {digest.window_hours} hours*", ""]

    overview = str(digest.stats.get("overview") or "")
    if overview:
        lines += [overview, ""]

    if digest.is_empty:
        lines += ["Nothing worth your time came through today.", ""]
        return "\n".join(lines) + _markdown_stats(digest)

    for index, cluster in enumerate(digest.clusters, 1):
        link = f"[{cluster.headline}]({cluster.lead.item.url})" if cluster.lead.item.url else cluster.headline
        lines.append(f"## {index}. {link}")
        lines.append("")
        if cluster.body:
            lines += [cluster.body, ""]
        for claim in cluster.claims[:4]:
            lines.append(f"- {claim}")
        if cluster.claims:
            lines.append("")
        lines.append(f"<sub>{_source_line(cluster)} · `{cluster.lead.item.id}`</sub>")
        lines.append("")

    if digest.skimmed:
        lines += ["## Also, briefly", ""]
        for entry in digest.skimmed:
            summary = entry.verdict.summary or entry.item.title
            link = f"[{summary}]({entry.item.url})" if entry.item.url else summary
            lines.append(f"- {link} <sub>{entry.item.source} · `{entry.item.id}`</sub>")
        lines.append("")

    return "\n".join(lines) + _markdown_stats(digest)


def _markdown_stats(digest: Digest) -> str:
    stats = digest.stats
    return (
        "\n---\n\n"
        f"<sub>{stats.get('fetched', 0)} items fetched · {stats.get('new', 0)} new · "
        f"{stats.get('filtered_out', 0)} filtered out · {stats.get('muted', 0)} muted · "
        f"{stats.get('tool_calls', 0)} Composio calls</sub>\n"
    )


# -- html ----------------------------------------------------------------

_CSS = """
:root { color-scheme: light dark; --fg:#16181d; --muted:#5f6672; --bg:#fbfbfa;
        --card:#fff; --line:#e6e6e3; --accent:#3b5bdb; }
@media (prefers-color-scheme: dark) {
  :root { --fg:#e8e8e6; --muted:#9aa0ab; --bg:#16181c; --card:#1e2126;
          --line:#2c3038; --accent:#8da2fb; }
}
* { box-sizing: border-box; }
body { margin:0; padding:2rem 1rem; background:var(--bg); color:var(--fg);
       font:16px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
.wrap { max-width: 42rem; margin:0 auto; }
h1 { font-size:1.5rem; margin:0 0 .25rem; letter-spacing:-.01em; }
.stamp { color:var(--muted); font-size:.85rem; margin-bottom:1.5rem; }
.overview { font-size:1.05rem; padding:1rem 1.1rem; background:var(--card);
            border:1px solid var(--line); border-left:3px solid var(--accent);
            border-radius:6px; margin-bottom:1.75rem; }
article { background:var(--card); border:1px solid var(--line); border-radius:8px;
          padding:1.1rem 1.25rem; margin-bottom:1rem; }
article h2 { font-size:1.05rem; margin:0 0 .5rem; line-height:1.35; }
article h2 a { color:var(--fg); text-decoration:none; }
article h2 a:hover { color:var(--accent); }
article p { margin:0 0 .65rem; }
ul { margin:.5rem 0; padding-left:1.15rem; }
li { margin:.2rem 0; }
.meta { color:var(--muted); font-size:.8rem; font-family:ui-monospace, SFMono-Regular, Menlo, monospace; }
h3 { font-size:.95rem; text-transform:uppercase; letter-spacing:.06em;
     color:var(--muted); margin:2rem 0 .75rem; }
.skim li { margin:.5rem 0; }
.skim a { color:var(--fg); }
footer { color:var(--muted); font-size:.78rem; margin-top:2rem;
         border-top:1px solid var(--line); padding-top:1rem; }
"""


def to_html(digest: Digest) -> str:
    stamp = digest.generated_at.astimezone().strftime("%A %d %B %Y, %H:%M")
    out = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>Your brief</title>",
        f"<style>{_CSS}</style></head><body><div class='wrap'>",
        "<h1>Your brief</h1>",
        f"<div class='stamp'>{_esc(stamp)} — last {digest.window_hours} hours</div>",
    ]

    overview = str(digest.stats.get("overview") or "")
    if overview:
        out.append(f"<div class='overview'>{_esc(overview)}</div>")

    if digest.is_empty:
        out.append("<article><p>Nothing worth your time came through today.</p></article>")

    for cluster in digest.clusters:
        headline = _esc(cluster.headline)
        if cluster.lead.item.url:
            headline = f"<a href='{_esc(cluster.lead.item.url)}'>{headline}</a>"
        out.append(f"<article><h2>{headline}</h2>")
        if cluster.body:
            out.append(f"<p>{_esc(cluster.body)}</p>")
        if cluster.claims:
            claims = "".join(f"<li>{_esc(claim)}</li>" for claim in cluster.claims[:4])
            out.append(f"<ul>{claims}</ul>")
        out.append(
            f"<div class='meta'>{_esc(_source_line(cluster))} · {_esc(cluster.lead.item.id)}</div></article>"
        )

    if digest.skimmed:
        out.append("<h3>Also, briefly</h3><ul class='skim'>")
        for entry in digest.skimmed:
            summary = _esc(entry.verdict.summary or entry.item.title)
            if entry.item.url:
                summary = f"<a href='{_esc(entry.item.url)}'>{summary}</a>"
            out.append(
                f"<li>{summary}<br><span class='meta'>{_esc(entry.item.source)} · {_esc(entry.item.id)}</span></li>"
            )
        out.append("</ul>")

    stats = digest.stats
    out.append(
        "<footer>"
        f"{stats.get('fetched', 0)} fetched · {stats.get('new', 0)} new · "
        f"{stats.get('filtered_out', 0)} filtered out · {stats.get('muted', 0)} muted · "
        f"{stats.get('tool_calls', 0)} Composio calls"
        "</footer></div></body></html>"
    )
    return "".join(out)


# -- shared --------------------------------------------------------------


def _source_line(cluster: Cluster) -> str:
    sources = cluster.sources
    if len(sources) == 1:
        return sources[0]
    shown = ", ".join(sources[:3])
    extra = f" +{len(sources) - 3}" if len(sources) > 3 else ""
    return f"{len(sources)} sources: {shown}{extra}"


def _wrap(text: str, indent: str = "", hanging: str | None = None, width: int = 84) -> str:
    import textwrap

    return textwrap.fill(
        text,
        width=width,
        initial_indent=indent,
        subsequent_indent=hanging if hanging is not None else indent,
    )


def _esc(text: object) -> str:
    return html.escape(str(text), quote=True)


RENDERERS = {"terminal": to_terminal, "markdown": to_markdown, "html": to_html}
