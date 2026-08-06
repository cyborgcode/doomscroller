# doomscroller

A personal bot that reads your feeds so you don't have to.

It pulls from your social accounts and newsletters through [Composio](https://composio.dev),
throws away the engagement bait, merges the same story told five times into one
entry, and hands back a short brief. It keeps a profile of what you actually
react to, so the filtering sharpens over time instead of staying generic.

```
Your brief — Thu 06 Aug, 09:12 (last 24h)

Quiet day outside databases; the Postgres 18 release is the only thing with legs.

1. Postgres 18 ships asynchronous I/O on Linux
   Corroborated across HN and r/rust. The io_uring path is opt-in behind a
   GUC, so upgrades won't change behaviour by default.
   • Async I/O uses io_uring on Linux, with a fallback worker pool elsewhere.
   • Sequential-scan read latency drops ~40% on NVMe in the release notes.
   3 sources: hackernews, reddit:rust, lobsters  [dcb6e5afe6007f71]
   https://www.postgresql.org/about/news/...

Also, briefly
  · Rust 1.90 stabilises async closures and 12 const fns.
    lobsters  [f169ef61fae5b6f2]

52 fetched · 31 new · 14 filtered out · 3 muted · 6 tool calls · 41,208 tokens (33,900 cached)
react with: doomscroller feedback <id> up|down|save|mute
```

## How it works

```
sources ──▶ mute ──▶ triage ──▶ score ──▶ noise floor ──▶ cluster ──▶ synthesise ──▶ deliver
(Composio)   (free)   (Claude)  (profile)   (config)      (dedup)      (Claude)      (Composio)
                          │                     ▲
                          └── cached in SQLite  └── learned from your feedback
```

**Triage** runs once per item: Claude classifies it (news / analysis / opinion /
promotion / drama / meme), scores how much of it is engagement machinery versus
information, extracts the standalone factual claims, and compresses it to one
line. Verdicts are cached by item id, so a post that shows up in tomorrow's
window too is never paid for twice.

**Scoring** combines six components — interest match, substance, a noise
penalty, freshness, engagement, and learned per-source trust. Every component's
contribution is recorded on the item, so a ranking can always be explained.

**Clustering** merges duplicates using canonical URLs, token overlap, and
character trigrams, so five accounts covering one story become one entry that
mentions the corroboration.

**Synthesis** runs once per brief over the survivors, writing the entries you
actually read — using only the claims extracted during triage, and flagging
where sources disagree or a claim rests on a single account.

## Setup

```bash
git clone https://github.com/cyborgcode/doomscroller && cd doomscroller
python -m venv .venv && source .venv/bin/activate
pip install -e .

cp .env.example .env              # add COMPOSIO_API_KEY and ANTHROPIC_API_KEY
cp config.example.yaml config.yaml

doomscroller check                # verify credentials and sources
doomscroller brief                # first brief
```

`config.yaml` ships with two RSS feeds enabled, which need no accounts and no
Composio quota — so `doomscroller brief` does something useful before you've
connected anything.

### Connecting accounts

```bash
doomscroller auth reddit          # prints an OAuth URL; open it, approve
doomscroller auth gmail
doomscroller check --probe        # actually fetches, shows item counts
```

Then enable the matching sources in `config.yaml`.

### Tool slugs

Each Composio-backed source calls one tool, and the defaults here are only
defaults — providers rename them. If a source returns nothing, find the real
slug and pin it:

```bash
doomscroller tools reddit
```

```yaml
- id: reddit:rust
  platform: reddit
  subreddit: rust
  slug: REDDIT_RETRIEVE_REDDIT_POST   # whatever `doomscroller tools` showed
```

## Teaching it

The bot starts from the `interests:` list in your config and gets more specific
from there. Every entry in a brief carries a short id; react to it and the
ranker adjusts.

```bash
doomscroller feedback dcb6e5af up      # more like this
doomscroller feedback 91c2ba07 down    # less like this
doomscroller feedback 4fe10a3b mute    # strongly less — the topic, not just the post
doomscroller feedback dcb6e5af save    # strongest positive

doomscroller profile                   # what it currently believes about you
doomscroller history --days 3          # what you were shown, with ids
```

Feedback is folded into the profile automatically at the start of the next
brief. Three weight tables are learned — vocabulary, topics, and per-source
trust — and each pass decays existing weights slightly before adding new
evidence. That decay is the point: interests you stop reinforcing fade, so the
profile tracks what you care about now rather than accumulating everything you
ever clicked.

Weights are scaled against the profile's own distribution rather than a fixed
constant, so a young profile discriminates just as sharply as a mature one —
without it, everything you taught it in week one would sit too close to neutral
to outrank freshness.

## Tuning

The two knobs that matter most:

```yaml
ranking:
  noise_ceiling: 0.72     # lower = harsher filter
  substance_floor: 0.25   # raise to demand more of every item
```

If too much gets through, lower `noise_ceiling` toward 0.5. If the brief is
thin, raise it. `doomscroller brief --dry-run --include-seen` re-ranks
yesterday's items without refetching, which makes tuning cheap.

Add `mute:` terms for anything you never want to see. Mutes are matched before
anything reaches Claude, so muted noise costs no tokens.

## Cost

Composio's free tier is 20,000 tool calls/month. One brief costs roughly one
call per configured source, so six sources daily is about 180 calls/month —
well inside the free tier.

Claude spend is dominated by triage, which is per-item. Three things hold it
down: the triage system prompt is cached across every batch and every run,
verdicts are cached in SQLite so repeat items cost nothing, and mutes drop items
before they're sent. Every brief prints its own token usage, including how much
was served from cache.

To cut it further, lower `models.triage_effort`, reduce source `limit`s, or
narrow the window. To spend less on the volume pass specifically, point
`models.triage` at a smaller model and leave `models.synthesis` on Opus — that
change is yours to make, not one the tool makes for you.

## Running it daily

```cron
0 8 * * * cd ~/doomscroller && .venv/bin/doomscroller brief >> ~/.doomscroller.log 2>&1
```

Configure a `delivery:` target other than `console` first — `file`, `gmail`,
`telegram`, `slack`, or `discord`.

## Adding a source

Any Composio toolkit works without writing code:

```yaml
- id: notion-weekly
  platform: composio
  slug: NOTION_SEARCH_NOTION_PAGE
  arguments: {query: "weekly review"}
  title_keys: [title, name]
  body_keys: [content]
```

For a platform that deserves real field mapping, subclass `ComposioSource` in
`doomscroller/sources/platforms.py` — implement `build_arguments` and `to_item`,
then register it in `COMPOSIO_SOURCES`.

## Known limits

- **Dedup is deliberately conservative.** Two headlines that mean the same thing
  while sharing almost no wording ("Postgres 18 ships async I/O" vs "PostgreSQL
  18 released with asynchronous I/O support") stay separate. Every threshold low
  enough to merge them also merges genuinely different stories about the same
  subject, and a duplicate costs you one line while a bad merge hides a story.
- **Default tool slugs drift.** They're a starting point; `doomscroller tools`
  is the source of truth for your account.
- **Triage judgements are the model's.** `noise` and `substance` are calibrated
  by prompt, not by a labelled dataset. Check a few briefs with
  `--dry-run --include-seen` before trusting the filter with things you'd mind
  missing.
- **No LLM, no filtering.** Without `ANTHROPIC_API_KEY` the pipeline still runs,
  but items get neutral heuristic verdicts — you get deduplication and
  recency ranking, not noise removal.

## Development

```bash
pip install -e ".[dev]"
pytest                                  # 101 tests, no network or API keys needed
doomscroller brief --no-llm --dry-run   # exercise the pipeline offline
```

## Licence

MIT
