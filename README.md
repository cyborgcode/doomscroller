# doomscroller

A personal bot that reads your feeds so you don't have to.

You point it at your accounts. Once a day it pulls everything new, throws away
the engagement bait, merges the same story told five times into one entry, and
hands you a short brief. You react to a few entries; it gets more selective
about what reaches you next time.

The reading is done by **DeepSeek V4 Flash on [NVIDIA NIM](https://build.nvidia.com/deepseek-ai/deepseek-v4-flash)**,
which is free. Feeds are reached through [Composio](https://composio.dev), whose
free tier is 20,000 tool calls a month — about a hundred times what daily use
needs. Claude is supported as an alternative backend, one line of config.

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

52 fetched · 31 new · 14 filtered out · 3 muted · 6 tool calls · 41,208 tokens
react with: doomscroller feedback <id> up|down|save|mute
```

---

## What it actually does

Here is the full journey of one Reddit post, from your feed to your brief.

**1. Fetched.** Each configured source makes one API call through Composio and
gets back raw records. Sources run in parallel; one that fails is reported and
skipped rather than taking the run down. The post is normalised into a common
shape — title, body, url, author, timestamp, engagement count — and given a
stable 16-character id derived from its platform and native id. That id is how
the bot recognises the same post tomorrow, and it's what you type to react.

**2. Checked against mutes.** If the text matches anything in your `mute:` list,
it's dropped here — before any model sees it, so muted noise costs nothing.

**3. Checked against history.** If that id is already in the local database from
a previous run, it's skipped. You never see the same post twice.

**4. Triaged.** New items go to the model in batches of 12. For each one it
returns:

| Field | What it is |
|---|---|
| `kind` | news, analysis, opinion, promotion, drama, meme, question, or unknown |
| `noise` | 0–1: how much is engagement machinery rather than information |
| `substance` | 0–1: how much you'd actually learn |
| `claims` | Standalone factual assertions, each readable without the original post |
| `topics` | 1–4 subject tags like `databases`, `ai-policy` |
| `entities` | The people, companies, and products it's actually about |
| `summary` | One sentence, under 30 words, that carries the point |

The verdict is cached in SQLite against the item id. If that post is still in
tomorrow's window, it is never triaged — or paid for — again.

**5. Scored.** Six components, each roughly 0–1, combined with the weights in
your config:

| Component | Default weight | What it measures |
|---|---|---|
| `interest` | 1.0 | Match against your learned profile |
| `noise` | 0.9 | Reward for *not* being engagement bait |
| `substance` | 0.8 | The triage substance score |
| `freshness` | 0.5 | Exponential decay, 14-hour half-life |
| `source` | 0.4 | Learned per-source trust |
| `engagement` | 0.25 | Log-scaled upvotes — a weak prior, not a verdict |

Each component's contribution is kept on the item, so a ranking can always be
explained rather than just asserted.

**6. Filtered.** Four rules decide whether it survives at all:

- `noise` above `noise_ceiling` (default 0.72) — dropped, whatever else it has
  going for it
- `substance` below `substance_floor` (default 0.25) — dropped
- `kind` is promotion or meme with `substance` below 0.6 — dropped
- `kind` is drama and your interest score is below 0.55 — dropped, because
  drama about something you've never signalled on is precisely what you asked
  to be spared

**7. Clustered.** Survivors are grouped by canonical URL (tracking parameters
stripped), token overlap, and character trigrams. Five accounts covering one
story become one entry that mentions the corroboration, and a cluster spanning
multiple platforms gets a modest score bump for being independently reported.

**8. Written up.** The top clusters go to the model once more, which writes the
entries you read — using only the claims extracted during triage, merging
across sources, and flagging where sources disagree or a claim rests on a single
unverified account. Below them, the rest get one line each.

**9. Delivered.** To your terminal, a file, or back out through Composio to
Gmail, Telegram, Slack, or Discord.

**10. Learned from.** Whatever you react to is folded into your profile at the
start of the next run.

---

## Quickstart

```bash
git clone https://github.com/cyborgcode/doomscroller && cd doomscroller
python -m venv .venv && source .venv/bin/activate
pip install -e .

cp .env.example .env              # add NVIDIA_API_KEY and COMPOSIO_API_KEY
cp config.example.yaml config.yaml

doomscroller check                # verify credentials and sources
doomscroller brief                # your first brief
```

Two keys, both free:

- **`NVIDIA_API_KEY`** from [build.nvidia.com](https://build.nvidia.com) — does
  the reading.
- **`COMPOSIO_API_KEY`** from [app.composio.dev](https://app.composio.dev) —
  reaches your accounts. Not needed if you only use RSS.

`config.yaml` ships with two RSS feeds enabled, which need no accounts and no
Composio quota. `doomscroller brief` does something useful before you've
connected anything — start there, then add accounts once you like the shape of
the output.

### Connecting an account

```bash
doomscroller auth reddit          # prints an OAuth URL; open it, approve
doomscroller check --probe        # actually fetches — shows real item counts
```

Then enable the matching source in `config.yaml` (several are present but
`enabled: false`).

---

## Daily use

The whole loop is two commands.

```bash
doomscroller brief                     # morning: read it
doomscroller feedback dcb6e5af up      # react to two or three entries
doomscroller feedback 91c2ba07 down
```

That's it. Feedback is applied automatically at the start of the next brief —
there's nothing to run in between. Two or three reactions a day is enough; the
profile moves noticeably within a week.

Occasionally worth running:

```bash
doomscroller profile                   # what it currently believes about you
doomscroller history --days 3          # what you were shown, with ids
```

`history` matters because ids scroll out of your terminal. If you remember
something was good but the brief is gone, `history` gets the id back.

---

## Commands

| Command | What it does |
|---|---|
| `brief` | Build and deliver a digest. The default — bare `doomscroller` does this. |
| `feedback <id> <signal>` | Record a reaction. Signals below. |
| `profile` | Show the learned interest profile: topics, vocabulary, source trust. |
| `history [--days N]` | List what you were shown, with ids, newest first. |
| `check [--probe]` | Verify config, credentials, and every source. `--probe` really fetches. |
| `auth <toolkit>` | Start a Composio OAuth flow; prints a URL to open. |
| `tools <toolkit>` | List the real Composio tool slugs available to your account. |
| `learn` | Fold pending feedback into the profile now, instead of at the next brief. |

`brief` flags:

| Flag | Effect |
|---|---|
| `--dry-run` | Print to the terminal and deliver nowhere. |
| `--no-llm` | Skip the model entirely — dedup and recency ranking only. |
| `--include-seen` | Re-rank items from previous runs instead of only new ones. |
| `--hours N` | Override the config time window for this run. |
| `--format html\|markdown\|terminal` | Change the `--dry-run` output format. |

Global: `--config PATH` to use a different config file, `-v` to see what each
stage is doing.

**The tuning combination worth knowing** is `--dry-run --include-seen`. It
re-ranks yesterday's items without refetching and without delivering, so you can
change a threshold and immediately see what it would have done differently.

---

## Teaching it

Every entry in a brief carries a short id. Six signals, with the weight each
carries:

| Signal | Weight | Use when |
|---|---|---|
| `save` | +1.2 | Strongest positive — this is exactly what you want |
| `up` | +1.0 | More like this |
| `open` | +0.4 | Weak positive, for scripting a "clicked the link" signal |
| `skip` | −0.3 | Weak negative |
| `down` | −1.0 | Less like this |
| `mute` | −2.0 | Strong negative, and hits the item's *topics* twice as hard |

`down` and `mute` differ in what they blame. `down` says this post was bad;
`mute` says the subject is. Use `mute` when you don't want the topic at all,
`down` when it was just a weak post about something you do care about.

Three tables are learned:

- **vocabulary** — the words that recur in things you react to
- **topics** — the tags triage assigned
- **source trust** — which feeds are earning their place

Each learning pass decays existing weights slightly before adding new evidence.
That decay is the point: interests you stop reinforcing fade, so the profile
tracks what you care about *now* rather than accumulating everything you ever
clicked. Weights are scaled against the profile's own distribution rather than a
fixed constant, so a young profile discriminates just as sharply as a mature
one — without that, everything you taught it in week one would sit too close to
neutral to outrank freshness.

Before any feedback exists, the `interests:` list in your config seeds the
profile. It stays weaker than anything learned from real reactions.

---

## Configuration

Everything lives in `config.yaml`; secrets live in `.env`.

### Top level

| Key | Default | Meaning |
|---|---|---|
| `user_id` | `default` | Which Composio account set to use |
| `window_hours` | `24` | How far back each brief looks |
| `db_path` | `doomscroller.db` | SQLite file: seen items, feedback, profile |
| `interests` | `[]` | Seed topics, used until real feedback exists |
| `mute` | `[]` | Case-insensitive substrings; matched before the model runs |

### `ranking:`

| Key | Default | Meaning |
|---|---|---|
| `noise_ceiling` | `0.72` | Above this, an item never reaches you. Lower = harsher. |
| `substance_floor` | `0.25` | Below this, it isn't worth a line |
| `headline_count` | `8` | Fully written-up stories |
| `skim_count` | `15` | One-liners after them |
| `half_life_hours` | `14` | Freshness decay rate |
| `weights` | see above | Per-component score weights |

If too much gets through, lower `noise_ceiling` toward 0.5. If the brief is
thin, raise it. These two do most of the work; the weights are for fine-tuning
after you've lived with it a while.

### `models:`

| Key | Default | Meaning |
|---|---|---|
| `provider` | `nvidia_nim` | `nvidia_nim` or `anthropic` |
| `triage` | provider default | Model for the per-item pass |
| `synthesis` | provider default | Model for the once-per-brief pass |
| `triage_effort` | `low` | Reasoning depth for triage |
| `synthesis_effort` | `high` | Reasoning depth for the write-up |
| `triage_batch_size` | `12` | Items per triage request |
| `max_tokens` | `16000` | Output cap per request |

Any other key here is passed to the provider — `base_url` for a self-hosted NIM
container, `thinking: false` to disable DeepSeek's reasoning mode, `timeout`.

### `sources:`

Each entry needs a `platform`; everything else is platform-specific.

```yaml
sources:
  - id: lobsters              # what appears in the brief; defaults to platform
    platform: rss
    url: https://lobste.rs/rss
    limit: 40                 # max items to take
    weight: 1.0               # multiplies every score from this source
    enabled: true
```

Platforms: `rss`, `hackernews`, `reddit`, `twitter` (`x` is an alias),
`youtube`, `gmail`, `linkedin`, and `composio` as a generic escape hatch.

| Platform | Needs |
|---|---|
| `rss` | `url` |
| `reddit` | `subreddit`, optionally `sort`, `time_filter` |
| `twitter` | `query`, or `user_id` / `list_id` |
| `youtube` | `query` or `channel_id` |
| `gmail` | `query` (defaults to `category:updates newer_than:1d`) |
| `hackernews`, `linkedin` | nothing |

### `delivery:`

```yaml
delivery:
  - kind: console
  - kind: file
    format: markdown          # or html
    path: briefs/{date}.{ext}
  - kind: gmail
    to: you@example.com
  - kind: telegram
    chat_id: "123456789"
  - kind: slack
    channel: "#briefs"
  - kind: discord
    channel_id: "987654321"
```

Multiple targets work. A channel that fails is reported; the others still get
the brief.

---

## Running it daily

```cron
0 8 * * * cd ~/doomscroller && .venv/bin/doomscroller brief >> ~/.doomscroller.log 2>&1
```

Configure a `delivery:` target other than `console` first, or the output goes
nowhere you'll see it.

---

## Tool slugs

Each Composio-backed source calls one named tool. The defaults here are only
defaults — providers rename them, and a renamed slug looks like a source that
returns nothing rather than an error. If a source is quiet, check:

```bash
doomscroller tools reddit
```

```yaml
- id: reddit:rust
  platform: reddit
  subreddit: rust
  slug: REDDIT_RETRIEVE_REDDIT_POST   # whatever `doomscroller tools` showed
```

---

## Switching providers

```yaml
models:
  provider: anthropic     # was: nvidia_nim
```

Model ids follow the provider automatically, so you don't need to know both
spellings. Set them explicitly to override — e.g. `deepseek-ai/deepseek-v4-pro`
for synthesis while triage stays on Flash.

| | NVIDIA NIM | Anthropic |
|---|---|---|
| Default model | `deepseek-ai/deepseek-v4-flash` | `claude-opus-5` |
| Key | `NVIDIA_API_KEY` | `ANTHROPIC_API_KEY` |
| Schema enforcement | `nvext.guided_json` (xgrammar) | `output_config.format` |
| Reasoning depth | `reasoning_effort`: none/high/max | `effort`: low→max |
| Prompt caching | no | yes |

`triage_effort` and `synthesis_effort` use this project's vocabulary
(`low`/`medium`/`high`/`xhigh`/`max`) on both; the NIM provider maps them onto
its three `reasoning_effort` values (`low`→`none`, `medium`/`high`→`high`,
`xhigh`/`max`→`max`).

### NIM specifics worth knowing

- **`chat_template_kwargs` is mandatory.** DeepSeek V4 on NIM requires
  `{enable_thinking, thinking}` at the root of the payload; omit it and the
  request *hangs* rather than erroring, which in a nightly cron looks like the
  bot silently dying. The provider always sends it, and a test pins that.
- **Schema output uses `guided_json`, not `response_format`.** NVIDIA recommends
  it, and unlike `response_format: {"type": "json_object"}` it actually enforces
  the schema instead of permitting any valid JSON including `{}`.
- **Reasoning arrives on `reasoning_content`**, not inside the content, so the
  content parses as clean JSON. Inline `<think>` blocks and markdown fences are
  stripped anyway, since self-hosted containers vary.
- **Self-hosting works**: point `models.base_url` at your own container.

---

## Cost

**Composio** free tier is 20,000 tool calls/month. One brief costs roughly one
call per configured source, so six sources daily is about 180 calls/month.

**NVIDIA NIM** serves DeepSeek V4 Flash free on
[build.nvidia.com](https://build.nvidia.com/deepseek-ai/deepseek-v4-flash) —
rate-limited rather than metered, so the constraint is requests per minute, not
spend. `triage_batch_size` is what keeps you under it: 12 items per request
means ~3 requests for a 40-item day.

The one real cost difference against Claude is that **NIM has no prompt
caching**, so the triage system prompt is re-sent on every batch. Two things
blunt that: verdicts are cached in SQLite so repeat items cost nothing, and
mutes drop items before they're sent. Raising `triage_batch_size` amortises the
system prompt over more items — 12–20 is a reasonable range. Every brief prints
its token usage, and `doomscroller check` tells you whether your provider caches.

---

## Where your data goes

Worth knowing, since this reads your inbox and your timeline.

- **Locally**, `doomscroller.db` stores item titles, bodies, urls, your feedback,
  and the learned profile. It's gitignored. Items older than 45 days can be
  pruned; nothing is uploaded anywhere by this tool.
- **To Composio** go your OAuth grants and the API calls that fetch your feeds.
- **To NVIDIA (or Anthropic)** go the title and first 1,500 characters of the
  body of every item that survives muting and isn't already cached — including
  email subjects and snippets if you enable the Gmail source.

If some source is more sensitive than you want leaving the machine, don't enable
it, or narrow its `query` so only the intended mail matches. `mute:` also runs
before anything is sent, so it's a real privacy control and not just a display
filter.

---

## Troubleshooting

**The brief is empty.** Check the counters at the bottom. `0 fetched` means the
sources returned nothing — run `doomscroller check --probe`. A high
`filtered out` means the noise floor is doing its job too well; raise
`noise_ceiling`. A high `already seen` is normal on a second run in the same day.

**One source returns nothing.** Almost always a renamed tool slug. Run
`doomscroller tools <toolkit>` and pin the real one.

**Everything looks mediocre and nothing is filtered.** Your provider key is
probably missing or wrong. The pipeline degrades to heuristic verdicts rather
than failing, so a broken key looks like a weak brief, not an error.
`doomscroller check` catches it.

**A request hangs on NIM.** If you've edited the provider, confirm
`chat_template_kwargs` is still being sent — that's the failure mode.

**Ids don't work.** They're from the brief and expire when the store is pruned.
`doomscroller history` lists current ones.

---

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

---

## Known limits

- **Dedup is deliberately conservative.** Two headlines that mean the same thing
  while sharing almost no wording ("Postgres 18 ships async I/O" vs "PostgreSQL
  18 released with asynchronous I/O support") stay separate. Every threshold low
  enough to merge them also merges genuinely different stories about the same
  subject, and a duplicate costs you one line while a bad merge hides a story.
- **Default tool slugs drift.** They're a starting point; `doomscroller tools`
  is the source of truth for your account.
- **Triage judgements are the model's.** `noise` and `substance` are calibrated
  by prompt, not against a labelled dataset. Check a few briefs with
  `--dry-run --include-seen` before trusting the filter with things you'd mind
  missing.
- **No model, no filtering.** Without a provider key — or if the endpoint is
  down — you get deduplication and recency ranking, not noise removal.
- **It only sees what you point it at.** There's no discovery. A story nobody in
  your configured sources covered will not appear.

---

## Development

```bash
pip install -e ".[dev]"
pytest                                  # 138 tests, no network or API keys needed
doomscroller brief --no-llm --dry-run   # exercise the pipeline with no model at all
```

| Path | What's in it |
|---|---|
| `sources/` | One class per platform; `base.py` has the tolerant field pickers |
| `providers/` | Model backends behind one interface |
| `pipeline/distill.py` | Both prompts and their schemas |
| `pipeline/rank.py` | Scoring and the noise floor |
| `pipeline/dedup.py` | Clustering |
| `learn.py` | Feedback → profile |
| `store.py` | SQLite |

## Licence

MIT
