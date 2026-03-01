# Speakyer

Podcast language learning pipeline. Speakyer downloads podcast episodes,
transcribes them with Whisper, extracts vocabulary tagged by CEFR level (A1–C2),
generates audio clips for each word in context, and pushes flashcards to Anki.

**Language support:** Starting with German. Additional languages are planned —
the pipeline is designed to be language-agnostic from the ground up.

---

## How it works

1. Speakyer reads your `sources.yaml` to find podcast RSS feeds
2. New episodes are downloaded and transcribed locally using Whisper (fast on Apple Silicon)
3. Vocabulary is extracted, lemmatized, and tagged with CEFR level using spaCy
4. Audio clips are extracted for each word's context sentence
5. Flashcards with audio and context sentences are pushed to Anki via AnkiConnect
6. Cards are tagged by level (`cefr::B2`), podcast, and date — study what you want in Anki

---

## Prerequisites

- macOS with Apple Silicon (M1/M2/M3) — transcription uses `mlx-whisper`
- Python 3.11+
- [Anki desktop](https://apps.ankiweb.net/) with AnkiConnect add-on
- ffmpeg: `brew install ffmpeg`

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/jamieforrest/speakyer
cd speakyer
pip install pip-tools
pip-compile requirements.in
pip-sync requirements.txt
pip install -e .
```

### 2. Set a HuggingFace token

mlx-whisper downloads Whisper model weights from HuggingFace Hub on first use.
A token is not strictly required but avoids rate limiting and speeds up downloads.

1. Create a read-only token at https://huggingface.co/settings/tokens
2. Add it to your shell profile (`~/.zshrc` or `~/.bash_profile`):

```bash
export HF_TOKEN=hf_...
```

Or add it to a `.env` file in the project root — Speakyer loads it automatically on startup (`.env` is gitignored). Copy `.env.example` to get started:

```bash
cp .env.example .env
# then edit .env and fill in your HF_TOKEN
```

### 3. Download the spaCy language model

For German (default):

```bash
python -m spacy download de_core_news_lg
```

### 4. Install AnkiConnect in Anki desktop

1. Open Anki → Tools → Add-ons → Get Add-ons
2. Enter code: `2055492159`
3. Restart Anki

Anki must be running whenever you run `speakyer run` or `speakyer audio-update`.

### 5. Configure podcast sources

Edit `sources.yaml` to define your podcast feeds (see [Sources](#sources) below).

### 6. Seed the CEFR vocabulary table

```bash
speakyer seed-cefr
```

This loads the bundled `data/cefr/de.csv` word list so the NLP pipeline can tag
vocabulary by CEFR level. You can also provide a custom CSV:

```bash
speakyer seed-cefr path/to/custom.csv
```

### 7. Run the pipeline

```bash
speakyer run
```

This fetches new episodes, transcribes, extracts vocabulary, generates cards
with audio clips, and exports everything to Anki in one pass.

---

## Daily usage

```bash
speakyer run
```

That's it. This fetches new episodes, transcribes, extracts vocabulary,
generates audio clips, and exports cards to Anki — all in one pass.
Anki must be running.

If you want to process a single source or episode:

```bash
speakyer run --source tagesschau
speakyer run --episode-id 42
```

---

## CLI commands

All interaction goes through the `speakyer` CLI:

| Command | Description |
|---|---|
| `speakyer run` | Full pipeline: fetch, transcribe, NLP, cards, audio, export |
| `speakyer audio-update` | Backfill audio clips or resync fields/tags on existing cards |
| `speakyer fix-clips` | Remove audio clips from Whisper-hallucinated segments |
| `speakyer episodes` | List fetched episodes with source and processing status |
| `speakyer transcript <id>` | Pretty-print a transcript with timestamps |
| `speakyer words <id>` | Display vocabulary grouped by CEFR level |
| `speakyer cards <id>` | Preview Anki cards for an episode |
| `speakyer db` | Inspect the database: table counts and recent rows |
| `speakyer seed-cefr [path]` | Seed the CEFR vocabulary table from CSV |

### Common options

```bash
speakyer run --dry-run                 # preview without writing
speakyer run --stage transcribe        # run up to a specific stage
speakyer run --source tagesschau       # process one source
speakyer run --episode-id 1            # process one episode
speakyer run --whisper-model mlx-community/whisper-small-mlx  # faster model
speakyer audio-update --resync-tags    # re-push tags to all cards
speakyer words 1 --level B1           # filter to a CEFR level
speakyer words 1 --unique             # deduplicated lemmas only
speakyer transcript 1 --words         # show word-level timestamps
speakyer cards 1 --status pending     # filter by card status
```

---

## Sources

`sources.yaml` defines the podcast feeds Speakyer subscribes to. Example:

```yaml
sources:
  - name: tagesschau
    rss_url: https://www.tagesschau.de/multimedia/sendung/tagesschau_20_uhr/podcast-ts2000-audio-100~podcast.xml
    language: de
    active: true

  - name: deutschlandfunk-nachrichten
    rss_url: https://www.deutschlandfunk.de/podcast-nachrichten.3184.de.podcast.xml
    language: de
    active: false
```

The `language` field is passed to Whisper and the NLP pipeline. On first run,
Speakyer seeds the `sources` table from this file. To add a new podcast, add an
entry and run the pipeline — it will be picked up automatically.

---

## Development

### Dependency management (pip-tools)

```bash
# After adding a dependency to pyproject.toml or requirements.in:
pip-compile requirements.in           # regenerate requirements.txt
pip-compile requirements-dev.in       # regenerate requirements-dev.txt
pip-sync requirements.txt             # sync your environment

# Install dev tools:
pip-sync requirements-dev.txt
```

### Running tests

```bash
# First time: generate and install dev dependencies
pip install pip-tools
pip-compile requirements-dev.in -o requirements-dev.txt
pip install -r requirements-dev.txt

# Run the test suite
python -m pytest
```

### Environment variable overrides

| Variable | Default | Description |
|---|---|---|
| `SPEAKYER_DB_PATH` | `./speakyer.db` | SQLite database path |
| `SPEAKYER_DATA_DIR` | `./data` | Root for audio, transcripts, clips |
| `SPEAKYER_SOURCES_YAML` | `./sources.yaml` | Podcast source definitions |
| `SPEAKYER_WHISPER_MODEL` | `mlx-community/whisper-large-v3-mlx` | Whisper model identifier |
| `SPEAKYER_SPACY_MODEL` | `de_core_news_lg` | spaCy model for NLP/lemmatisation |
| `SPEAKYER_ANKI_CONNECT_URL` | `http://localhost:8765` | AnkiConnect endpoint |

---

## Milestones

| Milestone | Description | Status |
|---|---|---|
| 0 | Project foundation: schema, config, storage abstraction | Done |
| 1 | Source configuration & audio download | Done |
| 2 | Whisper transcription (local, Apple Silicon) | Done |
| 3 | NLP pipeline + CEFR vocabulary tagging | Done |
| 4 | Anki card generation | Done |
| 5 | Pipeline runner + AnkiConnect export | Done |
| 6 | Hardening, idempotency, v1 complete | In progress |
| 7 | Audio clips in flashcards | Done |

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
