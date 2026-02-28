# Speakyer

Podcast language learning pipeline. Speakyer downloads podcast episodes,
transcribes them with Whisper, extracts vocabulary tagged by CEFR level (A1–C2),
and pushes flashcards to Anki.

**Language support:** Starting with German. Additional languages are planned —
the pipeline is designed to be language-agnostic from the ground up.

**Current status:** Milestone 2 — transcription complete.
NLP and card generation are coming in subsequent milestones.
See the [milestone plan](#milestones) below.

---

## How it works

1. Speakyer reads your `sources.yaml` to find podcast RSS feeds
2. New episodes are downloaded and transcribed locally using Whisper (fast on Apple Silicon)
3. Vocabulary is extracted, lemmatized, and tagged with CEFR level using spaCy
4. Flashcards with context sentences are pushed to Anki via AnkiConnect
5. Cards are tagged by level (`cefr::B2`), podcast, and date — study what you want in Anki

---

## Prerequisites

- macOS with Apple Silicon (M1/M2/M3) — transcription uses `mlx-whisper`
- Python 3.11+
- [Anki desktop](https://apps.ankiweb.net/)
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

Or add it to a `.env` file in the project root (already gitignored) and `source .env` before running.

### 3. Download the spaCy language model

For German (default):

```bash
python -m spacy download de_core_news_lg
```

### 5. Install AnkiConnect in Anki desktop

1. Open Anki → Tools → Add-ons → Get Add-ons
2. Enter code: `2055492159`
3. Restart Anki

Anki must be running whenever you run `speakyer run`.

### 6. Configure podcast sources

Edit `sources.yaml` to define your podcast feeds (see [Sources](#sources) below).

### 7. Initialise the database

```bash
python scripts/db_inspect.py
```

This creates `speakyer.db` and shows the empty schema.

---

## Running

```bash
python scripts/run_pipeline.py                         # all active sources
python scripts/run_pipeline.py --source tagesschau    # one source only
python scripts/run_pipeline.py --dry-run              # preview without writing
python scripts/run_pipeline.py --stage download       # download only
python scripts/run_pipeline.py --stage transcribe     # download + transcribe
python scripts/list_episodes.py                       # check what was fetched
python scripts/show_transcript.py <episode_id>        # view transcript
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

### Inspection scripts

These scripts evolve alongside the milestones as new data is available.

| Script | What it shows |
|---|---|
| `python scripts/db_inspect.py` | Table counts and recent rows |
| `python scripts/list_episodes.py` | Fetched episodes with status *(Milestone 1)* |
| `python scripts/show_transcript.py <id>` | Transcript with timestamps *(Milestone 2)* |
| `python scripts/show_words.py <id>` | Vocabulary grouped by CEFR level *(Milestone 3)* |
| `python scripts/show_cards.py <id>` | Card preview before Anki export *(Milestone 4)* |

### Environment variable overrides

| Variable | Default | Description |
|---|---|---|
| `SPEAKYER_DB_PATH` | `./speakyer.db` | SQLite database path |
| `SPEAKYER_DATA_DIR` | `./data` | Root for audio, transcripts, clips |
| `SPEAKYER_SOURCES_YAML` | `./sources.yaml` | Podcast source definitions |
| `SPEAKYER_WHISPER_MODEL` | `mlx-community/whisper-large-v3-mlx` | Whisper model identifier |
| `SPEAKYER_ANKI_CONNECT_URL` | `http://localhost:8765` | AnkiConnect endpoint |

---

## Milestones

| Milestone | Description | Status |
|---|---|---|
| 0 | Project foundation: schema, config, storage abstraction | ✅ Done |
| 1 | Source configuration & audio download | ✅ Done |
| 2 | Whisper transcription (local, Apple Silicon) | ✅ Done |
| 3 | NLP pipeline + CEFR vocabulary tagging | Pending |
| 4 | Anki card generation | Pending |
| 5 | Pipeline runner + AnkiConnect export | Pending |
| 6 | Hardening, idempotency, v1 complete | Pending |
| 7 | Audio clips in flashcards (v1.1) | Pending |

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
