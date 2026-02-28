# Speakyer

Podcast vocabulary learning pipeline for German learners. Speakyer downloads
podcast episodes, transcribes them with Whisper, extracts vocabulary tagged by
CEFR level (A1–C2), and pushes flashcards to Anki.

**Current status:** Milestone 0 — project foundation. The pipeline is not yet
implemented. See the [milestone plan](#milestones) below.

---

## How it works

1. Speakyer reads your `sources.yaml` to find podcast RSS feeds
2. New episodes are downloaded and transcribed locally using Whisper (fast on Apple Silicon)
3. German vocabulary is extracted, lemmatized, and tagged with CEFR level using spaCy
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

### 2. Download the German spaCy model

```bash
python -m spacy download de_core_news_lg
```

### 3. Install AnkiConnect in Anki desktop

1. Open Anki → Tools → Add-ons → Get Add-ons
2. Enter code: `2055492159`
3. Restart Anki

Anki must be running whenever you run `speakyer run`.

### 4. Configure podcast sources

Edit `sources.yaml` to define your podcast feeds (see [Sources](#sources) below).

### 5. Initialise the database

```bash
python scripts/db_inspect.py
```

This creates `speakyer.db` and shows the empty schema.

---

## Running

```bash
speakyer run                        # process all active sources
speakyer run --source tagesschau    # one source only
speakyer run --dry-run              # preview without writing
```

---

## Sources

`sources.yaml` defines the podcast feeds Speakyer subscribes to. Example:

```yaml
sources:
  - name: tagesschau
    rss_url: https://www.tagesschau.de/multimedia/podcast/ts100s.xml
    language: de
    active: true

  - name: deutschlandfunk
    rss_url: https://www.deutschlandfunk.de/podcast-nachrichten.3184.de.podcast.xml
    language: de
    active: false
```

On first run, Speakyer seeds the `sources` table from this file. To add a new
podcast later, add an entry to `sources.yaml` and run `speakyer run` — it will
be picked up automatically.

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
| 1 | Source configuration & audio download | Pending |
| 2 | Whisper transcription (local, Apple Silicon) | Pending |
| 3 | German NLP pipeline + CEFR vocabulary tagging | Pending |
| 4 | Anki card generation | Pending |
| 5 | Pipeline runner + AnkiConnect export | Pending |
| 6 | Hardening, idempotency, v1 complete | Pending |
| 7 | Audio clips in flashcards (v1.1) | Pending |

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
