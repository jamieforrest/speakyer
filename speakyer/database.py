import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator

from speakyer.config import config

# Register a custom datetime adapter/converter so that:
# 1. datetime objects are stored as "YYYY-MM-DD HH:MM:SS" (space separator),
#    which is what SQLite expects for TIMESTAMP columns.
# 2. We avoid the Python 3.12 deprecation warning about the default converter.
sqlite3.register_adapter(datetime, lambda dt: dt.strftime("%Y-%m-%d %H:%M:%S"))
sqlite3.register_converter(
    "TIMESTAMP",
    lambda b: datetime.fromisoformat(b.decode().replace(" ", "T")),
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    rss_url TEXT NOT NULL,
    language TEXT NOT NULL DEFAULT 'de',
    transcript_strategy TEXT NOT NULL DEFAULT 'whisper',
    active INTEGER NOT NULL DEFAULT 1,
    user_id INTEGER REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY,
    source_id INTEGER NOT NULL REFERENCES sources(id),
    guid TEXT NOT NULL UNIQUE,
    title TEXT,
    published_at TIMESTAMP,
    audio_url TEXT,
    audio_path TEXT,
    status TEXT NOT NULL DEFAULT 'fetched',
    user_id INTEGER REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS transcripts (
    id INTEGER PRIMARY KEY,
    episode_id INTEGER NOT NULL REFERENCES episodes(id),
    raw_json_path TEXT,
    language_detected TEXT,
    duration_seconds REAL,
    user_id INTEGER REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS transcript_segments (
    id INTEGER PRIMARY KEY,
    transcript_id INTEGER NOT NULL REFERENCES transcripts(id),
    segment_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    start_time REAL NOT NULL,
    end_time REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS words (
    id INTEGER PRIMARY KEY,
    episode_id INTEGER NOT NULL REFERENCES episodes(id),
    transcript_segment_id INTEGER REFERENCES transcript_segments(id),
    surface_form TEXT NOT NULL,
    lemma TEXT NOT NULL,
    pos TEXT,
    cefr_level TEXT,
    example_sentence TEXT,
    start_time REAL,
    user_id INTEGER REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS cards (
    id INTEGER PRIMARY KEY,
    word_id INTEGER NOT NULL REFERENCES words(id),
    anki_note_id INTEGER,
    deck_name TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    audio_clip_path TEXT,
    user_id INTEGER REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    exported_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS cefr_words (
    lemma TEXT PRIMARY KEY,
    level TEXT NOT NULL,
    pos TEXT
);
"""

TABLES = [
    "users",
    "sources",
    "episodes",
    "transcripts",
    "transcript_segments",
    "words",
    "cards",
    "cefr_words",
]


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or config.db_path
    conn = sqlite3.connect(
        str(path),
        detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def db(db_path: Path | None = None) -> Generator[sqlite3.Connection, None, None]:
    conn = get_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init(db_path: Path | None = None) -> None:
    """Create all tables. Safe to call multiple times (uses IF NOT EXISTS)."""
    conn = get_connection(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()
