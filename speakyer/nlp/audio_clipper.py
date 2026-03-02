"""Audio clip extractor for Speakyer.

Given a word_id, extracts a padded MP3 clip covering the full sentence that
contains the word.  The clip is written to ``data/clips/{word_id}.mp3``.

pydub and ffmpeg must be installed for this module to function::

    pip install pydub
    brew install ffmpeg   # macOS

Usage::

    from speakyer.nlp.audio_clipper import AudioClipper

    clipper = AudioClipper()
    clip_path = clipper.extract(word_id=42)   # returns Path or None
"""

from __future__ import annotations

import json
from pathlib import Path

from speakyer.config import config
from speakyer.database import db
from speakyer.storage import LocalStorage
from speakyer.transcription.base import Transcript

_PADDING_MS = 500       # milliseconds added before/after the target sentence
_CLIPS_REL_DIR = "clips"
_MIN_MS_PER_WORD = 250  # below this → likely a Whisper hallucination


def _clip_duration_ms(path: "Path") -> int | None:
    """Return the duration of an MP3 clip in milliseconds, or None on error.

    Tries mutagen first (fast header-only read); falls back to pydub which is
    already a required dependency.  Returns None if both fail so the caller
    can treat the clip as acceptable rather than deleting it unnecessarily.
    """
    try:
        from mutagen.mp3 import MP3  # type: ignore[import]
        return int(MP3(path).info.length * 1000)
    except Exception:
        pass
    try:
        from pydub import AudioSegment  # type: ignore[import]
        return len(AudioSegment.from_file(str(path)))
    except Exception:
        return None


def purge_hallucinated_clips(
    storage: "LocalStorage | None" = None,
) -> list[dict]:
    """Find cards whose clips came from hallucinated segments and remove them.

    Returns a list of dicts ``{"card_id": int, "clip_path": str}`` for every
    card that was purged.
    """
    if storage is None:
        storage = LocalStorage(config.data_dir)

    with db() as conn:
        rows = conn.execute(
            """
            SELECT c.id              AS card_id,
                   c.audio_clip_path,
                   w.example_sentence AS sentence_text,
                   w.start_time       AS sentence_start,
                   w.sentence_end_time,
                   ts.text            AS seg_text,
                   ts.start_time      AS seg_start,
                   ts.end_time        AS seg_end
            FROM cards c
            JOIN words w              ON w.id  = c.word_id
            LEFT JOIN transcript_segments ts ON ts.id = w.transcript_segment_id
            WHERE c.audio_clip_path IS NOT NULL
            """,
        ).fetchall()

        purged: list[dict] = []
        for row in rows:
            # Use sentence-level boundaries when available.
            sent_start = row["sentence_start"] if row["sentence_start"] is not None else row["seg_start"]
            sent_end = row["sentence_end_time"] if row["sentence_end_time"] is not None else row["seg_end"]
            sent_text = row["sentence_text"] or row["seg_text"] or ""

            if sent_start is None or sent_end is None:
                continue

            word_count = len(sent_text.split()) if sent_text else 1
            min_duration_ms = word_count * _MIN_MS_PER_WORD
            actual_duration_ms = (sent_end - sent_start) * 1000
            if actual_duration_ms >= min_duration_ms:
                continue

            # Delete the clip file if it exists on disk.
            clip_rel = row["audio_clip_path"]
            abs_clip = storage.absolute_path(clip_rel)
            if abs_clip.exists():
                abs_clip.unlink()

            conn.execute(
                "UPDATE cards SET audio_clip_path = NULL, status = 'exported' WHERE id = ?",
                (row["card_id"],),
            )
            purged.append({"card_id": row["card_id"], "clip_path": clip_rel})

    return purged


class AudioClipper:
    """Extract audio clips for Anki flashcards.

    Clips span the full sentence containing the target word (using
    ``words.start_time`` / ``words.sentence_end_time`` set by NLPStage).
    For words processed before sentence merging was introduced, falls back
    to the single transcript segment boundaries via the stored Whisper JSON.

    All dependencies are injectable for testing without real audio files or a
    populated database.
    """

    def __init__(
        self,
        storage: LocalStorage | None = None,
        clips_dir: str = _CLIPS_REL_DIR,
        padding_ms: int = _PADDING_MS,
        db_path: "Path | None" = None,
    ) -> None:
        self._storage = storage
        self.clips_dir = clips_dir
        self.padding_ms = padding_ms
        self._db_path = db_path
        # Per-instance caches so repeated calls for the same episode only load
        # the audio file and transcript JSON once each.
        self._audio_cache: dict[str, object] = {}   # path → AudioSegment
        self._transcript_cache: dict[str, Transcript] = {}

    @property
    def storage(self) -> LocalStorage:
        if self._storage is None:
            self._storage = LocalStorage(config.data_dir)
        return self._storage

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, word_id: int) -> Path | None:
        """Extract an audio clip for *word_id*.

        Returns the absolute Path of the written ``.mp3`` clip, or ``None``
        if prerequisites are not met (missing audio, implausible timestamps).

        When ``sentence_end_time`` is stored on the word the clip spans the
        full merged sentence; otherwise falls back to the single transcript
        segment boundaries (legacy path for episodes processed before sentence
        merging was introduced).
        """
        row = self._load_row(word_id)
        if row is None:
            return None

        audio_path = self._resolve_audio(row["audio_path"])
        if audio_path is None:
            return None

        # Prefer sentence-level boundaries set by NLPStage after segment merging.
        sentence_end = row["sentence_end_time"]
        if sentence_end is not None:
            sent_start = row["sentence_start"] if row["sentence_start"] is not None else row["seg_start"]
            sent_end = sentence_end
            sent_text = row["sentence_text"] or row["seg_text"] or ""
        else:
            # Legacy fallback: use single segment boundaries from the JSON.
            sent_start = row["seg_start"]
            sent_end = row["seg_end"]
            sent_text = row["seg_text"] or ""

            # Load the transcript JSON to resolve exact segment boundaries.
            json_path = self.storage.absolute_path(row["raw_json_path"])
            if not json_path.exists():
                return None
            json_key = str(json_path)
            if json_key not in self._transcript_cache:
                self._transcript_cache[json_key] = Transcript.from_dict(
                    json.loads(json_path.read_bytes())
                )
            transcript = self._transcript_cache[json_key]
            # Find exact boundaries in the JSON (within 50 ms tolerance).
            seg = next(
                (
                    s
                    for s in transcript.segments
                    if abs(s.start - sent_start) < 0.05 and abs(s.end - sent_end) < 0.05
                ),
                None,
            )
            if seg is not None:
                sent_start = seg.start
                sent_end = seg.end

        # Skip clips from implausibly short time spans — likely hallucinations.
        word_count = len(sent_text.split()) if sent_text else 1
        min_duration_ms = word_count * _MIN_MS_PER_WORD
        actual_duration_ms = (sent_end - sent_start) * 1000
        if actual_duration_ms < min_duration_ms:
            return None

        start_ms = max(0, int(sent_start * 1000) - self.padding_ms)
        end_ms = int(sent_end * 1000) + self.padding_ms
        expected_ms = end_ms - start_ms

        # Idempotent: return existing clip if its duration matches expectations.
        # A >1 s discrepancy means the file is stale (cut from old timestamps)
        # and must be regenerated.
        rel_clip = f"{self.clips_dir}/{word_id}.mp3"
        if self.storage.exists(rel_clip):
            abs_clip = self.storage.absolute_path(rel_clip)
            actual_ms = _clip_duration_ms(abs_clip)
            if actual_ms is None or abs(actual_ms - expected_ms) <= 1000:
                return abs_clip
            # Stale — delete and fall through to re-extract.
            abs_clip.unlink()

        return self._write_clip(audio_path, start_ms, end_ms, word_id)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_row(self, word_id: int):
        """Return a sqlite3.Row with all fields needed for clip extraction."""
        db_ctx = db(self._db_path) if self._db_path is not None else db()
        with db_ctx as conn:
            return conn.execute(
                """
                SELECT w.surface_form,
                       w.start_time       AS sentence_start,
                       w.sentence_end_time,
                       w.example_sentence AS sentence_text,
                       ts.start_time      AS seg_start,
                       ts.end_time        AS seg_end,
                       ts.text            AS seg_text,
                       t.raw_json_path,
                       e.audio_path
                FROM words w
                LEFT JOIN transcript_segments ts ON ts.id = w.transcript_segment_id
                LEFT JOIN transcripts t          ON t.id  = ts.transcript_id
                JOIN episodes e                  ON e.id  = w.episode_id
                WHERE w.id = ?
                """,
                (word_id,),
            ).fetchone()

    def _resolve_audio(self, audio_path: str | None) -> Path | None:
        if not audio_path:
            return None
        p = self.storage.absolute_path(audio_path)
        return p if p.exists() else None

    def _write_clip(
        self, audio_path: Path, start_ms: int, end_ms: int, word_id: int
    ) -> Path:
        try:
            from pydub import AudioSegment  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "pydub is required for audio clip extraction.\n"
                "Install it with: pip install pydub\n"
                "ffmpeg must also be installed and on your PATH (brew install ffmpeg)."
            ) from exc

        audio_key = str(audio_path)
        if audio_key not in self._audio_cache:
            self._audio_cache[audio_key] = AudioSegment.from_file(str(audio_path))
        audio = self._audio_cache[audio_key]
        clip = audio[start_ms:end_ms]
        buf = clip.export(format="mp3")

        rel_path = f"{self.clips_dir}/{word_id}.mp3"
        self.storage.write(rel_path, buf.read())
        return self.storage.absolute_path(rel_path)
