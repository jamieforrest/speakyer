"""Audio clip extractor for Speakyer.

Given a word_id, loads the stored Whisper JSON transcript, locates the
containing segment, and uses pydub to extract a padded MP3 clip.  The
clip is written to ``data/clips/{word_id}.mp3``.

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

_PADDING_MS = 500       # milliseconds added before/after the target segment
_CLIPS_REL_DIR = "clips"
_MIN_MS_PER_WORD = 250  # below this → likely a Whisper hallucination


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
            SELECT c.id            AS card_id,
                   c.audio_clip_path,
                   ts.text         AS seg_text,
                   ts.start_time   AS seg_start,
                   ts.end_time     AS seg_end
            FROM cards c
            JOIN words w              ON w.id  = c.word_id
            JOIN transcript_segments ts ON ts.id = w.transcript_segment_id
            WHERE c.audio_clip_path IS NOT NULL
            """,
        ).fetchall()

        purged: list[dict] = []
        for row in rows:
            word_count = len(row["seg_text"].split()) if row["seg_text"] else 1
            min_duration_ms = word_count * _MIN_MS_PER_WORD
            actual_duration_ms = (row["seg_end"] - row["seg_start"]) * 1000
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

    All dependencies are injectable for testing without real audio files or a
    populated database.
    """

    def __init__(
        self,
        storage: LocalStorage | None = None,
        clips_dir: str = _CLIPS_REL_DIR,
        padding_ms: int = _PADDING_MS,
    ) -> None:
        self._storage = storage
        self.clips_dir = clips_dir
        self.padding_ms = padding_ms
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
        if prerequisites are not met (missing transcript link, JSON, or audio).
        """
        row = self._load_row(word_id)
        if row is None:
            return None

        json_path = self.storage.absolute_path(row["raw_json_path"])
        if not json_path.exists():
            return None

        audio_path = self._resolve_audio(row["audio_path"])
        if audio_path is None:
            return None

        # Skip segments that are implausibly short for their text length —
        # these are almost always Whisper hallucinations with bad timestamps.
        word_count = len(row["seg_text"].split()) if row["seg_text"] else 1
        min_duration_ms = word_count * _MIN_MS_PER_WORD
        actual_duration_ms = (row["seg_end"] - row["seg_start"]) * 1000
        if actual_duration_ms < min_duration_ms:
            return None

        # Idempotent: return existing clip without re-extracting.
        rel_clip = f"{self.clips_dir}/{word_id}.mp3"
        if self.storage.exists(rel_clip):
            return self.storage.absolute_path(rel_clip)

        # Cache transcript and audio per path so repeated calls for the same
        # episode only load each file once.
        json_key = str(json_path)
        if json_key not in self._transcript_cache:
            self._transcript_cache[json_key] = Transcript.from_dict(
                json.loads(json_path.read_bytes())
            )
        transcript = self._transcript_cache[json_key]

        start_ms, end_ms = self._boundaries(
            transcript,
            seg_start=row["seg_start"],
            seg_end=row["seg_end"],
        )
        return self._write_clip(audio_path, start_ms, end_ms, word_id)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_row(self, word_id: int):
        """Return a sqlite3.Row with all fields needed for clip extraction."""
        with db() as conn:
            return conn.execute(
                """
                SELECT w.surface_form,
                       ts.start_time AS seg_start,
                       ts.end_time   AS seg_end,
                       ts.text       AS seg_text,
                       t.raw_json_path,
                       e.audio_path
                FROM words w
                JOIN transcript_segments ts ON ts.id = w.transcript_segment_id
                JOIN transcripts t          ON t.id  = ts.transcript_id
                JOIN episodes e             ON e.id  = w.episode_id
                WHERE w.id = ?
                """,
                (word_id,),
            ).fetchone()

    def _resolve_audio(self, audio_path: str | None) -> Path | None:
        if not audio_path:
            return None
        p = self.storage.absolute_path(audio_path)
        return p if p.exists() else None

    def _boundaries(
        self,
        transcript: Transcript,
        seg_start: float,
        seg_end: float,
    ) -> tuple[int, int]:
        """Return ``(start_ms, end_ms)`` for the clip.

        Finds the matching transcript segment by start/end proximity and returns
        the full segment boundaries expanded by ``self.padding_ms`` on each side,
        so the clip covers the entire displayed sentence.
        """
        seg = next(
            (
                s
                for s in transcript.segments
                if abs(s.start - seg_start) < 0.05 and abs(s.end - seg_end) < 0.05
            ),
            None,
        )

        if seg is None:
            # Segment not found in JSON (shouldn't happen, but degrade gracefully).
            start_ms = max(0, int(seg_start * 1000) - self.padding_ms)
            end_ms = int(seg_end * 1000) + self.padding_ms
            return start_ms, end_ms

        # Use the full segment so the clip covers the whole displayed sentence.
        start_ms = max(0, int(seg.start * 1000) - self.padding_ms)
        end_ms = int(seg.end * 1000) + self.padding_ms
        return start_ms, end_ms

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
