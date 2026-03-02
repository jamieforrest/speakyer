"""Tests for transcription package: base dataclasses, LocalWhisperTranscriber, TranscribeStage."""

import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from speakyer.database import db, init
from speakyer.pipeline.base import PipelineContext
from speakyer.pipeline.stages import TranscribeStage
from speakyer.sources.base import Episode, SourceConfig
from speakyer.storage import LocalStorage
from speakyer.transcription.base import Transcript, TranscriptSegment, Transcriber, WordTimestamp
from speakyer.transcription.whisper_local import LocalWhisperTranscriber

FAKE_SOURCE = SourceConfig(
    id=1,
    name="test",
    rss_url="https://example.com/feed.xml",
    language="de",
    transcript_strategy="whisper",
    active=True,
)

# A minimal mlx-whisper-style result dict.
FAKE_MLX_RESULT = {
    "text": " Guten Morgen Deutschland.",
    "language": "de",
    "segments": [
        {
            "start": 0.0,
            "end": 2.5,
            "text": " Guten Morgen Deutschland.",
            "words": [
                {"word": " Guten", "start": 0.0, "end": 0.8, "probability": 0.99},
                {"word": " Morgen", "start": 0.8, "end": 1.5, "probability": 0.98},
                {"word": " Deutschland.", "start": 1.5, "end": 2.5, "probability": 0.97},
            ],
        }
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def seed_episode(
    tmp_db: Path,
    audio_path: str | None = None,
    audio_url: str | None = "https://example.com/ep1.mp3",
) -> Episode:
    with db(tmp_db) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO sources (id, name, rss_url) VALUES (1, 'test', 'https://x.com')"
        )
        cursor = conn.execute(
            "INSERT INTO episodes (source_id, guid, title, audio_url, audio_path, status) "
            "VALUES (1, 'g1', 'Ep 1', ?, ?, 'downloaded')",
            (audio_url, audio_path),
        )
    return Episode(
        id=cursor.lastrowid,
        guid="g1",
        source_id=1,
        title="Ep 1",
        audio_url=audio_url,
        audio_path=audio_path,
    )


def make_fake_transcriber(transcript: Transcript | None = None) -> Transcriber:
    """Return a Transcriber that yields a fixed Transcript without calling mlx-whisper."""
    if transcript is None:
        transcript = LocalWhisperTranscriber("unused")._parse_result(FAKE_MLX_RESULT)

    class FakeTranscriber(Transcriber):
        def transcribe(self, audio_path: Path, language: str | None = None) -> Transcript:
            return transcript

    return FakeTranscriber()


def run_stage(
    ep: Episode,
    storage: LocalStorage,
    tmp_db: Path,
    *,
    transcriber: Transcriber | None = None,
    dry_run: bool = False,
    audio_path: Path | None = None,
) -> PipelineContext:
    ctx = PipelineContext(
        source_config=FAKE_SOURCE,
        episode=ep,
        dry_run=dry_run,
        audio_path=audio_path,
    )
    with (
        patch("speakyer.pipeline.stages.LocalStorage", return_value=storage),
        patch("speakyer.pipeline.stages.db", lambda: db(tmp_db)),
    ):
        return TranscribeStage(transcriber=transcriber or make_fake_transcriber()).run(ctx)


# ---------------------------------------------------------------------------
# Transcript dataclass tests
# ---------------------------------------------------------------------------


class TestTranscriptDataclasses:
    def test_word_timestamp_fields(self):
        w = WordTimestamp(word="Hallo", start=0.1, end=0.5, probability=0.95)
        assert w.word == "Hallo"
        assert w.probability == 0.95

    def test_transcript_segment_defaults_empty_words(self):
        seg = TranscriptSegment(index=0, text="Hallo", start=0.0, end=1.0)
        assert seg.words == []

    def test_to_dict_round_trip(self):
        original = LocalWhisperTranscriber("unused")._parse_result(FAKE_MLX_RESULT)
        restored = Transcript.from_dict(original.to_dict())
        assert restored.text == original.text
        assert restored.language == original.language
        assert restored.duration == original.duration
        assert len(restored.segments) == len(original.segments)
        seg = restored.segments[0]
        assert seg.text == "Guten Morgen Deutschland."
        assert len(seg.words) == 3
        assert seg.words[0].word == "Guten"
        assert seg.words[0].probability == pytest.approx(0.99)

    def test_from_dict_missing_probability_defaults_to_zero(self):
        data = {
            "text": "Test",
            "language": "de",
            "duration": 1.0,
            "segments": [
                {
                    "index": 0,
                    "text": "Test",
                    "start": 0.0,
                    "end": 1.0,
                    "words": [{"word": "Test", "start": 0.0, "end": 1.0}],
                }
            ],
        }
        t = Transcript.from_dict(data)
        assert t.segments[0].words[0].probability == 0.0


# ---------------------------------------------------------------------------
# LocalWhisperTranscriber._parse_result tests
# ---------------------------------------------------------------------------


class TestLocalWhisperTranscriber:
    def test_parse_result_produces_correct_transcript(self):
        transcriber = LocalWhisperTranscriber("unused-model")
        t = transcriber._parse_result(FAKE_MLX_RESULT)

        assert t.text == "Guten Morgen Deutschland."
        assert t.language == "de"
        assert t.duration == pytest.approx(2.5)
        assert len(t.segments) == 1

        seg = t.segments[0]
        assert seg.index == 0
        assert seg.text == "Guten Morgen Deutschland."
        assert seg.start == pytest.approx(0.0)
        assert seg.end == pytest.approx(2.5)
        assert len(seg.words) == 3

    def test_parse_result_strips_leading_space_from_words(self):
        transcriber = LocalWhisperTranscriber("unused-model")
        t = transcriber._parse_result(FAKE_MLX_RESULT)
        # mlx-whisper returns " Guten" with a leading space; we strip it
        assert t.segments[0].words[0].word == "Guten"

    def test_parse_result_empty_segments_gives_zero_duration(self):
        transcriber = LocalWhisperTranscriber("unused-model")
        t = transcriber._parse_result({"text": "", "language": None, "segments": []})
        assert t.duration == 0.0
        assert t.segments == []

    def test_parse_result_segment_without_words(self):
        transcriber = LocalWhisperTranscriber("unused-model")
        result = {
            "text": "Hallo",
            "language": "de",
            "segments": [{"start": 0.0, "end": 1.0, "text": " Hallo", "words": []}],
        }
        t = transcriber._parse_result(result)
        assert t.segments[0].words == []

    def test_transcribe_raises_when_mlx_unavailable(self, tmp_path):
        audio_file = tmp_path / "ep.mp3"
        audio_file.write_bytes(b"\x00")
        transcriber = LocalWhisperTranscriber("unused-model")
        with patch("speakyer.transcription.whisper_local._MLX_AVAILABLE", False):
            with pytest.raises(ImportError, match="mlx-whisper is not installed"):
                transcriber.transcribe(audio_file)


# ---------------------------------------------------------------------------
# TranscribeStage integration tests
# ---------------------------------------------------------------------------


class TestTranscribeStage:
    def test_transcribes_writes_json_and_db_records(self, tmp_db, tmp_storage, tmp_path):
        audio_file = tmp_path / "ep.mp3"
        audio_file.write_bytes(b"ID3" + b"\x00" * 100)

        ep = seed_episode(tmp_db)
        ctx = run_stage(ep, tmp_storage, tmp_db, audio_path=audio_file)

        # Context updated
        assert ctx.transcript is not None
        assert ctx.transcript.language == "de"
        assert len(ctx.transcript.segments) == 1

        # Raw JSON written to storage
        raw_path = f"transcripts/{ep.id}.json"
        assert tmp_storage.exists(raw_path)
        raw = json.loads(tmp_storage.read(raw_path).decode())
        assert raw["language"] == "de"
        assert len(raw["segments"]) == 1

        # DB: transcripts row
        conn = sqlite3.connect(str(tmp_db))
        row = conn.execute(
            "SELECT language_detected, duration_seconds, raw_json_path "
            "FROM transcripts WHERE episode_id = ?",
            (ep.id,),
        ).fetchone()
        assert row[0] == "de"
        assert row[1] == pytest.approx(2.5)
        assert row[2] == raw_path

        # DB: transcript_segments rows
        segments = conn.execute(
            "SELECT segment_index, text, start_time, end_time FROM transcript_segments "
            "WHERE transcript_id = (SELECT id FROM transcripts WHERE episode_id = ?)",
            (ep.id,),
        ).fetchall()
        assert len(segments) == 1
        assert segments[0][1] == "Guten Morgen Deutschland."

        # Episode status updated
        status = conn.execute(
            "SELECT status FROM episodes WHERE id = ?", (ep.id,)
        ).fetchone()[0]
        assert status == "transcribed"
        conn.close()

    def test_idempotent_skips_if_transcript_exists(self, tmp_db, tmp_storage, tmp_path):
        audio_file = tmp_path / "ep.mp3"
        audio_file.write_bytes(b"ID3")
        ep = seed_episode(tmp_db)

        # First run — inserts transcript.
        run_stage(ep, tmp_storage, tmp_db, audio_path=audio_file)

        # Second run — should skip; no duplicate rows.
        fake = make_fake_transcriber()
        call_count = [0]

        class CountingTranscriber(Transcriber):
            def transcribe(self, audio_path: Path, language: str | None = None) -> Transcript:
                call_count[0] += 1
                return fake.transcribe(audio_path, language)

        run_stage(ep, tmp_storage, tmp_db, transcriber=CountingTranscriber(), audio_path=audio_file)
        assert call_count[0] == 0

        conn = sqlite3.connect(str(tmp_db))
        count = conn.execute(
            "SELECT COUNT(*) FROM transcripts WHERE episode_id = ?", (ep.id,)
        ).fetchone()[0]
        conn.close()
        assert count == 1

    def test_dry_run_makes_no_writes(self, tmp_db, tmp_storage, tmp_path):
        audio_file = tmp_path / "ep.mp3"
        audio_file.write_bytes(b"ID3")
        ep = seed_episode(tmp_db)

        call_count = [0]

        class CountingTranscriber(Transcriber):
            def transcribe(self, audio_path: Path, language: str | None = None) -> Transcript:
                call_count[0] += 1
                return make_fake_transcriber().transcribe(audio_path, language)

        ctx = run_stage(
            ep, tmp_storage, tmp_db,
            transcriber=CountingTranscriber(),
            dry_run=True,
            audio_path=audio_file,
        )

        assert call_count[0] == 0
        assert ctx.transcript is None

        conn = sqlite3.connect(str(tmp_db))
        count = conn.execute(
            "SELECT COUNT(*) FROM transcripts WHERE episode_id = ?", (ep.id,)
        ).fetchone()[0]
        conn.close()
        assert count == 0

    def test_skips_when_audio_not_found(self, tmp_db, tmp_storage):
        # No audio_path set on ctx, no audio_path on episode either.
        ep = seed_episode(tmp_db, audio_path=None)
        call_count = [0]

        class CountingTranscriber(Transcriber):
            def transcribe(self, audio_path: Path, language: str | None = None) -> Transcript:
                call_count[0] += 1
                return make_fake_transcriber().transcribe(audio_path, language)

        ctx = run_stage(ep, tmp_storage, tmp_db, transcriber=CountingTranscriber())

        assert call_count[0] == 0
        assert ctx.transcript is None

    def test_uses_ctx_audio_path_over_episode_path(self, tmp_db, tmp_storage, tmp_path):
        """ctx.audio_path (set by DownloadStage) takes priority over ep.audio_path."""
        ctx_audio = tmp_path / "ctx_audio.mp3"
        ctx_audio.write_bytes(b"ID3")

        ep = seed_episode(tmp_db, audio_path="audio/some_other.mp3")

        received_paths = []

        class RecordingTranscriber(Transcriber):
            def transcribe(self, audio_path: Path, language: str | None = None) -> Transcript:
                received_paths.append(audio_path)
                return make_fake_transcriber().transcribe(audio_path, language)

        run_stage(ep, tmp_storage, tmp_db, transcriber=RecordingTranscriber(), audio_path=ctx_audio)
        assert len(received_paths) == 1
        assert received_paths[0] == ctx_audio
