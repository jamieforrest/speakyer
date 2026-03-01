"""Tests for AudioClipper, AudioClipStage, CardUpdateStage, and purge_hallucinated_clips."""

import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from speakyer.database import db
from speakyer.nlp.audio_clipper import AudioClipper, purge_hallucinated_clips
from speakyer.pipeline.base import PipelineContext
from speakyer.cards.anki_connect import AnkiConnectExporter
from speakyer.pipeline.stages import AudioClipStage, CardUpdateStage
from speakyer.sources.base import Episode, SourceConfig
from speakyer.storage import LocalStorage
from speakyer.transcription.base import Transcript, TranscriptSegment, WordTimestamp

FAKE_SOURCE = SourceConfig(
    id=1,
    name="tagesschau",
    rss_url="https://example.com/feed.xml",
    language="de",
    transcript_strategy="whisper",
    active=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_transcript(seg_start=0.0, seg_end=2.5, words=None) -> Transcript:
    """Build a minimal Transcript with one segment."""
    if words is None:
        words = [
            WordTimestamp(word="Welt", start=0.5, end=1.0, probability=0.99),
        ]
    seg = TranscriptSegment(index=0, text="Die Welt ist groß.", start=seg_start, end=seg_end, words=words)
    return Transcript(text="Die Welt ist groß.", language="de", duration=seg_end, segments=[seg])


def seed_db(tmp_db: Path, *, card_status="exported", anki_note_id=99, audio_clip_path=None):
    """Seed a source, episode, transcript, segment, word, and card. Returns episode id."""
    with db(tmp_db) as conn:
        conn.execute("INSERT OR IGNORE INTO sources (id, name, rss_url) VALUES (1, 'tagesschau', 'x')")
        ep_id = conn.execute(
            "INSERT INTO episodes (source_id, guid, title, audio_path, status) "
            "VALUES (1, 'g1', 'Ep 1', 'audio/1.mp3', 'exported')"
        ).lastrowid
        tr_id = conn.execute(
            "INSERT INTO transcripts (episode_id, raw_json_path, language_detected, duration_seconds) "
            "VALUES (?, 'transcripts/1.json', 'de', 2.5)",
            (ep_id,),
        ).lastrowid
        seg_id = conn.execute(
            "INSERT INTO transcript_segments (transcript_id, segment_index, text, start_time, end_time) "
            "VALUES (?, 0, 'Die Welt ist groß.', 0.0, 2.5)",
            (tr_id,),
        ).lastrowid
        w_id = conn.execute(
            "INSERT INTO words (episode_id, transcript_segment_id, surface_form, lemma, pos, "
            "cefr_level, example_sentence, start_time) VALUES (?, ?, 'Welt', 'welt', 'NOUN', 'A1', "
            "'Die Welt ist groß.', 0.0)",
            (ep_id, seg_id),
        ).lastrowid
        conn.execute(
            "INSERT INTO cards (word_id, deck_name, status, anki_note_id, audio_clip_path) "
            "VALUES (?, 'Speakyer', ?, ?, ?)",
            (w_id, card_status, anki_note_id, audio_clip_path),
        )
    return ep_id


def make_storage_with_assets(tmp_path: Path, transcript: Transcript) -> LocalStorage:
    """Create a LocalStorage with a fake audio file and transcript JSON."""
    storage = LocalStorage(tmp_path / "data")
    storage.write("audio/1.mp3", b"FAKE_AUDIO")
    storage.write("transcripts/1.json", json.dumps(transcript.to_dict()).encode())
    return storage


# ---------------------------------------------------------------------------
# AudioClipper unit tests
# ---------------------------------------------------------------------------


class TestAudioClipperBoundaries:
    def setup_method(self):
        self.clipper = AudioClipper(padding_ms=100)

    def test_uses_full_segment_bounds(self):
        transcript = make_transcript(seg_start=0.0, seg_end=2.5)
        start_ms, end_ms = self.clipper._boundaries(transcript, 0.0, 2.5)
        # Segment: 0ms - 100 pad = 0 (clamped); 2500ms + 100 = 2600ms
        assert start_ms == 0
        assert end_ms == 2600

    def test_segment_not_found_uses_raw_times(self):
        transcript = make_transcript(seg_start=5.0, seg_end=7.0)
        # Ask for boundaries of a different segment that doesn't exist in transcript
        start_ms, end_ms = self.clipper._boundaries(transcript, 10.0, 12.0)
        assert start_ms == 9900   # 10000 - 100
        assert end_ms == 12100    # 12000 + 100

    def test_start_ms_clamped_to_zero(self):
        # Segment very close to start of audio
        transcript = make_transcript(seg_start=0.0, seg_end=1.0)
        start_ms, end_ms = self.clipper._boundaries(transcript, 0.0, 1.0)
        assert start_ms == 0  # clamped, not negative
        assert end_ms == 1100

    def test_segment_with_words_still_uses_segment_bounds(self):
        words = [WordTimestamp(word="Welt", start=0.5, end=1.0, probability=0.99)]
        transcript = make_transcript(seg_start=0.0, seg_end=2.5, words=words)
        start_ms, end_ms = self.clipper._boundaries(transcript, 0.0, 2.5)
        # Should use full segment, not word boundaries
        assert start_ms == 0
        assert end_ms == 2600


class TestAudioClipperExtract:
    def test_returns_none_when_segment_link_missing(self, tmp_db):
        """Words without a transcript_segment_id (NULL join) return None."""
        with db(tmp_db) as conn:
            conn.execute("INSERT INTO sources (id, name, rss_url) VALUES (1, 's', 'x')")
            ep_id = conn.execute(
                "INSERT INTO episodes (source_id, guid, title, audio_path, status) "
                "VALUES (1, 'g', 'E', 'audio/1.mp3', 'exported')"
            ).lastrowid
            w_id = conn.execute(
                "INSERT INTO words (episode_id, surface_form, lemma, pos, example_sentence) "
                "VALUES (?, 'Welt', 'welt', 'NOUN', 'x')",
                (ep_id,),
            ).lastrowid

        clipper = AudioClipper()
        with patch("speakyer.nlp.audio_clipper.db", lambda: db(tmp_db)):
            result = clipper.extract(w_id)
        assert result is None

    def test_returns_clip_path_on_success(self, tmp_db, tmp_path):
        transcript = make_transcript()
        storage = make_storage_with_assets(tmp_path, transcript)
        ep_id = seed_db(tmp_db)

        clipper = AudioClipper(storage=storage, padding_ms=100)

        mock_seg = MagicMock()
        mock_seg.__getitem__ = lambda s, k: [0, 500][["start_ms", "end_ms"].index(k)] if isinstance(k, str) else None
        mock_export = MagicMock()
        mock_export.read.return_value = b"CLIP"

        with patch("speakyer.nlp.audio_clipper.db", lambda: db(tmp_db)):
            with patch("speakyer.nlp.audio_clipper.AudioClipper._write_clip") as mock_write:
                mock_write.return_value = storage.absolute_path("clips/1.mp3")
                result = clipper.extract(1)  # word_id=1

        assert result is not None

    def test_idempotent_returns_existing_clip(self, tmp_db, tmp_path):
        """If the clip already exists, return it without re-extracting."""
        transcript = make_transcript()
        storage = make_storage_with_assets(tmp_path, transcript)
        storage.write("clips/1.mp3", b"EXISTING_CLIP")
        ep_id = seed_db(tmp_db)

        clipper = AudioClipper(storage=storage)
        with patch("speakyer.nlp.audio_clipper.db", lambda: db(tmp_db)):
            with patch.object(clipper, "_write_clip") as mock_write:
                result = clipper.extract(1)

        mock_write.assert_not_called()
        assert result == storage.absolute_path("clips/1.mp3")

    def test_returns_none_when_json_missing(self, tmp_db, tmp_path):
        storage = LocalStorage(tmp_path / "data")
        storage.write("audio/1.mp3", b"AUDIO")
        # No transcripts/1.json written
        ep_id = seed_db(tmp_db)

        clipper = AudioClipper(storage=storage)
        with patch("speakyer.nlp.audio_clipper.db", lambda: db(tmp_db)):
            result = clipper.extract(1)
        assert result is None

    def test_returns_none_when_audio_missing(self, tmp_db, tmp_path):
        transcript = make_transcript()
        storage = LocalStorage(tmp_path / "data")
        storage.write("transcripts/1.json", json.dumps(transcript.to_dict()).encode())
        # No audio/1.mp3 written
        ep_id = seed_db(tmp_db)

        clipper = AudioClipper(storage=storage)
        with patch("speakyer.nlp.audio_clipper.db", lambda: db(tmp_db)):
            result = clipper.extract(1)
        assert result is None

    def test_returns_none_for_implausibly_short_segment(self, tmp_db, tmp_path):
        """Segments shorter than 250ms/word are likely Whisper hallucinations."""
        with db(tmp_db) as conn:
            conn.execute("INSERT OR IGNORE INTO sources (id, name, rss_url) VALUES (1, 'tagesschau', 'x')")
            ep_id = conn.execute(
                "INSERT INTO episodes (source_id, guid, title, audio_path, status) "
                "VALUES (1, 'g2', 'Ep 2', 'audio/2.mp3', 'exported')"
            ).lastrowid
            tr_id = conn.execute(
                "INSERT INTO transcripts (episode_id, raw_json_path, language_detected, duration_seconds) "
                "VALUES (?, 'transcripts/2.json', 'de', 10.0)",
                (ep_id,),
            ).lastrowid
            # 5-word segment squeezed into 0.88s → 176ms/word < 250ms/word threshold
            seg_id = conn.execute(
                "INSERT INTO transcript_segments (transcript_id, segment_index, text, start_time, end_time) "
                "VALUES (?, 0, 'Ausbildung ist eine große Herausforderung.', 5.0, 5.88)",
                (tr_id,),
            ).lastrowid
            w_id = conn.execute(
                "INSERT INTO words (episode_id, transcript_segment_id, surface_form, lemma, pos, "
                "cefr_level, example_sentence, start_time) VALUES (?, ?, 'Herausforderung', "
                "'herausforderung', 'NOUN', 'B2', 'Ausbildung ist eine große Herausforderung.', 5.0)",
                (ep_id, seg_id),
            ).lastrowid

        transcript = make_transcript(seg_start=5.0, seg_end=5.88)
        storage = make_storage_with_assets(tmp_path / "short", transcript)
        # rewrite transcript path to match the episode
        storage.write("transcripts/2.json", json.dumps(transcript.to_dict()).encode())

        clipper = AudioClipper(storage=storage)
        with patch("speakyer.nlp.audio_clipper.db", lambda: db(tmp_db)):
            result = clipper.extract(w_id)
        assert result is None


# ---------------------------------------------------------------------------
# AudioClipStage tests
# ---------------------------------------------------------------------------


class TestAudioClipStage:
    def _run(self, ep_id, tmp_db, tmp_path, *, dry_run=False, clipper=None):
        ep = Episode(id=ep_id, guid="g1", source_id=1, title="Ep 1", status="exported")
        ctx = PipelineContext(source_config=FAKE_SOURCE, episode=ep, dry_run=dry_run)
        stage = AudioClipStage(clipper=clipper)
        with patch("speakyer.pipeline.stages.db", lambda: db(tmp_db)):
            return stage.run(ctx)

    def test_skips_when_no_cards_need_clips(self, tmp_db, tmp_path, capsys):
        ep_id = seed_db(tmp_db, card_status="exported", audio_clip_path="clips/1.mp3")
        self._run(ep_id, tmp_db, tmp_path)
        assert "[skip]" in capsys.readouterr().out

    def test_dry_run_makes_no_writes(self, tmp_db, tmp_path):
        ep_id = seed_db(tmp_db, card_status="pending", audio_clip_path=None)
        mock_clipper = MagicMock()
        self._run(ep_id, tmp_db, tmp_path, dry_run=True, clipper=mock_clipper)
        mock_clipper.extract.assert_not_called()

        conn = sqlite3.connect(str(tmp_db))
        clip_path = conn.execute("SELECT audio_clip_path FROM cards").fetchone()[0]
        conn.close()
        assert clip_path is None

    def test_stores_relative_clip_path_on_success(self, tmp_db, tmp_path):
        ep_id = seed_db(tmp_db, card_status="pending", audio_clip_path=None)
        storage = LocalStorage(tmp_path / "data")

        mock_clipper = MagicMock()
        mock_clipper.storage.base_dir = storage.base_dir
        mock_clipper.extract.return_value = storage.base_dir / "clips" / "1.mp3"

        self._run(ep_id, tmp_db, tmp_path, clipper=mock_clipper)

        conn = sqlite3.connect(str(tmp_db))
        clip_path = conn.execute("SELECT audio_clip_path FROM cards").fetchone()[0]
        conn.close()
        assert clip_path == "clips/1.mp3"

    def test_skips_card_when_clipper_returns_none(self, tmp_db, tmp_path, capsys):
        ep_id = seed_db(tmp_db, card_status="pending", audio_clip_path=None)

        mock_clipper = MagicMock()
        mock_clipper.extract.return_value = None

        self._run(ep_id, tmp_db, tmp_path, clipper=mock_clipper)

        out = capsys.readouterr().out
        assert "skipped" in out

        conn = sqlite3.connect(str(tmp_db))
        clip_path = conn.execute("SELECT audio_clip_path FROM cards").fetchone()[0]
        conn.close()
        assert clip_path is None


# ---------------------------------------------------------------------------
# CardUpdateStage tests
# ---------------------------------------------------------------------------


class TestCardUpdateStage:
    def _run(self, ep_id, tmp_db, *, dry_run=False, url="http://localhost:8765"):
        ep = Episode(id=ep_id, guid="g1", source_id=1, title="Ep 1", status="exported")
        ctx = PipelineContext(source_config=FAKE_SOURCE, episode=ep, dry_run=dry_run)
        stage = CardUpdateStage(url=url)
        with patch("speakyer.pipeline.stages.db", lambda: db(tmp_db)):
            return stage.run(ctx)

    def test_skips_when_no_exported_cards_with_clips(self, tmp_db, capsys):
        ep_id = seed_db(tmp_db, card_status="exported", anki_note_id=99, audio_clip_path=None)
        self._run(ep_id, tmp_db)
        assert "[skip]" in capsys.readouterr().out

    def test_dry_run_makes_no_calls(self, tmp_db, tmp_path, capsys):
        ep_id = seed_db(tmp_db, card_status="exported", anki_note_id=99, audio_clip_path="clips/1.mp3")
        with patch("speakyer.pipeline.stages._invoke") as mock_invoke:
            self._run(ep_id, tmp_db, dry_run=True)
        mock_invoke.assert_not_called()

    def test_calls_updateNote_and_sets_status(self, tmp_db, tmp_path):
        ep_id = seed_db(tmp_db, card_status="exported", anki_note_id=99, audio_clip_path="clips/1.mp3")

        with patch("speakyer.pipeline.stages._invoke") as mock_invoke:
            # modelNames returns the model so createModel is skipped.
            mock_invoke.return_value = [AnkiConnectExporter.MODEL_NAME]
            with patch("speakyer.pipeline.stages.LocalStorage") as mock_storage_cls:
                mock_storage = MagicMock()
                mock_storage.absolute_path.return_value = Path("/data/clips/1.mp3")
                mock_storage_cls.return_value = mock_storage
                self._run(ep_id, tmp_db)

        actions = [call[0][0] for call in mock_invoke.call_args_list]
        assert "updateNote" in actions
        update_call = next(c for c in mock_invoke.call_args_list if c[0][0] == "updateNote")
        note = update_call[1]["note"]
        assert note["id"] == 99
        assert "audio" in note
        assert note["audio"][0]["filename"] == "speakyer_99.mp3"
        assert "tags" in note
        assert "speakyer" in note["tags"]

        conn = sqlite3.connect(str(tmp_db))
        status = conn.execute("SELECT status FROM cards").fetchone()[0]
        conn.close()
        assert status == "audio_updated"

    def test_connection_error_aborts_gracefully(self, tmp_db, tmp_path, capsys):
        ep_id = seed_db(tmp_db, card_status="exported", anki_note_id=99, audio_clip_path="clips/1.mp3")

        def _invoke_side_effect(action, *args, **kwargs):
            if action == "modelNames":
                return [AnkiConnectExporter.MODEL_NAME]
            raise ConnectionError("Anki down")

        with patch("speakyer.pipeline.stages._invoke", side_effect=_invoke_side_effect):
            with patch("speakyer.pipeline.stages.LocalStorage") as mock_storage_cls:
                mock_storage = MagicMock()
                mock_storage.absolute_path.return_value = Path("/data/clips/1.mp3")
                mock_storage_cls.return_value = mock_storage
                self._run(ep_id, tmp_db)

        out = capsys.readouterr().out
        assert "[error]" in out

        conn = sqlite3.connect(str(tmp_db))
        status = conn.execute("SELECT status FROM cards").fetchone()[0]
        conn.close()
        assert status == "exported"  # unchanged on failure


# ---------------------------------------------------------------------------
# _build_note audio attachment test
# ---------------------------------------------------------------------------


class TestBuildNoteAudioAttachment:
    def _make_row(self, **kwargs):
        defaults = {
            "card_id": 1, "word_id": 7, "deck_name": "Speakyer",
            "lemma": "welt", "surface_form": "Welt", "pos": "NOUN",
            "cefr_level": "A1", "example_sentence": "Die Welt ist groß.",
            "episode_title": "Ep 1", "published_at": None, "source_name": "tagesschau",
            "audio_clip_path": None,
        }
        defaults.update(kwargs)
        return defaults

    def test_no_audio_key_when_no_clip(self):
        from speakyer.cards.anki_connect import AnkiConnectExporter
        row = self._make_row()
        note = AnkiConnectExporter()._build_note(row)
        assert "audio" not in note

    def test_audio_key_present_when_clip_path_provided(self):
        from speakyer.cards.anki_connect import AnkiConnectExporter
        row = self._make_row(audio_clip_abs_path="/data/clips/7.mp3")
        note = AnkiConnectExporter()._build_note(row)
        assert "audio" in note
        assert note["audio"][0]["path"] == "/data/clips/7.mp3"
        assert note["audio"][0]["filename"] == "speakyer_7.mp3"
        assert note["audio"][0]["fields"] == ["Front"]


# ---------------------------------------------------------------------------
# purge_hallucinated_clips tests
# ---------------------------------------------------------------------------


def _seed_card_with_segment(
    tmp_db: Path,
    tmp_path: Path,
    *,
    seg_text: str,
    seg_start: float,
    seg_end: float,
    clip_path: str | None = "clips/1.mp3",
) -> int:
    """Seed a card linked to a segment with the given timing. Returns card id."""
    with db(tmp_db) as conn:
        conn.execute("INSERT OR IGNORE INTO sources (id, name, rss_url) VALUES (1, 'src', 'x')")
        ep_id = conn.execute(
            "INSERT INTO episodes (source_id, guid, title, audio_path, status) "
            "VALUES (1, 'g-purge', 'Ep', 'audio/1.mp3', 'exported')"
        ).lastrowid
        tr_id = conn.execute(
            "INSERT INTO transcripts (episode_id, raw_json_path) VALUES (?, 'transcripts/1.json')",
            (ep_id,),
        ).lastrowid
        seg_id = conn.execute(
            "INSERT INTO transcript_segments (transcript_id, segment_index, text, start_time, end_time) "
            "VALUES (?, 0, ?, ?, ?)",
            (tr_id, seg_text, seg_start, seg_end),
        ).lastrowid
        w_id = conn.execute(
            "INSERT INTO words (episode_id, transcript_segment_id, surface_form, lemma, pos, "
            "example_sentence, start_time) VALUES (?, ?, 'Welt', 'welt', 'NOUN', ?, ?)",
            (ep_id, seg_id, seg_text, seg_start),
        ).lastrowid
        card_id = conn.execute(
            "INSERT INTO cards (word_id, deck_name, status, audio_clip_path) VALUES (?, 'Speakyer', 'exported', ?)",
            (w_id, clip_path),
        ).lastrowid
    return card_id


class TestPurgeHallucinatedClips:
    def test_purges_card_with_short_segment(self, tmp_db, tmp_path):
        """A 5-word segment in 0.88s (176ms/word) should be purged."""
        storage = LocalStorage(tmp_path / "data")
        storage.write("clips/1.mp3", b"BAD_CLIP")

        card_id = _seed_card_with_segment(
            tmp_db, tmp_path,
            seg_text="Ausbildung ist eine große Herausforderung.",
            seg_start=5.0, seg_end=5.88,
            clip_path="clips/1.mp3",
        )

        with patch("speakyer.nlp.audio_clipper.db", lambda: db(tmp_db)):
            purged = purge_hallucinated_clips(storage=storage)

        assert len(purged) == 1
        assert purged[0]["card_id"] == card_id
        assert not storage.exists("clips/1.mp3")

        # DB should have audio_clip_path cleared and status reset to 'exported'.
        with db(tmp_db) as conn:
            row = conn.execute("SELECT audio_clip_path, status FROM cards WHERE id = ?", (card_id,)).fetchone()
        assert row["audio_clip_path"] is None
        assert row["status"] == "exported"

    def test_leaves_good_clips_alone(self, tmp_db, tmp_path):
        """A 4-word segment in 2.5s (625ms/word) should NOT be purged."""
        storage = LocalStorage(tmp_path / "data")
        storage.write("clips/1.mp3", b"GOOD_CLIP")

        card_id = _seed_card_with_segment(
            tmp_db, tmp_path,
            seg_text="Die Welt ist groß.",
            seg_start=0.0, seg_end=2.5,
            clip_path="clips/1.mp3",
        )

        with patch("speakyer.nlp.audio_clipper.db", lambda: db(tmp_db)):
            purged = purge_hallucinated_clips(storage=storage)

        assert len(purged) == 0
        assert storage.exists("clips/1.mp3")

        with db(tmp_db) as conn:
            row = conn.execute("SELECT audio_clip_path FROM cards WHERE id = ?", (card_id,)).fetchone()
        assert row["audio_clip_path"] == "clips/1.mp3"

    def test_handles_missing_clip_file_gracefully(self, tmp_db, tmp_path):
        """If the clip file is already gone from disk, still clear the DB."""
        storage = LocalStorage(tmp_path / "data")
        # Don't create the file — simulate it already being deleted.

        card_id = _seed_card_with_segment(
            tmp_db, tmp_path,
            seg_text="Ausbildung ist eine große Herausforderung.",
            seg_start=5.0, seg_end=5.88,
            clip_path="clips/1.mp3",
        )

        with patch("speakyer.nlp.audio_clipper.db", lambda: db(tmp_db)):
            purged = purge_hallucinated_clips(storage=storage)

        assert len(purged) == 1

        with db(tmp_db) as conn:
            row = conn.execute("SELECT audio_clip_path FROM cards WHERE id = ?", (card_id,)).fetchone()
        assert row["audio_clip_path"] is None

    def test_skips_cards_without_clips(self, tmp_db, tmp_path):
        """Cards with audio_clip_path = NULL are not touched."""
        storage = LocalStorage(tmp_path / "data")

        _seed_card_with_segment(
            tmp_db, tmp_path,
            seg_text="Ausbildung ist eine große Herausforderung.",
            seg_start=5.0, seg_end=5.88,
            clip_path=None,
        )

        with patch("speakyer.nlp.audio_clipper.db", lambda: db(tmp_db)):
            purged = purge_hallucinated_clips(storage=storage)

        assert len(purged) == 0
