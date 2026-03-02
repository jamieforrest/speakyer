"""Tests for Speakyer CLI commands.

Uses Click's CliRunner to invoke commands against an in-memory temp database.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from speakyer.cli import main
from speakyer.database import db, init
from speakyer.sources.base import Episode, SourceConfig
from speakyer.storage import LocalStorage


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

FAKE_SOURCE = SourceConfig(
    id=1,
    name="tagesschau",
    rss_url="https://example.com/feed.xml",
    language="de",
    transcript_strategy="whisper",
    active=True,
)


def _seed_source(tmp_db: Path) -> None:
    with db(tmp_db) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO sources (id, name, rss_url, language, active) "
            "VALUES (1, 'tagesschau', 'https://example.com/feed.xml', 'de', 1)"
        )


def _seed_episode(tmp_db: Path, *, status: str = "exported", published_at=None) -> int:
    _seed_source(tmp_db)
    with db(tmp_db) as conn:
        ep_id = conn.execute(
            "INSERT INTO episodes (source_id, guid, title, published_at, audio_path, status) "
            "VALUES (1, 'g1', 'Test Episode', ?, 'audio/1.mp3', ?)",
            (published_at, status),
        ).lastrowid
    return ep_id


def _seed_full_chain(tmp_db: Path, *, card_status="exported", anki_note_id=99,
                     audio_clip_path=None, published_at=None) -> dict:
    """Seed source → episode → transcript → segment → word → card. Returns IDs."""
    _seed_source(tmp_db)
    with db(tmp_db) as conn:
        ep_id = conn.execute(
            "INSERT INTO episodes (source_id, guid, title, published_at, audio_path, status) "
            "VALUES (1, 'g1', 'Test Episode', ?, 'audio/1.mp3', 'exported')",
            (published_at,),
        ).lastrowid
        tr_id = conn.execute(
            "INSERT INTO transcripts (episode_id, raw_json_path, language_detected, duration_seconds) "
            "VALUES (?, 'transcripts/1.json', 'de', 10.0)",
            (ep_id,),
        ).lastrowid
        seg_id = conn.execute(
            "INSERT INTO transcript_segments (transcript_id, segment_index, text, start_time, end_time) "
            "VALUES (?, 0, 'Die Welt ist groß.', 0.0, 2.5)",
            (tr_id,),
        ).lastrowid
        w_id = conn.execute(
            "INSERT INTO words (episode_id, transcript_segment_id, surface_form, lemma, pos, "
            "cefr_level, example_sentence, start_time) VALUES (?, ?, 'Welt', 'welt', 'NOUN', "
            "'A1', 'Die Welt ist groß.', 0.5)",
            (ep_id, seg_id),
        ).lastrowid
        card_id = conn.execute(
            "INSERT INTO cards (word_id, deck_name, status, anki_note_id, audio_clip_path) "
            "VALUES (?, 'Speakyer', ?, ?, ?)",
            (w_id, card_status, anki_note_id, audio_clip_path),
        ).lastrowid
    return {"ep_id": ep_id, "tr_id": tr_id, "seg_id": seg_id, "w_id": w_id, "card_id": card_id}


@pytest.fixture
def runner():
    return CliRunner()


# ---------------------------------------------------------------------------
# speakyer audio-update
# ---------------------------------------------------------------------------


class TestAudioUpdateCLI:
    """Tests for the ``speakyer audio-update`` command."""

    def test_no_episodes_with_cards(self, runner, patched_config, tmp_db):
        """Exits cleanly when no episodes have cards."""
        result = runner.invoke(main, ["audio-update"])
        assert result.exit_code == 0
        assert "No episodes with cards found" in result.output

    def test_finds_episodes_and_runs_pipeline(self, runner, patched_config, tmp_db, tmp_path):
        """Reconstructs Episode/SourceConfig from SQL rows and runs stages."""
        ids = _seed_full_chain(tmp_db, card_status="exported", anki_note_id=99,
                               audio_clip_path=None)
        with patch("speakyer.pipeline.stages.AUDIO_UPDATE_PIPELINE") as mock_pipeline:
            mock_stage_cls = MagicMock()
            mock_stage = MagicMock()
            mock_stage.run.return_value = MagicMock()  # return a ctx
            mock_stage_cls.return_value = mock_stage
            mock_pipeline.__iter__ = lambda self: iter([mock_stage_cls])

            result = runner.invoke(main, ["audio-update"])

        assert result.exit_code == 0
        assert "Running audio-update pipeline" in result.output
        mock_stage.run.assert_called_once()
        # Verify the PipelineContext was built correctly
        ctx = mock_stage.run.call_args[0][0]
        assert ctx.episode.id == ids["ep_id"]
        assert ctx.episode.guid == "g1"
        assert ctx.source_config.name == "tagesschau"
        assert ctx.source_config.language == "de"

    def test_filters_by_source_name(self, runner, patched_config, tmp_db):
        """--source filters to matching source name."""
        _seed_full_chain(tmp_db, card_status="exported")
        result = runner.invoke(main, ["audio-update", "--source", "nonexistent"])
        assert result.exit_code == 0
        assert "No episodes with cards found" in result.output

    def test_filters_by_episode_id(self, runner, patched_config, tmp_db):
        """--episode-id filters to specific episode."""
        _seed_full_chain(tmp_db, card_status="exported")
        result = runner.invoke(main, ["audio-update", "--episode-id", "999"])
        assert result.exit_code == 0
        assert "No episodes with cards found" in result.output

    def test_dry_run_flag_propagated(self, runner, patched_config, tmp_db, tmp_path):
        """--dry-run flag is passed to PipelineContext."""
        _seed_full_chain(tmp_db, card_status="exported", audio_clip_path=None)

        with patch("speakyer.pipeline.stages.AUDIO_UPDATE_PIPELINE") as mock_pipeline:
            mock_stage_cls = MagicMock()
            mock_stage = MagicMock()
            mock_stage.run.return_value = MagicMock()
            mock_stage_cls.return_value = mock_stage
            mock_pipeline.__iter__ = lambda self: iter([mock_stage_cls])

            result = runner.invoke(main, ["audio-update", "--dry-run"])

        assert result.exit_code == 0
        ctx = mock_stage.run.call_args[0][0]
        assert ctx.dry_run is True

    def test_resync_tags_includes_audio_updated_status(self, runner, patched_config, tmp_db):
        """--resync-tags widens the status filter to include audio_updated cards."""
        _seed_full_chain(tmp_db, card_status="audio_updated", audio_clip_path="clips/1.mp3")

        with patch("speakyer.pipeline.stages.AUDIO_UPDATE_PIPELINE") as mock_pipeline:
            mock_stage_cls = MagicMock()
            mock_stage = MagicMock()
            mock_stage.run.return_value = MagicMock()
            mock_stage_cls.return_value = mock_stage
            mock_pipeline.__iter__ = lambda self: iter([mock_stage_cls])

            result = runner.invoke(main, ["audio-update", "--resync-tags"])

        assert result.exit_code == 0
        assert "Running audio-update pipeline" in result.output
        ctx = mock_stage.run.call_args[0][0]
        assert ctx.resync_tags is True

    def test_status_filter_excludes_audio_updated_without_resync(self, runner, patched_config, tmp_db):
        """Without --resync-tags, audio_updated cards are NOT selected."""
        _seed_full_chain(tmp_db, card_status="audio_updated", audio_clip_path="clips/1.mp3")
        result = runner.invoke(main, ["audio-update"])
        assert result.exit_code == 0
        assert "No episodes with cards found" in result.output


# ---------------------------------------------------------------------------
# speakyer run
# ---------------------------------------------------------------------------


class TestRunCLI:
    """Tests for the ``speakyer run`` command."""

    def test_no_sources_exits_with_message(self, runner, patched_config, tmp_db, tmp_path):
        """Exits cleanly when no sources.yaml exists."""
        result = runner.invoke(main, ["run"])
        assert result.exit_code == 1
        assert "No active sources" in result.output

    def test_source_not_found(self, runner, patched_config, tmp_db, tmp_path):
        """--source with nonexistent name exits with error."""
        # Create a sources.yaml with one source
        sources_yaml = tmp_path / "sources.yaml"
        sources_yaml.write_text(
            "sources:\n"
            "  - name: tagesschau\n"
            "    rss_url: https://example.com/feed.xml\n"
            "    language: de\n"
            "    active: true\n"
        )
        result = runner.invoke(main, ["run", "--source", "nonexistent"])
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_episode_id_not_found(self, runner, patched_config, tmp_db):
        """--episode-id with nonexistent ID exits with error."""
        result = runner.invoke(main, ["run", "--episode-id", "999"])
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_episode_id_runs_stages(self, runner, patched_config, tmp_db, tmp_path):
        """--episode-id fetches the episode and runs all stages on it."""
        _seed_source(tmp_db)
        ep_id = _seed_episode(tmp_db, status="fetched")

        with patch("speakyer.cli._load_stages") as mock_load:
            mock_stage = MagicMock()
            mock_stage.run.side_effect = lambda ctx: ctx
            mock_load.return_value = [mock_stage]

            result = runner.invoke(main, ["run", "--episode-id", str(ep_id)])

        assert result.exit_code == 0
        assert "Processing episode" in result.output
        mock_stage.run.assert_called_once()

    def test_since_flag_invalid_date(self, runner, patched_config, tmp_db):
        """--since with bad format exits with error."""
        result = runner.invoke(main, ["run", "--since", "not-a-date"])
        assert result.exit_code == 1
        assert "Invalid date format" in result.output

    def test_since_flag_filters_episodes(self, runner, patched_config, tmp_db, tmp_path):
        """--since filters out episodes published before the given date."""
        sources_yaml = tmp_path / "sources.yaml"
        sources_yaml.write_text(
            "sources:\n"
            "  - name: tagesschau\n"
            "    rss_url: https://example.com/feed.xml\n"
            "    language: de\n"
            "    active: true\n"
        )

        old_ep = Episode(
            guid="old1", source_id=1, title="Old",
            published_at=datetime(2025, 1, 1), audio_url="http://x/old.mp3",
        )
        new_ep = Episode(
            guid="new1", source_id=1, title="New",
            published_at=datetime(2026, 3, 1), audio_url="http://x/new.mp3",
        )

        with patch("speakyer.sources.rss.RSSPodcastSource") as mock_rss_cls:
            mock_rss = MagicMock()
            mock_rss.get_new_episodes.return_value = [old_ep, new_ep]
            mock_rss_cls.return_value = mock_rss

            with patch("speakyer.cli._load_stages") as mock_load:
                mock_stage = MagicMock()
                mock_stage.run.side_effect = lambda ctx: ctx
                mock_load.return_value = [mock_stage]

                result = runner.invoke(main, ["run", "--since", "2026-01-01", "--dry-run"])

        assert result.exit_code == 0
        # Only 1 new episode should be included (the old one is filtered)
        assert "1 new episode" in result.output

    def test_stage_flag_limits_stages(self, runner, patched_config, tmp_db, tmp_path):
        """--stage transcribe only loads stages up to and including transcribe."""
        ep_id = _seed_episode(tmp_db, status="fetched")

        with patch("speakyer.cli._load_stages") as mock_load:
            mock_stage = MagicMock()
            mock_stage.run.side_effect = lambda ctx: ctx
            mock_load.return_value = [mock_stage]

            result = runner.invoke(main, ["run", "--episode-id", str(ep_id), "--stage", "transcribe"])

        assert result.exit_code == 0
        # _load_stages should be called with ["download", "transcribe"]
        mock_load.assert_called_once_with(["download", "transcribe"])

    def test_whisper_model_override(self, runner, patched_config, tmp_db, tmp_path):
        """--whisper-model sets the config and echoes the model name."""
        result = runner.invoke(main, ["run", "--episode-id", "999",
                                       "--whisper-model", "mlx-community/whisper-small-mlx"])
        # Episode won't be found, but the model message should appear first
        assert "whisper-small-mlx" in result.output

    def test_dry_run_skips_inserts(self, runner, patched_config, tmp_db, tmp_path):
        """--dry-run does not insert new episodes into the database."""
        sources_yaml = tmp_path / "sources.yaml"
        sources_yaml.write_text(
            "sources:\n"
            "  - name: tagesschau\n"
            "    rss_url: https://example.com/feed.xml\n"
            "    language: de\n"
            "    active: true\n"
        )

        new_ep = Episode(
            guid="new1", source_id=1, title="New",
            published_at=datetime(2026, 3, 1), audio_url="http://x/new.mp3",
        )

        with patch("speakyer.sources.rss.RSSPodcastSource") as mock_rss_cls:
            mock_rss = MagicMock()
            mock_rss.get_new_episodes.return_value = [new_ep]
            mock_rss_cls.return_value = mock_rss

            with patch("speakyer.cli._load_stages") as mock_load:
                mock_stage = MagicMock()
                mock_stage.run.side_effect = lambda ctx: ctx
                mock_load.return_value = [mock_stage]

                result = runner.invoke(main, ["run", "--dry-run"])

        assert result.exit_code == 0
        # Verify no episode was inserted
        conn = sqlite3.connect(str(tmp_db))
        count = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        conn.close()
        assert count == 0


# ---------------------------------------------------------------------------
# speakyer fix-clips
# ---------------------------------------------------------------------------


class TestFixClipsCLI:
    def test_no_bad_clips(self, runner, patched_config, tmp_db):
        """Exits cleanly when no bad clips found."""
        result = runner.invoke(main, ["fix-clips"])
        assert result.exit_code == 0
        assert "No bad clips found" in result.output

    def test_purges_bad_clips(self, runner, patched_config, tmp_db, tmp_path):
        """Reports purged clips when hallucinated segments are found."""
        storage = LocalStorage(tmp_path / "data")
        storage.write("clips/1.mp3", b"BAD")

        # Seed a card with a hallucinated segment (5 words in 0.88s)
        with db(tmp_db) as conn:
            conn.execute("INSERT INTO sources (id, name, rss_url) VALUES (1, 'src', 'x')")
            ep_id = conn.execute(
                "INSERT INTO episodes (source_id, guid, title, status) "
                "VALUES (1, 'g', 'Ep', 'exported')"
            ).lastrowid
            tr_id = conn.execute(
                "INSERT INTO transcripts (episode_id, raw_json_path) VALUES (?, 'transcripts/1.json')",
                (ep_id,),
            ).lastrowid
            seg_id = conn.execute(
                "INSERT INTO transcript_segments (transcript_id, segment_index, text, start_time, end_time) "
                "VALUES (?, 0, 'Ausbildung ist eine große Herausforderung.', 5.0, 5.88)",
                (tr_id,),
            ).lastrowid
            w_id = conn.execute(
                "INSERT INTO words (episode_id, transcript_segment_id, surface_form, lemma, pos, "
                "example_sentence, start_time) VALUES (?, ?, 'Herausforderung', 'herausforderung', "
                "'NOUN', 'Ausbildung ist eine große Herausforderung.', 5.0)",
                (ep_id, seg_id),
            ).lastrowid
            conn.execute(
                "INSERT INTO cards (word_id, deck_name, status, audio_clip_path) "
                "VALUES (?, 'Speakyer', 'exported', 'clips/1.mp3')",
                (w_id,),
            )

        result = runner.invoke(main, ["fix-clips"])
        assert result.exit_code == 0
        assert "Removed 1 bad clip" in result.output


# ---------------------------------------------------------------------------
# speakyer episodes
# ---------------------------------------------------------------------------


class TestEpisodesCLI:
    def test_no_episodes(self, runner, patched_config, tmp_db):
        result = runner.invoke(main, ["episodes"])
        assert result.exit_code == 0
        assert "No episodes found" in result.output

    def test_lists_episodes(self, runner, patched_config, tmp_db):
        _seed_episode(tmp_db, status="exported", published_at=datetime(2026, 3, 1))
        result = runner.invoke(main, ["episodes"])
        assert result.exit_code == 0
        assert "tagesschau" in result.output
        assert "2026-03-01" in result.output
        assert "Test Episode" in result.output

    def test_published_at_string_handled(self, runner, patched_config, tmp_db):
        """published_at stored as a string (not datetime) doesn't crash."""
        _seed_source(tmp_db)
        with db(tmp_db) as conn:
            # Insert with raw string (bypassing the datetime adapter)
            conn.execute(
                "INSERT INTO episodes (source_id, guid, title, published_at, status) "
                "VALUES (1, 'gstr', 'String Date', '2026-03-01 10:00:00', 'fetched')"
            )
        result = runner.invoke(main, ["episodes"])
        assert result.exit_code == 0
        assert "2026-03-01" in result.output


# ---------------------------------------------------------------------------
# speakyer transcript
# ---------------------------------------------------------------------------


class TestTranscriptCLI:
    def test_episode_not_found(self, runner, patched_config, tmp_db):
        result = runner.invoke(main, ["transcript", "999"])
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_no_transcript(self, runner, patched_config, tmp_db):
        ep_id = _seed_episode(tmp_db)
        result = runner.invoke(main, ["transcript", str(ep_id)])
        assert result.exit_code == 1
        assert "No transcript found" in result.output

    def test_shows_segments(self, runner, patched_config, tmp_db):
        ids = _seed_full_chain(tmp_db)
        result = runner.invoke(main, ["transcript", str(ids["ep_id"])])
        assert result.exit_code == 0
        assert "Die Welt ist" in result.output


# ---------------------------------------------------------------------------
# speakyer words
# ---------------------------------------------------------------------------


class TestWordsCLI:
    def test_episode_not_found(self, runner, patched_config, tmp_db):
        result = runner.invoke(main, ["words", "999"])
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_no_words(self, runner, patched_config, tmp_db):
        ep_id = _seed_episode(tmp_db)
        result = runner.invoke(main, ["words", str(ep_id)])
        assert result.exit_code == 0
        assert "No words found" in result.output

    def test_shows_words_grouped_by_level(self, runner, patched_config, tmp_db):
        ids = _seed_full_chain(tmp_db)
        result = runner.invoke(main, ["words", str(ids["ep_id"])])
        assert result.exit_code == 0
        assert "A1" in result.output
        assert "Welt" in result.output

    def test_level_filter(self, runner, patched_config, tmp_db):
        ids = _seed_full_chain(tmp_db)
        result = runner.invoke(main, ["words", str(ids["ep_id"]), "--level", "B2"])
        assert result.exit_code == 0
        assert "No words found" in result.output


# ---------------------------------------------------------------------------
# speakyer cards
# ---------------------------------------------------------------------------


class TestCardsCLI:
    def test_episode_not_found(self, runner, patched_config, tmp_db):
        result = runner.invoke(main, ["cards", "999"])
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_no_cards(self, runner, patched_config, tmp_db):
        ep_id = _seed_episode(tmp_db, status="transcribed")
        result = runner.invoke(main, ["cards", str(ep_id)])
        assert result.exit_code == 0
        assert "No cards found" in result.output

    def test_shows_cards(self, runner, patched_config, tmp_db):
        ids = _seed_full_chain(tmp_db, card_status="pending")
        result = runner.invoke(main, ["cards", str(ids["ep_id"])])
        assert result.exit_code == 0
        assert "welt" in result.output
        assert "Front:" in result.output
        assert "Back:" in result.output

    def test_status_filter(self, runner, patched_config, tmp_db):
        ids = _seed_full_chain(tmp_db, card_status="exported")
        result = runner.invoke(main, ["cards", str(ids["ep_id"]), "--status", "pending"])
        assert result.exit_code == 0
        assert "No cards found" in result.output


# ---------------------------------------------------------------------------
# speakyer db
# ---------------------------------------------------------------------------


class TestDbCLI:
    def test_shows_tables(self, runner, patched_config, tmp_db):
        result = runner.invoke(main, ["db"])
        assert result.exit_code == 0
        assert "sources" in result.output
        assert "episodes" in result.output
        assert "cards" in result.output

    def test_shows_row_counts(self, runner, patched_config, tmp_db):
        _seed_source(tmp_db)
        result = runner.invoke(main, ["db"])
        assert result.exit_code == 0
        assert "1 row" in result.output


# ---------------------------------------------------------------------------
# speakyer seed-cefr
# ---------------------------------------------------------------------------


class TestSeedCefrCLI:
    def test_default_csv_seeding(self, runner, patched_config, tmp_db, tmp_path):
        """Seeds from a CSV file and reports counts."""
        csv_path = tmp_path / "cefr.csv"
        csv_path.write_text("lemma,level,pos\nwelt,A1,NOUN\nhaus,A2,NOUN\n")

        result = runner.invoke(main, ["seed-cefr", str(csv_path)])
        assert result.exit_code == 0
        assert "Seeded 2 CEFR words" in result.output
        assert "A1: 1" in result.output
        assert "A2: 1" in result.output

    def test_invalid_rows_skipped(self, runner, patched_config, tmp_db, tmp_path):
        csv_path = tmp_path / "cefr.csv"
        csv_path.write_text("lemma,level,pos\nwelt,A1,NOUN\n,INVALID,\n")

        result = runner.invoke(main, ["seed-cefr", str(csv_path)])
        assert result.exit_code == 0
        assert "Seeded 1 CEFR words (skipped 1 invalid" in result.output

    def test_csv_not_found(self, runner, patched_config, tmp_db):
        result = runner.invoke(main, ["seed-cefr", "/nonexistent/path.csv"])
        assert result.exit_code == 2  # Click reports missing path as usage error

    def test_idempotent_replace(self, runner, patched_config, tmp_db, tmp_path):
        """Running seed-cefr twice with same data doesn't duplicate rows."""
        csv_path = tmp_path / "cefr.csv"
        csv_path.write_text("lemma,level,pos\nwelt,A1,NOUN\n")

        runner.invoke(main, ["seed-cefr", str(csv_path)])
        result = runner.invoke(main, ["seed-cefr", str(csv_path)])
        assert result.exit_code == 0

        conn = sqlite3.connect(str(tmp_db))
        count = conn.execute("SELECT COUNT(*) FROM cefr_words").fetchone()[0]
        conn.close()
        assert count == 1
