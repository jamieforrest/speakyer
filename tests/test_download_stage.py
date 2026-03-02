"""Tests for DownloadStage."""

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from speakyer.database import db, init
from speakyer.pipeline.base import PipelineContext
from speakyer.pipeline.stages import DownloadStage
from speakyer.sources.base import Episode, SourceConfig
from speakyer.storage import LocalStorage

FAKE_SOURCE = SourceConfig(
    id=1,
    name="test",
    rss_url="https://example.com/feed.xml",
    language="de",
    transcript_strategy="whisper",
    active=True,
)

FAKE_AUDIO = b"ID3" + b"\x00" * 100


def seed_episode(tmp_db: Path, audio_url: str | None = "https://example.com/ep1.mp3") -> Episode:
    with db(tmp_db) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO sources (id, name, rss_url) VALUES (1, 'test', 'https://x.com')"
        )
        cursor = conn.execute(
            "INSERT INTO episodes (source_id, guid, title, audio_url, status) VALUES (1, 'g1', 'Ep 1', ?, 'fetched')",
            (audio_url,),
        )
    return Episode(id=cursor.lastrowid, guid="g1", source_id=1, title="Ep 1", audio_url=audio_url)


def run_stage(ep: Episode, storage: LocalStorage, tmp_db: Path, dry_run: bool = False) -> PipelineContext:
    """Run DownloadStage with patched config and db pointing at tmp fixtures."""
    ctx = PipelineContext(source_config=FAKE_SOURCE, episode=ep, dry_run=dry_run)

    with (
        patch("speakyer.pipeline.stages.LocalStorage", return_value=storage),
        patch("speakyer.pipeline.stages.db", lambda: db(tmp_db)),
    ):
        return DownloadStage().run(ctx)


class TestDownloadStage:
    def test_downloads_audio_writes_file_and_updates_db(self, tmp_db, tmp_storage):
        ep = seed_episode(tmp_db)
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_content.return_value = [FAKE_AUDIO]

        with patch("requests.get", return_value=mock_resp):
            ctx = run_stage(ep, tmp_storage, tmp_db)

        assert ctx.audio_path is not None
        assert ctx.audio_path.exists()
        assert ctx.audio_path.read_bytes() == FAKE_AUDIO

        # DB status updated
        conn = sqlite3.connect(str(tmp_db))
        row = conn.execute("SELECT status, audio_path FROM episodes WHERE id = ?", (ep.id,)).fetchone()
        conn.close()
        assert row[0] == "downloaded"
        assert row[1] == f"audio/{ep.id}.mp3"

    def test_skips_episode_without_audio_url(self, tmp_db, tmp_storage):
        ep = seed_episode(tmp_db, audio_url=None)
        with patch("requests.get") as mock_get:
            ctx = run_stage(ep, tmp_storage, tmp_db)
        mock_get.assert_not_called()
        assert ctx.audio_path is None

    def test_dry_run_does_not_download(self, tmp_db, tmp_storage):
        ep = seed_episode(tmp_db)
        with patch("requests.get") as mock_get:
            ctx = run_stage(ep, tmp_storage, tmp_db, dry_run=True)
        mock_get.assert_not_called()
        assert ctx.audio_path is None

    def test_skips_already_downloaded_episode(self, tmp_db, tmp_storage):
        ep = seed_episode(tmp_db)
        tmp_storage.write(f"audio/{ep.id}.mp3", FAKE_AUDIO)

        with patch("requests.get") as mock_get:
            ctx = run_stage(ep, tmp_storage, tmp_db)

        mock_get.assert_not_called()
        assert ctx.audio_path == tmp_storage.absolute_path(f"audio/{ep.id}.mp3")
