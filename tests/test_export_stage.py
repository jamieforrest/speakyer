"""Tests for ExportStage and AnkiConnectExporter."""

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from speakyer.cards.anki_connect import AnkiConnectExporter
from speakyer.database import db
from speakyer.exporters.base import CardExporter
from speakyer.pipeline.base import PipelineContext
from speakyer.pipeline.stages import ExportStage
from speakyer.sources.base import Episode, SourceConfig

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


def seed_episode_with_cards(
    tmp_db: Path,
    words: list[tuple[str, str, str | None]],  # (surface, lemma, cefr)
    *,
    ep_status: str = "cards_pending",
    card_status: str = "pending",
) -> Episode:
    """Insert source, episode, word, and card rows. Returns the Episode."""
    with db(tmp_db) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO sources (id, name, rss_url) VALUES (1, 'tagesschau', 'https://x.com')"
        )
        ep_cur = conn.execute(
            "INSERT INTO episodes (source_id, guid, title, status) VALUES (1, 'g1', 'Ep 1', ?)",
            (ep_status,),
        )
        ep_id = ep_cur.lastrowid

        for surface, lemma, cefr in words:
            w_cur = conn.execute(
                "INSERT INTO words (episode_id, surface_form, lemma, pos, cefr_level, example_sentence) "
                "VALUES (?, ?, ?, 'NOUN', ?, 'Das ist ein Beispiel.')",
                (ep_id, surface, lemma, cefr),
            )
            conn.execute(
                "INSERT INTO cards (word_id, deck_name, status) VALUES (?, 'Speakyer', ?)",
                (w_cur.lastrowid, card_status),
            )

    return Episode(id=ep_id, guid="g1", source_id=1, title="Ep 1", status=ep_status)


class FakeExporter(CardExporter):
    """Exporter that records calls and returns fake note IDs."""

    def __init__(self, note_ids: list[int] | None = None):
        self.calls: list = []
        self._note_ids = note_ids  # None means auto-assign sequential IDs

    def export(self, cards):
        raise NotImplementedError

    def export_rows(self, rows: list) -> list[int]:
        self.calls.append(list(rows))
        if self._note_ids is not None:
            return self._note_ids
        return list(range(1001, 1001 + len(rows)))


def run_stage(
    ep: Episode,
    tmp_db: Path,
    *,
    dry_run: bool = False,
    exporter: CardExporter | None = None,
) -> PipelineContext:
    ctx = PipelineContext(source_config=FAKE_SOURCE, episode=ep, dry_run=dry_run)
    stage = ExportStage(exporter=exporter or FakeExporter())
    with patch("speakyer.pipeline.stages.db", lambda: db(tmp_db)):
        return stage.run(ctx)


# ---------------------------------------------------------------------------
# AnkiConnectExporter unit tests
# ---------------------------------------------------------------------------


class TestAnkiConnectExporterNoteBuilding:
    def _make_row(self, **kwargs):
        """Return a dict that quacks like sqlite3.Row."""
        defaults = {
            "card_id": 1,
            "word_id": 1,
            "deck_name": "Speakyer",
            "lemma": "welt",
            "surface_form": "Welt",
            "pos": "NOUN",
            "cefr_level": "A1",
            "example_sentence": "Die Welt ist groß.",
            "episode_title": "Ep 1",
            "published_at": None,
            "source_name": "tagesschau",
        }
        defaults.update(kwargs)
        return defaults

    def test_front_bolds_surface_form(self):
        row = self._make_row(surface_form="Welt", example_sentence="Die Welt ist groß.")
        exp = AnkiConnectExporter()
        front = exp._build_front(row)
        assert front == "Die <b>Welt</b> ist groß."

    def test_front_no_match_returns_sentence(self):
        row = self._make_row(surface_form="xyz", example_sentence="Die Welt ist groß.")
        front = AnkiConnectExporter._build_front(row)
        assert front == "Die Welt ist groß."

    def test_front_empty_sentence(self):
        row = self._make_row(example_sentence=None)
        front = AnkiConnectExporter._build_front(row)
        assert front == ""

    def test_back_includes_lemma_pos_cefr_source(self):
        row = self._make_row()
        back = AnkiConnectExporter._build_back(row)
        assert "welt" in back
        assert "NOUN" in back
        assert "A1" in back
        assert "tagesschau" in back

    def test_back_omits_cefr_when_none(self):
        row = self._make_row(cefr_level=None)
        back = AnkiConnectExporter._build_back(row)
        assert "CEFR" not in back

    def test_tags_include_speakyer_and_cefr(self):
        row = self._make_row(cefr_level="B1", source_name="tagesschau")
        tags = AnkiConnectExporter._build_tags(row)
        assert "speakyer" in tags
        assert "cefr::B1" in tags
        assert "podcast::tagesschau" in tags

    def test_tags_no_cefr_tag_when_none(self):
        row = self._make_row(cefr_level=None)
        tags = AnkiConnectExporter._build_tags(row)
        assert not any(t.startswith("cefr::") for t in tags)

    def test_tags_include_date(self):
        row = self._make_row(published_at="2026-02-28T10:00:00")
        tags = AnkiConnectExporter._build_tags(row)
        assert "date::2026-02-28" in tags

    def test_tags_no_date_tag_when_none(self):
        row = self._make_row(published_at=None)
        tags = AnkiConnectExporter._build_tags(row)
        assert not any(t.startswith("date::") for t in tags)

    def test_note_structure(self):
        row = self._make_row()
        exp = AnkiConnectExporter()
        note = exp._build_note(row)
        assert note["deckName"] == "Speakyer"
        assert note["modelName"] == "Basic"
        assert "Front" in note["fields"]
        assert "Back" in note["fields"]
        assert isinstance(note["tags"], list)
        assert note["options"]["allowDuplicate"] is True


class TestAnkiConnectExporterHTTP:
    def test_export_rows_sends_correct_payload(self):
        exp = AnkiConnectExporter(url="http://localhost:8765")
        row = {
            "card_id": 1, "word_id": 1, "deck_name": "Speakyer",
            "lemma": "welt", "surface_form": "Welt", "pos": "NOUN",
            "cefr_level": "A1", "example_sentence": "Die Welt ist groß.",
            "episode_title": "Ep 1", "published_at": None, "source_name": "tagesschau",
        }

        with patch("speakyer.cards.anki_connect.requests.post") as mock_post:
            mock_post.return_value.json.return_value = {"result": [42], "error": None}
            mock_post.return_value.raise_for_status = MagicMock()

            # Also mock createDeck call
            mock_post.return_value.json.side_effect = [
                {"result": "Speakyer", "error": None},  # createDeck
                {"result": [42], "error": None},         # addNotes
            ]

            note_ids = exp.export_rows([row])

        assert note_ids == [42]
        assert mock_post.call_count == 2  # createDeck + addNotes

    def test_connection_refused_raises_connection_error(self):
        import requests as _requests
        exp = AnkiConnectExporter(url="http://localhost:8765")
        row = {
            "card_id": 1, "word_id": 1, "deck_name": "Speakyer",
            "lemma": "welt", "surface_form": "Welt", "pos": "NOUN",
            "cefr_level": None, "example_sentence": "Welt.", "episode_title": "E",
            "published_at": None, "source_name": "src",
        }
        with patch(
            "speakyer.cards.anki_connect.requests.post",
            side_effect=_requests.exceptions.ConnectionError,
        ):
            with pytest.raises(ConnectionError, match="AnkiConnect"):
                exp.export_rows([row])

    def test_anki_error_raises_runtime_error(self):
        exp = AnkiConnectExporter(url="http://localhost:8765")
        row = {
            "card_id": 1, "word_id": 1, "deck_name": "Speakyer",
            "lemma": "welt", "surface_form": "Welt", "pos": "NOUN",
            "cefr_level": None, "example_sentence": "Welt.", "episode_title": "E",
            "published_at": None, "source_name": "src",
        }
        with patch("speakyer.cards.anki_connect.requests.post") as mock_post:
            mock_post.return_value.raise_for_status = MagicMock()
            mock_post.return_value.json.return_value = {
                "result": None, "error": "collection is not available"
            }
            with pytest.raises(RuntimeError, match="AnkiConnect error"):
                exp.export_rows([row])


# ---------------------------------------------------------------------------
# ExportStage integration tests
# ---------------------------------------------------------------------------


class TestExportStage:
    def test_exports_pending_cards_and_stores_note_ids(self, tmp_db):
        ep = seed_episode_with_cards(tmp_db, [("Welt", "welt", "A1"), ("Tag", "tag", "A1")])
        run_stage(ep, tmp_db, exporter=FakeExporter(note_ids=[1001, 1002]))

        conn = sqlite3.connect(str(tmp_db))
        rows = conn.execute("SELECT anki_note_id, status FROM cards ORDER BY id").fetchall()
        conn.close()
        assert rows[0] == (1001, "exported")
        assert rows[1] == (1002, "exported")

    def test_episode_status_updated_to_exported(self, tmp_db):
        ep = seed_episode_with_cards(tmp_db, [("Welt", "welt", "A1")])
        run_stage(ep, tmp_db)

        conn = sqlite3.connect(str(tmp_db))
        status = conn.execute("SELECT status FROM episodes WHERE id = ?", (ep.id,)).fetchone()[0]
        conn.close()
        assert status == "exported"

    def test_idempotent_skips_when_no_pending_cards(self, tmp_db):
        ep = seed_episode_with_cards(tmp_db, [("Welt", "welt", "A1")], card_status="exported")
        exporter = FakeExporter()
        run_stage(ep, tmp_db, exporter=exporter)

        assert exporter.calls == []  # exporter never called

    def test_dry_run_makes_no_writes(self, tmp_db):
        ep = seed_episode_with_cards(tmp_db, [("Welt", "welt", "A1")])
        exporter = FakeExporter()
        run_stage(ep, tmp_db, dry_run=True, exporter=exporter)

        assert exporter.calls == []

        conn = sqlite3.connect(str(tmp_db))
        status = conn.execute("SELECT status FROM cards").fetchone()[0]
        ep_status = conn.execute("SELECT status FROM episodes WHERE id = ?", (ep.id,)).fetchone()[0]
        conn.close()
        assert status == "pending"
        assert ep_status == "cards_pending"

    def test_anki_connection_error_is_handled_gracefully(self, tmp_db):
        ep = seed_episode_with_cards(tmp_db, [("Welt", "welt", "A1")])

        broken = FakeExporter()
        broken.export_rows = MagicMock(side_effect=ConnectionError("Anki not running"))

        # Should not raise — stage catches ConnectionError and prints it.
        ctx = run_stage(ep, tmp_db, exporter=broken)

        # Cards remain pending — nothing written.
        conn = sqlite3.connect(str(tmp_db))
        status = conn.execute("SELECT status FROM cards").fetchone()[0]
        conn.close()
        assert status == "pending"

    def test_rejected_notes_marked_failed(self, tmp_db):
        """If AnkiConnect returns 0 for a note (duplicate), mark it failed."""
        ep = seed_episode_with_cards(
            tmp_db,
            [("Welt", "welt", "A1"), ("Tag", "tag", "A1")],
        )
        # First note accepted, second rejected (duplicate in Anki).
        run_stage(ep, tmp_db, exporter=FakeExporter(note_ids=[1001, 0]))

        conn = sqlite3.connect(str(tmp_db))
        statuses = [r[0] for r in conn.execute("SELECT status FROM cards ORDER BY id").fetchall()]
        conn.close()
        assert statuses == ["exported", "failed"]

    def test_exported_at_is_set(self, tmp_db):
        ep = seed_episode_with_cards(tmp_db, [("Welt", "welt", "A1")])
        run_stage(ep, tmp_db)

        conn = sqlite3.connect(str(tmp_db))
        exported_at = conn.execute("SELECT exported_at FROM cards").fetchone()[0]
        conn.close()
        assert exported_at is not None
