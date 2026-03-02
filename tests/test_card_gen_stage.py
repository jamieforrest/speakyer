"""Tests for CardGenStage and the Card dataclass."""

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from speakyer.cards.base import Card
from speakyer.database import db
from speakyer.pipeline.base import PipelineContext
from speakyer.pipeline.stages import CardGenStage
from speakyer.sources.base import Episode, SourceConfig

FAKE_SOURCE = SourceConfig(
    id=1,
    name="test",
    rss_url="https://example.com/feed.xml",
    language="de",
    transcript_strategy="whisper",
    active=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def seed_episode_with_words(
    tmp_db: Path,
    words: list[tuple[str, str | None]],  # (lemma, cefr_level)
    *,
    status: str = "analyzed",
) -> Episode:
    """Insert source, episode, and word rows. Returns the Episode."""
    with db(tmp_db) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO sources (id, name, rss_url) VALUES (1, 'test', 'https://x.com')"
        )
        ep_cur = conn.execute(
            "INSERT INTO episodes (source_id, guid, title, status) VALUES (1, 'g1', 'Ep 1', ?)",
            (status,),
        )
        ep_id = ep_cur.lastrowid

        for lemma, cefr in words:
            conn.execute(
                "INSERT INTO words (episode_id, surface_form, lemma, pos, cefr_level, example_sentence) "
                "VALUES (?, ?, ?, 'NOUN', ?, 'Example.')",
                (ep_id, lemma.capitalize(), lemma, cefr),
            )

    return Episode(id=ep_id, guid="g1", source_id=1, title="Ep 1", status=status)


def run_stage(ep: Episode, tmp_db: Path, *, dry_run: bool = False) -> PipelineContext:
    ctx = PipelineContext(source_config=FAKE_SOURCE, episode=ep, dry_run=dry_run)
    with patch("speakyer.pipeline.stages.db", lambda: db(tmp_db)):
        return CardGenStage().run(ctx)


# ---------------------------------------------------------------------------
# Card dataclass
# ---------------------------------------------------------------------------


class TestCardDataclass:
    def test_defaults(self):
        c = Card(word_id=1, deck_name="Speakyer")
        assert c.status == "pending"
        assert c.id is None

    def test_fields(self):
        c = Card(word_id=42, deck_name="Speakyer", status="exported", id=7)
        assert c.word_id == 42
        assert c.deck_name == "Speakyer"
        assert c.status == "exported"
        assert c.id == 7


# ---------------------------------------------------------------------------
# CardGenStage
# ---------------------------------------------------------------------------


class TestCardGenStage:
    def test_creates_one_card_per_unique_lemma(self, tmp_db):
        ep = seed_episode_with_words(
            tmp_db,
            [("morgen", "A1"), ("nachricht", "B1"), ("morgen", "A1")],  # duplicate
        )
        ctx = run_stage(ep, tmp_db)

        assert len(ctx.cards) == 2  # morgen + nachricht (duplicate morgen skipped)

        conn = sqlite3.connect(str(tmp_db))
        rows = conn.execute(
            "SELECT c.id FROM cards c JOIN words w ON c.word_id = w.id WHERE w.episode_id = ?",
            (ep.id,),
        ).fetchall()
        conn.close()
        assert len(rows) == 2

    def test_episode_status_updated_to_cards_pending(self, tmp_db):
        ep = seed_episode_with_words(tmp_db, [("welt", "A1")])
        run_stage(ep, tmp_db)

        conn = sqlite3.connect(str(tmp_db))
        status = conn.execute(
            "SELECT status FROM episodes WHERE id = ?", (ep.id,)
        ).fetchone()[0]
        conn.close()
        assert status == "cards_pending"

    def test_deck_name_is_speakyer(self, tmp_db):
        ep = seed_episode_with_words(tmp_db, [("tag", "A1")])
        run_stage(ep, tmp_db)

        conn = sqlite3.connect(str(tmp_db))
        deck = conn.execute("SELECT deck_name FROM cards").fetchone()[0]
        conn.close()
        assert deck == "Speakyer"

    def test_card_status_defaults_to_pending(self, tmp_db):
        ep = seed_episode_with_words(tmp_db, [("land", "A1")])
        run_stage(ep, tmp_db)

        conn = sqlite3.connect(str(tmp_db))
        status = conn.execute("SELECT status FROM cards").fetchone()[0]
        conn.close()
        assert status == "pending"

    def test_globally_deduplicates_across_episodes(self, tmp_db):
        """A lemma that already has a card from a prior episode is skipped."""
        with db(tmp_db) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO sources (id, name, rss_url) VALUES (1, 'test', 'https://x.com')"
            )
            # Episode 1
            ep1_cur = conn.execute(
                "INSERT INTO episodes (source_id, guid, title, status) VALUES (1, 'g1', 'Ep 1', 'analyzed')"
            )
            ep1_id = ep1_cur.lastrowid
            w_cur = conn.execute(
                "INSERT INTO words (episode_id, surface_form, lemma, pos, example_sentence) "
                "VALUES (?, 'Morgen', 'morgen', 'NOUN', 'Example.')",
                (ep1_id,),
            )
            # Manually create a card for ep1's word.
            conn.execute("INSERT INTO cards (word_id, deck_name) VALUES (?, 'Speakyer')", (w_cur.lastrowid,))

            # Episode 2 — same lemma "morgen"
            ep2_cur = conn.execute(
                "INSERT INTO episodes (source_id, guid, title, status) VALUES (1, 'g2', 'Ep 2', 'analyzed')"
            )
            ep2_id = ep2_cur.lastrowid
            conn.execute(
                "INSERT INTO words (episode_id, surface_form, lemma, pos, example_sentence) "
                "VALUES (?, 'Morgen', 'morgen', 'NOUN', 'Another example.')",
                (ep2_id,),
            )

        ep2 = Episode(id=ep2_id, guid="g2", source_id=1, title="Ep 2", status="analyzed")
        ctx = run_stage(ep2, tmp_db)

        # morgen already has a card — nothing new should be created
        assert len(ctx.cards) == 0

        conn = sqlite3.connect(str(tmp_db))
        total_cards = conn.execute("SELECT COUNT(*) FROM cards").fetchone()[0]
        conn.close()
        assert total_cards == 1  # only the original from ep1

    def test_idempotent_skips_if_cards_exist(self, tmp_db):
        ep = seed_episode_with_words(tmp_db, [("haus", "A1")])
        run_stage(ep, tmp_db)  # first run

        ctx = run_stage(ep, tmp_db)  # second run — should skip
        assert ctx.cards == []

        conn = sqlite3.connect(str(tmp_db))
        count = conn.execute(
            "SELECT COUNT(*) FROM cards c JOIN words w ON c.word_id = w.id WHERE w.episode_id = ?",
            (ep.id,),
        ).fetchone()[0]
        conn.close()
        assert count == 1  # still just one card

    def test_dry_run_makes_no_writes(self, tmp_db):
        ep = seed_episode_with_words(tmp_db, [("wasser", "A1")])
        ctx = run_stage(ep, tmp_db, dry_run=True)

        assert ctx.cards == []

        conn = sqlite3.connect(str(tmp_db))
        card_count = conn.execute("SELECT COUNT(*) FROM cards").fetchone()[0]
        ep_status = conn.execute(
            "SELECT status FROM episodes WHERE id = ?", (ep.id,)
        ).fetchone()[0]
        conn.close()
        assert card_count == 0
        assert ep_status == "analyzed"  # unchanged

    def test_skips_when_no_words(self, tmp_db):
        """Episode with no words in the words table is silently skipped."""
        with db(tmp_db) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO sources (id, name, rss_url) VALUES (1, 'test', 'https://x.com')"
            )
            cur = conn.execute(
                "INSERT INTO episodes (source_id, guid, title, status) VALUES (1, 'gx', 'Empty', 'transcribed')"
            )
        ep = Episode(id=cur.lastrowid, guid="gx", source_id=1, title="Empty")

        ctx = run_stage(ep, tmp_db)
        assert ctx.cards == []

        conn = sqlite3.connect(str(tmp_db))
        count = conn.execute("SELECT COUNT(*) FROM cards").fetchone()[0]
        conn.close()
        assert count == 0

    def test_card_word_id_points_to_first_occurrence(self, tmp_db):
        """When a lemma appears twice, the card should reference the first word row."""
        with db(tmp_db) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO sources (id, name, rss_url) VALUES (1, 'test', 'https://x.com')"
            )
            ep_cur = conn.execute(
                "INSERT INTO episodes (source_id, guid, title, status) VALUES (1, 'g1', 'Ep', 'analyzed')"
            )
            ep_id = ep_cur.lastrowid
            first = conn.execute(
                "INSERT INTO words (episode_id, surface_form, lemma, pos, example_sentence) "
                "VALUES (?, 'Morgen', 'morgen', 'NOUN', 'First.')",
                (ep_id,),
            ).lastrowid
            conn.execute(
                "INSERT INTO words (episode_id, surface_form, lemma, pos, example_sentence) "
                "VALUES (?, 'Morgen', 'morgen', 'NOUN', 'Second.')",
                (ep_id,),
            )

        ep = Episode(id=ep_id, guid="g1", source_id=1, title="Ep", status="analyzed")
        run_stage(ep, tmp_db)

        conn = sqlite3.connect(str(tmp_db))
        word_id = conn.execute("SELECT word_id FROM cards").fetchone()[0]
        conn.close()
        assert word_id == first
